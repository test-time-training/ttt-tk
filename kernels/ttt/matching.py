import torch
import test_time_training
import multiprocessing
import time


def gelu_bwd(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff


def compare_outputs(output_hw, output_ref, name):
    abs_diff = torch.abs(output_hw - output_ref)
    max_diff = torch.max(torch.abs(output_hw - output_ref)).item()
    median_diff = torch.median(torch.abs(output_hw - output_ref)).item()
    
    # Avoid division by zero and calculate relative absolute error
    with torch.no_grad():
        nonzero_mask = output_ref != 0
        relative_error = torch.zeros_like(output_ref)
        relative_error[nonzero_mask] = abs_diff[nonzero_mask] / torch.abs(output_ref[nonzero_mask])
        max_relative_error = torch.max(relative_error).item()
        median_relative_error = torch.median(relative_error).item()

    print(f"{name} - Max Difference: {max_diff}, Median Difference: {median_diff}, "
          f"Max Relative Error: {max_relative_error}, Median Relative Error: {median_relative_error}")


def compute_mini_batch_no_dual(
    W1, 
    b1, 
    W2, 
    b2, 
    xq_mb, 
    xk_mb, 
    xv_mb, 
    eta_mb,
    ttt_norm_weight,
    ttt_norm_bias,
):
    """
    Mini batch forward for TTT MLP.

    xq_mb: [CS, F]
    xk_mb: [CS, F]
    xv_mb: [CS, F]
    W1: [F, K]
    b1: [1, K]
    W2: [K, F]
    b2: [1, F]

    Dimension Key:
    B: Batch size
    H: Num of heads
    CS: Mini-batch size
    F: Head dimension
    K: Expansion dimension
    """
    num_heads = xk_mb.shape[1]
    head_dim = xk_mb.shape[-1]

    # Inner model forward
    Z1 = xk_mb @ W1 + b1
    X2 = torch.nn.functional.gelu(Z1, approximate="tanh")
    Z2 = X2 @ W2 + b2

    reconstruction_target = xv_mb - xk_mb

    ln_weight = ttt_norm_weight.reshape(num_heads, 1, head_dim)
    ln_bias = ttt_norm_bias.reshape(num_heads, 1, head_dim)

    # Stage 2: LnFusedL2BWD

    eps = 1e-8
    mu_fused = Z2.mean(dim=-1, keepdim=True)
    var_fused = Z2.var(dim=-1, keepdim=True, unbiased=False)

    std_fused = torch.sqrt(var_fused + eps)
    x_hat_fused = (Z2 - mu_fused) / std_fused

    y = ln_weight * x_hat_fused + ln_bias
    grad_output_fused = y - reconstruction_target
    grad_x_hat_fused = grad_output_fused * ln_weight

    grad_l_wrt_Z2 = (
        (1.0 / head_dim)
        * (
            head_dim * grad_x_hat_fused
            - grad_x_hat_fused.sum(dim=-1, keepdim=True)
            - x_hat_fused * (grad_x_hat_fused * x_hat_fused).sum(dim=-1, keepdim=True)
        )
        / std_fused
    )

    # Gradient calculation
    # grad_l_wrt_Z2 = xv_mb - Z2
    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2.transpose(-1,-2) * gelu_bwd(Z1)

    # Weight updates
    last_eta_mini_batch = eta_mb[:, :, -1, :, None]

    W1_next = W1 - (last_eta_mini_batch * xk_mb).transpose(-1,-2) @ grad_l_wrt_Z1
    b1_next = b1 - (last_eta_mini_batch * grad_l_wrt_Z1).sum(dim=-2, keepdim=True)

    W2_next = W2 - (last_eta_mini_batch * X2).transpose(-1,-2) @ grad_l_wrt_Z2
    b2_next = b2 - (last_eta_mini_batch * grad_l_wrt_Z2).sum(dim=-2, keepdim=True)

    # Post grad forward
    Z1_bar = xq_mb @ W1_next + b1_next
    X2_bar = torch.nn.functional.gelu(Z1_bar, approximate="tanh")
    Z2_bar = X2_bar @ W2_next + b2_next

    # Ln
    mu_ln = Z2_bar.mean(dim=-1, keepdim=True)
    var_ln = Z2_bar.var(dim=-1, keepdim=True, unbiased=False)
    std_ln = torch.sqrt(var_ln + eps)
    x_hat_ln = (Z2_bar - mu_ln) / std_ln

    Z2_bar_ln = ln_weight * x_hat_ln + ln_bias

    # Residual
    XQW_mini_batch = xq_mb + Z2_bar_ln

    return XQW_mini_batch, W1_next, b1_next, W2_next, b2_next




def main():
    torch.manual_seed(0)
    # Define shapes
    B = 1
    NH = 48
    K = 4
    
    seq_len = 2048
    mini_batch_size = 64
    NC = seq_len // mini_batch_size
    checkpoint_group_size = NC // K

    head_dim = 64
    expansion_dim = 256
    shard_size = 4

    dtype = torch.bfloat16
    full_dtype = torch.float32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_inputs(dtype):
        torch.manual_seed(0)
        # Create inputs
        xq = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
        xk = torch.randn(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
        xv = torch.ones(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()

        eta = torch.randn(B, NH, NC, mini_batch_size, mini_batch_size, dtype=dtype, device=device).contiguous() * 0.02
        last_eta = eta[:, :, :, -1, :].contiguous()

        ttt_norm_weight = torch.ones(1, NH, 1, head_dim, dtype=dtype, device=device).contiguous()
        ttt_norm_bias = torch.randn(1, NH, 1, head_dim, dtype=dtype, device=device).contiguous() * 0.02

        W1 = torch.randn(B, NH, head_dim, expansion_dim, dtype=dtype, device=device).contiguous() * 0.02
        b1 = torch.randn(B, NH, 1, expansion_dim, dtype=dtype, device=device).contiguous() * 0.02
        W2 = torch.randn(B, NH, expansion_dim, head_dim, dtype=dtype, device=device).contiguous() * 0.02
        b2 = torch.randn(B, NH, 1, head_dim, dtype=dtype, device=device).contiguous() * 0.02

        return xq, xk, xv, eta, last_eta, ttt_norm_weight, ttt_norm_bias, W1, b1, W2, b2

    W1_checkpoints = torch.empty(B, NH, K, head_dim, expansion_dim, dtype=full_dtype, device=device).contiguous()
    b1_checkpoints = torch.empty(B, NH, K, 1, expansion_dim, dtype=full_dtype, device=device).contiguous()
    W2_checkpoints = torch.empty(B, NH, K, expansion_dim, head_dim, dtype=full_dtype, device=device).contiguous()
    b2_checkpoints = torch.empty(B, NH, K, 1, head_dim, dtype=full_dtype, device=device).contiguous()

    W1_checkpoints_ref = torch.empty(B, NH, K, head_dim, expansion_dim, dtype=full_dtype, device=device).contiguous()
    b1_checkpoints_ref = torch.empty(B, NH, K, 1, expansion_dim, dtype=full_dtype, device=device).contiguous()
    W2_checkpoints_ref = torch.empty(B, NH, K, expansion_dim, head_dim, dtype=full_dtype, device=device).contiguous()
    b2_checkpoints_ref = torch.empty(B, NH, K, 1, head_dim, dtype=full_dtype, device=device).contiguous()
    W1_checkpoints_ref_bf = torch.empty(B, NH, K, head_dim, expansion_dim, dtype=dtype, device=device).contiguous()
    b1_checkpoints_ref_bf = torch.empty(B, NH, K, 1, expansion_dim, dtype=dtype, device=device).contiguous()
    W2_checkpoints_ref_bf = torch.empty(B, NH, K, expansion_dim, head_dim, dtype=dtype, device=device).contiguous()
    b2_checkpoints_ref_bf = torch.empty(B, NH, K, 1, head_dim, dtype=dtype, device=device).contiguous()

    # Create output buffers
    output_ref = torch.zeros(B, NH, NC, mini_batch_size, head_dim, dtype=full_dtype, device=device).contiguous()
    output_ref_bf = torch.zeros(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
    Z2_bar_pt_shard = torch.zeros(NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()
    output_tk = torch.zeros(B, NH, NC, mini_batch_size, head_dim, dtype=dtype, device=device).contiguous()

    xq, xk, xv, eta, last_eta, ttt_norm_weight, ttt_norm_bias, W1, b1, W2, b2 = get_inputs(torch.float32)

    xq = xq.to(torch.bfloat16).contiguous()
    xk = xk.to(torch.bfloat16).contiguous()
    xv = xv.to(torch.bfloat16).contiguous()
    last_eta = last_eta.to(torch.bfloat16)
    eta = eta.to(torch.bfloat16)

    test_time_training.ttt_forward(
        xq,
        xk,
        xv,
        last_eta,
        ttt_norm_weight,
        ttt_norm_bias,
        W1,
        b1,
        W2,
        b2,
        W1_checkpoints,
        b1_checkpoints,
        W2_checkpoints,
        b2_checkpoints,
        output_tk,
        checkpoint_group_size
    )

    xq, xk, xv, eta, last_eta, ttt_norm_weight, ttt_norm_bias, W1, b1, W2, b2 = get_inputs(torch.float32)

    W1_curr, b1_curr, W2_curr, b2_curr = (W1, b1, W2, b2)

    # Compute mini-batches for PyTorch
    for i in range(NC):
        if i % checkpoint_group_size == 0:
            checkpoint_idx = i // checkpoint_group_size
            W1_checkpoints_ref[:, :, checkpoint_idx] = W1_curr
            b1_checkpoints_ref[:, :, checkpoint_idx] = b1_curr
            W2_checkpoints_ref[:, :, checkpoint_idx] = W2_curr
            b2_checkpoints_ref[:, :, checkpoint_idx] = b2_curr

        xq_mb = xq[:,:,i]
        xk_mb = xk[:,:,i]
        xv_mb = xv[:,:,i]
        eta_mb = eta[:, :, i]

        (
            output_ref[:, :, i],
            W1_curr,
            b1_curr,
            W2_curr,
            b2_curr
        ) = compute_mini_batch_no_dual(
            W1_curr, 
            b1_curr, 
            W2_curr, 
            b2_curr, 
            xq_mb, 
            xk_mb, 
            xv_mb, 
            eta_mb,
            ttt_norm_weight,
            ttt_norm_bias
        )

    # Compare outputs
    print("Comparing Outputs")
    compare_outputs(output_tk, output_ref, "Output")

    compare_outputs(W1_checkpoints[:, :, -1], W1_checkpoints_ref[:, :, -1], "W1")
    compare_outputs(b1_checkpoints[:, :, -1], b1_checkpoints_ref[:, :, -1], "b1")
    compare_outputs(W2_checkpoints[:, :, -1], W2_checkpoints_ref[:, :, -1], "W2")
    compare_outputs(b2_checkpoints[:, :, -1], b2_checkpoints_ref[:, :, -1], "b2")

    breakpoint()


if __name__ == "__main__":
    main()
