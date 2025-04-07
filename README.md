# TTT

TTT is a repository for test-time training kernels.

## Installation

Installation requires CUDA drivers and toolkit (v 12.3+) and gcc 11+.

### From source
```bash
python setup.py install
```

We recommend you use conda to install your environment.

## Usage

Currently, we only support non-causal TTT-MLP kernels with head dimension of 64. Remat is automatically supported with these kernels.

Here is an example on how to invoke the kernels.

```python
import test_time_training as ttt


# Both ttt-mlp
ttt.ttt_forward(
    XQ_batch.contiguous(),
    XK_batch.contiguous(),
    XV_batch.contiguous(),
    last_eta.contiguous(),
    ttt_norm_weight.contiguous(),
    ttt_norm_bias.contiguous(),
    W1_init.contiguous(),
    b1_init.contiguous(),
    W2_init.contiguous(),
    b2_init.contiguous(),
    W1_checkpoints.contiguous(),
    b1_checkpoints.contiguous(),
    W2_checkpoints.contiguous(),
    b2_checkpoints.contiguous(),
    XQW_batch.contiguous(),
    checkpoint_group_size
)

ttt.ttt_backward(
    # Forward inputs
    XQ_batch.contiguous(),
    XK_batch.contiguous(),
    XV_batch.contiguous(),
    last_eta.contiguous(),
    ttt_norm_weight.contiguous(),
    ttt_norm_bias.contiguous(),
    # Checkpoints
    W1_checkpoints.contiguous(),
    b1_checkpoints.contiguous(),
    W2_checkpoints.contiguous(),
    b2_checkpoints.contiguous(),
    XQW_batch.contiguous(),
    # Rematted Buffers
    W1_init_group.contiguous(),
    b1_init_group.contiguous(),
    W2_init_group.contiguous(),
    b2_init_group.contiguous(),
    x_hat_ln_group.contiguous(),
    std_ln_group.contiguous(),
    X2_group.contiguous(),
    Z1_group.contiguous(),
    Z1_bar_group.contiguous(),
    X2_bar_group.contiguous(),
    grad_l_wrt_Z2_group.contiguous(),
    grad_l_wrt_Z1_group.contiguous(),
    x_hat_fused_group.contiguous(),
    grad_x_hat_fused_group.contiguous(),
    grad_output_fused_group.contiguous(),
    std_fused_group.contiguous(),
    # Upstream grads
    grad_L_W1_last.contiguous(),
    grad_L_b1_last.contiguous(),
    grad_L_W2_last.contiguous(),
    grad_L_b2_last.contiguous(),
    grad_L_XQW_batch.contiguous(),
    # Output grads
    grad_L_ttt_norm_weight.contiguous(),
    grad_L_ttt_norm_bias.contiguous(),
    grad_L_W1_init.contiguous(),
    grad_L_b1_init.contiguous(),
    grad_L_W2_init.contiguous(),
    grad_L_b2_init.contiguous(),
    grad_L_last_eta.contiguous(),
    grad_L_XQ.contiguous(),
    grad_L_XK.contiguous(),
    grad_L_XV.contiguous(),
    checkpoint_group_size
)
```

Note that these kernels do not support non-contiguous tensors.

## Thunderkittens
This repository is forked from Thunderkittens (https://github.com/HazyResearch/ThunderKittens).
Thunderkittens was used and modified for kernel development.


## Notes on implementation

These kernels use distributed shared memory to implement tensor-parallelism and sharding. The hidden states are sharded across SMs to save shared memory. 

These kernels also use input staging and pipelining to hide latencies for global reads.

We also used mixed precision to perform the matmuls in bf16 for tensor core usage and also kept hidden state (and grads) accumulation and layer norm computation in float32.

Kernel code can be found in `./kernels/`.