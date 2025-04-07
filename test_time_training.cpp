#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>

/*

HOW TO REGISTER YOUR OWN, CUSTOM SET OF KERNELS:

1. Decide on the identifier which will go in config.py. For example, "attn_inference" is the identifier for the first set below.
2. Add the identifier to the dict of sources in config.py.
3. Add the identifier to the list of kernels you want compiled.
4. The macro defined here, when that kernel is compiled, will be "TK_COMPILE_{IDENTIFIER_IN_ALL_CAPS}." You need to add two chunks to this file.
4a. the extern declaration at the top.
4b. the registration of the function into the module.
*/



#ifdef TK_COMPILE_TTT
extern torch::Tensor ttt_forward(
    const torch::Tensor XQ,
    const torch::Tensor XK,
    const torch::Tensor XV,
    const torch::Tensor last_eta,
    const torch::Tensor ttt_norm_weight,
    const torch::Tensor ttt_norm_bias,
    const torch::Tensor W1,
    const torch::Tensor b1,
    const torch::Tensor W2,
    const torch::Tensor b2,
    const torch::Tensor W1_checkpoints,
    const torch::Tensor b1_checkpoints,
    const torch::Tensor W2_checkpoints,
    const torch::Tensor b2_checkpoints,
    const torch::Tensor Out,
    const int checkpoint_group_size
);
#endif


#ifdef TK_COMPILE_TTT_BACKWARD
extern torch::Tensor ttt_backward(
    const torch::Tensor XQ,
    const torch::Tensor XK,
    const torch::Tensor XV,
    const torch::Tensor last_eta,
    const torch::Tensor ttt_norm_weight,
    const torch::Tensor ttt_norm_bias,
    const torch::Tensor W1_checkpoints,
    const torch::Tensor b1_checkpoints,
    const torch::Tensor W2_checkpoints,
    const torch::Tensor b2_checkpoints,
    const torch::Tensor Out,
    const torch::Tensor W1_init_group,
    const torch::Tensor b1_init_group,
    const torch::Tensor W2_init_group,
    const torch::Tensor b2_init_group,
    const torch::Tensor x_hat_ln_group,
    const torch::Tensor std_ln_group,
    const torch::Tensor X2_group,
    const torch::Tensor Z1_group,
    const torch::Tensor Z1_bar_group,
    const torch::Tensor X2_bar_group,
    const torch::Tensor grad_l_wrt_Z2_group,
    const torch::Tensor grad_l_wrt_Z1_group,
    const torch::Tensor x_hat_fused_group,
    const torch::Tensor grad_x_hat_fused_group,
    const torch::Tensor grad_output_fused_group,
    const torch::Tensor std_fused_group,
    const torch::Tensor grad_L_W1_last,
    const torch::Tensor grad_L_b1_last,
    const torch::Tensor grad_L_W2_last,
    const torch::Tensor grad_L_b2_last,
    const torch::Tensor grad_L_XQW_mini_batch,
    const torch::Tensor grad_L_ttt_norm_weight,
    const torch::Tensor grad_L_ttt_norm_bias,
    const torch::Tensor grad_L_W1_init,
    const torch::Tensor grad_L_b1_init,
    const torch::Tensor grad_L_W2_init,
    const torch::Tensor grad_L_b2_init,
    const torch::Tensor grad_L_last_eta,
    const torch::Tensor grad_L_XQ,
    const torch::Tensor grad_L_XK,
    const torch::Tensor grad_L_XV,
    const int checkpoint_group_size
);
#endif


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test Time Training Kernels"; // optional module docstring

#ifdef TK_COMPILE_TTT
    m.def("ttt_forward", &ttt_forward, "TTT Forward.");
#endif

#ifdef TK_COMPILE_TTT_BACKWARD
    m.def("ttt_backward", &ttt_backward, "TTT Backward.");
#endif
}