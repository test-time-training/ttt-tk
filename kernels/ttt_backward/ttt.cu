#include "cooperative_groups.h"
#include "kittens.cuh"
#include <iostream>
#include <cstdio>

// Build torch entrypoint
#ifdef TORCH_COMPILE
#define TK_COMPILE_TTT_BACKWARD
#endif

// Kernel config
constexpr int TP = (4);
constexpr int CONSUMER_WARPGROUPS = (1);
constexpr int PRODUCER_WARPGROUPS = (1);
constexpr int NUM_WARPGROUPS = (CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS);
constexpr int NUM_WORKERS = (NUM_WARPGROUPS * kittens::WARPGROUP_WARPS);

using namespace kittens; // dangerous as thunderkittens could overload cuda terms

// Reduce a tensor across cluster
template<ducks::st::all T>
__device__ static inline void tp_reduce(
    T &src, 
    T &reduction_buffer, 
    kittens::semaphore& dsmem_semaphore1, 
    kittens::semaphore& dsmem_semaphore2, 
    const int idx
)
{
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    const int tp = cluster.block_rank();
    static_assert(TP == 4, "Reduction is only implemented for tp=4.");
    const int warpid = kittens::warpid(); // Global warp ID
    const int wg_warpid = warpid % kittens::WARPGROUP_WARPS; // Warp ID within Warpgroup

    if (wg_warpid == 0) tma::expect_bytes(dsmem_semaphore1, sizeof(reduction_buffer));
    warpgroup::sync(1);
    tma::cluster::sync();
    if (wg_warpid == 0) tma::cluster::store_async(reduction_buffer, src, tp ^ 1, dsmem_semaphore1);
    kittens::wait(dsmem_semaphore1, idx % 2);

    warpgroup::sync(1);
    tma::cluster::sync();

    warpgroup::add(src, src, reduction_buffer);
    warpgroup::sync(1);

    if (wg_warpid == 0) tma::expect_bytes(dsmem_semaphore2, sizeof(reduction_buffer));
    tma::cluster::sync();
    if (wg_warpid == 0) tma::cluster::store_async(reduction_buffer, src, tp ^ 3, dsmem_semaphore2);
    kittens::wait(dsmem_semaphore2, idx % 2);

    warpgroup::sync(1);
    tma::cluster::sync();

    warpgroup::add(src, src, reduction_buffer);
    warpgroup::sync(1);
}

// For producer to signal tp, not functional besides preventing deadlock
__device__ static inline void tp_reduce_arrive()
{
    static_assert(TP == 4, "Reduction is only implemented for tp=4.");
    // need two here since we need two global reads to reduce 4
    tma::cluster::arrive_aligned();
    tma::cluster::arrive_aligned();
    tma::cluster::arrive_aligned();
    tma::cluster::arrive_aligned();
}

// General TTT metadata
template <int head_dim> struct bwd_ttt_mlp_ker_tile_dims {
    constexpr static int mini_batch_size = 64;
    constexpr static int F = head_dim;
    constexpr static int stages = (2);
};

// Globals with memory layout metadata
template <int head_dim> struct bwd_globals {
    using tile_dims = bwd_ttt_mlp_ker_tile_dims<head_dim>;
    // Tiles
    using CS_F_tile_type = st_bf<tile_dims::mini_batch_size, tile_dims::F>;
    using F_F_tile_type = st_bf<tile_dims::F, tile_dims::F>;
    using CS_F_tile_acc_type = st_fl<tile_dims::mini_batch_size, tile_dims::F>;
    using F_F_tile_acc_type = st_fl<tile_dims::F, tile_dims::F>;

    // Vectors
    using CS_vec_type = sv_bf<tile_dims::mini_batch_size>;
    using F_vec_type = sv_bf<tile_dims::F>;
    using CS_vec_acc_type = sv_fl<tile_dims::mini_batch_size>;
    using F_vec_acc_type = sv_fl<tile_dims::F>;
    using std_vec_acc_type = sv_fl<16>;

    // Global memory layout
    using qkvo_gl = gl<bf16, -1, -1, -1, -1, CS_F_tile_type>;
    using qkvo_acc_gl = gl<float, -1, -1, -1, -1, CS_F_tile_acc_type>;

    using last_eta_gl = gl<bf16, -1, -1, -1, -1, CS_vec_type>;

    using ttt_norm_weight_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;
    using ttt_norm_bias_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;

    using w1_gl = gl<float, -1, -1, -1, -1, F_F_tile_acc_type>;
    using b1_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;
    using w2_gl = gl<float, -1, -1, -1, -1, F_F_tile_acc_type>;
    using b2_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;

    // Remat checkpoints
    using w1_checkpoints_gl = gl<float, -1, -1, -1, -1, F_F_tile_acc_type>;
    using b1_checkpoints_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;
    using w2_checkpoints_gl = gl<float, -1, -1, -1, -1, F_F_tile_acc_type>;
    using b2_checkpoints_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;

    using std_gl = gl<float, -1, -1, -1, -1, std_vec_acc_type>;
    using std_test_gl = gl<float, -1, -1, -1, -1, CS_vec_acc_type>;

    qkvo_gl q;
    qkvo_gl k;
    qkvo_gl v;

    // This is output, used for debugging. Not o projection.
    qkvo_gl o;

    last_eta_gl last_eta;

    ttt_norm_weight_gl ttt_norm_weight;
    ttt_norm_bias_gl ttt_norm_bias;

    w1_checkpoints_gl w1_checkpoints;
    b1_checkpoints_gl b1_checkpoints;
    w2_checkpoints_gl w2_checkpoints;
    b2_checkpoints_gl b2_checkpoints;

    // rematted activations
    w1_gl W1_init_group;
    b1_gl b1_init_group;
    w2_gl W2_init_group;
    b2_gl b2_init_group;
    qkvo_gl x_hat_ln_group;
    std_gl std_ln_group;
    std_test_gl std_ln_test_group;
    qkvo_gl X2_group;
    qkvo_gl Z1_group;
    qkvo_gl Z1_bar_group;
    qkvo_gl X2_bar_group;
    qkvo_gl grad_l_wrt_Z2_group;
    qkvo_gl grad_l_wrt_Z1_group;
    qkvo_gl x_hat_fused_group;
    qkvo_gl grad_x_hat_fused_group;
    qkvo_gl grad_output_fused_group;
    std_gl std_fused_group;
    std_test_gl std_fused_test_group;

    // Upstream grads
    w1_gl grad_L_W1_last;
    b1_gl grad_L_b1_last;
    w2_gl grad_L_W2_last;
    b2_gl grad_L_b2_last;
    qkvo_gl grad_L_XQW_mini_batch;

    // Output grads
    ttt_norm_weight_gl grad_L_ttt_norm_weight;
    ttt_norm_bias_gl grad_L_ttt_norm_bias;
    w1_gl grad_L_W1_init;
    b1_gl grad_L_b1_init;
    w2_gl grad_L_W2_init;
    b2_gl grad_L_b2_init;
    last_eta_gl grad_L_last_eta;
    qkvo_gl grad_L_XQ;
    qkvo_gl grad_L_XK;
    qkvo_gl grad_L_XV;

    const int num_mini_batch;
    const int seq_len;
    const int num_checkpoints;
    const int checkpoint_group_size;
};

template <int head_dim>
__cluster_dims__(TP)
__global__ __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
void bwd_ttt_mlp_ker(const __grid_constant__ bwd_globals<head_dim> g) {
    using globals = bwd_globals<head_dim>;
    using K = bwd_ttt_mlp_ker_tile_dims<head_dim>;

    using CS_F_tile_type = globals::CS_F_tile_type;
    using F_F_tile_type = globals::F_F_tile_type;
    using CS_F_tile_acc_type = globals::CS_F_tile_acc_type;
    using F_F_tile_acc_type = globals::F_F_tile_acc_type;

    // Vectors
    using CS_vec_type = globals::CS_vec_type;
    using F_vec_type = globals::F_vec_type;
    using CS_vec_acc_type = globals::CS_vec_acc_type;
    using F_vec_acc_type = globals::F_vec_acc_type;

    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    const int tp = cluster.block_rank();

    // For input indexing
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.z;
    const int n_minibatch = g.seq_len / (K::mini_batch_size);
    const int n_remat_groups = g.num_checkpoints;
    const int checkpoint_group_size = g.checkpoint_group_size;

    // Block info
    const int warpid = kittens::warpid(); // Global warp ID
    const int wg_warpid = warpid % kittens::WARPGROUP_WARPS; // Warp ID within Warpgroup
    const int warpgroupid = warpid / kittens::WARPGROUP_WARPS; // Warpgroup ID
    const int tp_shard_rank = tp;
    const bool is_producer = warpgroupid == NUM_WARPGROUPS - 1;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);

    if (is_producer)
    {
        // I believe there is a restriction on this, the register count here is ignored somewhat
        warpgroup::decrease_registers<24>(); // producer needs less registers due to tma
    }
    else
    {
        warpgroup::increase_registers<256>(); // consumer needs all of the registers
    }

    // Shared memory for hidden states
    F_F_tile_acc_type(&w1_smem) = al.allocate<F_F_tile_acc_type>();
    F_vec_acc_type(&b1_smem) = al.allocate<F_vec_acc_type>();
    F_F_tile_acc_type(&w2_smem) = al.allocate<F_F_tile_acc_type>();
    F_vec_acc_type(&b2_smem) = al.allocate<F_vec_acc_type>();

    // Shared memory for inputs (staged)
    CS_F_tile_type(&q_smem)[K::stages] = al.allocate<CS_F_tile_type, K::stages>();
    CS_F_tile_type(&k_smem)[K::stages] = al.allocate<CS_F_tile_type, K::stages>();
    CS_F_tile_type(&v_smem)[K::stages] = al.allocate<CS_F_tile_type, K::stages>();
    CS_vec_type(&last_eta_smem)[K::stages] = al.allocate<CS_vec_type, K::stages>();

    // Shared memory for ttt norm params
    F_vec_acc_type(&ttt_norm_weight_smem) = al.allocate<F_vec_acc_type>();
    F_vec_acc_type(&ttt_norm_bias_smem) = al.allocate<F_vec_acc_type>();

    // Extra shared memory for intermediate computations
    CS_F_tile_type(&z1_smem) = al.allocate<CS_F_tile_type>();
    CS_F_tile_type(&x2_smem) = al.allocate<CS_F_tile_type>();
    CS_F_tile_type(&ln_tile_smem) = al.allocate<CS_F_tile_type>();
    CS_F_tile_type(&grad_l_z1_smem) = al.allocate<CS_F_tile_type>();
    sv_fl<16>(&ln_smem)[4] = al.allocate<sv_fl<16>, 4>();
    CS_F_tile_type(&matmul_smem) = al.allocate<CS_F_tile_type>();
    CS_F_tile_acc_type(&b_acc_smem) = al.allocate<CS_F_tile_acc_type>();
    CS_F_tile_type(&cs_f_store_smem) = al.allocate<CS_F_tile_type>();
    CS_F_tile_type(&cs_f_store2_smem) = al.allocate<CS_F_tile_type>();

    // Backward-backward smem
    F_F_tile_acc_type(&grad_L_W1_smem) = al.allocate<F_F_tile_acc_type>();
    F_vec_acc_type(&grad_L_b1_smem) = al.allocate<F_vec_acc_type>();
    F_F_tile_acc_type(&grad_L_W2_smem) = al.allocate<F_F_tile_acc_type>();
    F_vec_acc_type(&grad_L_b2_smem) = al.allocate<F_vec_acc_type>();
    CS_vec_type(&grad_L_last_eta_smem) = al.allocate<CS_vec_type>();
    F_vec_acc_type(&grad_L_ttt_norm_weight_smem) = al.allocate<F_vec_acc_type>();
    F_vec_acc_type(&grad_L_ttt_norm_bias_smem) = al.allocate<F_vec_acc_type>();
    CS_vec_acc_type(&std_ln_smem) = al.allocate<CS_vec_acc_type>();
    CS_vec_acc_type(&std_fused_smem) = al.allocate<CS_vec_acc_type>();

    // Reinterpretations for intermediates
    auto(&bwd_q_smem) = q_smem[0];
    auto(&bwd_k_smem) = k_smem[0];
    auto(&bwd_last_eta_smem) = last_eta_smem[0];
    auto(&grad_L_XQW_mini_batch_smem) = v_smem[0];
    auto(&x_hat_ln_smem) = z1_smem;
    auto(&grad_L_Z2_bar_smem) = x2_smem;
    auto(&z1_bar_smem) = grad_l_z1_smem;
    auto(&grad_L_Z1_bar_smem) = q_smem[1];
    auto(&x2_bar_smem) = k_smem[1];
    auto(&grad_l_wrt_Z2_smem) = v_smem[1];
    auto(&x2_bwd_smem) = z1_smem;
    auto(&grad_l_wrt_Z1_smem) = grad_l_z1_smem;
    auto(&grad_L_grad_l_wrt_Z1_smem) = q_smem[0];
    auto(&grad_L_grad_l_wrt_Z2_smem) = k_smem[1];
    auto(&z1_bwd_smem) = x2_smem;
    auto(&x_hat_fused_smem) = v_smem[0];
    auto(&grad_L_grad_x_hat_fused_smem) = q_smem[1];
    auto(&grad_L_reconstruction_target_smem) = grad_l_z1_smem;
    auto(&grad_output_fused_smem) = cs_f_store2_smem;
    auto(&grad_L_x_hat_fused_smem) = q_smem[1];
    auto(&grad_x_hat_fused_smem) = grad_l_z1_smem;
    auto(&grad_L_Z2_smem) = k_smem[1];
    auto(&grad_L_Z1_smem) = v_smem[0];
    auto(&grad_L_XK_smem) = q_smem[1];
    auto(&reduction_buffer) = matmul_smem;
    auto(&z2_smem) = grad_l_z1_smem;
    auto(&grad_l_z2_smem) = v_smem[0];


    // Create locks to make sure reads aren't premature
    __shared__ kittens::semaphore 
        w1_arrived,
        w2_arrived,
        b1_arrived,
        b2_arrived,
        q_sem_arrived[K::stages],
        k_sem_arrived[K::stages], 
        v_sem_arrived[K::stages],
        last_eta_sem_arrived[K::stages],
        ttt_norm_weight_arrived,
        ttt_norm_bias_arrived,
        compute_done[K::stages],
        dsmem_semaphore1,
        dsmem_semaphore2,
        second_dsmem_semaphore1,
        second_dsmem_semaphore2,
        forward_done,
        backward_done,
        bwd_compute_done,
        grad_L_W1_arrived,
        grad_L_b1_arrived,
        grad_L_W2_arrived,
        grad_L_b2_arrived,
        grad_L_XQW_mini_batch_sem_arrived,
        bwd_q_sem_arrived,
        bwd_k_sem_arrived, 
        bwd_last_eta_sem_arrived,
        old_weights_done,
        x_hat_ln_arrived,
        std_ln_arrived,
        std_fused_arrived,
        z1_bar_group_arrived,
        x2_bar_group_arrived,
        grad_l_wrt_Z2_arrived,
        x2_bwd_arrived,
        grad_l_wrt_Z1_arrived,
        z1_bwd_arrived,
        x_hat_fused_arrived,
        grad_output_fused_arrived,
        grad_x_hat_fused_arrived,
        grad_L_XQW_mini_batch_sem_freed,
        x_hat_ln_freed,
        grad_L_Z2_bar_freed,
        z1_bar_freed,
        grad_L_reconstruction_target_freed,
        cs_f_store2_smem_freed,
        w1_remat_smem_arrived,
        b1_remat_smem_arrived,
        w2_remat_smem_arrived,
        b2_remat_smem_arrived,
        bwd_dsmem_semaphore1,
        bwd_dsmem_semaphore2;


    if (threadIdx.x == 0) {
        init_semaphore(w1_arrived, 0, 1);
        init_semaphore(b1_arrived, 0, 1);
        init_semaphore(w2_arrived, 0, 1);
        init_semaphore(b2_arrived, 0, 1);
        init_semaphore(ttt_norm_weight_arrived, 0, 1);
        init_semaphore(ttt_norm_bias_arrived, 0, 1);
        init_semaphore(dsmem_semaphore1, 0, 1);
        init_semaphore(dsmem_semaphore2, 0, 1);
        init_semaphore(second_dsmem_semaphore1, 0, 1);
        init_semaphore(second_dsmem_semaphore2, 0, 1);
        for (int i = 0; i < K::stages; i++) {
            init_semaphore(q_sem_arrived[i], 0, 1);
            init_semaphore(k_sem_arrived[i], 0, 1);
            init_semaphore(v_sem_arrived[i], 0, 1);
            init_semaphore(last_eta_sem_arrived[i], 0, 1);
            init_semaphore(compute_done[i], CONSUMER_WARPGROUPS, 0);
        }

        // Flow synchronization
        init_semaphore(forward_done, 0, 1);
        init_semaphore(backward_done, 0, 1);
        init_semaphore(bwd_compute_done, 0, 1);

        // Upstream gradients
        init_semaphore(grad_L_W1_arrived, 0, 1);
        init_semaphore(grad_L_b1_arrived, 0, 1);
        init_semaphore(grad_L_W2_arrived, 0, 1);
        init_semaphore(grad_L_b2_arrived, 0, 1);

        // Loading in pipelined inputs
        init_semaphore(old_weights_done, 0, 1);
        init_semaphore(grad_L_XQW_mini_batch_sem_arrived, 0, 1);
        init_semaphore(bwd_q_sem_arrived, 0, 1);
        init_semaphore(bwd_k_sem_arrived, 0, 1);
        init_semaphore(bwd_last_eta_sem_arrived, 0, 1);
        init_semaphore(x_hat_ln_arrived, 0, 1);
        init_semaphore(std_ln_arrived, 0, 1);
        init_semaphore(std_fused_arrived, 0, 1);
        init_semaphore(z1_bar_group_arrived, 0, 1);
        init_semaphore(x2_bar_group_arrived, 0, 1);
        init_semaphore(grad_l_wrt_Z2_arrived, 0, 1);
        init_semaphore(x2_bwd_arrived, 0, 1);
        init_semaphore(grad_l_wrt_Z1_arrived, 0, 1);
        init_semaphore(z1_bwd_arrived, 0, 1);
        init_semaphore(x_hat_fused_arrived, 0, 1);
        init_semaphore(grad_x_hat_fused_arrived, 0, 1);
        init_semaphore(grad_output_fused_arrived, 0, 1);

        // For pipelining
        init_semaphore(grad_L_XQW_mini_batch_sem_freed, 0, 1);
        init_semaphore(x_hat_ln_freed, 0, 1);
        init_semaphore(grad_L_Z2_bar_freed, 0, 1);
        init_semaphore(z1_bar_freed, 0, 1);
        init_semaphore(grad_L_reconstruction_target_freed, 0, 1);
        init_semaphore(cs_f_store2_smem_freed, 0, 1);

        // Make sure next hidden state is done loading in
        init_semaphore(w1_remat_smem_arrived, 0, 1);
        init_semaphore(b1_remat_smem_arrived, 0, 1);
        init_semaphore(w2_remat_smem_arrived, 0, 1);
        init_semaphore(b2_remat_smem_arrived, 0, 1);

        // For tp reduce
        init_semaphore(bwd_dsmem_semaphore1, 0, 1);
        init_semaphore(bwd_dsmem_semaphore2, 0, 1);

        // ttt norm params
        tma::expect_bytes(ttt_norm_weight_arrived, sizeof(ttt_norm_weight_smem));
        tma::expect_bytes(ttt_norm_bias_arrived, sizeof(ttt_norm_bias_smem));
        tma::load_async(ttt_norm_weight_smem, g.ttt_norm_weight, {0, head_idx, 0, 0}, ttt_norm_weight_arrived);
        tma::load_async(ttt_norm_bias_smem, g.ttt_norm_bias, {0, head_idx, 0, 0}, ttt_norm_bias_arrived);

        // ttt norm params
        tma::expect_bytes(grad_L_W1_arrived, sizeof(grad_L_W1_smem));
        tma::expect_bytes(grad_L_b1_arrived, sizeof(grad_L_b1_smem));
        tma::expect_bytes(grad_L_W2_arrived, sizeof(grad_L_W2_smem));
        tma::expect_bytes(grad_L_b2_arrived, sizeof(grad_L_b2_smem));
        tma::load_async(grad_L_W1_smem, g.grad_L_W1_last, {batch_idx, head_idx, 0, tp_shard_rank}, grad_L_W1_arrived);
        tma::load_async(grad_L_b1_smem, g.grad_L_b1_last, {batch_idx, head_idx, 0, tp_shard_rank}, grad_L_b1_arrived);
        tma::load_async(grad_L_W2_smem, g.grad_L_W2_last, {batch_idx, head_idx, tp_shard_rank, 0}, grad_L_W2_arrived);
        tma::load_async(grad_L_b2_smem, g.grad_L_b2_last, {batch_idx, head_idx, 0, 0}, grad_L_b2_arrived);       
    }
    __syncthreads();

    // Allow producer to start loading in 
    if (!is_producer)
    {
        kittens::wait(ttt_norm_weight_arrived, 0);
        kittens::wait(ttt_norm_bias_arrived, 0);

        kittens::wait(grad_L_W1_arrived, 0);
        kittens::wait(grad_L_b1_arrived, 0);
        kittens::wait(grad_L_W2_arrived, 0);
        kittens::wait(grad_L_b2_arrived, 0);

        warpgroup::zero(grad_L_ttt_norm_weight_smem);
        warpgroup::zero(grad_L_ttt_norm_bias_smem);
    }

    int semaphore_idx = 0;
    int bwd_semaphore_idx = 0;
    
    for (int checkpoint_idx = g.num_checkpoints - 1; checkpoint_idx >= 0; --checkpoint_idx)
    {
        if (is_producer && warpid == NUM_WORKERS - 4)
        {
            int4 curr_checkpoint = {batch_idx, head_idx, checkpoint_idx, tp_shard_rank};
            const int sharded_checkpoint_offset = checkpoint_idx * TP + tp_shard_rank;
            const int4 W2_checkpoint = {batch_idx, head_idx, sharded_checkpoint_offset, 0};
            
            // Load hidden states from checkpoint
            tma::expect_bytes(w1_arrived, sizeof(w1_smem));
            tma::expect_bytes(b1_arrived, sizeof(b1_smem));
            tma::expect_bytes(w2_arrived, sizeof(w2_smem));
            tma::expect_bytes(b2_arrived, sizeof(b2_smem));
            tma::load_async(w1_smem, g.w1_checkpoints, curr_checkpoint, w1_arrived);
            tma::load_async(b1_smem, g.b1_checkpoints, curr_checkpoint, b1_arrived);
            tma::load_async(w2_smem, g.w2_checkpoints, W2_checkpoint, w2_arrived);
            tma::load_async(b2_smem, g.b2_checkpoints, {batch_idx, head_idx, checkpoint_idx, 0}, b2_arrived);
        }
        else if (!is_producer)
        {
            const int semaphore_phase = (g.num_checkpoints - checkpoint_idx - 1) % 2;
            kittens::wait(w1_arrived, semaphore_phase);
            kittens::wait(b1_arrived, semaphore_phase);
            kittens::wait(w2_arrived, semaphore_phase);
            kittens::wait(b2_arrived, semaphore_phase);
        }

        // backward-forward
        for (int mini_batch_in_group_idx = 0; mini_batch_in_group_idx < g.checkpoint_group_size; ++mini_batch_in_group_idx)
        {
            const int global_mini_batch_idx = checkpoint_idx * checkpoint_group_size + mini_batch_in_group_idx;
            // Num of mini-batches might not be perfect multiple of remat group size
            if (global_mini_batch_idx >= g.num_mini_batch) continue;

            if (is_producer)
            {
                if (warpid == NUM_WORKERS - 4)
                {
                    int4 tile_idx = {batch_idx, head_idx, global_mini_batch_idx, 0};

                    const int curr_stage = semaphore_idx % K::stages;

                    // Wait on previous stage to finish computation
                    if (semaphore_idx / K::stages > 0)
                    {
                        const int last_idx = semaphore_idx - K::stages;
                        kittens::wait(compute_done[last_idx % K::stages], (last_idx / K::stages) % 2);
                    }
                    
                    tma::expect_bytes(k_sem_arrived[curr_stage], sizeof(k_smem[0]));
                    tma::load_async(k_smem[curr_stage], g.k, tile_idx, k_sem_arrived[curr_stage]);
                    tma::expect_bytes(v_sem_arrived[curr_stage], sizeof(v_smem[0]));
                    tma::load_async(v_smem[curr_stage], g.v, tile_idx, v_sem_arrived[curr_stage]);
                    tma::expect_bytes(q_sem_arrived[curr_stage], sizeof(q_smem[0]));
                    tma::load_async(q_smem[curr_stage], g.q, tile_idx, q_sem_arrived[curr_stage]);
                    tma::expect_bytes(last_eta_sem_arrived[curr_stage], sizeof(last_eta_smem[0]));
                    tma::load_async(last_eta_smem[curr_stage], g.last_eta, tile_idx, last_eta_sem_arrived[curr_stage]);

                    tp_reduce_arrive(); // might be slow to do this, idk
                    tp_reduce_arrive(); // Should be two reduces
                }            
            }
            else
            {
                if (wg_warpid == 0) {
                    tma::store_async(g.W1_init_group, w1_smem, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank});
                    tma::store_async(g.b1_init_group, b1_smem, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank});
                    tma::store_async(g.W2_init_group, w2_smem, {batch_idx, head_idx, mini_batch_in_group_idx * TP + tp_shard_rank, 0});
                    tma::store_async(g.b2_init_group, b2_smem, {batch_idx, head_idx, mini_batch_in_group_idx, 0});
                    tma::store_commit_group();
                    tma::store_async_wait();
                }

                rt_fl<16, K::F> cs_cs_fl_reg;
                rt_fl<16, K::F> cs_cs_fl_reg2;
                typeof(cs_cs_fl_reg)::row_vec cs_row_fl_reg;
                typeof(cs_cs_fl_reg)::col_vec cs_col_fl_reg;

                const int curr_stage = (semaphore_idx) % K::stages;
                const int semaphore_phase = (semaphore_idx / K::stages) % 2;
                
                // Hidden state forward
                kittens::wait(k_sem_arrived[curr_stage], semaphore_phase);
                zero(cs_cs_fl_reg);
                warpgroup::load(cs_cs_fl_reg2, w1_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mm_AB(cs_cs_fl_reg, k_smem[curr_stage], matmul_smem);
                load(cs_row_fl_reg, b1_smem);
                warpgroup::mma_async_wait();
                add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                warpgroup::store(z1_smem, cs_cs_fl_reg);
                
                gelu(cs_cs_fl_reg, cs_cs_fl_reg);
                warpgroup::store(x2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                if (wg_warpid == 0) {
                    tma::store_async(g.Z1_group, z1_smem, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank});
                    tma::store_async(g.X2_group, x2_smem, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank});
                    tma::store_commit_group();
                }

                tma::store_async_wait();

                zero(cs_cs_fl_reg);
                warpgroup::load(cs_cs_fl_reg2, w2_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mm_AB(cs_cs_fl_reg, x2_smem, matmul_smem);
                warpgroup::mma_async_wait();
                // Only add b2 to one of the sharded Z2
                // Else, post reduction it will result in adding 4*b2
                if (tp_shard_rank == 0) {
                    load(cs_row_fl_reg, b2_smem);
                    add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                }
                warpgroup::store(z2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+4);

                // Reductions
                tp_reduce(z2_smem, reduction_buffer, dsmem_semaphore1, dsmem_semaphore2, semaphore_idx);

                // LN
                // mean
                zero(cs_col_fl_reg);
                warpgroup::load(cs_cs_fl_reg, z2_smem);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                div(cs_col_fl_reg, cs_col_fl_reg, static_cast<float>(head_dim)); // mu_fused

                sub_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg); // Z2 - mu_fused, first part of xhat
                warpgroup::store(ln_tile_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                // var
                warpgroup::load(cs_cs_fl_reg, z2_smem);
                sub_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg);
                mul(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg);

                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                div(cs_col_fl_reg, cs_col_fl_reg, static_cast<float>(head_dim)); // var
                add(cs_col_fl_reg, cs_col_fl_reg, 1e-8f); 
                sqrt(cs_col_fl_reg, cs_col_fl_reg); // std
                store(ln_smem[wg_warpid], cs_col_fl_reg);
                warpgroup::sync(warpgroupid+1);

                // finish x_hat
                warpgroup::load(cs_cs_fl_reg, ln_tile_smem);
                div_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg); // final x_hat
                warpgroup::store(z2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                if (wg_warpid == 0)
                {
                    tma::store_async(g.x_hat_fused_group, z2_smem, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank});
                }

                tma::store_async(g.std_fused_group, ln_smem[wg_warpid], {batch_idx, head_idx, mini_batch_in_group_idx, wg_warpid});
                tma::store_commit_group();
                tma::store_async_wait();
                
                // compute y
                load(cs_row_fl_reg, ttt_norm_weight_smem);
                mul_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                load(cs_row_fl_reg, ttt_norm_bias_smem);
                add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg); // y
                
                // LN BWD
                // grad_output
                kittens::wait(v_sem_arrived[curr_stage], semaphore_phase);
                warpgroup::load(cs_cs_fl_reg2, v_smem[curr_stage]);
                sub(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                warpgroup::load(cs_cs_fl_reg2, k_smem[curr_stage]);
                add(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                warpgroup::store(cs_f_store_smem, cs_cs_fl_reg);

                // grad x_hat
                load(cs_row_fl_reg, ttt_norm_weight_smem);
                mul_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                warpgroup::store(cs_f_store2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                if (wg_warpid == 0) {
                    tma::store_async(g.grad_output_fused_group, cs_f_store_smem, {batch_idx, head_idx, mini_batch_in_group_idx, 0});
                    tma::store_async(g.grad_x_hat_fused_group, cs_f_store2_smem, {batch_idx, head_idx, mini_batch_in_group_idx, 0});
                    tma::store_commit_group();
                    
                }
                tma::store_async_wait();

                warpgroup::load(cs_cs_fl_reg2, z2_smem);
                mul(cs_cs_fl_reg2, cs_cs_fl_reg, cs_cs_fl_reg2);
                zero(cs_col_fl_reg);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg2);
                warpgroup::load(cs_cs_fl_reg2, z2_smem);
                mul_row(cs_cs_fl_reg2, cs_cs_fl_reg2, cs_col_fl_reg); // 3rd line, not negative

                zero(cs_col_fl_reg);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                add_row(cs_cs_fl_reg2, cs_cs_fl_reg2, cs_col_fl_reg);

                mul(cs_cs_fl_reg, cs_cs_fl_reg, static_cast<float>(head_dim));
                sub(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                div(cs_cs_fl_reg, cs_cs_fl_reg, static_cast<float>(head_dim));

                load(cs_col_fl_reg, ln_smem[wg_warpid]);
                div_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg);
                warpgroup::store(cs_f_store_smem, cs_cs_fl_reg); // untouched grad_l_wrt_Z2
                mul(cs_cs_fl_reg, cs_cs_fl_reg, -1.0f); // negate to prepare for grad step
                kittens::wait(last_eta_sem_arrived[curr_stage], semaphore_phase);

                warpgroup::store(grad_l_z2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mul_row(grad_l_z2_smem, grad_l_z2_smem, last_eta_smem[curr_stage]);

                if (wg_warpid == 0) {
                    tma::store_async(g.grad_l_wrt_Z2_group, cs_f_store_smem, {batch_idx, head_idx, mini_batch_in_group_idx, 0});
                    tma::store_commit_group();
                }
                tma::store_async_wait();

                // Calculate grad_l_wrt_Z1
                zero(cs_cs_fl_reg);
                warpgroup::load(cs_cs_fl_reg2, w2_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mm_ABt(cs_cs_fl_reg, grad_l_z2_smem, matmul_smem);
                warpgroup::mma_async_wait();
                warpgroup::store(grad_l_z1_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                warpgroup::load(cs_cs_fl_reg, z1_smem);
                gelu_bwd(cs_cs_fl_reg, cs_cs_fl_reg);
                warpgroup::store(z1_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mul(grad_l_z1_smem, grad_l_z1_smem, z1_smem);

                // recalc grad_l_wrt_z1 without eta for fwd comp
                zero(cs_cs_fl_reg);
                warpgroup::mm_ABt(cs_cs_fl_reg, cs_f_store_smem, matmul_smem);
                warpgroup::mma_async_wait();
                warpgroup::store(cs_f_store_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mul(cs_f_store_smem, cs_f_store_smem, z1_smem); // already gelu bwd for Z1
                warpgroup::sync(warpgroupid+1);

                if (wg_warpid == 0) {
                    tma::store_async(g.grad_l_wrt_Z1_group, cs_f_store_smem, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank});
                    tma::store_commit_group();
                }
                tma::store_async_wait();
                
                // Update W2
                warpgroup::load(cs_cs_fl_reg, w2_smem);
                warpgroup::mma_AtB(cs_cs_fl_reg, x2_smem, grad_l_z2_smem);
                warpgroup::mma_async_wait();
                warpgroup::store(w2_smem, cs_cs_fl_reg);

                // Update b2
                warpgroup::copy(b_acc_smem, grad_l_z2_smem);
                warpgroup::col_sum(b2_smem, b_acc_smem, b2_smem);

                // Update W1
                warpgroup::load(cs_cs_fl_reg, w1_smem);
                warpgroup::mma_AtB(cs_cs_fl_reg, k_smem[curr_stage], grad_l_z1_smem);
                warpgroup::mma_async_wait();
                warpgroup::store(w1_smem, cs_cs_fl_reg);

                // Update b1
                warpgroup::copy(b_acc_smem, grad_l_z1_smem);
                warpgroup::col_sum(b1_smem, b_acc_smem, b1_smem);

                warpgroup::sync(warpgroupid+1);

                // Compute output
                kittens::wait(q_sem_arrived[curr_stage], semaphore_phase);
                zero(cs_cs_fl_reg);
                warpgroup::load(cs_cs_fl_reg2, w1_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mm_AB(cs_cs_fl_reg, q_smem[curr_stage], matmul_smem);
                load(cs_row_fl_reg, b1_smem);
                warpgroup::mma_async_wait();
                
                add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                warpgroup::store(z1_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                gelu(cs_cs_fl_reg, cs_cs_fl_reg);
                warpgroup::store(x2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                if (wg_warpid == 0) {
                    tma::store_async(g.Z1_bar_group, z1_smem, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank});
                    tma::store_async(g.X2_bar_group, x2_smem, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank});
                    tma::store_commit_group();
                }
                tma::store_async_wait();

                zero(cs_cs_fl_reg);
                warpgroup::load(cs_cs_fl_reg2, w2_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mm_AB(cs_cs_fl_reg, x2_smem, matmul_smem);
                warpgroup::mma_async_wait();
                // Only add b2 to one of the sharded Z2
                // Else, post reduction it will result in adding 4*b2
                if (tp_shard_rank == 0) {
                    load(cs_row_fl_reg, b2_smem);
                    add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                }
                warpgroup::store(z2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                // Reduction
                tp_reduce(z2_smem, reduction_buffer, second_dsmem_semaphore1, second_dsmem_semaphore2, semaphore_idx);

                // mean
                zero(cs_col_fl_reg);
                warpgroup::load(cs_cs_fl_reg, z2_smem);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                div(cs_col_fl_reg, cs_col_fl_reg, static_cast<float>(head_dim)); // mu_fused

                sub_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg); // Z2 - mu_fused, first part of xhat
                warpgroup::store(ln_tile_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                // var
                warpgroup::load(cs_cs_fl_reg, z2_smem);
                sub_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg);
                mul(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg);

                // zero(cs_col_fl_reg);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                div(cs_col_fl_reg, cs_col_fl_reg, static_cast<float>(head_dim)); // var
                add(cs_col_fl_reg, cs_col_fl_reg, 1e-8f); 
                sqrt(cs_col_fl_reg, cs_col_fl_reg); // std
                store(ln_smem[wg_warpid], cs_col_fl_reg);

                // finish x_hat
                warpgroup::load(cs_cs_fl_reg, ln_tile_smem);
                div_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg); // final x_hat
                warpgroup::store(z2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                if (tp_shard_rank == 0 && wg_warpid == 0)
                {
                    tma::store_async(g.x_hat_ln_group, z2_smem, {batch_idx, head_idx, mini_batch_in_group_idx, 0});
                }
                tma::store_async(g.std_ln_group, ln_smem[wg_warpid], {batch_idx, head_idx, mini_batch_in_group_idx, wg_warpid});
                tma::store_commit_group();
                tma::store_async_wait();

                if (warpgroup::laneid() == 0) arrive(compute_done[curr_stage], 1);
                
            }

            ++semaphore_idx;
        }

        // Need to synchronize in order to reuse shared memory and not overstep.
        // At this point, hidden states should be the last in the checkpoint group.
        if (is_producer && warpid == NUM_WORKERS - 4)
        {
            const int checkpoint_phase = (g.num_checkpoints - checkpoint_idx - 1) % 2;
            kittens::wait(forward_done, checkpoint_phase);
        }
        else
        {
            if (warpgroup::laneid() == 0) arrive(forward_done, 1);
        }

        // backward-backward
        for (int mini_batch_in_group_idx = g.checkpoint_group_size - 1; mini_batch_in_group_idx >= 0; --mini_batch_in_group_idx)
        {
            const int global_mini_batch_idx = checkpoint_idx * checkpoint_group_size + mini_batch_in_group_idx;
            // Num of mini-batches might not be perfect multiple of remat group size
            if (global_mini_batch_idx >= g.num_mini_batch) continue;

            if (is_producer)
            {
                if (warpid == NUM_WORKERS - 4)
                {
                    const int bwd_semaphore_phase = bwd_semaphore_idx % 2;

                    int4 tile_idx = {batch_idx, head_idx, global_mini_batch_idx, 0};

                    // Wait on previous stage to finish computation
                    if (bwd_semaphore_idx > 0)
                    {
                        const int last_idx = bwd_semaphore_idx - 1;
                        kittens::wait(bwd_compute_done, last_idx % 2);
                    }

                    tma::expect_bytes(grad_L_XQW_mini_batch_sem_arrived, sizeof(grad_L_XQW_mini_batch_smem));
                    tma::load_async(grad_L_XQW_mini_batch_smem, g.grad_L_XQW_mini_batch, tile_idx, grad_L_XQW_mini_batch_sem_arrived);
                    tma::expect_bytes(bwd_k_sem_arrived, sizeof(bwd_k_smem));
                    tma::load_async(bwd_k_smem, g.k, tile_idx, bwd_k_sem_arrived);
                    tma::expect_bytes(bwd_q_sem_arrived, sizeof(bwd_q_smem));
                    tma::load_async(bwd_q_smem, g.q, tile_idx, bwd_q_sem_arrived);
                    tma::expect_bytes(bwd_last_eta_sem_arrived, sizeof(bwd_last_eta_smem));
                    tma::load_async(bwd_last_eta_smem, g.last_eta, tile_idx, bwd_last_eta_sem_arrived);

                    tma::expect_bytes(std_ln_arrived, sizeof(std_ln_smem));
                    tma::load_async(std_ln_smem, g.std_ln_test_group, {batch_idx, head_idx, mini_batch_in_group_idx, 0}, std_ln_arrived);

                    tma::expect_bytes(x_hat_ln_arrived, sizeof(x_hat_ln_smem));
                    tma::load_async(x_hat_ln_smem, g.x_hat_ln_group, {batch_idx, head_idx, mini_batch_in_group_idx, 0}, x_hat_ln_arrived);

                    tma::expect_bytes(z1_bar_group_arrived, sizeof(z1_bar_smem));
                    tma::load_async(z1_bar_smem, g.Z1_bar_group, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank}, z1_bar_group_arrived);

                    tma::expect_bytes(x2_bar_group_arrived, sizeof(x2_bar_smem));
                    tma::load_async(x2_bar_smem, g.X2_bar_group, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank}, x2_bar_group_arrived);

                    tma::expect_bytes(grad_l_wrt_Z2_arrived, sizeof(grad_l_wrt_Z2_smem));
                    tma::load_async(grad_l_wrt_Z2_smem, g.grad_l_wrt_Z2_group, {batch_idx, head_idx, mini_batch_in_group_idx, 0}, grad_l_wrt_Z2_arrived);
                    
                    kittens::wait(x_hat_ln_freed, bwd_semaphore_phase);

                    tma::expect_bytes(x2_bwd_arrived, sizeof(x2_bwd_smem));
                    tma::load_async(x2_bwd_smem, g.X2_group, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank}, x2_bwd_arrived);

                    kittens::wait(z1_bar_freed, bwd_semaphore_phase);

                    tma::expect_bytes(std_fused_arrived, sizeof(std_fused_smem));
                    tma::load_async(std_fused_smem, g.std_fused_test_group, {batch_idx, head_idx, mini_batch_in_group_idx, 0}, std_fused_arrived);

                    tma::expect_bytes(grad_l_wrt_Z1_arrived, sizeof(grad_l_wrt_Z1_smem));
                    tma::load_async(grad_l_wrt_Z1_smem, g.grad_l_wrt_Z1_group, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank}, grad_l_wrt_Z1_arrived);
                    
                    kittens::wait(grad_L_Z2_bar_freed, bwd_semaphore_phase);

                    tma::expect_bytes(z1_bwd_arrived, sizeof(z1_bwd_smem));
                    tma::load_async(z1_bwd_smem, g.Z1_group, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank}, z1_bwd_arrived);

                    // load in new weights
                    kittens::wait(old_weights_done, bwd_semaphore_phase);
                    tma::expect_bytes(w1_remat_smem_arrived, sizeof(w1_smem));
                    tma::load_async(w1_smem, g.W1_init_group, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank}, w1_remat_smem_arrived);
                    tma::expect_bytes(b1_remat_smem_arrived, sizeof(b1_smem));
                    tma::load_async(b1_smem, g.b1_init_group, {batch_idx, head_idx, mini_batch_in_group_idx, tp_shard_rank}, b1_remat_smem_arrived);
                    tma::expect_bytes(w2_remat_smem_arrived, sizeof(w2_smem));
                    tma::load_async(w2_smem, g.W2_init_group, {batch_idx, head_idx, mini_batch_in_group_idx * TP + tp_shard_rank, 0}, w2_remat_smem_arrived);
                    tma::expect_bytes(b2_remat_smem_arrived, sizeof(b2_smem));
                    tma::load_async(b2_smem, g.b2_init_group, {batch_idx, head_idx, mini_batch_in_group_idx, 0}, b2_remat_smem_arrived);

                    kittens::wait(grad_L_XQW_mini_batch_sem_freed, bwd_semaphore_phase);

                    tma::expect_bytes(x_hat_fused_arrived, sizeof(x_hat_fused_smem));
                    tma::load_async(x_hat_fused_smem, g.x_hat_fused_group, {batch_idx, head_idx, mini_batch_in_group_idx, 0}, x_hat_fused_arrived);

                    tp_reduce_arrive();

                    kittens::wait(cs_f_store2_smem_freed, bwd_semaphore_phase);

                    tma::expect_bytes(grad_output_fused_arrived, sizeof(grad_output_fused_smem));
                    tma::load_async(grad_output_fused_smem, g.grad_output_fused_group, {batch_idx, head_idx, mini_batch_in_group_idx, 0}, grad_output_fused_arrived);

                    kittens::wait(grad_L_reconstruction_target_freed, bwd_semaphore_phase);

                    tma::expect_bytes(grad_x_hat_fused_arrived, sizeof(grad_x_hat_fused_smem));
                    tma::load_async(grad_x_hat_fused_smem, g.grad_x_hat_fused_group, {batch_idx, head_idx, mini_batch_in_group_idx, 0}, grad_x_hat_fused_arrived);
                }
            }
            else // consumer
            {
                // Try to reuse same registers for better tracking of register usage
                rt_fl<16, K::F> cs_cs_fl_reg;
                rt_fl<16, K::F> cs_cs_fl_reg2;
                typeof(cs_cs_fl_reg)::row_vec cs_row_fl_reg;
                typeof(cs_cs_fl_reg)::col_vec cs_col_fl_reg;
                typeof(cs_cs_fl_reg)::col_vec cs_col_fl_reg2;

                const int bwd_semaphore_phase = bwd_semaphore_idx % 2;
                
                kittens::wait(grad_L_XQW_mini_batch_sem_arrived, bwd_semaphore_phase);
                kittens::wait(x_hat_ln_arrived, bwd_semaphore_phase);
                warpgroup::load(cs_cs_fl_reg, grad_L_XQW_mini_batch_smem);

                warpgroup::load(cs_cs_fl_reg2, x_hat_ln_smem);
                mul(cs_cs_fl_reg2, cs_cs_fl_reg2, cs_cs_fl_reg);
                warpgroup::store(b_acc_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                warpgroup::col_sum(grad_L_ttt_norm_weight_smem, b_acc_smem, grad_L_ttt_norm_weight_smem);
                warpgroup::sync(warpgroupid+1);
                warpgroup::store(b_acc_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                warpgroup::col_sum(grad_L_ttt_norm_bias_smem, b_acc_smem, grad_L_ttt_norm_bias_smem);

                // grad_L_x_hat_ln
                load(cs_row_fl_reg, ttt_norm_weight_smem);
                mul_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg); // grad_L_x_hat_ln

                // grad_L_Z2_bar
                warpgroup::load(cs_cs_fl_reg2, x_hat_ln_smem);
                mul(cs_cs_fl_reg2, cs_cs_fl_reg, cs_cs_fl_reg2);
                zero(cs_col_fl_reg);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg2);
                warpgroup::load(cs_cs_fl_reg2, x_hat_ln_smem);
                mul_row(cs_cs_fl_reg2, cs_cs_fl_reg2, cs_col_fl_reg); // 3rd line, not negative

                if (warpgroup::laneid() == 0) arrive(x_hat_ln_freed, 1);

                zero(cs_col_fl_reg);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                add_row(cs_cs_fl_reg2, cs_cs_fl_reg2, cs_col_fl_reg);

                mul(cs_cs_fl_reg, cs_cs_fl_reg, static_cast<float>(head_dim));
                sub(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                div(cs_cs_fl_reg, cs_cs_fl_reg, static_cast<float>(head_dim));

                kittens::wait(std_ln_arrived, bwd_semaphore_phase);
                warpgroup::load(cs_col_fl_reg, std_ln_smem);
                div_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg);
                warpgroup::store(grad_L_Z2_bar_smem, cs_cs_fl_reg); // grad_L_Z2_bar
                warpgroup::sync(warpgroupid+1);

                // grad_L_Z1_bar
                warpgroup::load(cs_cs_fl_reg2, w2_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg2);
                zero(cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mma_ABt(cs_cs_fl_reg, grad_L_Z2_bar_smem, matmul_smem); // grad_L_X2_bar

                kittens::wait(z1_bar_group_arrived, bwd_semaphore_phase);
                warpgroup::load(cs_cs_fl_reg2, z1_bar_smem);
                if (warpgroup::laneid() == 0) arrive(z1_bar_freed, 1);
                warpgroup::mma_async_wait();
                gelu_bwd(cs_cs_fl_reg2, cs_cs_fl_reg2);
                mul(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2); // grad_L_Z1_bar
                warpgroup::store(grad_L_Z1_bar_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                // grad_L_W1_last
                warpgroup::load(cs_cs_fl_reg2, grad_L_W1_smem);
                kittens::wait(bwd_q_sem_arrived, bwd_semaphore_phase);
                warpgroup::sync(warpgroupid+1);

                warpgroup::mma_AtB(cs_cs_fl_reg2, bwd_q_smem, grad_L_Z1_bar_smem);
                warpgroup::mma_async_wait();

                // grad_L_b1_last
                warpgroup::store(b_acc_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                warpgroup::col_sum(grad_L_b1_smem, b_acc_smem, grad_L_b1_smem);

                warpgroup::store(grad_L_W1_smem, cs_cs_fl_reg2);
                warpgroup::store(cs_f_store_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);

                // grad_L_W2_last
                warpgroup::load(cs_cs_fl_reg2, grad_L_W2_smem);
                kittens::wait(x2_bar_group_arrived, bwd_semaphore_phase);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mma_AtB(cs_cs_fl_reg2, x2_bar_smem, grad_L_Z2_bar_smem);
                warpgroup::mma_async_wait();

                // grad_L_b2_last
                warpgroup::load(cs_cs_fl_reg, grad_L_Z2_bar_smem);
                warpgroup::store(b_acc_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                warpgroup::col_sum(grad_L_b2_smem, b_acc_smem, grad_L_b2_smem);

                warpgroup::store(grad_L_W2_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);

                if (warpgroup::laneid() == 0) arrive(grad_L_Z2_bar_freed, 1);

                // grad_L_XQ_mini_batch
                warpgroup::load(cs_cs_fl_reg, w1_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                if (warpgroup::laneid() == 0) arrive(old_weights_done, 1);

                if (tp_shard_rank == 0)
                {
                    warpgroup::load(cs_cs_fl_reg, grad_L_XQW_mini_batch_smem);
                }
                else
                {
                    zero(cs_cs_fl_reg);
                }
                warpgroup::sync(warpgroupid+1);
                if (warpgroup::laneid() == 0) arrive(grad_L_XQW_mini_batch_sem_freed, 1);

                warpgroup::mma_ABt(cs_cs_fl_reg, grad_L_Z1_bar_smem, matmul_smem);
                warpgroup::mma_async_wait();

                warpgroup::store(cs_f_store2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                if (wg_warpid == 0)
                {
                    tma::store_add_async(g.grad_L_XQ, cs_f_store2_smem, {batch_idx, head_idx, global_mini_batch_idx, 0});
                    tma::store_commit_group();
                }
                tma::store_async_wait();

                // grad_L_last_eta_in_mini_batch
                warpgroup::load(cs_cs_fl_reg, grad_L_W2_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg);
                zero(cs_cs_fl_reg);
                kittens::wait(grad_l_wrt_Z2_arrived, bwd_semaphore_phase);
                kittens::wait(x2_bwd_arrived, bwd_semaphore_phase);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mma_ABt(cs_cs_fl_reg, grad_l_wrt_Z2_smem, matmul_smem);
                warpgroup::load(cs_cs_fl_reg2, x2_bwd_smem);
                zero(cs_col_fl_reg2);
                warpgroup::mma_async_wait();
                mul(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                row_sum(cs_col_fl_reg2, cs_cs_fl_reg);
                warpgroup::store(grad_L_last_eta_smem, cs_col_fl_reg2); // first line
                warpgroup::sync(warpgroupid+1);

                load(cs_row_fl_reg, grad_L_b2_smem);
                warpgroup::load(cs_cs_fl_reg, grad_l_wrt_Z2_smem);
                mul_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                warpgroup::sync(warpgroupid+1);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                if (tp_shard_rank == 0)
                {
                    add(cs_col_fl_reg2, cs_col_fl_reg2, cs_col_fl_reg); // second line
                }
                

                warpgroup::load(cs_cs_fl_reg, grad_L_W1_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg);
                zero(cs_cs_fl_reg);
                kittens::wait(grad_l_wrt_Z1_arrived, bwd_semaphore_phase);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mma_ABt(cs_cs_fl_reg, grad_l_wrt_Z1_smem, matmul_smem);
                warpgroup::load(cs_cs_fl_reg2, bwd_k_smem);
                warpgroup::mma_async_wait();
                mul(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                add(cs_col_fl_reg2, cs_col_fl_reg2, cs_col_fl_reg); // third line

                load(cs_row_fl_reg, grad_L_b1_smem);
                warpgroup::load(cs_cs_fl_reg, grad_l_wrt_Z1_smem);
                mul_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                add(cs_col_fl_reg2, cs_col_fl_reg2, cs_col_fl_reg);
                mul(cs_col_fl_reg2, cs_col_fl_reg2, -1.0f); // fourth line
                warpgroup::store(grad_L_last_eta_smem, cs_col_fl_reg2);

                warpgroup::sync(warpgroupid+1);
                if (wg_warpid == 0) {
                    tma::store_add_async(g.grad_L_last_eta, grad_L_last_eta_smem, {batch_idx, head_idx, global_mini_batch_idx, 0});
                    tma::store_commit_group();
                }

                // grad_L_grad_l_wrt_Z1
                warpgroup::mul_row(cs_f_store_smem, bwd_k_smem, bwd_last_eta_smem); 
                warpgroup::load(cs_cs_fl_reg, grad_L_W1_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg);   
                zero(cs_cs_fl_reg);  
                warpgroup::sync(warpgroupid+1);       
                warpgroup::mma_AB(cs_cs_fl_reg, cs_f_store_smem, matmul_smem);
                load(cs_row_fl_reg, grad_L_b1_smem);
                broadcast_col(cs_cs_fl_reg2, cs_row_fl_reg);
                warpgroup::load(cs_col_fl_reg, bwd_last_eta_smem);
                mul_row(cs_cs_fl_reg2, cs_cs_fl_reg2, cs_col_fl_reg);
                warpgroup::mma_async_wait();
                add(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                mul(cs_cs_fl_reg, cs_cs_fl_reg, -1.0f);
                warpgroup::store(grad_L_grad_l_wrt_Z1_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                // grad_L_grad_l_wrt_Z2
                warpgroup::mul_row(cs_f_store_smem, x2_bwd_smem, bwd_last_eta_smem);
                warpgroup::load(cs_cs_fl_reg, grad_L_W2_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg);     
                zero(cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);       
                warpgroup::mma_AB(cs_cs_fl_reg, cs_f_store_smem, matmul_smem);
                load(cs_row_fl_reg, grad_L_b2_smem);
                broadcast_col(cs_cs_fl_reg2, cs_row_fl_reg);
                warpgroup::load(cs_col_fl_reg, bwd_last_eta_smem);
                mul_row(cs_cs_fl_reg2, cs_cs_fl_reg2, cs_col_fl_reg);
                warpgroup::mma_async_wait();
                if (tp_shard_rank == 0)
                {
                    add(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                }
                warpgroup::store(grad_L_grad_l_wrt_Z2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
               
                warpgroup::load(cs_cs_fl_reg, grad_L_grad_l_wrt_Z1_smem);
                kittens::wait(z1_bwd_arrived, bwd_semaphore_phase);
                warpgroup::load(cs_cs_fl_reg2, z1_bwd_smem);
                gelu_bwd(cs_cs_fl_reg2, cs_cs_fl_reg2);
                mul(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                kittens::wait(w2_remat_smem_arrived, bwd_semaphore_phase);
                warpgroup::load(cs_cs_fl_reg2, w2_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg2);
                warpgroup::store(cs_f_store2_smem, cs_cs_fl_reg);
                zero(cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mma_AB(cs_cs_fl_reg2, cs_f_store2_smem, matmul_smem);
                warpgroup::load(cs_cs_fl_reg, grad_L_grad_l_wrt_Z2_smem);
                warpgroup::mma_async_wait();

                sub(cs_cs_fl_reg2, cs_cs_fl_reg2, cs_cs_fl_reg);
                warpgroup::store(grad_L_grad_l_wrt_Z2_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);

                tp_reduce(grad_L_grad_l_wrt_Z2_smem, reduction_buffer, bwd_dsmem_semaphore1, bwd_dsmem_semaphore2, bwd_semaphore_idx);

                // grad_L_XK_mini_batch
                zero(cs_cs_fl_reg);
                warpgroup::load(cs_cs_fl_reg2, grad_L_W1_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mma_ABt(cs_cs_fl_reg, grad_l_wrt_Z1_smem, matmul_smem);
                warpgroup::load(cs_col_fl_reg, bwd_last_eta_smem);
                warpgroup::mma_async_wait();
                mul_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg);
                mul(cs_cs_fl_reg, cs_cs_fl_reg, -1.0f);
                warpgroup::store(cs_f_store_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                if (wg_warpid == 0)
                {
                    tma::store_add_async(g.grad_L_XK, cs_f_store_smem, {batch_idx, head_idx, global_mini_batch_idx, 0});
                    tma::store_commit_group();
                }

                if (warpgroup::laneid() == 0) arrive(cs_f_store2_smem_freed, 1);
                
                // grad_L_grad_x_hat_fused
                warpgroup::load(cs_cs_fl_reg, grad_L_grad_l_wrt_Z2_smem);
                kittens::wait(x_hat_fused_arrived, bwd_semaphore_phase);
                warpgroup::load(cs_cs_fl_reg2, x_hat_fused_smem);
                mul(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                mul_row(cs_cs_fl_reg2, cs_cs_fl_reg2, cs_col_fl_reg);

                warpgroup::load(cs_cs_fl_reg, grad_L_grad_l_wrt_Z2_smem);
                zero(cs_col_fl_reg2);
                row_sum(cs_col_fl_reg2, cs_cs_fl_reg);
                add_row(cs_cs_fl_reg2, cs_cs_fl_reg2, cs_col_fl_reg2);
                div(cs_cs_fl_reg2, cs_cs_fl_reg2, static_cast<float>(-1*head_dim));

                add(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                kittens::wait(std_fused_arrived, bwd_semaphore_phase);
                warpgroup::load(cs_col_fl_reg, std_fused_smem);
                div_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg);
                warpgroup::store(grad_L_grad_x_hat_fused_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                // grad_L_reconstruction_target
                load(cs_row_fl_reg, ttt_norm_weight_smem);
                mul_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                warpgroup::store(cs_f_store_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                mul(cs_cs_fl_reg2, cs_cs_fl_reg, -1.0f);
                warpgroup::store(grad_L_reconstruction_target_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);

                if (tp_shard_rank == 0 && wg_warpid == 0)
                {
                    tma::store_add_async(g.grad_L_XV, grad_L_reconstruction_target_smem, {batch_idx, head_idx, global_mini_batch_idx, 0});
                    tma::store_add_async(g.grad_L_XK, cs_f_store_smem, {batch_idx, head_idx, global_mini_batch_idx, 0});
                    tma::store_commit_group();
                }
                tma::store_async_wait();

                warpgroup::sync(warpgroupid+1);
                // grad_L_y already calculated above
                // grad_L_x_hat_fused
                mul_col(cs_cs_fl_reg2, cs_cs_fl_reg, cs_row_fl_reg);
                warpgroup::store(cs_f_store_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);

                // do ttt_norm_bias
                warpgroup::store(b_acc_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                warpgroup::col_sum(grad_L_ttt_norm_bias_smem, b_acc_smem, grad_L_ttt_norm_bias_smem);
                warpgroup::sync(warpgroupid+1);

                // And do ttt_norm_weight
                kittens::wait(grad_output_fused_arrived, bwd_semaphore_phase);
                warpgroup::load(cs_cs_fl_reg2, x_hat_fused_smem);
                mul(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                warpgroup::store(b_acc_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                warpgroup::load(cs_cs_fl_reg, grad_output_fused_smem);
                warpgroup::load(cs_cs_fl_reg2, grad_L_grad_x_hat_fused_smem);
                mul(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                warpgroup::load(cs_cs_fl_reg2, b_acc_smem);
                if (warpgroup::laneid() == 0) arrive(grad_L_reconstruction_target_freed, 1);
                add(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                warpgroup::store(b_acc_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                warpgroup::col_sum(grad_L_ttt_norm_weight_smem, b_acc_smem, grad_L_ttt_norm_weight_smem);
                warpgroup::sync(warpgroupid+1);

                kittens::wait(grad_x_hat_fused_arrived, bwd_semaphore_phase);
                warpgroup::load(cs_cs_fl_reg, grad_x_hat_fused_smem);
                warpgroup::load(cs_cs_fl_reg2, x_hat_fused_smem);
                mul(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                zero(cs_col_fl_reg2);
                row_sum(cs_col_fl_reg2, cs_cs_fl_reg);
                warpgroup::load(cs_cs_fl_reg, grad_L_grad_l_wrt_Z2_smem);
                mul(cs_cs_fl_reg2, cs_cs_fl_reg, cs_cs_fl_reg2);
                mul_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg2);
                zero(cs_col_fl_reg);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg2);
                warpgroup::load(cs_cs_fl_reg2, grad_x_hat_fused_smem);
                mul_row(cs_cs_fl_reg2, cs_cs_fl_reg2, cs_col_fl_reg);
                add(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                
                warpgroup::load(cs_col_fl_reg, std_fused_smem);
                div_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg);
                warpgroup::sync(warpgroupid+1);
                div(cs_cs_fl_reg, cs_cs_fl_reg, -1.0f * static_cast<float>(head_dim));
                warpgroup::load(cs_cs_fl_reg2, cs_f_store_smem);
                add(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                warpgroup::store(grad_L_x_hat_fused_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                // grad_L_std
                warpgroup::load(cs_cs_fl_reg2, x_hat_fused_smem);
                mul(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                warpgroup::mul(grad_L_grad_l_wrt_Z2_smem, grad_L_grad_l_wrt_Z2_smem, grad_l_wrt_Z2_smem);
                warpgroup::load(cs_cs_fl_reg2, grad_L_grad_l_wrt_Z2_smem);
                add(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                warpgroup::load(cs_col_fl_reg, std_fused_smem);
                div_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg);
                mul(cs_cs_fl_reg, cs_cs_fl_reg, -1.0f);

                // grad_L_Z2
                warpgroup::load(cs_cs_fl_reg2, x_hat_fused_smem);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                mul_row(cs_cs_fl_reg2, cs_cs_fl_reg2, cs_col_fl_reg);
                warpgroup::load(cs_cs_fl_reg, grad_L_x_hat_fused_smem);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                warpgroup::load(cs_col_fl_reg2, std_fused_smem);
                div(cs_col_fl_reg, cs_col_fl_reg, cs_col_fl_reg2);
                sub_row(cs_cs_fl_reg2, cs_cs_fl_reg2, cs_col_fl_reg);
                div(cs_cs_fl_reg2, cs_cs_fl_reg2, static_cast<float>(head_dim));
                warpgroup::load(cs_cs_fl_reg, grad_L_x_hat_fused_smem);
                div_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg2);
                add(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                warpgroup::store(grad_L_Z2_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                // grad_L_X2
                kittens::wait(w2_remat_smem_arrived, bwd_semaphore_phase);
                warpgroup::load(cs_cs_fl_reg, w2_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                zero(cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mma_ABt(cs_cs_fl_reg, grad_L_Z2_smem, matmul_smem);  
                warpgroup::mma_async_wait();

                if (wg_warpid == 0)
                {
                    tma::store_add_async(g.o, grad_L_Z2_smem, {batch_idx, head_idx, global_mini_batch_idx, 0});
                    tma::store_commit_group();
                }
                
                warpgroup::load(cs_cs_fl_reg2, grad_L_W2_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                zero(cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mma_ABt(cs_cs_fl_reg2, grad_l_wrt_Z2_smem, matmul_smem);
                warpgroup::load(cs_col_fl_reg, bwd_last_eta_smem);
                warpgroup::mma_async_wait();
                mul_row(cs_cs_fl_reg2, cs_cs_fl_reg2, cs_col_fl_reg);
                sub(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);

                // grad_L_W2_init
                warpgroup::load(cs_cs_fl_reg2, z1_bwd_smem);
                gelu_bwd(cs_cs_fl_reg2, cs_cs_fl_reg2);
                warpgroup::store(cs_f_store_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mul(cs_f_store_smem, cs_f_store_smem, grad_L_grad_l_wrt_Z1_smem);
                warpgroup::load(cs_cs_fl_reg2, grad_L_W2_smem);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mma_AtB(cs_cs_fl_reg2, cs_f_store_smem, grad_l_wrt_Z2_smem);
                warpgroup::mma_async_wait();
                warpgroup::store(grad_L_W2_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);

                // grad_L_Z1
                warpgroup::load(cs_cs_fl_reg2, z1_bwd_smem);
                gelu_bwd(cs_cs_fl_reg2, cs_cs_fl_reg2);
                mul(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);

                warpgroup::load(cs_cs_fl_reg2, w2_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                zero(cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mma_ABt(cs_cs_fl_reg2, grad_l_wrt_Z2_smem, matmul_smem);
                warpgroup::gelu_bwd_bwd(z1_bwd_smem, z1_bwd_smem);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mul(grad_L_grad_l_wrt_Z1_smem, grad_L_grad_l_wrt_Z1_smem, z1_bwd_smem);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mma_async_wait();
                warpgroup::store(matmul_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mul(matmul_smem, matmul_smem, grad_L_grad_l_wrt_Z1_smem);
                warpgroup::sync(warpgroupid+1);
                warpgroup::load(cs_cs_fl_reg2, matmul_smem);
                add(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                warpgroup::store(grad_L_Z1_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                
                // grad_L_XK
                kittens::wait(w1_remat_smem_arrived, bwd_semaphore_phase);
                warpgroup::load(cs_cs_fl_reg, w1_smem);
                warpgroup::store(matmul_smem, cs_cs_fl_reg);
                zero(cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mma_ABt(cs_cs_fl_reg, grad_L_Z1_smem, matmul_smem);
                warpgroup::mma_async_wait();

                warpgroup::store(grad_L_XK_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                if (wg_warpid == 0)
                {
                    tma::store_add_async(g.grad_L_XK, grad_L_XK_smem, {batch_idx, head_idx, global_mini_batch_idx, 0});
                    tma::store_commit_group();
                }

                // grad_L_W2_smem
                warpgroup::load(cs_cs_fl_reg, grad_L_W2_smem);
                warpgroup::mma_AtB(cs_cs_fl_reg, x2_bwd_smem, grad_L_Z2_smem);

                // grad_L_b2_last
                warpgroup::load(cs_cs_fl_reg2, grad_L_Z2_smem);
                warpgroup::store(b_acc_smem, cs_cs_fl_reg2);
                warpgroup::sync(warpgroupid+1);
                warpgroup::col_sum(grad_L_b2_smem, b_acc_smem, grad_L_b2_smem);
                warpgroup::sync(warpgroupid+1);

                warpgroup::mma_async_wait();
                warpgroup::store(grad_L_W2_smem, cs_cs_fl_reg);
                warpgroup::load(cs_cs_fl_reg2, grad_L_W1_smem);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mma_AtB(cs_cs_fl_reg2, bwd_k_smem, grad_L_Z1_smem);
        
                // grad_L_b1_last
                warpgroup::load(cs_cs_fl_reg, grad_L_Z1_smem);
                warpgroup::store(b_acc_smem, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                warpgroup::col_sum(grad_L_b1_smem, b_acc_smem, grad_L_b1_smem);
                warpgroup::sync(warpgroupid+1);

                warpgroup::mma_async_wait();
                warpgroup::store(grad_L_W1_smem, cs_cs_fl_reg2);

                if (warpgroup::laneid() == 0) arrive(bwd_compute_done, 1);
            }

            ++bwd_semaphore_idx;
        }

        if (is_producer && warpid == NUM_WORKERS - 4)
        {
            const int checkpoint_phase = (g.num_checkpoints - checkpoint_idx - 1) % 2;
            kittens::wait(backward_done, checkpoint_phase);
        }
        else
        {
            if (warpgroup::laneid() == 0) arrive(backward_done, 1);
        }
    }

    // Store out grad of weights
    // Using consumer here for synchronization
    if (!is_producer && wg_warpid == 0)
    {
        tma::store_async(g.grad_L_W1_init, grad_L_W1_smem, {batch_idx, head_idx, 0, tp_shard_rank});
        tma::store_async(g.grad_L_b1_init, grad_L_b1_smem, {batch_idx, head_idx, 0, tp_shard_rank});
        tma::store_async(g.grad_L_W2_init, grad_L_W2_smem, {batch_idx, head_idx, tp_shard_rank, 0});
        tma::store_async(g.grad_L_b2_init, grad_L_b2_smem, {batch_idx, head_idx, 0, 0});
        if (tp_shard_rank == 0)
        {
            tma::store_async(g.grad_L_ttt_norm_weight, grad_L_ttt_norm_weight_smem, {batch_idx, head_idx, 0, 0});
            tma::store_async(g.grad_L_ttt_norm_bias, grad_L_ttt_norm_bias_smem, {batch_idx, head_idx, 0, 0});
        }
        
        tma::store_commit_group();
    }
}

#if TORCH_COMPILE

#include "common/pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

torch::Tensor ttt_backward(
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
    const torch::Tensor Out, // Unused
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
) {
    constexpr int F = 64;
    constexpr int K = 4;
    const unsigned long B = XQ.size(0);
    const unsigned long H = XQ.size(1);
    const unsigned long NH = XQ.size(1);
    const unsigned long T = XQ.size(2) * XQ.size(3); // seq len
    const unsigned long NC = XQ.size(2);
    const unsigned long CS = XQ.size(3);
    const unsigned long num_checkpoints = static_cast<int>(W1_checkpoints.size(2));
    
    // Only perform the checks on the first call.
    static bool checks_done = false;
    if (!checks_done) {
        TORCH_CHECK(XQ.device().is_cuda() && XQ.is_contiguous() && XQ.dim() == 5 && XQ.size(4) == F, "Encountered error with XQ, please check the shape and if it is contiguous.");
        TORCH_CHECK(XK.device().is_cuda() && XK.is_contiguous() && XK.dim() == 5 && XK.size(4) == F, "Encountered error with XK, please check the shape and if it is contiguous.");
        TORCH_CHECK(XV.device().is_cuda() && XV.is_contiguous() && XV.dim() == 5 && XV.size(4) == F, "Encountered error with XV, please check the shape and if it is contiguous.");
        TORCH_CHECK(W1_checkpoints.device().is_cuda() && W1_checkpoints.is_contiguous() && W1_checkpoints.dim() == 5 && W1_checkpoints.size(0) == B && W1_checkpoints.size(1) == H && W1_checkpoints.size(2) == num_checkpoints && W1_checkpoints.size(3) == F && W1_checkpoints.size(4) == F*K, "Encountered error with W1_checkpoints, please check the shape and if it is contiguous.");
        TORCH_CHECK(W2_checkpoints.device().is_cuda() && W2_checkpoints.is_contiguous() && W2_checkpoints.dim() == 5 && W2_checkpoints.size(0) == B && W2_checkpoints.size(1) == H && W2_checkpoints.size(2) == num_checkpoints && W2_checkpoints.size(3) == F*K && W2_checkpoints.size(4) == F, "Encountered error with W2_checkpoints, please check the shape and if it is contiguous.");

        TORCH_CHECK(ttt_norm_weight.device().is_cuda() && ttt_norm_weight.is_contiguous() && ttt_norm_weight.dim() == 4 && ttt_norm_weight.size(0) == 1 && ttt_norm_weight.size(1) == H && ttt_norm_weight.size(2) == 1 && ttt_norm_weight.size(2) == 1 && ttt_norm_weight.size(3) == F, "Encountered error with ttt_norm_weight, please check the shape and if it is contiguous.");

        TORCH_CHECK(last_eta.device().is_cuda() && last_eta.is_contiguous() &&
                last_eta.dim() == 5 && last_eta.size(4) == 1,
                "Encountered error with last_eta, please check the shape and if it is contiguous.");

        // Check ttt_norm_bias: expecting a 4-D tensor with shape [1, H, 1, F]
        TORCH_CHECK(ttt_norm_bias.device().is_cuda() && ttt_norm_bias.is_contiguous() &&
                    ttt_norm_bias.dim() == 4 && ttt_norm_bias.size(0) == 1 &&
                    ttt_norm_bias.size(1) == H && ttt_norm_bias.size(2) == 1 &&
                    ttt_norm_bias.size(3) == F,
                    "Encountered error with ttt_norm_bias, please check the shape and if it is contiguous.");

        // Check b1_checkpoints: expecting a 5-D tensor with shape [B, NH, K, 1, F*4]
        TORCH_CHECK(b1_checkpoints.device().is_cuda() && b1_checkpoints.is_contiguous() &&
                    b1_checkpoints.dim() == 5 && b1_checkpoints.size(0) == B &&
                    b1_checkpoints.size(1) == NH && b1_checkpoints.size(2) == num_checkpoints &&
                    b1_checkpoints.size(3) == 1 && b1_checkpoints.size(4) == F * 4,
                    "Encountered error with b1_checkpoints, please check the shape and if it is contiguous.");

        // Check b2_checkpoints: expecting a 5-D tensor with shape [B, NH, num_checkpoints, 1, F]
        TORCH_CHECK(b2_checkpoints.device().is_cuda() && b2_checkpoints.is_contiguous() &&
                    b2_checkpoints.dim() == 5 && b2_checkpoints.size(0) == B &&
                    b2_checkpoints.size(1) == NH && b2_checkpoints.size(2) == num_checkpoints &&
                    b2_checkpoints.size(3) == 1 && b2_checkpoints.size(4) == F,
                    "Encountered error with b2_checkpoints, please check the shape and if it is contiguous.");

        // Check W1_init_group: expecting a 5-D tensor with shape [B, NH, checkpoint_group_size, F, F*4]
        TORCH_CHECK(W1_init_group.device().is_cuda() && W1_init_group.is_contiguous() &&
                    W1_init_group.dim() == 5 && W1_init_group.size(0) == B &&
                    W1_init_group.size(1) == NH && W1_init_group.size(2) == checkpoint_group_size &&
                    W1_init_group.size(3) == F && W1_init_group.size(4) == F * 4,
                    "Encountered error with W1_init_group, please check the shape and if it is contiguous.");

        // Check b1_init_group: expecting a 5-D tensor with shape [B, NH, checkpoint_group_size, 1, F*4]
        TORCH_CHECK(b1_init_group.device().is_cuda() && b1_init_group.is_contiguous() &&
                    b1_init_group.dim() == 5 && b1_init_group.size(0) == B &&
                    b1_init_group.size(1) == NH && b1_init_group.size(2) == checkpoint_group_size &&
                    b1_init_group.size(3) == 1 && b1_init_group.size(4) == F * 4,
                    "Encountered error with b1_init_group, please check the shape and if it is contiguous.");

        // Check W2_init_group: expecting a 5-D tensor with shape [B, NH, checkpoint_group_size, F*4, F]
        TORCH_CHECK(W2_init_group.device().is_cuda() && W2_init_group.is_contiguous() &&
                    W2_init_group.dim() == 5 && W2_init_group.size(0) == B &&
                    W2_init_group.size(1) == NH && W2_init_group.size(2) == checkpoint_group_size &&
                    W2_init_group.size(3) == F * 4 && W2_init_group.size(4) == F,
                    "Encountered error with W2_init_group, please check the shape and if it is contiguous.");

        // Check b2_init_group: expecting a 5-D tensor with shape [B, NH, checkpoint_group_size, 1, F]
        TORCH_CHECK(b2_init_group.device().is_cuda() && b2_init_group.is_contiguous() &&
                    b2_init_group.dim() == 5 && b2_init_group.size(0) == B &&
                    b2_init_group.size(1) == NH && b2_init_group.size(2) == checkpoint_group_size &&
                    b2_init_group.size(3) == 1 && b2_init_group.size(4) == F,
                    "Encountered error with b2_init_group, please check the shape and if it is contiguous.");

        // Check x_hat_ln_group: expecting a 5-D tensor with shape [B, NH, checkpoint_group_size, CS, F]
        TORCH_CHECK(x_hat_ln_group.device().is_cuda() && x_hat_ln_group.is_contiguous() &&
                    x_hat_ln_group.dim() == 5 && x_hat_ln_group.size(0) == B &&
                    x_hat_ln_group.size(1) == NH && x_hat_ln_group.size(2) == checkpoint_group_size &&
                    x_hat_ln_group.size(3) == CS && x_hat_ln_group.size(4) == F,
                    "Encountered error with x_hat_ln_group, please check the shape and if it is contiguous.");

        // Check std_ln_group: expecting a 5-D tensor with shape [B, NH, checkpoint_group_size, CS, 1]
        TORCH_CHECK(std_ln_group.device().is_cuda() && std_ln_group.is_contiguous() &&
                    std_ln_group.dim() == 5 && std_ln_group.size(0) == B &&
                    std_ln_group.size(1) == NH && std_ln_group.size(2) == checkpoint_group_size &&
                    std_ln_group.size(3) == CS && std_ln_group.size(4) == 1,
                    "Encountered error with std_ln_group, please check the shape and if it is contiguous.");

        // Check X2_group: expecting a 5-D tensor with shape [B, NH, checkpoint_group_size, CS, F*4]
        TORCH_CHECK(X2_group.device().is_cuda() && X2_group.is_contiguous() &&
                    X2_group.dim() == 5 && X2_group.size(0) == B &&
                    X2_group.size(1) == NH && X2_group.size(2) == checkpoint_group_size &&
                    X2_group.size(3) == CS && X2_group.size(4) == F * 4,
                    "Encountered error with X2_group, please check the shape and if it is contiguous.");

        // Check Z1_group: expecting a 5-D tensor with shape [B, NH, checkpoint_group_size, CS, F*4]
        TORCH_CHECK(Z1_group.device().is_cuda() && Z1_group.is_contiguous() &&
                    Z1_group.dim() == 5 && Z1_group.size(0) == B &&
                    Z1_group.size(1) == NH && Z1_group.size(2) == checkpoint_group_size &&
                    Z1_group.size(3) == CS && Z1_group.size(4) == F * 4,
                    "Encountered error with Z1_group, please check the shape and if it is contiguous.");

        // Check Z1_bar_group: expecting a 5-D tensor with shape [B, NH, checkpoint_group_size, CS, F*4]
        TORCH_CHECK(Z1_bar_group.device().is_cuda() && Z1_bar_group.is_contiguous() &&
                    Z1_bar_group.dim() == 5 && Z1_bar_group.size(0) == B &&
                    Z1_bar_group.size(1) == NH && Z1_bar_group.size(2) == checkpoint_group_size &&
                    Z1_bar_group.size(3) == CS && Z1_bar_group.size(4) == F * 4,
                    "Encountered error with Z1_bar_group, please check the shape and if it is contiguous.");

        // Check X2_bar_group: expecting a 5-D tensor with shape [B, NH, checkpoint_group_size, CS, F*4]
        TORCH_CHECK(X2_bar_group.device().is_cuda() && X2_bar_group.is_contiguous() &&
                    X2_bar_group.dim() == 5 && X2_bar_group.size(0) == B &&
                    X2_bar_group.size(1) == NH && X2_bar_group.size(2) == checkpoint_group_size &&
                    X2_bar_group.size(3) == CS && X2_bar_group.size(4) == F * 4,
                    "Encountered error with X2_bar_group, please check the shape and if it is contiguous.");

        // Check grad_l_wrt_Z2_group: expecting a 5-D tensor with shape [B, NH, checkpoint_group_size, CS, F]
        TORCH_CHECK(grad_l_wrt_Z2_group.device().is_cuda() && grad_l_wrt_Z2_group.is_contiguous() &&
                    grad_l_wrt_Z2_group.dim() == 5 && grad_l_wrt_Z2_group.size(0) == B &&
                    grad_l_wrt_Z2_group.size(1) == NH && grad_l_wrt_Z2_group.size(2) == checkpoint_group_size &&
                    grad_l_wrt_Z2_group.size(3) == CS && grad_l_wrt_Z2_group.size(4) == F,
                    "Encountered error with grad_l_wrt_Z2_group, please check the shape and if it is contiguous.");

        // Check grad_l_wrt_Z1_group: expecting a 5-D tensor with shape [B, NH, checkpoint_group_size, CS, F*4]
        TORCH_CHECK(grad_l_wrt_Z1_group.device().is_cuda() && grad_l_wrt_Z1_group.is_contiguous() &&
                    grad_l_wrt_Z1_group.dim() == 5 && grad_l_wrt_Z1_group.size(0) == B &&
                    grad_l_wrt_Z1_group.size(1) == NH && grad_l_wrt_Z1_group.size(2) == checkpoint_group_size &&
                    grad_l_wrt_Z1_group.size(3) == CS && grad_l_wrt_Z1_group.size(4) == F * 4,
                    "Encountered error with grad_l_wrt_Z1_group, please check the shape and if it is contiguous.");

        // Check x_hat_fused_group: expecting a 5-D tensor with shape [B, NH, checkpoint_group_size, CS, F]
        TORCH_CHECK(x_hat_fused_group.device().is_cuda() && x_hat_fused_group.is_contiguous() &&
                    x_hat_fused_group.dim() == 5 && x_hat_fused_group.size(0) == B &&
                    x_hat_fused_group.size(1) == NH && x_hat_fused_group.size(2) == checkpoint_group_size &&
                    x_hat_fused_group.size(3) == CS && x_hat_fused_group.size(4) == F,
                    "Encountered error with x_hat_fused_group, please check the shape and if it is contiguous.");

        // Check grad_x_hat_fused_group: expecting a 5-D tensor with shape [B, NH, checkpoint_group_size, CS, F]
        TORCH_CHECK(grad_x_hat_fused_group.device().is_cuda() && grad_x_hat_fused_group.is_contiguous() &&
                    grad_x_hat_fused_group.dim() == 5 && grad_x_hat_fused_group.size(0) == B &&
                    grad_x_hat_fused_group.size(1) == NH && grad_x_hat_fused_group.size(2) == checkpoint_group_size &&
                    grad_x_hat_fused_group.size(3) == CS && grad_x_hat_fused_group.size(4) == F,
                    "Encountered error with grad_x_hat_fused_group, please check the shape and if it is contiguous.");

        // Check grad_output_fused_group: expecting a 5-D tensor with shape [B, NH, checkpoint_group_size, CS, F]
        TORCH_CHECK(grad_output_fused_group.device().is_cuda() && grad_output_fused_group.is_contiguous() &&
                    grad_output_fused_group.dim() == 5 && grad_output_fused_group.size(0) == B &&
                    grad_output_fused_group.size(1) == NH && grad_output_fused_group.size(2) == checkpoint_group_size &&
                    grad_output_fused_group.size(3) == CS && grad_output_fused_group.size(4) == F,
                    "Encountered error with grad_output_fused_group, please check the shape and if it is contiguous.");

        // Check std_fused_group: expecting a 5-D tensor with shape [B, NH, checkpoint_group_size, CS, 1]
        TORCH_CHECK(std_fused_group.device().is_cuda() && std_fused_group.is_contiguous() &&
                    std_fused_group.dim() == 5 && std_fused_group.size(0) == B &&
                    std_fused_group.size(1) == NH && std_fused_group.size(2) == checkpoint_group_size &&
                    std_fused_group.size(3) == CS && std_fused_group.size(4) == 1,
                    "Encountered error with std_fused_group, please check the shape and if it is contiguous.");

        // Check grad_L_W1_last: expecting a 4-D tensor with shape [B, NH, F, F*4]
        TORCH_CHECK(grad_L_W1_last.device().is_cuda() && grad_L_W1_last.is_contiguous() &&
                    grad_L_W1_last.dim() == 4 && grad_L_W1_last.size(0) == B &&
                    grad_L_W1_last.size(1) == NH && grad_L_W1_last.size(2) == F &&
                    grad_L_W1_last.size(3) == F * 4,
                    "Encountered error with grad_L_W1_last, please check the shape and if it is contiguous.");

        // Check grad_L_b1_last: expecting a 4-D tensor with shape [B, NH, 1, F*4]
        TORCH_CHECK(grad_L_b1_last.device().is_cuda() && grad_L_b1_last.is_contiguous() &&
                    grad_L_b1_last.dim() == 4 && grad_L_b1_last.size(0) == B &&
                    grad_L_b1_last.size(1) == NH && grad_L_b1_last.size(2) == 1 &&
                    grad_L_b1_last.size(3) == F * 4,
                    "Encountered error with grad_L_b1_last, please check the shape and if it is contiguous.");

        // Check grad_L_W2_last: expecting a 4-D tensor with shape [B, NH, F*4, F]
        TORCH_CHECK(grad_L_W2_last.device().is_cuda() && grad_L_W2_last.is_contiguous() &&
                    grad_L_W2_last.dim() == 4 && grad_L_W2_last.size(0) == B &&
                    grad_L_W2_last.size(1) == NH && grad_L_W2_last.size(2) == F * 4 &&
                    grad_L_W2_last.size(3) == F,
                    "Encountered error with grad_L_W2_last, please check the shape and if it is contiguous.");

        // Check grad_L_b2_last: expecting a 4-D tensor with shape [B, NH, 1, F]
        TORCH_CHECK(grad_L_b2_last.device().is_cuda() && grad_L_b2_last.is_contiguous() &&
                    grad_L_b2_last.dim() == 4 && grad_L_b2_last.size(0) == B &&
                    grad_L_b2_last.size(1) == NH && grad_L_b2_last.size(2) == 1 &&
                    grad_L_b2_last.size(3) == F,
                    "Encountered error with grad_L_b2_last, please check the shape and if it is contiguous.");

        // Check grad_L_XQW_mini_batch: expecting a 5-D tensor with shape [B, NH, NC, CS, F]
        TORCH_CHECK(grad_L_XQW_mini_batch.device().is_cuda() && grad_L_XQW_mini_batch.is_contiguous() &&
                    grad_L_XQW_mini_batch.dim() == 5 && grad_L_XQW_mini_batch.size(0) == B &&
                    grad_L_XQW_mini_batch.size(1) == NH && grad_L_XQW_mini_batch.size(2) == NC &&
                    grad_L_XQW_mini_batch.size(3) == CS && grad_L_XQW_mini_batch.size(4) == F,
                    "Encountered error with grad_L_XQW_mini_batch, please check the shape and if it is contiguous.");

        // Check grad_L_ttt_norm_weight: expecting a 4-D tensor with shape [B, NH, 1, F]
        TORCH_CHECK(grad_L_ttt_norm_weight.device().is_cuda() && grad_L_ttt_norm_weight.is_contiguous() &&
                    grad_L_ttt_norm_weight.dim() == 4 && grad_L_ttt_norm_weight.size(0) == B &&
                    grad_L_ttt_norm_weight.size(1) == NH && grad_L_ttt_norm_weight.size(2) == 1 &&
                    grad_L_ttt_norm_weight.size(3) == F,
                    "Encountered error with grad_L_ttt_norm_weight, please check the shape and if it is contiguous.");

        // Check grad_L_ttt_norm_bias: expecting a 4-D tensor with shape [B, NH, 1, F]
        TORCH_CHECK(grad_L_ttt_norm_bias.device().is_cuda() && grad_L_ttt_norm_bias.is_contiguous() &&
                    grad_L_ttt_norm_bias.dim() == 4 && grad_L_ttt_norm_bias.size(0) == B &&
                    grad_L_ttt_norm_bias.size(1) == NH && grad_L_ttt_norm_bias.size(2) == 1 &&
                    grad_L_ttt_norm_bias.size(3) == F,
                    "Encountered error with grad_L_ttt_norm_bias, please check the shape and if it is contiguous.");

        // Check grad_L_W1_init: expecting a 4-D tensor with shape [B, NH, F, F*4]
        TORCH_CHECK(grad_L_W1_init.device().is_cuda() && grad_L_W1_init.is_contiguous() &&
                    grad_L_W1_init.dim() == 4 && grad_L_W1_init.size(0) == B &&
                    grad_L_W1_init.size(1) == NH && grad_L_W1_init.size(2) == F &&
                    grad_L_W1_init.size(3) == F * 4,
                    "Encountered error with grad_L_W1_init, please check the shape and if it is contiguous.");

        // Check grad_L_b1_init: expecting a 4-D tensor with shape [B, NH, 1, F*4]
        TORCH_CHECK(grad_L_b1_init.device().is_cuda() && grad_L_b1_init.is_contiguous() &&
                    grad_L_b1_init.dim() == 4 && grad_L_b1_init.size(0) == B &&
                    grad_L_b1_init.size(1) == NH && grad_L_b1_init.size(2) == 1 &&
                    grad_L_b1_init.size(3) == F * 4,
                    "Encountered error with grad_L_b1_init, please check the shape and if it is contiguous.");

        // Check grad_L_W2_init: expecting a 4-D tensor with shape [B, NH, F*4, F]
        TORCH_CHECK(grad_L_W2_init.device().is_cuda() && grad_L_W2_init.is_contiguous() &&
                    grad_L_W2_init.dim() == 4 && grad_L_W2_init.size(0) == B &&
                    grad_L_W2_init.size(1) == NH && grad_L_W2_init.size(2) == F * 4 &&
                    grad_L_W2_init.size(3) == F,
                    "Encountered error with grad_L_W2_init, please check the shape and if it is contiguous.");

        // Check grad_L_b2_init: expecting a 4-D tensor with shape [B, NH, 1, F]
        TORCH_CHECK(grad_L_b2_init.device().is_cuda() && grad_L_b2_init.is_contiguous() &&
                    grad_L_b2_init.dim() == 4 && grad_L_b2_init.size(0) == B &&
                    grad_L_b2_init.size(1) == NH && grad_L_b2_init.size(2) == 1 &&
                    grad_L_b2_init.size(3) == F,
                    "Encountered error with grad_L_b2_init, please check the shape and if it is contiguous.");

        // Check grad_L_last_eta: expecting a 5-D tensor with shape [B, NH, NC, CS, 1]
        TORCH_CHECK(grad_L_last_eta.device().is_cuda() && grad_L_last_eta.is_contiguous() &&
                    grad_L_last_eta.dim() == 5 && grad_L_last_eta.size(0) == B &&
                    grad_L_last_eta.size(1) == NH && grad_L_last_eta.size(2) == NC &&
                    grad_L_last_eta.size(3) == CS && grad_L_last_eta.size(4) == 1,
                    "Encountered error with grad_L_last_eta, please check the shape and if it is contiguous.");

        // Check grad_L_XQ: expecting a 5-D tensor with shape [B, NH, NC, CS, F]
        TORCH_CHECK(grad_L_XQ.device().is_cuda() && grad_L_XQ.is_contiguous() &&
                    grad_L_XQ.dim() == 5 && grad_L_XQ.size(0) == B &&
                    grad_L_XQ.size(1) == NH && grad_L_XQ.size(2) == NC &&
                    grad_L_XQ.size(3) == CS && grad_L_XQ.size(4) == F,
                    "Encountered error with grad_L_XQ, please check the shape and if it is contiguous.");

        // Check grad_L_XK: expecting a 5-D tensor with shape [B, NH, NC, CS, F]
        TORCH_CHECK(grad_L_XK.device().is_cuda() && grad_L_XK.is_contiguous() &&
                    grad_L_XK.dim() == 5 && grad_L_XK.size(0) == B &&
                    grad_L_XK.size(1) == NH && grad_L_XK.size(2) == NC &&
                    grad_L_XK.size(3) == CS && grad_L_XK.size(4) == F,
                    "Encountered error with grad_L_XK, please check the shape and if it is contiguous.");

        // Check grad_L_XV: expecting a 5-D tensor with shape [B, NH, NC, CS, F]
        TORCH_CHECK(grad_L_XV.device().is_cuda() && grad_L_XV.is_contiguous() &&
                    grad_L_XV.dim() == 5 && grad_L_XV.size(0) == B &&
                    grad_L_XV.size(1) == NH && grad_L_XV.size(2) == NC &&
                    grad_L_XV.size(3) == CS && grad_L_XV.size(4) == F,
                    "Encountered error with grad_L_XV, please check the shape and if it is contiguous.");
        
        checks_done = true;
    }


    using globals = bwd_globals<F>;

    using CS_F_tile_type = globals::CS_F_tile_type;
    using F_F_tile_type = globals::F_F_tile_type;
    using CS_F_tile_acc_type = globals::CS_F_tile_acc_type;
    using F_F_tile_acc_type = globals::F_F_tile_acc_type;

    // Vectors
    using CS_vec_type = globals::CS_vec_type;
    using F_vec_type = globals::F_vec_type;
    using CS_vec_acc_type = globals::CS_vec_acc_type;
    using F_vec_acc_type = globals::F_vec_acc_type;
    using std_vec_acc_type = globals::std_vec_acc_type;

    using CS_F_tile_gl = gl<bf16, -1, -1, -1, -1, CS_F_tile_type>;
    using F_F_tile_gl = gl<bf16, -1, -1, -1, -1, F_F_tile_type>;
    using CS_F_tile_acc_gl = gl<float, -1, -1, -1, -1, CS_F_tile_acc_type>;
    using F_F_tile_acc_gl = gl<float, -1, -1, -1, -1, F_F_tile_acc_type>;

    using CS_vec_gl = gl<bf16, -1, -1, -1, -1, CS_vec_type>;
    using F_vec_gl = gl<bf16, -1, -1, -1, -1, F_vec_type>;
    using CS_vec_acc_gl = gl<float, -1, -1, -1, -1, CS_vec_acc_type>;
    using F_vec_acc_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;
    using std_gl = gl<float, -1, -1, -1, -1, std_vec_acc_type>;

    CS_F_tile_gl q_gl{reinterpret_cast<bf16*>(XQ.data_ptr<at::BFloat16>()), B, H, T, F};
    CS_F_tile_gl k_gl{reinterpret_cast<bf16*>(XK.data_ptr<at::BFloat16>()), B, H, T, F};
    CS_F_tile_gl v_gl{reinterpret_cast<bf16*>(XV.data_ptr<at::BFloat16>()), B, H, T, F};
    CS_F_tile_gl o_gl{reinterpret_cast<bf16*>(Out.data_ptr<at::BFloat16>()), B, H, T, F};

    CS_vec_gl last_eta_gl{reinterpret_cast<bf16*>(last_eta.data_ptr<at::BFloat16>()), B, H, NC, CS};

    F_vec_acc_gl ttt_norm_weight_gl{reinterpret_cast<float*>(ttt_norm_weight.data_ptr<float>()), 1, H, 1, F};
    F_vec_acc_gl ttt_norm_bias_gl{reinterpret_cast<float*>(ttt_norm_bias.data_ptr<float>()), 1, H, 1, F};

    F_F_tile_acc_gl w1_checkpoints_gl{reinterpret_cast<float*>(W1_checkpoints.data_ptr<float>()), B, H, num_checkpoints*F, F*K};
    F_vec_acc_gl b1_checkpoints_gl{reinterpret_cast<float*>(b1_checkpoints.data_ptr<float>()), B, H, num_checkpoints, F*K};
    F_F_tile_acc_gl w2_checkpoints_gl{reinterpret_cast<float*>(W2_checkpoints.data_ptr<float>()), B, H, num_checkpoints*F*K, F};
    F_vec_acc_gl b2_checkpoints_gl{reinterpret_cast<float*>(b2_checkpoints.data_ptr<float>()), B, H, num_checkpoints, F};

    // Rematted activations
    F_F_tile_acc_gl W1_init_group_gl{reinterpret_cast<float*>(W1_init_group.data_ptr<float>()), B, H, checkpoint_group_size*F, F*K};
    F_vec_acc_gl b1_init_group_gl{reinterpret_cast<float*>(b1_init_group.data_ptr<float>()), B, H, checkpoint_group_size, F*K};
    F_F_tile_acc_gl W2_init_group_gl{reinterpret_cast<float*>(W2_init_group.data_ptr<float>()), B, H, checkpoint_group_size*F*K, F};
    F_vec_acc_gl b2_init_group_gl{reinterpret_cast<float*>(b2_init_group.data_ptr<float>()), B, H, checkpoint_group_size, F};
    CS_F_tile_gl x_hat_ln_group_gl{reinterpret_cast<bf16*>(x_hat_ln_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F};
    std_gl std_ln_group_gl{reinterpret_cast<float*>(std_ln_group.data_ptr<float>()), B, H, checkpoint_group_size, CS};
    CS_vec_acc_gl std_ln_test_group_gl{reinterpret_cast<float*>(std_ln_group.data_ptr<float>()), B, H, checkpoint_group_size, CS};
    CS_F_tile_gl X2_group_gl{reinterpret_cast<bf16*>(X2_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F*K};
    CS_F_tile_gl Z1_group_gl{reinterpret_cast<bf16*>(Z1_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F*K};
    CS_F_tile_gl Z1_bar_group_gl{reinterpret_cast<bf16*>(Z1_bar_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F*K};
    CS_F_tile_gl X2_bar_group_gl{reinterpret_cast<bf16*>(X2_bar_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F*K};
    CS_F_tile_gl grad_l_wrt_Z2_group_gl{reinterpret_cast<bf16*>(grad_l_wrt_Z2_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F};
    CS_F_tile_gl grad_l_wrt_Z1_group_gl{reinterpret_cast<bf16*>(grad_l_wrt_Z1_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F*K};
    CS_F_tile_gl x_hat_fused_group_gl{reinterpret_cast<bf16*>(x_hat_fused_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F};
    CS_F_tile_gl grad_x_hat_fused_group_gl{reinterpret_cast<bf16*>(grad_x_hat_fused_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F};
    CS_F_tile_gl grad_output_fused_group_gl{reinterpret_cast<bf16*>(grad_output_fused_group.data_ptr<at::BFloat16>()), B, H, checkpoint_group_size*CS, F};
    std_gl std_fused_group_gl{reinterpret_cast<float*>(std_fused_group.data_ptr<float>()), B, H, checkpoint_group_size, CS};
    CS_vec_acc_gl std_fused_test_group_gl{reinterpret_cast<float*>(std_fused_group.data_ptr<float>()), B, H, checkpoint_group_size, CS};

    // Upstream grads
    F_F_tile_acc_gl grad_L_W1_last_gl{reinterpret_cast<float*>(grad_L_W1_last.data_ptr<float>()), B, H, F, F*K};
    F_vec_acc_gl grad_L_b1_last_gl{reinterpret_cast<float*>(grad_L_b1_last.data_ptr<float>()), B, H, 1, F*K};
    F_F_tile_acc_gl grad_L_W2_last_gl{reinterpret_cast<float*>(grad_L_W2_last.data_ptr<float>()), B, H, F*K, F};
    F_vec_acc_gl grad_L_b2_last_gl{reinterpret_cast<float*>(grad_L_b2_last.data_ptr<float>()), B, H, 1, F};
    CS_F_tile_gl grad_L_XQW_mini_batch_gl{reinterpret_cast<bf16*>(grad_L_XQW_mini_batch.data_ptr<at::BFloat16>()), B, H, T, F};

    // Output grads
    F_vec_acc_gl grad_L_ttt_norm_weight_gl{reinterpret_cast<float*>(grad_L_ttt_norm_weight.data_ptr<float>()), B, H, 1, F};
    F_vec_acc_gl grad_L_ttt_norm_bias_gl{reinterpret_cast<float*>(grad_L_ttt_norm_bias.data_ptr<float>()), B, H, 1, F};
    F_F_tile_acc_gl grad_L_W1_init_gl{reinterpret_cast<float*>(grad_L_W1_init.data_ptr<float>()), B, H, F, F*K};
    F_vec_acc_gl grad_L_b1_init_gl{reinterpret_cast<float*>(grad_L_b1_init.data_ptr<float>()), B, H, 1, F*K};
    F_F_tile_acc_gl grad_L_W2_init_gl{reinterpret_cast<float*>(grad_L_W2_init.data_ptr<float>()), B, H, F*K, F};
    F_vec_acc_gl grad_L_b2_init_gl{reinterpret_cast<float*>(grad_L_b2_init.data_ptr<float>()), B, H, 1, F};
    CS_vec_gl grad_L_last_eta_gl{reinterpret_cast<bf16*>(grad_L_last_eta.data_ptr<at::BFloat16>()), B, H, NC, CS};
    CS_F_tile_gl grad_L_XQ_gl{reinterpret_cast<bf16*>(grad_L_XQ.data_ptr<at::BFloat16>()), B, H, T, F};
    CS_F_tile_gl grad_L_XK_gl{reinterpret_cast<bf16*>(grad_L_XK.data_ptr<at::BFloat16>()), B, H, T, F};
    CS_F_tile_gl grad_L_XV_gl{reinterpret_cast<bf16*>(grad_L_XV.data_ptr<at::BFloat16>()), B, H, T, F};

    globals g{
        q_gl, 
        k_gl, 
        v_gl, 
        o_gl, 
        last_eta_gl,
        ttt_norm_weight_gl,
        ttt_norm_bias_gl,
        w1_checkpoints_gl, 
        b1_checkpoints_gl,
        w2_checkpoints_gl, 
        b2_checkpoints_gl,
        W1_init_group_gl,
        b1_init_group_gl,
        W2_init_group_gl,
        b2_init_group_gl,
        x_hat_ln_group_gl,
        std_ln_group_gl,
        std_ln_test_group_gl,
        X2_group_gl,
        Z1_group_gl,
        Z1_bar_group_gl,
        X2_bar_group_gl,
        grad_l_wrt_Z2_group_gl,
        grad_l_wrt_Z1_group_gl,
        x_hat_fused_group_gl,
        grad_x_hat_fused_group_gl,
        grad_output_fused_group_gl,
        std_fused_group_gl,
        std_fused_test_group_gl,
        grad_L_W1_last_gl,
        grad_L_b1_last_gl,
        grad_L_W2_last_gl,
        grad_L_b2_last_gl,
        grad_L_XQW_mini_batch_gl,
        grad_L_ttt_norm_weight_gl,
        grad_L_ttt_norm_bias_gl,
        grad_L_W1_init_gl,
        grad_L_b1_init_gl,
        grad_L_W2_init_gl,
        grad_L_b2_init_gl,
        grad_L_last_eta_gl,
        grad_L_XQ_gl,
        grad_L_XK_gl,
        grad_L_XV_gl,
        static_cast<int>(NC),
        static_cast<int>(T),
        static_cast<int>(num_checkpoints),
        static_cast<int>(checkpoint_group_size)
    };

    auto stream = at::cuda::getCurrentCUDAStream().stream(); 

    constexpr long mem_size = kittens::MAX_SHARED_MEMORY;
    cudaFuncSetAttribute(
        bwd_ttt_mlp_ker<F>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    dim3 grid(TP, B, H);
    bwd_ttt_mlp_ker<F><<<grid, NUM_WORKERS*32, mem_size, stream>>>(g);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    // Ensure the kernel execution completes
    cudaStreamSynchronize(stream);
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("CUDA Kernel Execution Error: %s\n", cudaGetErrorString(syncErr));
    }

    return Out;
}

#else

#include "harness.cuh"

#endif
