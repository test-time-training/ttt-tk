#include "cooperative_groups.h"
#include "kittens.cuh"
#include <iostream>
#include <cstdio>

// Build torch entrypoint
#ifdef TORCH_COMPILE
#define TK_COMPILE_TTT_MLP_FORWARD_TP
#endif

constexpr int TP = (2);
constexpr int CONSUMER_WARPGROUPS = (2);
constexpr int PRODUCER_WARPGROUPS = (1);
constexpr int NUM_WARPGROUPS = (CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS);
constexpr int NUM_WORKERS = (NUM_WARPGROUPS * kittens::WARPGROUP_WARPS);

using namespace kittens;

template <int head_dim> struct fwd_ttt_mlp_ker_tile_dims {
    constexpr static int mini_batch_size = 64;
    constexpr static int F = head_dim;
    constexpr static int stages = (2);
};

template <int head_dim> struct fwd_globals {
    // Tiles
    using CS_F_tile_type = st_bf<fwd_ttt_mlp_ker_tile_dims<head_dim>::mini_batch_size, fwd_ttt_mlp_ker_tile_dims<head_dim>::F>;
    using F_F_tile_type = st_bf<fwd_ttt_mlp_ker_tile_dims<head_dim>::F, fwd_ttt_mlp_ker_tile_dims<head_dim>::F>;
    using CS_F_tile_acc_type = st_fl<fwd_ttt_mlp_ker_tile_dims<head_dim>::mini_batch_size, fwd_ttt_mlp_ker_tile_dims<head_dim>::F>;
    using F_F_tile_acc_type = st_fl<fwd_ttt_mlp_ker_tile_dims<head_dim>::F, fwd_ttt_mlp_ker_tile_dims<head_dim>::F>;

    // Vectors
    using CS_vec_type = sv_bf<fwd_ttt_mlp_ker_tile_dims<head_dim>::mini_batch_size>;
    using F_vec_type = sv_bf<fwd_ttt_mlp_ker_tile_dims<head_dim>::F>;
    using CS_vec_acc_type = sv_fl<fwd_ttt_mlp_ker_tile_dims<head_dim>::mini_batch_size>;
    using F_vec_acc_type = sv_fl<fwd_ttt_mlp_ker_tile_dims<head_dim>::F>;

    // Global memory layout
    using q_gl = gl<bf16, -1, -1, -1, -1, CS_F_tile_type>;
    using k_gl = gl<bf16, -1, -1, -1, -1, CS_F_tile_type>;
    using v_gl = gl<bf16, -1, -1, -1, -1, CS_F_tile_type>;
    using o_gl = gl<bf16, -1, -1, -1, -1, CS_F_tile_type>;

    using last_eta_gl = gl<bf16, -1, -1, -1, -1, CS_vec_type>;

    using ttt_norm_weight_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;
    using ttt_norm_bias_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;

    using w1_init_gl = gl<float, -1, -1, -1, -1, F_F_tile_acc_type>;
    using b1_init_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;
    using w2_init_gl = gl<float, -1, -1, -1, -1, F_F_tile_acc_type>;
    using b2_init_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;

    // Remat checkpoints
    using w1_checkpoints_gl = gl<float, -1, -1, -1, -1, F_F_tile_acc_type>;
    using b1_checkpoints_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;
    using w2_checkpoints_gl = gl<float, -1, -1, -1, -1, F_F_tile_acc_type>;
    using b2_checkpoints_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;

    q_gl q;
    k_gl k;
    v_gl v;
    o_gl o;

    last_eta_gl last_eta;

    ttt_norm_weight_gl ttt_norm_weight;
    ttt_norm_bias_gl ttt_norm_bias;

    w1_init_gl w1;
    b1_init_gl b1;
    w2_init_gl w2;
    b2_init_gl b2;

    w1_checkpoints_gl w1_checkpoints;
    b1_checkpoints_gl b1_checkpoints;
    w2_checkpoints_gl w2_checkpoints;
    b2_checkpoints_gl b2_checkpoints;

    const int seq_len;
    const int num_checkpoints;
    const int checkpoint_group_size;
};

template <int head_dim>
__cluster_dims__(TP)
__global__ __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
void fwd_ttt_mlp_ker(const __grid_constant__ fwd_globals<head_dim> g) {
    using K = fwd_ttt_mlp_ker_tile_dims<head_dim>;

    using CS_F_tile_type = fwd_globals<head_dim>::CS_F_tile_type;
    using F_F_tile_type = fwd_globals<head_dim>::F_F_tile_type;
    using CS_F_tile_acc_type = fwd_globals<head_dim>::CS_F_tile_acc_type;
    using F_F_tile_acc_type = fwd_globals<head_dim>::F_F_tile_acc_type;

    // Vectors
    using CS_vec_type = fwd_globals<head_dim>::CS_vec_type;
    using F_vec_type = fwd_globals<head_dim>::F_vec_type;
    using CS_vec_acc_type = fwd_globals<head_dim>::CS_vec_acc_type;
    using F_vec_acc_type = fwd_globals<head_dim>::F_vec_acc_type;

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
    const int cluster_wgid = warpgroupid + CONSUMER_WARPGROUPS * tp; // Cluster Warpgroup ID
    const int tp_shard_rank = cluster_wgid;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int *)&__shm[0]);

    // Shared memory for hidden states
    F_F_tile_acc_type(&w1_smem)[CONSUMER_WARPGROUPS] = al.allocate<F_F_tile_acc_type, CONSUMER_WARPGROUPS>();
    F_vec_acc_type(&b1_smem)[CONSUMER_WARPGROUPS] = al.allocate<F_vec_acc_type, CONSUMER_WARPGROUPS>();
    F_F_tile_acc_type(&w2_smem)[CONSUMER_WARPGROUPS] = al.allocate<F_F_tile_acc_type, CONSUMER_WARPGROUPS>();
    F_vec_acc_type(&b2_smem) = al.allocate<F_vec_acc_type>();
    
    // Shared memory for inputs (staged)
    CS_F_tile_type(&q_smem)[K::stages] = al.allocate<CS_F_tile_type, K::stages>();
    CS_F_tile_type(&k_smem)[K::stages] = al.allocate<CS_F_tile_type, K::stages>();
    CS_F_tile_type(&v_smem)[K::stages] = al.allocate<CS_F_tile_type, K::stages>();
    CS_vec_type(&last_eta_smem)[K::stages] = al.allocate<CS_vec_type, K::stages>();

    // Shared memory for ttt norm params
    F_vec_acc_type(&ttt_norm_weight_smem) = al.allocate<F_vec_acc_type>();
    F_vec_acc_type(&ttt_norm_bias_smem) = al.allocate<F_vec_acc_type>();
    
    // Shared memory for intermediates
    CS_F_tile_type(&z1_smem)[CONSUMER_WARPGROUPS] = al.allocate<CS_F_tile_type, CONSUMER_WARPGROUPS>();
    CS_F_tile_type(&x2_smem)[CONSUMER_WARPGROUPS] = al.allocate<CS_F_tile_type, CONSUMER_WARPGROUPS>();
    CS_F_tile_type(&grad_l_z1_smem)[CONSUMER_WARPGROUPS] = al.allocate<CS_F_tile_type, CONSUMER_WARPGROUPS>();
    sv_fl<16>(&ln_smem)[4] = al.allocate<sv_fl<16>, 4>();
    CS_F_tile_type(&matmul_smem)[CONSUMER_WARPGROUPS] = al.allocate<CS_F_tile_type, CONSUMER_WARPGROUPS>();
    CS_F_tile_acc_type(&b_acc_smem)[CONSUMER_WARPGROUPS] = al.allocate<CS_F_tile_acc_type, CONSUMER_WARPGROUPS>();

    // Reinterpretations for intermediates
    auto(&reduction_buffer) = matmul_smem[0];
    auto(&z2_smem)[CONSUMER_WARPGROUPS] = grad_l_z1_smem;
    auto(&grad_l_z2_smem)[K::stages] = v_smem;

    // Create locks to make sure reads aren't premature
    __shared__ kittens::semaphore 
        w1_arrived,
        w2_arrived,
        b1_arrived,
        b2_arrived,
        start_reduction,
        second_start_reduction,
        dsmem_semaphore,
        second_dsmem_semaphore,
        reduction_done,
        second_reduction_done,
        q_sem_arrived[K::stages],
        k_sem_arrived[K::stages], 
        v_sem_arrived[K::stages],
        last_eta_sem_arrived[K::stages],
        ttt_norm_weight_arrived,
        ttt_norm_bias_arrived,
        compute_done[K::stages];

    if (threadIdx.x == 0) {
        init_semaphore(w1_arrived, 0, 1);
        init_semaphore(b1_arrived, 0, 1);
        init_semaphore(w2_arrived, 0, 1);
        init_semaphore(b2_arrived, 0, 1);
        init_semaphore(ttt_norm_weight_arrived, 0, 1);
        init_semaphore(ttt_norm_bias_arrived, 0, 1);
        init_semaphore(dsmem_semaphore, 0, 1);
        init_semaphore(second_dsmem_semaphore, 0, 1);
        init_semaphore(reduction_done, CONSUMER_WARPGROUPS, 0);
        init_semaphore(second_reduction_done, CONSUMER_WARPGROUPS, 0);
        init_semaphore(start_reduction, CONSUMER_WARPGROUPS, 0);
        init_semaphore(second_start_reduction, CONSUMER_WARPGROUPS, 0);
        for (int i = 0; i < K::stages; i++) {
            init_semaphore(q_sem_arrived[i], 0, 1);
            init_semaphore(k_sem_arrived[i], 0, 1);
            init_semaphore(v_sem_arrived[i], 0, 1);
            init_semaphore(last_eta_sem_arrived[i], 0, 1);
            init_semaphore(compute_done[i], CONSUMER_WARPGROUPS, 0);
        }

        // Load hidden states across consumer warpgroups
        tma::expect_bytes(w1_arrived, sizeof(w1_smem));
        tma::expect_bytes(b1_arrived, sizeof(b1_smem));
        tma::expect_bytes(w2_arrived, sizeof(w2_smem));
        tma::expect_bytes(b2_arrived, sizeof(b2_smem));
        for (int wg = 0; wg < CONSUMER_WARPGROUPS; wg++) {
            tma::load_async(w1_smem[wg], g.w1, {batch_idx, head_idx, 0, wg + CONSUMER_WARPGROUPS * tp}, w1_arrived);
            tma::load_async(b1_smem[wg], g.b1, {batch_idx, head_idx, 0, wg + CONSUMER_WARPGROUPS * tp}, b1_arrived);
            tma::load_async(w2_smem[wg], g.w2, {batch_idx, head_idx, wg + CONSUMER_WARPGROUPS * tp, 0}, w2_arrived);
        }
        tma::load_async(b2_smem, g.b2, {batch_idx, head_idx, 0, 0}, b2_arrived);

        // ttt norm params
        tma::expect_bytes(ttt_norm_weight_arrived, sizeof(ttt_norm_weight_smem));
        tma::expect_bytes(ttt_norm_bias_arrived, sizeof(ttt_norm_bias_smem));
        tma::load_async(ttt_norm_weight_smem, g.ttt_norm_weight, {0, head_idx, 0, 0}, ttt_norm_weight_arrived);
        tma::load_async(ttt_norm_bias_smem, g.ttt_norm_bias, {0, head_idx, 0, 0}, ttt_norm_bias_arrived);

        // Preload minibatches
        for (int j = 0; j < K::stages - 1; j++) {
            int4 tile_idx = {batch_idx, head_idx, 0, 0};
            tma::expect_bytes(k_sem_arrived[j], sizeof(k_smem[j]));
            tma::load_async(k_smem[j], g.k, tile_idx, k_sem_arrived[j]);
            tma::expect_bytes(v_sem_arrived[j], sizeof(v_smem[j]));
            tma::load_async(v_smem[j], g.v, tile_idx, v_sem_arrived[j]);
            tma::expect_bytes(q_sem_arrived[j], sizeof(q_smem[j]));
            tma::load_async(q_smem[j], g.q, tile_idx, q_sem_arrived[j]);
            tma::expect_bytes(last_eta_sem_arrived[j], sizeof(last_eta_smem[j]));
            tma::load_async(last_eta_smem[j], g.last_eta, tile_idx, last_eta_sem_arrived[j]);
        }
    }
    __syncthreads();

    int pipe_idx = K::stages - 1;

    // First warp in last warpgroup is the producer
    if (warpgroupid == NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<24>();
        // Two reductions done
        tma::cluster::arrive_aligned();
        tma::cluster::arrive_aligned();
        tma::cluster::arrive_aligned();
        tma::cluster::arrive_aligned();

        int iters = n_minibatch - 1;
        if (warpid == NUM_WORKERS - 4) {
            for (auto idx = pipe_idx - 1; idx < iters; idx++) {
                int4 tile_idx = {batch_idx, head_idx, idx + 1, 0};

                const int curr_stage = (idx + 1) % K::stages;

                tma::expect_bytes(k_sem_arrived[curr_stage], sizeof(k_smem[0]));
                tma::load_async(k_smem[curr_stage], g.k, tile_idx, k_sem_arrived[curr_stage]);
                tma::expect_bytes(v_sem_arrived[curr_stage], sizeof(v_smem[0]));
                tma::load_async(v_smem[curr_stage], g.v, tile_idx, v_sem_arrived[curr_stage]);
                tma::expect_bytes(q_sem_arrived[curr_stage], sizeof(q_smem[0]));
                tma::load_async(q_smem[curr_stage], g.q, tile_idx, q_sem_arrived[curr_stage]);
                tma::expect_bytes(last_eta_sem_arrived[curr_stage], sizeof(last_eta_smem[0]));
                tma::load_async(last_eta_smem[curr_stage], g.last_eta, tile_idx, last_eta_sem_arrived[curr_stage]);

                // Wait on previous stage to finish computation
                kittens::wait(compute_done[idx % K::stages], (idx / K::stages) % 2);

                tma::cluster::arrive_aligned();
                tma::cluster::arrive_aligned();
                tma::cluster::arrive_aligned();
                tma::cluster::arrive_aligned();
            }
        }
    } else {
        warpgroup::increase_registers<240>();

        rt_fl<16, K::F> cs_cs_fl_reg;
        rt_fl<16, K::F> cs_cs_fl_reg2;
        typeof(cs_cs_fl_reg)::row_vec cs_row_fl_reg;
        typeof(cs_cs_fl_reg)::col_vec cs_col_fl_reg;

        kittens::wait(w1_arrived, 0);
        kittens::wait(b1_arrived, 0);
        kittens::wait(w2_arrived, 0);
        kittens::wait(b2_arrived, 0);
        kittens::wait(ttt_norm_weight_arrived, 0);
        kittens::wait(ttt_norm_bias_arrived, 0);

        for (auto idx = 0; idx < n_minibatch; idx++) {
            // Save hidden state checkpoint (W1)
            if (wg_warpid == 0 && idx % checkpoint_group_size == 0) {
                const int curr_checkpoint_idx = idx / checkpoint_group_size;
                int4 curr_checkpoint = {batch_idx, head_idx, curr_checkpoint_idx, tp_shard_rank};
                tma::store_async(g.w1_checkpoints, w1_smem[warpgroupid], curr_checkpoint);

                curr_checkpoint = {batch_idx, head_idx, curr_checkpoint_idx, tp_shard_rank};
                tma::store_async(g.b1_checkpoints, b1_smem[warpgroupid], curr_checkpoint);

                const int sharded_checkpoint_offset = curr_checkpoint_idx * TP * CONSUMER_WARPGROUPS + tp_shard_rank;
                curr_checkpoint = {batch_idx, head_idx, sharded_checkpoint_offset, 0};
                tma::store_async(g.w2_checkpoints, w2_smem[warpgroupid], curr_checkpoint);

                curr_checkpoint = {batch_idx, head_idx, curr_checkpoint_idx, 0};
                tma::store_async(g.b2_checkpoints, b2_smem, curr_checkpoint);

                tma::store_commit_group();
            }

            const int curr_stage = idx % K::stages;
            const int curr_stage_lock_phase = (idx / K::stages) % 2;

            // Hidden state forward
            kittens::wait(k_sem_arrived[curr_stage], (idx / K::stages) % 2);
            
            zero(cs_cs_fl_reg);
            warpgroup::load(cs_cs_fl_reg2, w1_smem[warpgroupid]);
            warpgroup::store(matmul_smem[warpgroupid], cs_cs_fl_reg2);
            warpgroup::sync(warpgroupid+1);
            warpgroup::mm_AB(cs_cs_fl_reg, k_smem[idx % K::stages], matmul_smem[warpgroupid]);
            load(cs_row_fl_reg, b1_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
            warpgroup::store(z1_smem[warpgroupid], cs_cs_fl_reg);
            warpgroup::sync(warpgroupid+1);
            
            gelu(cs_cs_fl_reg, cs_cs_fl_reg);
            warpgroup::store(x2_smem[warpgroupid], cs_cs_fl_reg);
            warpgroup::sync(warpgroupid+1);

            zero(cs_cs_fl_reg);
            warpgroup::load(cs_cs_fl_reg2, w2_smem[warpgroupid]);
            warpgroup::store(matmul_smem[warpgroupid], cs_cs_fl_reg2);
            warpgroup::sync(warpgroupid+1);
            warpgroup::mm_AB(cs_cs_fl_reg, x2_smem[warpgroupid], matmul_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            // Only add b2 to one of the sharded Z2
            // Else, post reduction it will result in adding 4*b2
            if (tp_shard_rank == 0) {
                load(cs_row_fl_reg, b2_smem);
                add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
            }
            warpgroup::store(z2_smem[warpgroupid], cs_cs_fl_reg);
            warpgroup::sync(warpgroupid+4);

            if (warpgroup::laneid() == 0) arrive(start_reduction, 1);
            kittens::wait(start_reduction, idx % 2);

            // Warpgroup 0 will perform reduction
            if (warpgroupid == 0) {
                // Reduce intra CTA Z2 before inter CTA reduce
                warpgroup::add(z2_smem[0], z2_smem[0], z2_smem[1]);
                warpgroup::sync(warpgroupid+4);
                if (wg_warpid == 0) tma::expect_bytes(dsmem_semaphore, sizeof(reduction_buffer));
                tma::cluster::sync();
                if (wg_warpid == 0) tma::cluster::store_async(reduction_buffer, z2_smem[0], tp ^ 1, dsmem_semaphore);

                // reduction here to do ln
                kittens::wait(dsmem_semaphore, idx % 2);

                tma::cluster::sync();
                warpgroup::add(z2_smem[0], z2_smem[0], reduction_buffer);

                // mean
                zero(cs_col_fl_reg);
                warpgroup::load(cs_cs_fl_reg, z2_smem[0]);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                div(cs_col_fl_reg, cs_col_fl_reg, static_cast<float>(head_dim)); // mu_fused

                sub_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg); // Z2 - mu_fused, first part of xhat
                warpgroup::store(z2_smem[1], cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                // var
                warpgroup::load(cs_cs_fl_reg, z2_smem[0]);
                sub_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg);
                mul(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg);

                // zero(cs_col_fl_reg);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                div(cs_col_fl_reg, cs_col_fl_reg, static_cast<float>(head_dim)); // var
                add(cs_col_fl_reg, cs_col_fl_reg, 1e-6f); 
                sqrt(cs_col_fl_reg, cs_col_fl_reg); // std
                store(ln_smem[wg_warpid], cs_col_fl_reg);
                warpgroup::sync(warpgroupid+1);

                // finish x_hat
                warpgroup::load(cs_cs_fl_reg, z2_smem[1]);
                div_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg); // final x_hat
                warpgroup::store(z2_smem[0], cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                
                // compute y
                load(cs_row_fl_reg, ttt_norm_weight_smem);
                mul_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                load(cs_row_fl_reg, ttt_norm_bias_smem);
                add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg); // y
                
                // grad_output
                warpgroup::load(cs_cs_fl_reg2, v_smem[idx % K::stages]);
                sub(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                warpgroup::load(cs_cs_fl_reg2, k_smem[idx % K::stages]);
                add(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);

                // grad x_hat
                load(cs_row_fl_reg, ttt_norm_weight_smem);
                mul_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
   
                warpgroup::load(cs_cs_fl_reg2, z2_smem[0]);
                mul(cs_cs_fl_reg2, cs_cs_fl_reg, cs_cs_fl_reg2);
                zero(cs_col_fl_reg);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg2);
                warpgroup::load(cs_cs_fl_reg2, z2_smem[0]);
                mul_row(cs_cs_fl_reg2, cs_cs_fl_reg2, cs_col_fl_reg); // 3rd line, not negative

                zero(cs_col_fl_reg);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                add_row(cs_cs_fl_reg2, cs_cs_fl_reg2, cs_col_fl_reg);

                mul(cs_cs_fl_reg, cs_cs_fl_reg, static_cast<float>(head_dim));
                sub(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                div(cs_cs_fl_reg, cs_cs_fl_reg, static_cast<float>(head_dim));

                load(cs_col_fl_reg, ln_smem[wg_warpid]);
                div_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg);
                mul(cs_cs_fl_reg, cs_cs_fl_reg, -1.0f); // negate to prepare for grad step
                kittens::wait(last_eta_sem_arrived[idx % K::stages], (idx / K::stages) % 2);

                warpgroup::store(grad_l_z2_smem[curr_stage], cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                warpgroup::mul_row(grad_l_z2_smem[curr_stage], grad_l_z2_smem[curr_stage], last_eta_smem[idx % K::stages]);
            }
            else {
                tma::cluster::arrive_aligned();
                tma::cluster::arrive_aligned();
            }

            // Wait on each other to complete
            if (warpgroup::laneid() == 0) arrive(reduction_done, 1);
            kittens::wait(reduction_done, idx % 2);

            // Calculate grad_l_wrt_Z1
            zero(cs_cs_fl_reg);
            warpgroup::load(cs_cs_fl_reg2, w2_smem[warpgroupid]);
            warpgroup::store(matmul_smem[warpgroupid], cs_cs_fl_reg2);
            warpgroup::sync(warpgroupid+1);
            warpgroup::mm_ABt(cs_cs_fl_reg, grad_l_z2_smem[curr_stage], matmul_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            warpgroup::store(grad_l_z1_smem[warpgroupid], cs_cs_fl_reg);
            warpgroup::sync(warpgroupid+1);
            warpgroup::load(cs_cs_fl_reg, z1_smem[warpgroupid]);
            gelu_bwd(cs_cs_fl_reg, cs_cs_fl_reg);
            warpgroup::store(z1_smem[warpgroupid], cs_cs_fl_reg);
            warpgroup::sync(warpgroupid+1);
            warpgroup::mul(grad_l_z1_smem[warpgroupid], grad_l_z1_smem[warpgroupid], z1_smem[warpgroupid]);
            
            // Update W2
            warpgroup::load(cs_cs_fl_reg, w2_smem[warpgroupid]);
            warpgroup::mma_AtB(cs_cs_fl_reg, x2_smem[warpgroupid], grad_l_z2_smem[curr_stage]);
            warpgroup::mma_async_wait();
            warpgroup::store(w2_smem[warpgroupid], cs_cs_fl_reg);
            warpgroup::sync(warpgroupid+1);

            // Update b2
            if (warpgroupid == 0)
            {
                warpgroup::copy(b_acc_smem[0], grad_l_z2_smem[curr_stage]);
                warpgroup::col_sum(b2_smem, b_acc_smem[0], b2_smem);
                warpgroup::sync(warpgroupid+1);
            }

            // Update W1
            warpgroup::load(cs_cs_fl_reg, w1_smem[warpgroupid]);
            warpgroup::mma_AtB(cs_cs_fl_reg, k_smem[idx % K::stages], grad_l_z1_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            warpgroup::store(w1_smem[warpgroupid], cs_cs_fl_reg);
            warpgroup::sync(warpgroupid+1);

            // Update b1
            warpgroup::copy(b_acc_smem[warpgroupid], grad_l_z1_smem[warpgroupid]);
            warpgroup::col_sum(b1_smem[warpgroupid], b_acc_smem[warpgroupid], b1_smem[warpgroupid]);
            warpgroup::sync(warpgroupid+1);

            // Compute output
            zero(cs_cs_fl_reg);
            kittens::wait(q_sem_arrived[idx % K::stages], (idx / K::stages) % 2);
            warpgroup::load(cs_cs_fl_reg2, w1_smem[warpgroupid]);
            warpgroup::store(matmul_smem[warpgroupid], cs_cs_fl_reg2);
            warpgroup::sync(warpgroupid+1);
            warpgroup::mm_AB(cs_cs_fl_reg, q_smem[idx % K::stages], matmul_smem[warpgroupid]);
            warpgroup::mma_async_wait();
            load(cs_row_fl_reg, b1_smem[warpgroupid]);
            warpgroup::sync(warpgroupid+1);
            add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
            warpgroup::sync(warpgroupid+1);
            warpgroup::store(z1_smem[warpgroupid], cs_cs_fl_reg);
            warpgroup::sync(warpgroupid+1);
            gelu(cs_cs_fl_reg, cs_cs_fl_reg);
            warpgroup::store(x2_smem[warpgroupid], cs_cs_fl_reg);
            warpgroup::sync(warpgroupid+1);

            warpgroup::load(cs_cs_fl_reg2, w2_smem[warpgroupid]);
            warpgroup::sync(warpgroupid+1);
            warpgroup::store(matmul_smem[warpgroupid], cs_cs_fl_reg2);
            zero(cs_cs_fl_reg);
            warpgroup::sync(warpgroupid+1);
            warpgroup::mm_AB(cs_cs_fl_reg, x2_smem[warpgroupid], matmul_smem[warpgroupid]);
            warpgroup::mma_async_wait();

            // Only add b2 to one of the sharded Z2
            // Else, post reduction it will result in adding 4*b2
            if (tp_shard_rank == 0) {
                load(cs_row_fl_reg, b2_smem);
                add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                warpgroup::sync(warpgroupid+1);
            }
            warpgroup::store(z2_smem[warpgroupid], cs_cs_fl_reg);
            warpgroup::sync(warpgroupid+1);

            if (warpgroup::laneid() == 0) arrive(second_start_reduction, 1);
            kittens::wait(second_start_reduction, idx % 2);

            // Warpgroup 0 will perform reduction
            if (warpgroupid == 0) {
                // Reduce intra CTA Z2 before inter CTA reduce
                warpgroup::add(z2_smem[0], z2_smem[0], z2_smem[1]);
                warpgroup::sync(warpgroupid+4);
                if (wg_warpid == 0) tma::expect_bytes(second_dsmem_semaphore, sizeof(reduction_buffer));
                tma::cluster::sync();
                if (wg_warpid == 0) tma::cluster::store_async(reduction_buffer, z2_smem[0], tp ^ 1, second_dsmem_semaphore);

                // reduction here to do ln
                kittens::wait(second_dsmem_semaphore, idx % 2);
                tma::cluster::sync();
                warpgroup::add(z2_smem[0], z2_smem[0], reduction_buffer);
                warpgroup::sync(warpgroupid+1);

                // mean
                zero(cs_col_fl_reg);
                warpgroup::load(cs_cs_fl_reg, z2_smem[0]);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                div(cs_col_fl_reg, cs_col_fl_reg, static_cast<float>(head_dim)); // mu_fused

                sub_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg); // Z2 - mu_fused, first part of xhat
                warpgroup::store(z2_smem[1], cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);

                // var
                warpgroup::load(cs_cs_fl_reg, z2_smem[0]);
                sub_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg);
                mul(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg);

                // zero(cs_col_fl_reg);
                row_sum(cs_col_fl_reg, cs_cs_fl_reg);
                div(cs_col_fl_reg, cs_col_fl_reg, static_cast<float>(head_dim)); // var
                add(cs_col_fl_reg, cs_col_fl_reg, 1e-6f); 
                sqrt(cs_col_fl_reg, cs_col_fl_reg); // std
                store(ln_smem[wg_warpid], cs_col_fl_reg);
                warpgroup::sync(warpgroupid+1);

                // finish x_hat
                warpgroup::load(cs_cs_fl_reg, z2_smem[1]);
                div_row(cs_cs_fl_reg, cs_cs_fl_reg, cs_col_fl_reg); // final x_hat
                warpgroup::sync(warpgroupid+1);
                warpgroup::store(z2_smem[0], cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                
                // compute y
                load(cs_row_fl_reg, ttt_norm_weight_smem);
                mul_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg);
                load(cs_row_fl_reg, ttt_norm_bias_smem);
                add_col(cs_cs_fl_reg, cs_cs_fl_reg, cs_row_fl_reg); // y

                // Residual
                warpgroup::load(cs_cs_fl_reg2, q_smem[idx % K::stages]);
                add(cs_cs_fl_reg, cs_cs_fl_reg, cs_cs_fl_reg2);
                
                warpgroup::store(z2_smem[0], cs_cs_fl_reg);
                warpgroup::sync(warpgroupid+1);
                // Store output to global
                if (tp_shard_rank == 0 && wg_warpid == 0) {
                    tma::store_async(g.o, z2_smem[0], {batch_idx, head_idx, idx, 0});
                    tma::store_commit_group();
                    tma::store_async_wait();
                }
            }
            else {
                tma::cluster::arrive_aligned();
                tma::cluster::arrive_aligned();
            }

            // Wait on each other to complete
            if (warpgroup::laneid() == 0) arrive(second_reduction_done, 1);
            kittens::wait(second_reduction_done, idx % 2);

            if (warpgroup::laneid() == 0) arrive(compute_done[idx % K::stages], 1);
        }
    }
}

#if TORCH_COMPILE

#include "common/pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

torch::Tensor ttt_forward(
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
) {
    constexpr int F = 64;
    constexpr int K = 4;
    unsigned long B = XQ.size(0);
    unsigned long H = XQ.size(1);
    unsigned long T = XQ.size(2) * XQ.size(3); // seq len
    unsigned long NC = XQ.size(2);
    unsigned long CS = XQ.size(3);
    unsigned long num_checkpoints = static_cast<int>(W1_checkpoints.size(2));
    
    TORCH_CHECK(XQ.device().is_cuda() && XQ.is_contiguous() && XQ.dim() == 5 && XQ.size(4) == F, "Invalid dims for XQ");
    TORCH_CHECK(XK.device().is_cuda() && XK.is_contiguous() && XK.dim() == 5 && XK.size(4) == F, "Invalid dims for XK");
    TORCH_CHECK(XV.device().is_cuda() && XV.is_contiguous() && XV.dim() == 5 && XV.size(4) == F, "Invalid dims for XV");
    TORCH_CHECK(W1.device().is_cuda() && W1.is_contiguous() && W1.dim() == 4 && W1.size(0) == B && W1.size(1) == H && W1.size(2) == F && W1.size(3) == F*K, "Invalid dims for W1");
    TORCH_CHECK(W2.device().is_cuda() && W2.is_contiguous() && W2.dim() == 4 && W2.size(0) == B && W2.size(1) == H && W2.size(2) == F*K && W2.size(3) == F, "Invalid dims for W2");
    TORCH_CHECK(W1_checkpoints.device().is_cuda() && W1_checkpoints.is_contiguous() && W1_checkpoints.dim() == 5 && W1_checkpoints.size(0) == B && W1_checkpoints.size(1) == H && W1_checkpoints.size(2) == num_checkpoints && W1_checkpoints.size(3) == F && W1_checkpoints.size(4) == F*K, "Invalid dims for W1_checkpoints");
    TORCH_CHECK(W2_checkpoints.device().is_cuda() && W2_checkpoints.is_contiguous() && W2_checkpoints.dim() == 5 && W2_checkpoints.size(0) == B && W2_checkpoints.size(1) == H && W2_checkpoints.size(2) == num_checkpoints && W2_checkpoints.size(3) == F*K && W2_checkpoints.size(4) == F, "Invalid dims for W2_checkpoints");
    TORCH_CHECK(Out.device().is_cuda() && Out.is_contiguous() && Out.dim() == 5 && Out.size(4) == F, "Invalid dims for Out");

    TORCH_CHECK(ttt_norm_weight.device().is_cuda() && ttt_norm_weight.is_contiguous() && ttt_norm_weight.dim() == 4 && ttt_norm_weight.size(0) == 1 && ttt_norm_weight.size(1) == H && ttt_norm_weight.size(2) == 1 && ttt_norm_weight.size(2) == 1 && ttt_norm_weight.size(3) == F, "Invalid dims for ttt_norm_weight");

    using globals = fwd_globals<F>;

    using CS_F_tile_type = globals::CS_F_tile_type;
    using F_F_tile_type = globals::F_F_tile_type;
    using CS_F_tile_acc_type = globals::CS_F_tile_acc_type;
    using F_F_tile_acc_type = globals::F_F_tile_acc_type;

    // Vectors
    using CS_vec_type = globals::CS_vec_type;
    using F_vec_type = globals::F_vec_type;
    using CS_vec_acc_type = globals::CS_vec_acc_type;
    using F_vec_acc_type = globals::F_vec_acc_type;

    using CS_F_tile_gl = gl<bf16, -1, -1, -1, -1, CS_F_tile_type>;
    using F_F_tile_gl = gl<bf16, -1, -1, -1, -1, F_F_tile_type>;
    using CS_F_tile_acc_gl = gl<float, -1, -1, -1, -1, CS_F_tile_acc_type>;
    using F_F_tile_acc_gl = gl<float, -1, -1, -1, -1, F_F_tile_acc_type>;

    using CS_vec_gl = gl<bf16, -1, -1, -1, -1, CS_vec_type>;
    using F_vec_gl = gl<bf16, -1, -1, -1, -1, F_vec_type>;
    using CS_vec_acc_gl = gl<float, -1, -1, -1, -1, CS_vec_acc_type>;
    using F_vec_acc_gl = gl<float, -1, -1, -1, -1, F_vec_acc_type>;

    CS_F_tile_gl q_gl{reinterpret_cast<bf16*>(XQ.data_ptr<at::BFloat16>()), B, H, T, F};
    CS_F_tile_gl k_gl{reinterpret_cast<bf16*>(XK.data_ptr<at::BFloat16>()), B, H, T, F};
    CS_F_tile_gl v_gl{reinterpret_cast<bf16*>(XV.data_ptr<at::BFloat16>()), B, H, T, F};
    CS_F_tile_gl o_gl{reinterpret_cast<bf16*>(Out.data_ptr<at::BFloat16>()), B, H, T, F};

    CS_vec_gl last_eta_gl{reinterpret_cast<bf16*>(last_eta.data_ptr<at::BFloat16>()), B, H, NC, CS};

    F_vec_acc_gl ttt_norm_weight_gl{reinterpret_cast<float*>(ttt_norm_weight.data_ptr<float>()), 1, H, 1, F};
    F_vec_acc_gl ttt_norm_bias_gl{reinterpret_cast<float*>(ttt_norm_bias.data_ptr<float>()), 1, H, 1, F};

    F_F_tile_acc_gl w1_init_gl{reinterpret_cast<float*>(W1.data_ptr<float>()), B, H, F, F*K};
    F_vec_acc_gl b1_init_gl{reinterpret_cast<float*>(b1.data_ptr<float>()), B, H, 1, F*K};
    F_F_tile_acc_gl w2_init_gl{reinterpret_cast<float*>(W2.data_ptr<float>()), B, H, F*K, F};
    F_vec_acc_gl b2_init_gl{reinterpret_cast<float*>(b2.data_ptr<float>()), B, H, 1, F};

    F_F_tile_acc_gl w1_checkpoints_gl{reinterpret_cast<float*>(W1_checkpoints.data_ptr<float>()), B, H, num_checkpoints*F, F*K};
    F_vec_acc_gl b1_checkpoints_gl{reinterpret_cast<float*>(b1_checkpoints.data_ptr<float>()), B, H, num_checkpoints, F*K};
    F_F_tile_acc_gl w2_checkpoints_gl{reinterpret_cast<float*>(W2_checkpoints.data_ptr<float>()), B, H, num_checkpoints*F*K, F};
    F_vec_acc_gl b2_checkpoints_gl{reinterpret_cast<float*>(b2_checkpoints.data_ptr<float>()), B, H, num_checkpoints, F};

    globals g{
        q_gl, 
        k_gl, 
        v_gl, 
        o_gl, 
        last_eta_gl,
        ttt_norm_weight_gl,
        ttt_norm_bias_gl,
        w1_init_gl, 
        b1_init_gl,
        w2_init_gl, 
        b2_init_gl,
        w1_checkpoints_gl, 
        b1_checkpoints_gl,
        w2_checkpoints_gl, 
        b2_checkpoints_gl,
        static_cast<int>(T),
        static_cast<int>(num_checkpoints),
        checkpoint_group_size
    };

    auto stream = at::cuda::getCurrentCUDAStream().stream(); 


    constexpr long mem_size = kittens::MAX_SHARED_MEMORY;
    cudaFuncSetAttribute(
        fwd_ttt_mlp_ker<F>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    dim3 grid(TP, B, H);
    fwd_ttt_mlp_ker<F><<<grid, NUM_WORKERS*32, mem_size, stream>>>(g);

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
}//*/

#else

#include "harness.cuh"

#endif
