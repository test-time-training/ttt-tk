#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

#include <cuda.h>
#include <iostream>

namespace kittens {
namespace tma {

/* ----------   Prefetch Tensor Map  ---------- */

/**
 * @brief Prefetches data from global memory into a shared memory tile, along with the tensormap.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination shared memory tile.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in] tile_row_idx The row coord of the requested tile. This is in units of complete tiles.
 * @param[in] tile_col_idx The column coord of the requested tile. This is in units of complete tiles.
 */
template<ducks::st::all ST, ducks::gl::all GL>
__device__ static inline void prefetch(ST &dst, const GL &src, const coord &idx) {
    if (::kittens::laneid()) {
        uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<ST>());
        int32_t crd0 = 0;
        int32_t crd1 = idx.r * (ST::rows);
        int32_t crd2 = idx.c * (ST::cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));
        int32_t crd3 = idx.d;
        int32_t crd4 = idx.b;

        asm volatile (
            "cp.async.bulk.prefetch.tensor.5d.L2.global.tile"
            " [%0, {%1, %2, %3, %4, %5}];"
            :
            : "l"(tma_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
            : "memory"
        );
    }
}

/* ----------   Async load and store data from gmem/smem  ---------- */

/**
 * @brief Asynchronously stores data into global memory from a shared memory tile.
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] tile_row_idx The row coord of the tile destination. This is in units of complete tiles.
 * @param[in] tile_col_idx The column coord of the tile destination. This is in units of complete tiles.
 */
template<ducks::st::all ST, ducks::gl::all GL>
__device__ static inline void store_async(const GL &dst, const ST &src, const coord &idx) {
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST>());
        uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
        int32_t crd0 = 0;
        int32_t crd1 = idx.r * (ST::rows);
        int32_t crd2 = idx.c * (ST::cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));
        int32_t crd3 = idx.d;
        int32_t crd4 = idx.b;

        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        asm volatile (
            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
            " [%0, {%2, %3, %4, %5, %6}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
            : "memory"
        );
    }
    store_commit_group();
}

/* ----------   Async reduction + store data from gmem/smem  ---------- */

/**
 * @brief Asynchronously performs an add reduction and stores the result into global memory from a shared memory tile.
 *
 * This function performs an asynchronous add reduction and copy operation using CUDA's cp.reduce.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] tile_row_idx The row coord of the tile destination. This is in units of complete tiles.
 * @param[in] tile_col_idx The column coord of the tile destination. This is in units of complete tiles.
 */
template<ducks::st::all ST, ducks::gl::all GL>
__device__ static inline void store_add_async(const GL &dst, const ST &src, const coord &idx) {
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST>());
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
        int32_t crd0 = 0;
        int32_t crd1 = idx.r * (ST::rows);
        int32_t crd2 = idx.c * (ST::cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));
        int32_t crd3 = idx.d;
        int32_t crd4 = idx.b;

        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        asm volatile (
            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group"
            " [%0, {%2, %3, %4, %5, %6}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
            : "memory"
        );
    }
    store_commit_group();
}

/**
 * @brief Asynchronously performs an min reduction and stores the result into global memory from a shared memory tile.
 *
 * This function performs an asynchronous min reduction and copy operation using CUDA's cp.reduce.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] tile_row_idx The row coord of the tile destination. This is in units of complete tiles.
 * @param[in] tile_col_idx The column coord of the tile destination. This is in units of complete tiles.
 */
template<ducks::st::all ST, ducks::gl::all GL>
__device__ static inline void store_min_async(const GL &dst, const ST &src, const coord &idx) {
    static_assert(!std::is_same_v<typename ST::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST>());
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
        int32_t crd0 = 0;
        int32_t crd1 = idx.r * (ST::rows);
        int32_t crd2 = idx.c * (ST::cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));
        int32_t crd3 = idx.d;
        int32_t crd4 = idx.b;

        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        asm volatile (
            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.bulk_group"
            " [%0, {%2, %3, %4, %5, %6}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
            : "memory"
        );
    }
    store_commit_group();
}

/**
 * @brief Asynchronously performs an max reduction and stores the result into global memory from a shared memory tile.
 *
 * This function performs an asynchronous max reduction and copy operation using CUDA's cp.reduce.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination tensormap address in global memory
 * @param[in] src_tma_map The source shared memory tile.
 * @param[in] tile_row_idx The row coord of the tile destination. This is in units of complete tiles.
 * @param[in] tile_col_idx The column coord of the tile destination. This is in units of complete tiles.
 */
template<ducks::st::all ST, ducks::gl::all GL>
__device__ static inline void store_max_async(const GL &dst, const ST &src, const coord &idx) {
    static_assert(!std::is_same_v<typename ST::dtype, float>, "TMA does not support async min/max reductions for fp32 types.");
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr = reinterpret_cast<uint64_t>(dst.template get_tma<ST>());
        uint32_t src_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&src));
        int32_t crd0 = 0;
        int32_t crd1 = idx.r * (ST::rows);
        int32_t crd2 = idx.c * (ST::cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));
        int32_t crd3 = idx.d;
        int32_t crd4 = idx.b;

        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        asm volatile (
            "cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.bulk_group"
            " [%0, {%2, %3, %4, %5, %6}], [%1];"
            :
            : "l"(tma_ptr), "r"(src_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
            : "memory"
        );
    }
    store_commit_group();
}

/**
 * @brief Asynchronously loads data from global memory into a shared memory tile.
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination shared memory tile.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in,out] bar The semaphore used for synchronization of the asynchronous copy.
 * @param[in] tile_row_idx The row coord of the requested tile. This is in units of complete tiles.
 * @param[in] tile_col_idx The column coord of the requested tile. This is in units of complete tiles.
 */
template<ducks::st::all ST, ducks::gl::all GL>
__device__ static inline void load_async(ST &dst, const GL &src, const coord &idx, semaphore& bar) {
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr = reinterpret_cast<uint64_t>(src.template get_tma<ST>());
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
        uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
        int32_t crd0 = 0;
        int32_t crd1 = idx.r * (ST::rows);
        int32_t crd2 = idx.c * (ST::cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));
        int32_t crd3 = idx.d;
        int32_t crd4 = idx.b;

        asm volatile (
            "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
            : "memory"
        );
    }
}

namespace cluster {

/**
 * @brief Asynchronously loads data from global memory into a shared memory tile, across a threadblock cluster
 *
 * This function performs an asynchronous copy operation using CUDA's cp.async.bulk.tensor instruction.
 *
 * @tparam ST A shared tile type with a TMA-compatible layout
 * @param[out] dst The destination shared memory tile.
 * @param[in] src_tma_map The source tensormap address in global memory
 * @param[in,out] bar The semaphore used for synchronization of the asynchronous copy.
 * @param[in] tile_row_idx The row coord of the requested tile. This is in units of complete tiles.
 * @param[in] tile_col_idx The column coord of the requested tile. This is in units of complete tiles.
 * @param[in] cluster_mask The mask of the clusters to broadcast to.
 */
template<ducks::st::all ST, ducks::gl::all GL>
__device__ static inline void load_async(ST &dst, const GL &src, const coord &idx, semaphore& bar, uint16_t cluster_mask) {
    if (::kittens::laneid() == 0) {
        uint64_t tma_ptr = reinterpret_cast<uint64_t>(src.template get_tma<ST>());
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
        uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
        int32_t crd0 = 0;
        int32_t crd1 = idx.r * (ST::rows);
        int32_t crd2 = idx.c * (ST::cols / (ST::swizzle_bytes / sizeof(typename ST::dtype)));
        int32_t crd3 = idx.d;
        int32_t crd4 = idx.b;

        asm volatile (
            "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
            " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
            "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4), "h"(cluster_mask)
            : "memory"
        );
    }
}

} // namespace cluster
} // namespace tma
} // namespace kittens