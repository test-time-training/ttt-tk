/**
 * @file
 * @brief Functions for a group to collaboratively transfer data directly between global memory and registers and back.
 */

/**
 * @brief Collaboratively loads data from a source array into row-major layout tiles.
 *
 * @tparam RT The row-major layout tile type.
 * @tparam U The data type of the source array.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param row_stride[in] The stride in elements between rows in the source array.
 */
template<ducks::rt::row_layout RT, ducks::gl::all GL>
__device__ inline static void load(RT &dst, const GL &src, const coord &idx) {
    using T2 = RT::dtype;
    using U = typename GL::dtype;
    using MEGA_RT = rt<typename RT::T, N_WARPS*RT::rows, RT::cols, typename RT::layout>; // the megatile for the original coord.
    U *src_ptr = (U*)&src.template get<MEGA_RT>(idx);
    const int row_stride = src.row_stride();
    using U2 = base_types::packing<U>::packed_type;
    int warp_laneid = threadIdx.x % 32;
    const int row_offset = dst.rows*warpid();
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row = row_offset + i*dst.tile_size + (warp_laneid / 4);
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size + 2*(warp_laneid % 4);
            dst.tiles[i][j].data[0] = base_types::convertor<T2, U2>::convert(*(U2*)(&src_ptr[(row+0)*row_stride + (col+0)]));
            dst.tiles[i][j].data[2] = base_types::convertor<T2, U2>::convert(*(U2*)(&src_ptr[(row+0)*row_stride + (col+8)]));
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size + 2*(warp_laneid % 4);
            dst.tiles[i][j].data[1] = base_types::convertor<T2, U2>::convert(*(U2*)(&src_ptr[(row+8)*row_stride + (col+0)]));
            dst.tiles[i][j].data[3] = base_types::convertor<T2, U2>::convert(*(U2*)(&src_ptr[(row+8)*row_stride + (col+8)]));
        }
    }
}
/**
 * @brief Collaboratively loads data from a source array into column-major layout tiles.
 *
 * @tparam RT The column-major layout tile type.
 * @tparam U The data type of the source array.
 * @param dst[out] The destination tile to load data into.
 * @param src[in] The source array to load data from.
 * @param row_stride[in] The stride in elements between rows in the source array.
 */
template<ducks::rt::col_layout RT, ducks::gl::all GL>
__device__ inline static void load(RT &dst, const GL &src, const coord &idx) {
    using T = typename RT::T;
    using U = typename GL::dtype;
    using MEGA_RT = rt<typename RT::T, N_WARPS*RT::rows, RT::cols, typename RT::layout>; // the megatile for the original coord.
    U *src_ptr = (U*)&src.template get<MEGA_RT>(idx);
    const int row_stride = src.row_stride();
    int warp_laneid = threadIdx.x % 32;
    const int row_offset = dst.rows*warpid();
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        int row = row_offset + i*dst.tile_size + 2*(warp_laneid % 4);
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size + (warp_laneid / 4);
            dst.tiles[i][j].data[0].x = base_types::convertor<T, U>::convert(src_ptr[(row+0)*row_stride + (col+0)]);
            dst.tiles[i][j].data[1].x = base_types::convertor<T, U>::convert(src_ptr[(row+0)*row_stride + (col+8)]);
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size + (warp_laneid / 4);
            dst.tiles[i][j].data[0].y = base_types::convertor<T, U>::convert(src_ptr[(row+1)*row_stride + (col+0)]);
            dst.tiles[i][j].data[1].y = base_types::convertor<T, U>::convert(src_ptr[(row+1)*row_stride + (col+8)]);
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size + (warp_laneid / 4);
            dst.tiles[i][j].data[2].x = base_types::convertor<T, U>::convert(src_ptr[(row+8)*row_stride + (col+0)]);
            dst.tiles[i][j].data[3].x = base_types::convertor<T, U>::convert(src_ptr[(row+8)*row_stride + (col+8)]);
        }
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            int col = j*dst.tile_size + (warp_laneid / 4);
            dst.tiles[i][j].data[2].y = base_types::convertor<T, U>::convert(src_ptr[(row+9)*row_stride + (col+0)]);
            dst.tiles[i][j].data[3].y = base_types::convertor<T, U>::convert(src_ptr[(row+9)*row_stride + (col+8)]);
        }
    }
}


/**
 * @brief Collaboratively stores data from register tiles to a destination array in global memory with a row-major layout.
 *
 * @tparam RT The register tile type with a row-major layout.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param row_stride[in] The stride in elements between rows in the destination array.
 */
template<ducks::rt::row_layout RT, ducks::gl::all GL>
__device__ inline static void store(GL &dst, const RT &src, const coord &idx) {
    using T2 = RT::dtype;
    using U = typename GL::dtype;
    using MEGA_RT = rt<typename RT::T, N_WARPS*RT::rows, RT::cols, typename RT::layout>; // the megatile for the original coord.
    U *dst_ptr = (U*)&dst.template get<MEGA_RT>(idx);
    const int row_stride = dst.row_stride();
    using U2 = base_types::packing<U>::packed_type;
    int warp_laneid = threadIdx.x % 32;
    const int row_offset = src.rows*warpid();
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = row_offset + i*src.tile_size + (warp_laneid / 4);
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size + 2*(warp_laneid % 4);
            *(U2*)(&dst_ptr[(row+0)*row_stride + (col+0)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
            *(U2*)(&dst_ptr[(row+0)*row_stride + (col+8)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[2]);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size + 2*(warp_laneid % 4);
            *(U2*)(&dst_ptr[(row+8)*row_stride + (col+0)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);
            *(U2*)(&dst_ptr[(row+8)*row_stride + (col+8)]) = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[3]);
        }
    }
}
/**
 * @brief Collaboratively stores data from register tiles to a destination array in global memory with a column-major layout.
 *
 * @tparam RT The register tile type with a column-major layout.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register tile to store data from.
 * @param row_stride[in] The stride in elements between rows in the destination array.
 */
template<ducks::rt::col_layout RT, ducks::gl::all GL>
__device__ inline static void store(GL &dst, const RT &src, const coord &idx) {
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;
    using MEGA_RT = rt<typename RT::T, N_WARPS*RT::rows, RT::cols, typename RT::layout>; // the megatile for the original coord.
    U *dst_ptr = (U*)&dst.template get<MEGA_RT>(idx);
    const int row_stride = dst.row_stride();
    int warp_laneid = threadIdx.x % 32;
    const int row_offset = src.rows*warpid();
    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        int row = row_offset + i*src.tile_size + 2*(warp_laneid % 4);
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size + (warp_laneid / 4);
            dst_ptr[(row+0)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
            dst_ptr[(row+0)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size + (warp_laneid / 4);
            dst_ptr[(row+1)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
            dst_ptr[(row+1)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size + (warp_laneid / 4);
            dst_ptr[(row+8)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].x);
            dst_ptr[(row+8)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].x);
        }
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = j*src.tile_size + (warp_laneid / 4);
            dst_ptr[(row+9)*row_stride + (col+0)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[2].y);
            dst_ptr[(row+9)*row_stride + (col+8)] = base_types::convertor<U, T>::convert(src.tiles[i][j].data[3].y);
        }
    }
}
