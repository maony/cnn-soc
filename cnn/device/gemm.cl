/*
* matrix mult
* M, N, K all modulo block size should be zero
* Pointer point to Matrix A, B should avoid aliasing
*
*/

#ifdef GROUP_XX
__kernel
__attribute((reqd_work_group_size(BS_GEMM, BS_GEMM, 1)))
__attribute((num_simd_work_items(SIMD_GEMM)))
void gemm(__global float *restrict C,
          __global float *restrict A,
          __global float *restrict B,
          int M, int K, int N) {
    const int Aw = K;
    const int Bw = N;

    // local storage for blocks of A and B
    __local float a_local[BS_GEMM][BS_GEMM];
    __local float b_local[BS_GEMM][BS_GEMM];

    // block index
    int x_block = get_group_id(0);
    int y_block = get_group_id(1);

    // local id, offset in a block
    int x_local = get_local_id(0);
    int y_local = get_local_id(1);

    // A split in horizontal, B else
    int a_start = Aw * BS_GEMM * y_block;
    int a_end   = a_start + Aw - 1;
    int b_start = BS_GEMM * x_block;
    int a_index = a_start;
    int b_index = b_start;
    float sum = 0.0f;

    // loop iteration to process one block
    for(; a_index <= a_end; a_index += BS_GEMM, b_index += (BS_GEMM * Bw)) {
        b_local[y_local][x_local] = B[b_index + Bw * y_local + x_local];
        a_local[y_local][x_local] = A[a_index + Aw * y_local + x_local];
        // b_local[x_local][y_local] = B[b_index + Bw * y_local + x_local];

        // synchronize in group
        barrier(CLK_LOCAL_MEM_FENCE);

        // dot product
        #pragma unroll
        for(int k = 0; k < BS_GEMM; k++) {
            sum += a_local[y_local][k] * b_local[k][x_local];
        }

        // synchronize
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = sum;
}
#else
__kernel
__attribute((reqd_work_group_size(WS_X, WS_Y, 1)))
__attribute((num_simd_work_items(SIMD)))
void gemm(__global float *restrict C,
          __global float *restrict A,
          __global float *restrict B,
          int M, int K, int N) {
    // local storage for blocks of A and B
    __local float a_local[WS_Y * WS_X];
    __local float b_local[WS_X * WS_Y];

    // block index
    int x_block = get_group_id(0);
    int y_block = get_group_id(1);

    // local id, offset in a block
    int x_local = get_local_id(0);
    int y_local = get_local_id(1);

    // A split in horizontal, B else
    int a_start = K * WS_Y * y_block;
    int b_start = WS_X * x_block;
    float sum = 0.0f;
    
    // loop iteration to process one block
    for(int i = 0; i < K / WS_MIN; i++, a_start += WS_MIN, b_start += (WS_Y * N)) {
        a_local[y_local * WS_X + x_local] = A[a_start + y_local * K + x_local];
        b_local[y_local * WS_X + x_local] = B[b_start + y_local * N + x_local];

        // synchronize in group
        barrier(CLK_LOCAL_MEM_FENCE);

        // dot product
        #pragma unroll
        for(int k = 0; k < WS_MIN; k++) {
            sum += a_local[y_local * WS_X + k] * b_local[x_local + WS_X * k];
        }

        // synchronize
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = sum;
}
#endif






