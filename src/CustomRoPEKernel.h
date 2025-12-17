/*
 * src/CustomRoPEKernel.h
 * 定義 Host 端呼叫 CUDA Kernel 的介面
 */

#ifndef CUSTOM_ROPE_KERNEL_H
#define CUSTOM_ROPE_KERNEL_H

#include <cuda_runtime.h>

// 啟動 RoPE Kernel 的 Host 函式
void launchCustomRoPE(
    const float* input,     // 輸入數據
    const float* cos,       // Cos 表
    const float* sin,       // Sin 表
    float* output,          // 輸出結果
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    cudaStream_t stream
);

#endif // CUSTOM_ROPE_KERNEL_H