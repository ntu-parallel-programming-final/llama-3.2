/*
 * src/CustomRoPEKernel.cu
 * CUDA Kernel 實作 - 修復廣播索引問題
 */

#include "CustomRoPEKernel.h"
#include <cuda_fp16.h>

__global__ void custom_rope_kernel(
    const float* input,
    const float* cos_table,
    const float* sin_table,
    float* output,
    int head_dim,
    int num_heads,
    int seq_len,
    int half_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 總元素數量檢查 (選用，但為了安全建議加上)
    // int total_elements = gridDim.x * blockDim.x; // 這裡簡化處理

    // 1. 計算 dim_idx (0 ~ head_dim-1)
    int dim_idx = idx % head_dim;

    // 2. 解決廣播問題 (Broadcasting Logic)
    // 輸入資料 Layout: [Batch, Heads, Seq, Dim]
    // Cos/Sin Layout: [1, 1, Seq, Dim] (實際上只存了 Seq * Dim 個元素)
    //
    // 我們需要計算 freq_idx，讓它對應到 Cos/Sin 表中的正確位置。
    // 在 [Batch, Heads, Seq, Dim] 的結構中，Seq * Dim 是一個完整的「頻率區塊」。
    // 每個 Head 和每個 Batch 都是重複使用這個區塊。
    
    int freq_block_size = seq_len * head_dim;
    int freq_idx = idx % freq_block_size; // 這就是關鍵修正！

    // 讀取 Input (使用全局 idx)
    float x_val = input[idx];
    
    // 讀取 Cos/Sin (使用循環的 freq_idx)
    float c = cos_table[freq_idx];
    float s = sin_table[freq_idx];

    float result;

    if (dim_idx < half_dim) {
        // 前半段 x1，需要找後半段 x2
        // 注意：x2 也在同一個 Head/Seq 內，所以 offset 是 half_dim
        float x2 = input[idx + half_dim];
        
        // 公式：x1 * cos - x2 * sin
        result = x_val * c - x2 * s;
    } else {
        // 後半段 x2，需要找前半段 x1
        float x1 = input[idx - half_dim];
        
        // 公式：x2 * cos + x1 * sin
        result = x_val * c + x1 * s;
    }

    output[idx] = result;
}

void launchCustomRoPE(
    const float* input,
    const float* cos,
    const float* sin,
    float* output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    cudaStream_t stream
) {
    size_t total_elements = (size_t)batch_size * seq_len * num_heads * head_dim;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    int half_dim = head_dim / 2;

    custom_rope_kernel<<<grid_size, block_size, 0, stream>>>(
        input, cos, sin, output, 
        head_dim, num_heads, seq_len, half_dim
    );
}