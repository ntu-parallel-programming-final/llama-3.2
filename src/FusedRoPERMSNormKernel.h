#pragma once
#include <cuda_runtime.h>

// 這是 C++ (.cpp) 與 CUDA (.cu) 溝通的介面
// 告訴 .cpp 檔：「別擔心，雖然你現在看不到實作，但真的有這個函式存在」
void FusedRoPERMSNormLaunch_FP32(
    cudaStream_t stream,
    const float* input,
    const float* gamma,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_size,
    float rope_base,
    int token_offset,
    float epsilon
);