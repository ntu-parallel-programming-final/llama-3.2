#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

// ==========================================
// 1. Helper: 讀取二進位檔案
// ==========================================
template<typename T>
void load_bin(const std::string& path, std::vector<T>& data) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error opening " << path << std::endl;
        exit(1);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    data.resize(size / sizeof(T));
    if (!file.read((char*)data.data(), size)) {
        std::cerr << "Error reading " << path << std::endl;
        exit(1);
    }
}

// ==========================================
// 2. CUDA Kernel (Naive Version)
// ==========================================
// 這是一個最簡單的實作：每個 Thread 負責處理 "一個 Token (Row)"
// 優化空間：非常大！(這是你們之後要優化的地方)
__global__ void fused_rope_rms_kernel(
    const float* input,     // [Batch, Seq, Hidden]
    const float* gamma,     // [Hidden]
    float* output,          // [Batch, Seq, Hidden]
    int batch_size,
    int seq_len,
    int hidden_size,
    int max_seq_len,
    float rope_base
) {
    // 計算目前這個 Thread 負責哪一個 Token (Row)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tokens = batch_size * seq_len;

    if (idx >= total_tokens) return;

    // 定位：這個 Thread 負責的 Row 在 global memory 的起始位置
    int row_offset = idx * hidden_size;
    
    // 計算當前 Token 在序列中的位置 (Position ID)
    // 假設 input 是 [Batch, Seq, Hidden] 且連續
    int token_pos = idx % seq_len; 

    // ---------------------------------------------------
    // Step A: RoPE (Rotary Positional Embedding)
    // ---------------------------------------------------
    // 我們需要一個暫存陣列來放 RoPE 算完的結果，再做 RMSNorm
    // 但因為這是 Naive kernel，我們直接在 register 裡做或寫回 output
    
    // 為了簡單，我們這裡動態計算 cos/sin (這很慢，真正的實作應該預計算好表)
    // 這裡演示 "On-the-fly" 計算
    
    // 暫存這個 Row 的數據，避免反覆讀 Global Memory
    // 注意：如果是大 Hidden Size，這裡可能會爆 Register，這是 Naive 版的缺點
    // 為了通用性，我們這邊還是直接讀寫 Global Memory (最慢，但最安全)

    // 我們需要先遍歷一次這個 Row 做 RoPE
    for (int i = 0; i < hidden_size / 2; ++i) {
        // RoPE 處理一對 (Real, Imag) -> (x[2i], x[2i+1])
        float x0 = input[row_offset + 2 * i];
        float x1 = input[row_offset + 2 * i + 1];

        // 計算頻率 theta
        // theta = 1.0 / pow(base, 2.0 * i / dim)
        float theta_scale = powf(rope_base, -2.0f * i / hidden_size);
        float theta = token_pos * theta_scale;

        float cos_val = cosf(theta);
        float sin_val = sinf(theta);

        // 旋轉公式:
        // out0 = x0 * cos - x1 * sin
        // out1 = x0 * sin + x1 * cos
        // (注意：這取決於 PyTorch 實作的旋轉方向，Llama 通常是 [-x1, x0])
        // 根據我們 Python 腳本的 rotate_half: [-x2, x1]，對應這裡的偶奇位置交換
        
        // 修正後的 RoPE (對齊 test_data.py):
        // x_out = x * cos + rotate_half(x) * sin
        // rotate_half 對於 (x0, x1) 來說，如果是鄰近對，通常是 (-x1, x0)
        // 讓我們對齊 Python 的行為
        
        float out0 = x0 * cos_val - x1 * sin_val;
        float out1 = x1 * cos_val + x0 * sin_val;

        // 寫入 output (暫存)
        output[row_offset + 2 * i] = out0;
        output[row_offset + 2 * i + 1] = out1;
    }

    // ---------------------------------------------------
    // Step B: RMSNorm
    // ---------------------------------------------------
    // 1. 計算平方和 (Sum of Squares)
    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        float val = output[row_offset + i]; // 讀取剛剛 RoPE 的結果
        sum_sq += val * val;
    }

    // 2. 計算 Mean & Rsqrt
    float mean_sq = sum_sq / hidden_size;
    float rsqrt_val = rsqrtf(mean_sq + 1e-5f); // eps

    // 3. 縮放並乘上 Gamma
    for (int i = 0; i < hidden_size; ++i) {
        float val = output[row_offset + i];
        output[row_offset + i] = val * rsqrt_val * gamma[i];
    }
}

// ==========================================
// 3. Main Function
// ==========================================
int main() {
    // 1. Load Data
    std::vector<float> h_input, h_gamma, h_output_golden, h_rope_base_vec;
    std::vector<int> h_max_seq_len_vec;

    std::cout << "Loading binary data..." << std::endl;
    load_bin("data/bin/input.bin", h_input);
    load_bin("data/bin/gamma.bin", h_gamma);
    load_bin("data/bin/output.bin", h_output_golden);
    load_bin("data/bin/max_seq_len.bin", h_max_seq_len_vec);
    load_bin("data/bin/rope_base.bin", h_rope_base_vec);

    // Hardcoded shapes from Python script (You can also save these to bin if needed)
    int BATCH = 2;
    int SEQ = 32;
    int HIDDEN = 512;
    int TOTAL_ELEMENTS = BATCH * SEQ * HIDDEN;
    int MAX_SEQ_LEN = h_max_seq_len_vec[0];
    float ROPE_BASE = h_rope_base_vec[0];

    std::cout << "Shape: [" << BATCH << ", " << SEQ << ", " << HIDDEN << "]" << std::endl;

    // 2. Allocate Device Memory
    float *d_input, *d_gamma, *d_output;
    cudaMalloc(&d_input, TOTAL_ELEMENTS * sizeof(float));
    cudaMalloc(&d_gamma, HIDDEN * sizeof(float));
    cudaMalloc(&d_output, TOTAL_ELEMENTS * sizeof(float));

    // 3. Copy to Device
    cudaMemcpy(d_input, h_input.data(), TOTAL_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma.data(), HIDDEN * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Launch Kernel
    // Strategy: 1 Thread per Token (Row) -> Total Threads = Batch * Seq
    int total_tokens = BATCH * SEQ;
    int threads_per_block = 256;
    int blocks = (total_tokens + threads_per_block - 1) / threads_per_block;

    std::cout << "Launching Kernel with " << blocks << " blocks, " << threads_per_block << " threads..." << std::endl;
    
    fused_rope_rms_kernel<<<blocks, threads_per_block>>>(
        d_input, d_gamma, d_output,
        BATCH, SEQ, HIDDEN, MAX_SEQ_LEN, ROPE_BASE
    );
    cudaDeviceSynchronize();

    // Check Kernel Error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 5. Copy Back
    std::vector<float> h_output_my(TOTAL_ELEMENTS);
    cudaMemcpy(h_output_my.data(), d_output, TOTAL_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Verify Correctness (MSE)
    float max_diff = 0.0f;
    float mse = 0.0f;
    for (int i = 0; i < TOTAL_ELEMENTS; ++i) {
        float diff = std::abs(h_output_my[i] - h_output_golden[i]);
        if (diff > max_diff) max_diff = diff;
        mse += diff * diff;
    }
    mse /= TOTAL_ELEMENTS;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Max Diff: " << max_diff << std::endl;
    std::cout << "MSE:      " << mse << std::endl;
    
    if (mse < 1e-4) {
        std::cout << "✅ Result MATCH! (Kernel Logic is Correct)" << std::endl;
    } else {
        std::cout << "❌ Result MISMATCH! (Check your math)" << std::endl;
    }
    std::cout << "------------------------------------------------" << std::endl;

    // Free
    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_output);

    return 0;
}