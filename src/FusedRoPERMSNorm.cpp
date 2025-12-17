#include "FusedRoPERMSNormPlugin.h" 
#include "FusedRoPERMSNormKernel.h"
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <cassert>

using namespace nvinfer1;
using namespace plugin;

// ----------------------------------------------------
// FusedRoPERMSNorm 輔助函式 (權重管理)
// ----------------------------------------------------

void FusedRoPERMSNorm::freeDeviceMemory() {
    if (mDeviceGamma) {
        cudaFree(mDeviceGamma);
        mDeviceGamma = nullptr;
    }
}

// 設置 Hidden Size (供 Creator 呼叫)
void FusedRoPERMSNorm::setHiddenSize(int hidden_size) {
    mHiddenSize = hidden_size;
}

// 將 Gamma 權重從 CPU 複製到 GPU
void FusedRoPERMSNorm::setDeviceWeights(const float* gamma_cpu) {
    freeDeviceMemory(); 
    
    // 假設權重為 FP32 (float)，大小為 mHiddenSize
    size_t size = mHiddenSize * sizeof(float); 
    
    if (cudaMalloc(&mDeviceGamma, size) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for gamma.");
    }

    if (cudaMemcpy(mDeviceGamma, gamma_cpu, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("Failed to copy gamma to GPU.");
    }
}

// ----------------------------------------------------
// FusedRoPERMSNorm 建構子與生命週期
// ----------------------------------------------------

// 1. 從 PluginCreator 建立 (引擎建構時，接收 PluginFieldCollection)
FusedRoPERMSNorm::FusedRoPERMSNorm(const std::string& name, const nvinfer1::PluginFieldCollection* fc) 
    : mPluginName(name) 
{
    // 在 Creator 中，我們將會呼叫 setHiddenSize 和 setDeviceWeights 來初始化
    // 這裡只負責基礎初始化
}

// 2. 從序列化資料建立 (引擎載入時)
FusedRoPERMSNorm::FusedRoPERMSNorm(const std::string& name, const void* data, size_t length) 
    : mPluginName(name)
{
    const char* d = static_cast<const char*>(data);
    
    // 1. 反序列化配置參數
    memcpy(&mHiddenSize, d, sizeof(int)); d += sizeof(int);
    memcpy(&mEpsilon, d, sizeof(float)); d += sizeof(float);
    memcpy(&mRoPEBase, d, sizeof(float)); d += sizeof(float);
    memcpy(&mMaxSeqLen, d, sizeof(int)); d += sizeof(int);

    // 2. 反序列化權重 (從 CPU 緩衝區複製到 GPU)
    size_t weightSize = mHiddenSize * sizeof(float);

    // 分配 GPU 記憶體
    cudaMalloc(&mDeviceGamma, weightSize);

    // 複製數據到 GPU
    cudaMemcpy(mDeviceGamma, d, weightSize, cudaMemcpyHostToDevice);
    d += weightSize;

    // 假設我們在序列化時只使用 kFLOAT
    mDataType = DataType::kFLOAT; 
    
    if (d != static_cast<const char*>(data) + length) {
        std::cerr << "Error: Deserialization length mismatch!" << std::endl;
    }
}

void FusedRoPERMSNorm::destroy() noexcept {
    freeDeviceMemory(); // 釋放 GPU 記憶體
    delete this;
}

IPluginV3* FusedRoPERMSNorm::clone() const noexcept {
    // 創建一個新的 Plugin 實例，然後複製所有狀態
    FusedRoPERMSNorm* newPlugin = new FusedRoPERMSNorm(mPluginName, nullptr);
    newPlugin->mHiddenSize = mHiddenSize;
    newPlugin->mEpsilon = mEpsilon;
    newPlugin->mRoPEBase = mRoPEBase;
    newPlugin->mMaxSeqLen = mMaxSeqLen;
    newPlugin->mDataType = mDataType;

    // 複製 GPU 權重
    if (mDeviceGamma) {
        size_t size = mHiddenSize * sizeof(float);
        cudaMalloc(&(newPlugin->mDeviceGamma), size);
        cudaMemcpy(newPlugin->mDeviceGamma, mDeviceGamma, size, cudaMemcpyDeviceToDevice);
    }
    return newPlugin;
}

// ----------------------------------------------------
// 步驟 3: 建構邏輯
// ----------------------------------------------------

DimsExprs FusedRoPERMSNorm::getOutputDimensions(
    int outputIndex, 
    const DimsExprs* inputs, 
    int nbInputs, 
    IExprBuilder& exprBuilder) const noexcept 
{
    // RoPE + RMSNorm 是同形狀操作 (OutputDims == InputDims)
    assert(nbInputs == 1);
    assert(outputIndex == 0);
    return inputs[0]; 
}

bool FusedRoPERMSNorm::supportsFormatCombination(
    int pos, 
    const PluginTensorDesc* inOut, 
    int nbInputs, 
    int nbOutputs) const noexcept 
{
    if (nbInputs != 1 || nbOutputs != 1) return false;
    
    // 假設我們只支援 kFLOAT 或 kHALF
    if (inOut[pos].desc.type != DataType::kFLOAT && inOut[pos].desc.type != DataType::kHALF) {
        return false;
    }
    
    // 確保輸入和輸出格式一致
    if (pos == 1) {
        return (inOut[pos].desc.type == inOut[0].desc.type) && 
               (inOut[pos].desc.format == inOut[0].desc.format) && 
               (inOut[pos].desc.format == TensorFormat::kLINEAR);
    }
    
    return inOut[pos].desc.format == TensorFormat::kLINEAR;
}

void FusedRoPERMSNorm::configurePlugin(
    const EsdPluginTensors* in, 
    int nbInputs, 
    const EsdPluginTensors* out, 
    int nbOutputs) noexcept 
{
    // 獲取並儲存資料型別
    mDataType = in[0].desc.type;
    
    // 在這裡，我們假設 mHiddenSize 已經在 Creator 中設定完成
    if (mHiddenSize <= 0) {
         // 應該拋出錯誤或紀錄日誌，但這裡使用簡單斷言
        assert(false && "Hidden size not initialized before configurePlugin.");
    }
    
    // TODO: 如果需要，可以在此處檢查輸入維度是否與 mHiddenSize 一致
}

// ----------------------------------------------------
// 步驟 4: 推論邏輯
// ----------------------------------------------------

// Fused RoPE/RMSNorm 應該不需要額外的暫存空間
size_t FusedRoPERMSNorm::getWorkspaceSize(
    const EsdPluginTensors* in, 
    int nbInputs, 
    const EsdPluginTensors* out, 
    int nbOutputs) const noexcept 
{
    return 0; 
}

int FusedRoPERMSNorm::enqueue(
    const EsdPluginTensors* in, 
    int nbInputs, 
    const EsdPluginTensors* out, 
    int nbOutputs, 
    void* workspace, 
    cudaStream_t stream) noexcept 
{
    // 1. 獲取 GPU 數據指標
    const void* input = in[0].data;
    void* output = out[0].data;
    const void* gamma = mDeviceGamma; 

    // 2. 獲取當前的動態維度
    Dims inputDims = in[0].desc.dims;
    
    // 假設輸入是 [B, S, D] 或 [S, D] (nbDims >= 2)
    int batch_size = inputDims.d[0];
    int seq_len = inputDims.d[1];
    
    // **注意**：token_offset 在 K/V Cache 流程中是動態的，這裡簡化為 0
    int token_offset = 0; 

    // 3. 呼叫 CUDA 核心
    if (mDataType == DataType::kFLOAT) {
        FusedRoPERMSNormLaunch_FP32(
            stream,
            static_cast<const float*>(input),
            static_cast<const float*>(gamma),
            static_cast<float*>(output),
            batch_size,
            seq_len,
            mHiddenSize, // D
            mRoPEBase,
            token_offset,
            mEpsilon
        );
    } else if (mDataType == DataType::kHALF) { 
        // TODO: 實作 FusedRoPERMSNormLaunch_FP16
    }

    return 0;
}

// ----------------------------------------------------
// 步驟 5: 序列化邏輯
// ----------------------------------------------------

size_t FusedRoPERMSNorm::getSerializationSize(
    const PluginTensorDesc* in, 
    int nbInputs, 
    const PluginTensorDesc* out, 
    int nbOutputs) const noexcept 
{
    // 序列化數據 = 配置參數 + 權重數據
    return sizeof(int)               // mHiddenSize
         + sizeof(float)             // mEpsilon
         + sizeof(float)             // mRoPEBase
         + sizeof(int)               // mMaxSeqLen
         + mHiddenSize * sizeof(float); // Gamma Weights
}

void FusedRoPERMSNorm::serialize(
    void* buffer, 
    size_t length, 
    const PluginTensorDesc* in, 
    int nbInputs, 
    const PluginTensorDesc* out, 
    int nbOutputs) const noexcept 
{
    char* d = static_cast<char*>(buffer);
    const char* const a = d;
    
    // 1. 序列化配置參數
    memcpy(d, &mHiddenSize, sizeof(int)); d += sizeof(int);
    memcpy(d, &mEpsilon, sizeof(float)); d += sizeof(float);
    memcpy(d, &mRoPEBase, sizeof(float)); d += sizeof(float);
    memcpy(d, &mMaxSeqLen, sizeof(int)); d += sizeof(int);

    // 2. 序列化權重 (從 GPU 複製到 CPU 緩衝區)
    size_t weightSize = mHiddenSize * sizeof(float);
    
    // Gamma
    cudaMemcpy(d, mDeviceGamma, weightSize, cudaMemcpyDeviceToHost);
    d += weightSize;
    
    assert(d == a + length); // 檢查序列化大小是否正確
}