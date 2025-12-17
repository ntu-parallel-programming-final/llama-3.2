#include "FusedRoPERMSNormPlugin.h"
#include <cstring>
#include <iostream>
#include <vector>

// 定義 Plugin 的名稱和版本，必須跟 ONNX 裡的對應
static const char* PLUGIN_NAME = "CustomRoPERMSNorm";
static const char* PLUGIN_VERSION = "1";

// --------------------------------------------------------
// FusedRoPERMSNormPlugin 實作
// --------------------------------------------------------

FusedRoPERMSNormPlugin::FusedRoPERMSNormPlugin(int max_seq_len, float rope_base)
    : mMaxSeqLen(max_seq_len), mRopeBase(rope_base) {}

// 反序列化建構子 (讀取 Engine 時使用)
FusedRoPERMSNormPlugin::FusedRoPERMSNormPlugin(const void* data, size_t length) {
    const char* d = reinterpret_cast<const char*>(data);
    mMaxSeqLen = *reinterpret_cast<const int*>(d); d += sizeof(int);
    mRopeBase = *reinterpret_cast<const float*>(d); d += sizeof(float);
}

// 1. 設定輸出維度 (Output Dimensions)
// 我們的 Plugin 輸出形狀跟輸入 Input(0) 完全一樣
DimsExprs FusedRoPERMSNormPlugin::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept {
    return inputs[0]; // Copy Input[0] dims to Output
}

// 2. 支援的資料格式 (Format Combination)
// 我們目前只支援 Float32 (Linear)
bool FusedRoPERMSNormPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    // 輸入: 0=Input, 1=Gamma, 2=MaxSeqLen(Int), 3=RopeBase(Float)
    // 輸出: 0=Output
    // 注意：ONNX 轉過來的 Constant 有時會被視為輸入 Tensor
    
    // 確保所有主要 Tensor 都是 Float32
    if (pos == 0 || pos == 1 || pos == nbInputs) { // Input, Gamma, Output
        return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
    
    // 輔助參數 Tensor (如果有的話)
    return true; 
}

// 3. 核心執行函式 (Enqueue) 🌟🌟🌟
int FusedRoPERMSNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    
    // 解析輸入維度
    // inputs[0] shape: [Batch, Seq, Hidden]
    int batch_size = inputDesc[0].dims.d[0];
    int seq_len = inputDesc[0].dims.d[1];
    int hidden_size = inputDesc[0].dims.d[2];

    // 取得資料指標
    const float* d_input = static_cast<const float*>(inputs[0]);
    const float* d_gamma = static_cast<const float*>(inputs[1]);
    float* d_output = static_cast<float*>(outputs[0]);

    // 呼叫我們的 CUDA Kernel
    FusedRoPERMSNormLaunch_FP32(
        stream,
        d_input,
        d_gamma,
        d_output,
        batch_size,
        seq_len,
        hidden_size,
        mRopeBase,
        0,       // token_offset (目前簡易版設為 0)
        1e-5f    // epsilon
    );

    return 0;
}

// 其他標準實作 (Boilerplate)
int FusedRoPERMSNormPlugin::getNbOutputs() const noexcept { return 1; }
void FusedRoPERMSNormPlugin::destroy() noexcept { delete this; }
const char* FusedRoPERMSNormPlugin::getPluginType() const noexcept { return PLUGIN_NAME; }
const char* FusedRoPERMSNormPlugin::getPluginVersion() const noexcept { return PLUGIN_VERSION; }
void FusedRoPERMSNormPlugin::setPluginNamespace(const char* pluginNamespace) noexcept { mNamespace = pluginNamespace; }
const char* FusedRoPERMSNormPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
DataType FusedRoPERMSNormPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept { return DataType::kFLOAT; }
size_t FusedRoPERMSNormPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept { return 0; }
IPluginV2DynamicExt* FusedRoPERMSNormPlugin::clone() const noexcept { return new FusedRoPERMSNormPlugin(mMaxSeqLen, mRopeBase); }
void FusedRoPERMSNormPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept {}

// 序列化
size_t FusedRoPERMSNormPlugin::getSerializationSize() const noexcept {
    return sizeof(int) + sizeof(float);
}
void FusedRoPERMSNormPlugin::serialize(void* buffer) const noexcept {
    char* d = reinterpret_cast<char*>(buffer);
    *reinterpret_cast<int*>(d) = mMaxSeqLen; d += sizeof(int);
    *reinterpret_cast<float*>(d) = mRopeBase; d += sizeof(float);
}

// --------------------------------------------------------
// Creator 實作 (工廠模式)
// --------------------------------------------------------

PluginFieldCollection FusedRoPERMSNormPluginCreator::mFC{};
std::vector<PluginField> FusedRoPERMSNormPluginCreator::mPluginAttributes;

FusedRoPERMSNormPluginCreator::FusedRoPERMSNormPluginCreator() {


    // ==========================================
    // 🔥🔥🔥 加入這段 Debug 訊息 🔥🔥🔥
    // ==========================================
    std::cerr << "\n\n";
    std::cerr << "****************************************************************" << std::endl;
    std::cerr << ">>> DEBUG: FusedRoPERMSNormPluginCreator has been LOADED! <<<" << std::endl;
    std::cerr << ">>> Plugin Name: " << PLUGIN_NAME << " <<<" << std::endl;
    std::cerr << "****************************************************************" << std::endl;
    std::cerr << "\n\n";
    // ==========================================
    
    mPluginAttributes.clear();
    // 這裡定義 ONNX 節點中可能的 Attribute
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* FusedRoPERMSNormPluginCreator::getPluginName() const noexcept { return PLUGIN_NAME; }
const char* FusedRoPERMSNormPluginCreator::getPluginVersion() const noexcept { return PLUGIN_VERSION; }
const PluginFieldCollection* FusedRoPERMSNormPluginCreator::getFieldNames() noexcept { return &mFC; }

// TensorRT 在讀取 ONNX 時會呼叫這個函式來建立 Plugin
IPluginV2* FusedRoPERMSNormPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
    // 預設值 (如果 ONNX 沒傳過來的話)
    int max_seq_len = 2048;
    float rope_base = 10000.0f;

    // 解析從 ONNX 傳來的輸入參數 (我們這裡簡化，直接讀取第一個輸入當參數，
    // 實際上這些數值通常是從 Constant Input 傳進來的)
    
    // 注意：因為我們在 Python 是把這些參數當成 "Input Tensor" 傳進來的，
    // 所以在 createPlugin 階段其實拿不到數值 (要等到 enqueue 執行期)。
    // 為了簡單起見，我們這裡先寫死預設值，或者你之後可以在 enqueue 裡動態讀取 Input[2] 和 Input[3]。
    
    return new FusedRoPERMSNormPlugin(max_seq_len, rope_base);
}

IPluginV2* FusedRoPERMSNormPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {
    return new FusedRoPERMSNormPlugin(serialData, serialLength);
}

void FusedRoPERMSNormPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept { mNamespace = pluginNamespace; }
const char* FusedRoPERMSNormPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// 註冊 Plugin Creator (這行最重要，沒有它 TRT 找不到外掛)
REGISTER_TENSORRT_PLUGIN(FusedRoPERMSNormPluginCreator);