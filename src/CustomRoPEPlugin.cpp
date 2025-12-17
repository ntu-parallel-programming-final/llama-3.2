/*
 * src/CustomRoPEPlugin.cpp
 * 實作 IPluginV2DynamicExt 方法
 */

#include "CustomRoPEPlugin.h"
#include <cstring>
#include <iostream>

// 建構子 (反序列化)
CustomRoPEPlugin::CustomRoPEPlugin(const void* data, size_t length) {
    // 由於我們沒有狀態 (權重)，這裡不需要做什麼
}

int CustomRoPEPlugin::getNbOutputs() const noexcept {
    return 1;
}

DimsExprs CustomRoPEPlugin::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept {
    // 輸出形狀 = 輸入 0 的形狀
    return inputs[0];
}

bool CustomRoPEPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
    // 要求所有輸入 (0, 1, 2) 和輸出 (0) 都是 Float 且 Linear
    bool typeOK = (inOut[pos].type == DataType::kFLOAT);
    bool formatOK = (inOut[pos].format == TensorFormat::kLINEAR);
    return typeOK && formatOK;
}

void CustomRoPEPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
    // 可以在這裡驗證維度
}

size_t CustomRoPEPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept {
    return 0;
}

int CustomRoPEPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
            const void* const* inputs, void* const* outputs,
            void* workspace, cudaStream_t stream) noexcept {
    
    const auto& dims = inputDesc[0].dims;
    int nbDims = dims.nbDims;
    int head_dim = dims.d[nbDims - 1]; // 最後一維永遠是 HeadDim

    // 預設初始化
    int batch_size = dims.d[0];
    int num_heads = 1;
    int seq_len = 1;

    // 修正：針對 [Batch, Heads, Seq, HeadDim] 格式解析
    if (nbDims == 4) {
        num_heads = dims.d[1];
        seq_len   = dims.d[2];
    } 
    // 防呆：如果是 3D [Batch, Seq, HeadDim] (很少見但可能)
    else if (nbDims == 3) {
        seq_len = dims.d[1];
    }

    const float* d_input = static_cast<const float*>(inputs[0]);
    const float* d_cos   = static_cast<const float*>(inputs[1]);
    const float* d_sin   = static_cast<const float*>(inputs[2]);
    float* d_output      = static_cast<float*>(outputs[0]);

    launchCustomRoPE(d_input, d_cos, d_sin, d_output, 
                     batch_size, seq_len, num_heads, head_dim, stream);

    return 0;
}

DataType CustomRoPEPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept {
    return inputTypes[0];
}

const char* CustomRoPEPlugin::getPluginType() const noexcept { return "CustomRoPE"; }
const char* CustomRoPEPlugin::getPluginVersion() const noexcept { return "1"; }

void CustomRoPEPlugin::destroy() noexcept { delete this; }

IPluginV2DynamicExt* CustomRoPEPlugin::clone() const noexcept {
    return new CustomRoPEPlugin(*this);
}

void CustomRoPEPlugin::setPluginNamespace(const char* pluginNamespace) noexcept { mNamespace = pluginNamespace; }
const char* CustomRoPEPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

size_t CustomRoPEPlugin::getSerializationSize() const noexcept { return 0; }
void CustomRoPEPlugin::serialize(void* buffer) const noexcept {}
int32_t CustomRoPEPlugin::initialize() noexcept {
    return 0;
}
void CustomRoPEPlugin::terminate() noexcept {}