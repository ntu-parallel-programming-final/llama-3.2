#pragma once

#include <vector>
#include <string>
#include <NvInferPlugin.h>
#include "FusedRoPERMSNormKernel.h" // 包含我們剛寫好的 Kernel 介面

using namespace nvinfer1;

// Plugin 類別：負責執行時期的邏輯
class FusedRoPERMSNormPlugin : public IPluginV2DynamicExt {
public:
    // 建構子 (Constructor)
    FusedRoPERMSNormPlugin(int max_seq_len, float rope_base);
    FusedRoPERMSNormPlugin(const void* data, size_t length); // 用於從 Engine 反序列化

    // IPluginV2DynamicExt 必要實作的方法
    int getNbOutputs() const noexcept override;
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2 必要實作
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    void destroy() noexcept override;
    IPluginV2DynamicExt* clone() const noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

    // 序列化 (Serialization)
    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;
    void serialize(void* buffer) const noexcept override;
    size_t getSerializationSize() const noexcept override;
    void terminate() noexcept override {}
    int initialize() noexcept override { return 0; }

private:
    std::string mNamespace;
    // 我們的自定義參數
    int mMaxSeqLen;
    float mRopeBase;
};

// Creator 類別：負責將 ONNX 節點轉換為 Plugin 實例
class FusedRoPERMSNormPluginCreator : public IPluginCreator {
public:
    FusedRoPERMSNormPluginCreator();
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};