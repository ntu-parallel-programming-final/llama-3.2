#ifndef CUSTOM_ROPE_PLUGIN_H
#define CUSTOM_ROPE_PLUGIN_H

#include "NvInfer.h"
#include "CustomRoPEKernel.h"
#include <string>
#include <vector>

using namespace nvinfer1;

class CustomRoPEPlugin : public IPluginV2DynamicExt {
public:
    CustomRoPEPlugin() = default;
    CustomRoPEPlugin(const void* data, size_t length);

    // IPluginV2DynamicExt Methods
    int getNbOutputs() const noexcept override;
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs,
                void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2 Methods
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override;
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    void destroy() noexcept override;
    IPluginV2DynamicExt* clone() const noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    
    // 🌟 修正：返回類型改為 int32_t
    int32_t initialize() noexcept override;
    void terminate() noexcept override;

private:
    std::string mNamespace;
};

class CustomRoPEPluginCreator : public IPluginCreator {
public:
    CustomRoPEPluginCreator();

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
    
    // 🌟 移除 getPluginDomain，改用 Namespace
    // const char* getPluginDomain() const noexcept override; 

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

#endif // CUSTOM_ROPE_PLUGIN_H