#include "CustomRoPEPlugin.h"
#include <iostream>

PluginFieldCollection CustomRoPEPluginCreator::mFC{};
std::vector<PluginField> CustomRoPEPluginCreator::mPluginAttributes;

CustomRoPEPluginCreator::CustomRoPEPluginCreator() {
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
    
    // 🌟 關鍵修正：在這裡設定 Domain
    // mNamespace = "com.custom.trt";
    mNamespace = "";
}

const char* CustomRoPEPluginCreator::getPluginName() const noexcept {
    return "CustomRoPE";
}

const char* CustomRoPEPluginCreator::getPluginVersion() const noexcept {
    return "1";
}

const PluginFieldCollection* CustomRoPEPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

// 🌟 移除 getPluginDomain 實作

IPluginV2* CustomRoPEPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept {
    auto* plugin = new CustomRoPEPlugin();
    plugin->setPluginNamespace(mNamespace.c_str()); // 確保 Plugin 繼承 Namespace
    return plugin;
}

IPluginV2* CustomRoPEPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {
    auto* plugin = new CustomRoPEPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void CustomRoPEPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* CustomRoPEPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}