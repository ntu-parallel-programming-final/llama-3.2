// FusedRoPERMSNormCreator.cpp

#include "FusedRoPERMSNormPlugin.h" // 修正：已更名為 Plugin.h
#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <cassert>

using namespace nvinfer1;
using namespace plugin;

// ------------------------------------------------------------------
// 步驟 6.1: FusedRoPERMSNormCreator 類別定義 (使用 IPluginCreator V2 介面)
// ------------------------------------------------------------------

// 繼承 IPluginCreator 介面
class FusedRoPERMSNormCreator : public IPluginCreator {
public:
    // IPluginCreator 介面方法
    const char* getPluginName() const noexcept override { return "CustomRoPERMSNorm"; }
    const char* getPluginVersion() const noexcept override { return "1.0"; }
    const PluginFieldCollection* getFieldNames() noexcept override;
    
    // 修正：使用 IPluginV2 簽名
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    // 修正：使用 IPluginV2 簽名
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override; 

    void setPluginNamespace(const char* pluginNamespace) noexcept override { mNamespace = pluginNamespace; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    PluginFieldCollection mFC;
    std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};


// ------------------------------------------------------------------
// 步驟 6.2: 獲取欄位名稱 (定義外掛程式的參數) - 保持不變
// ------------------------------------------------------------------

const PluginFieldCollection* FusedRoPERMSNormCreator::getFieldNames() noexcept {
    // 這裡我們不再需要 'gamma_weights'，因為我們在 Plugin.cpp 裡是通過 Input 拿的
    // 我們讓這個 Creator 保持簡潔
    mFC.nbFields = 0;
    mFC.fields = nullptr;
    return &mFC;
}


// ------------------------------------------------------------------
// 步驟 6.3: 創建外掛程式實例並初始化權重 (修正參數簽名)
// ------------------------------------------------------------------

// 修正：使用 IPluginV2 簽名，不傳遞 EsdPluginTensors
IPluginV2* FusedRoPERMSNormCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept 
{
    // 預設值 (如果 ONNX 沒傳過來的話)
    int max_seq_len = 2048;
    float rope_base = 10000.0f;
    
    // 注意：這裡無法獲得 gamma 權重和輸入張量的維度，因為 V2 介面沒有提供。
    // Gamma 權重必須在 Plugin 內部 (configurePlugin 或 initialize) 通過 inputs[1] 讀取。
    
    // 這裡只實例化 Plugin，並傳入 RoPE 參數
    try {
        return new FusedRoPERMSNormPlugin(max_seq_len, rope_base);
    } catch (const std::exception& e) {
        std::cerr << "Plugin creation failed: " << e.what() << std::endl;
        return nullptr;
    }
}


// ------------------------------------------------------------------
// 步驟 6.4: 從序列化數據建構
// ------------------------------------------------------------------

IPluginV2* FusedRoPERMSNormCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {
    try {
        return new FusedRoPERMSNormPlugin(serialData, serialLength);
    } catch (const std::exception& e) {
        std::cerr << "Plugin deserialization failed: " << e.what() << std::endl;
        return nullptr;
    }
}


// ------------------------------------------------------------------
// 步驟 6.5: 靜態註冊 (使用 V2 宏)
// ------------------------------------------------------------------

namespace {
    static FusedRoPERMSNormCreator gCreator;
    // 使用 V2 介面的宏進行註冊
    REGISTER_TENSORRT_PLUGIN(FusedRoPERMSNormCreator); 
}