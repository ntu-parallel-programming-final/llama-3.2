/*
 * src/CustomRoPE.cpp
 * 全局註冊 Plugin Creator
 */

#include "CustomRoPEPlugin.h"

// 實例化 Creator，這會自動觸發 TensorRT 的註冊機制
REGISTER_TENSORRT_PLUGIN(CustomRoPEPluginCreator);