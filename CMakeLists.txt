cmake_minimum_required(VERSION 3.10)
# 專案名稱改為 CustomRoPE 比較貼切，但這只是標籤，不影響生成的檔名
project(CustomRoPE LANGUAGES CXX CUDA)

# 1. 設定標準
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 2. 尋找 TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h 
    HINTS /usr/include/x86_64-linux-gnu /usr/include /usr/local/tensorrt/include)
find_library(TENSORRT_LIB nvinfer 
    HINTS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/tensorrt/lib)

if (NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
    message(FATAL_ERROR "TensorRT not found.")
endif()

# 3. 編譯參數
# 設置計算能力為 8.9 (RTX 40 系列)
set(COMPUTE_CAPABILITY "89") 

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -fPIC")

# 生成 SM 8.9 的機器碼 (cubin) 和 PTX
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${COMPUTE_CAPABILITY},code=compute_${COMPUTE_CAPABILITY}")

# 4. 包含路徑
# 加入 src 目錄，這樣 .cpp 檔案可以找到 .h 檔案
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(/usr/local/cuda/include)

# 5. 來源檔案設定 (對應我們剛剛拆分的 6 個檔案)
# 注意：這裡假設您的檔案都放在 src/ 資料夾下
set(PLUGIN_SRCS
    src/CustomRoPEKernel.cu
    src/CustomRoPEPlugin.cpp
    src/CustomRoPECreator.cpp
    src/CustomRoPE.cpp
    # .h 檔案不需要列在 add_library 中，只要 include_directories 正確即可
)

# 6. 建立 Shared Library (Plugin)
# 這裡將輸出的庫名稱改為 "CustomRoPE" (生成 libCustomRoPE.so)
add_library(CustomRoPE SHARED ${PLUGIN_SRCS})

# 7. 連結 Plugin
target_link_libraries(CustomRoPE PRIVATE ${TENSORRT_LIB} cudart)

# 8. (可選) 移除 Test Kernel 相關設定
# 除非您有針對 CustomRoPE 寫特定的單元測試，否則可以先註解掉，避免編譯錯誤
# set(TEST_SRCS src/test_kernel.cu)
# add_executable(test_kernel ${TEST_SRCS})
# target_link_libraries(test_kernel PRIVATE cudart)