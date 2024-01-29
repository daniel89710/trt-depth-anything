cmake_minimum_required(VERSION 3.1) 
project(trt-depth-anything VERSION 1.0)            
enable_language(CUDA)

set(CMAKE_CXX_COMPILIER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -Wno-write-strings -Wall")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O0 -g -Wno-write-strings -Wall")
#set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -g")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

add_compile_options(-D DEPTH_ANYTHING_STANDALONE)

include_directories(/usr/local/cuda/include include)
link_directories(/usr/local/cuda/lib64)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
find_package(OpenCV REQUIRED)


file(GLOB_RECURSE sources src/tensorrt_common/tensorrt_common.cpp src/tensorrt_common/simple_profiler.cpp src/tensorrt_depth_anything/tensorrt_depth_anything.cpp)
add_library(depth_estimatior SHARED ${sources})

target_link_libraries(depth_estimatior  nvinfer nvinfer_plugin nvparsers nvonnxparser cuda cudart  "stdc++fs")
target_include_directories(depth_estimatior PRIVATE extra/ modules/ ${OpenCV_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/include )

add_executable(trt-depth-anything src/main.cpp src/config_parser.cpp ${sources})

target_include_directories(trt-depth-anything PRIVATE include)
target_link_libraries(trt-depth-anything  ${OpenCV_LIBS} nvinfer nvinfer_plugin nvparsers nvonnxparser cuda cudart gflags boost_system boost_filesystem dl "stdc++fs" "stdc++")




