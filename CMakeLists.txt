cmake_minimum_required(VERSION 3.10)
project(yolo_fastestv2_opencv)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译器路径
set(CMAKE_C_COMPILER "C:/MinGW/bin/gcc.exe")
set(CMAKE_CXX_COMPILER "C:/MinGW/bin/g++.exe")

# 设置OpenCV路径
set(OpenCV_DIR "C:/opencv/build/x64/MinGW")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件，包含新增的姿态估计相关文件
add_executable (yolo_detector 
                main.cpp 
                yolo_fast.cpp 
                mobileHumanPose.cpp 
                utils_pose_estimation.cpp
                HumanTracker.cpp
)

# 链接库
target_link_libraries(yolo_detector 
    ${OpenCV_LIBS}
)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 复制模型和测试图像到输出目录
add_custom_command(TARGET yolo_detector POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/img"
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/img"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/yolofastv2.onnx"
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/yolofastv2.onnx"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/coco.names"
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/coco.names"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/mobile_human_pose_working_well_256x256.onnx"
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/mobile_human_pose_working_well_256x256.onnx"
)

# 确保将coco.names复制到可执行文件所在的目录
# 同时复制模型文件
add_custom_command(TARGET yolo_detector POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/coco.names"
        "$<TARGET_FILE_DIR:yolo_detector>/coco.names"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/yolofastv2.onnx"
        "$<TARGET_FILE_DIR:yolo_detector>/yolofastv2.onnx"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/mobile_human_pose_working_well_256x256.onnx"
        "$<TARGET_FILE_DIR:yolo_detector>/mobile_human_pose_working_well_256x256.onnx"
)

# 打印信息
message(STATUS "OpenCV库版本: ${OpenCV_VERSION}")
message(STATUS "OpenCV库: ${OpenCV_LIBS}")
message(STATUS "OpenCV包含目录: ${OpenCV_INCLUDE_DIRS}")

# 添加调试可视化宏定义
add_definitions(-D_DEBUG_VISUALIZATOIN)
