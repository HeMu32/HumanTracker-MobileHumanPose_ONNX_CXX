#ifndef POSE_DETECTOR_H
#define POSE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <vector>
#include <string>

#include "yolo_fast.h"
#include "mobileHumanPose.h"

class HumanTracker
{
public:
    // 构造函数
    HumanTracker(const std::string& poseModelPath, const std::string& yoloModelPath);
    
    // 析构函数
    ~HumanTracker();
    
    // 初始化检测线程
    void initDetectionThread();
    
    // 处理图像并显示结果
    void processImage(const cv::Mat& image);
    
private:
    // 模型
    MobileHumanPose pose_estimator;
    yolo_fast yolo_model;
    
    // 线程同步变量
    std::mutex mtxYolo;
    std::condition_variable condVarYolo;
    bool detection_done;
    cv::Mat thread_image;
    std::vector<cv::Vec4i> thread_boxes;
    std::thread* yolo_thread;
    bool thread_running;
    
    // YOLO检测线程函数
    void yoloDetectionThread();
};

#endif // POSE_DETECTOR_H