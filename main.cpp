#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "HumanTracker.h"

int main()
{
    // 创建PoseDetector实例
    HumanTracker detector("mobile_human_pose_working_well_256x256.onnx", "yolofastv2.onnx");
    
    // 初始化检测线程
    detector.initThreads();
    
    while (true)
    {
        // 遍历D:\VideoCache目录中的所有jpg文件
        std::string path = "D:\\VideoCache\\*.jpg";
        std::vector<cv::String> filenames;
        cv::glob(path, filenames, false);
        
        for (const auto& filename : filenames)
        {
            // 读取图像
            cv::Mat image = cv::imread(filename);
            if (!image.empty())
            {
                // 处理图像
                detector.estimate(image);
            }
        }
    }
    
    return 0;
}