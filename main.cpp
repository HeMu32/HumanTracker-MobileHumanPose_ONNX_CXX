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
    
    
    // 遍历D:\VideoCache目录中的所有jpg文件
    std::string path = "D:\\VideoCache\\Clip5\\*.jpg";
    std::vector<cv::String> filenames;
    cv::glob(path, filenames, false);
    
    // 按文件名排序（假设文件名包含数字序号）
    std::sort(filenames.begin(), filenames.end(), [](const cv::String& a, const cv::String& b) {
        // 从文件路径中提取文件名
        std::string fileA = a.substr(a.find_last_of("\\") + 1);
        std::string fileB = b.substr(b.find_last_of("\\") + 1);
        
        // 从文件名中提取数字部分
        std::string numA, numB;
        for (char c : fileA) {
            if (std::isdigit(c)) numA += c;
        }
        for (char c : fileB) {
            if (std::isdigit(c)) numB += c;
        }
        
        // 如果提取到了数字，按数字大小比较
        if (!numA.empty() && !numB.empty()) {
            return std::stoi(numA) < std::stoi(numB);
        }
        
        // 否则按字符串比较
        return a < b;
    });
    
    for (const auto& filename : filenames)
    {
        // 读取图像
        cv::Mat image = cv::imread(filename);
        if (!image.empty())
        {
            // 处理图像
            int ret = detector.estimate(image);
            if (ret == -1)
                printf ("TRACK LOST\n");
        }
    }
    
    return 0;
}