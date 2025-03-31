#pragma once

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "utils_pose_estimation.h"

// 前向声明
class YoloV5s;

class MobileHumanPose
{
public:
    MobileHumanPose(const std::string& model_path, 
                   const cv::Vec2f& focal_length = cv::Vec2f(1500, 1500), 
                   const cv::Vec2f& principal_points = cv::Vec2f(1280/2, 720/2));
    
    // 重载调用运算符，方便直接调用对象进行姿态估计
    std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> operator()(const cv::Mat& image, const cv::Vec4i& bbox, float abs_depth = 1.0);
    
    // 姿态估计主函数
    // 估计姿态 - 完整版（包含3D信息）
    std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> estimatePose(const cv::Mat& image, const cv::Vec4i& bbox, float abs_depth);
    
    /// @brief      
    /// @param image 
    /// @param bbox 
    /// @return         tuple <cv::Mat, cv::Mat>, 
    ///                 first for pose2d: x: pose_2d.at<float>(i, 0), y: pose_2d.at<float>(i, 1)
    ///                 second for score
    std::tuple<cv::Mat, cv::Mat> estimatePose2d(const cv::Mat& image, const cv::Vec4i& bbox);

    // 处理输出 - 仅计算2D姿态（更高效）
    std::tuple<cv::Mat, cv::Mat> processOutput2d(const cv::Mat& output, const cv::Vec4i& bbox);
    
private:
    // 初始化模型
    void initializeModel(const std::string& model_path);
    
    // 准备输入数据
    cv::Mat prepareInput(const cv::Mat& image, const cv::Vec4i& bbox);
    
    // 执行推理
    cv::Mat inference(const cv::Mat& input_tensor);
    
    // 处理输出数据
    std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> processOutput(const cv::Mat& output, float abs_depth, const cv::Vec4i& bbox);
    
    // 获取模型输入细节
    void getModelInputDetails();
    
    // 获取模型输出细节
    void getModelOutputDetails();

private:
    cv::Vec2f focal_length;
    cv::Vec2f principal_points;
    
    cv::dnn::Net net;
    
    // 输入相关参数, 从Netron获得
    std::string input_name;
    int channels        = 3;
    int input_height    = 256;
    int input_width     = 256;
    
    // 输出相关参数, 从Netron获得
    std::vector<std::string> output_names;
    int output_depth    = 672;
    int output_height   = 32;
    int output_width    = 32;
    
    // 图像尺寸
    int img_height;
    int img_width;
    int img_channels;
    
    // 工具类实例
    PoseEstimationUtils utils;
    
    // 关节数量
    const int joint_num = 21;
};

// 主函数示例
/*
void runPoseEstimation(const std::string& pose_model_path, 
                      const std::string& detector_model_path,
                      const std::string& input_image_path,
                      const std::string& output_image_path);
*/
