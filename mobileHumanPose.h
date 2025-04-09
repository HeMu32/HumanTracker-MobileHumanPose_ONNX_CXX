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
    MobileHumanPose(const std::string &model_path, 
                   const cv::Vec2f &focal_length = cv::Vec2f(1500, 1500), 
                   const cv::Vec2f &principal_points = cv::Vec2f(1280/2, 720/2));
    
    bool isModelEmpty() const {
    return net.empty();
    }
    // 重载调用运算符，方便直接调用对象进行姿态估计
    std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> operator()(const cv::Mat &image, const cv::Vec4i &bbox, float abs_depth = 1.0);
    
    /// @brief              An attempt to reproduce the 3d pose estimation. Failed
    /// @param image        Picture to be estimatted
    /// @param bbox         Bound box for the person to be estimatted
    /// @param abs_depth    Depth
    /// @return             tuple <cv::Mat, cv::Mat, cv::Mat, cv::Mat>: pose_2d, pose_3d, resized_heatmap, scores
    std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> estimatePose(const cv::Mat &image, const cv::Vec4i &bbox, float abs_depth);
    
    /// @brief      
    /// @param image    Picture to be dectected
    /// @param bbox     Bound box of a person, xyxy (not xywh!)
    /// @return         tuple <cv::Mat, cv::Mat>, pose2d, scores
    ///                 pose2d: x: pose_2d.at<float>(i, 0), y: pose_2d.at<float>(i, 1), 
    ///                 in regard of top left of the box, unit: pixel
    std::tuple<cv::Mat, cv::Mat> estimatePose2d(const cv::Mat &image, const cv::Vec4i &bbox);

    // 处理输出 - 仅计算2D姿态（更高效）

    /// @brief          Process model output into estimated 2d joints
    /// @param output   Output of MHP model
    /// @param bbox     box
    /// @return         tuple <cv::Mat, cv::Mat>, pose2d, scores
    ///                 pose2d: x: pose_2d.at<float>(i, 0), y: pose_2d.at<float>(i, 1), 
    ///                 in regard of top left of the box, unit: pixel
    std::tuple<cv::Mat, cv::Mat> processOutput2d(const cv::Mat &output, const cv::Vec4i &bbox);
    
private:
    /// @brief              Initialize model
    /// @param model_path 
    void initializeModel(const std::string &model_path);
    
    /// @brief          Prepare input to blob that the network accpects
    /// @param image    Picture to be detected
    /// @param bbox     Bound box for the preson
    /// @return         Bolb aka input tensor
    cv::Mat prepareInput(const cv::Mat &image, const cv::Vec4i &bbox);
    
    /// @brief              Interference Mobile Human Pose via cv::dnn
    /// @param input_tensor Input blob
    /// @return             Model outupt, 1*672*32*32 heatmap
    ///                     Actually 1*(21*32)*32*32, where each of 21 jints has a 32*32*32 heatmap
    cv::Mat inference(const cv::Mat &input_tensor);
    
    // 实现不能, 菜
    std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> processOutput(const cv::Mat &output, float abs_depth, const cv::Vec4i &bbox);
    
    // 获取模型输入细节, 其实信息是写死的
    void getModelInputDetails();
    
    // 获取模型输出细节, 其实信息是写死的
    void getModelOutputDetails();

private:
    cv::Vec2f focal_length;
    cv::Vec2f principal_points;
    
    cv::dnn::Net net;
    
    // 输入相关参数, 从Netron获得, 会被getModelInputDetails()覆写
    std::string input_name;
    int channels        = 3;
    int input_height    = 256;
    int input_width     = 256;
    
    // 输出相关参数, 从Netron获得, 会被getModelOutputDetails()覆写
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
void runPoseEstimation(const std::string &pose_model_path, 
                      const std::string &detector_model_path,
                      const std::string &input_image_path,
                      const std::string &output_image_path);
*/
