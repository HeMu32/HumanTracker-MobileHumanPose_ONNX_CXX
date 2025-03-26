#pragma once

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <opencv2/viz.hpp>

class PoseEstimationUtils
{
public:
    PoseEstimationUtils();
    
    // 裁剪图像
    cv::Mat cropImage(const cv::Mat& image, const cv::Vec4i& bbox);
    
    // 像素坐标转换为相机坐标
    cv::Mat pixel2cam(const cv::Mat& pixel_coord, const cv::Mat& depth, const cv::Vec2f& f, const cv::Vec2f& c);
    
    // 绘制骨架
    cv::Mat drawSkeleton(cv::Mat& img, const cv::Mat& keypoints, const cv::Mat& scores, float kp_thres = 0.02);
    
    // 绘制热力图
    cv::Mat drawHeatmap(const cv::Mat& img, const cv::Mat& img_heatmap);
/*
    // 可视化3D骨架
    cv::Mat vis3DMultipleSkeleton(const std::vector<cv::Mat>& kpt_3d, const std::vector<cv::Mat>& kpt_3d_vis, const std::string& filename = "");
*/
private:
    std::vector<std::string> joints_name;
    int joint_num;
    std::vector<std::pair<int, int>> skeleton;
    std::vector<cv::Scalar> colors_cv;
};