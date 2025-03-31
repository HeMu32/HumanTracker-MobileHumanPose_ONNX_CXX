#include "utils_pose_estimation.h"

PoseEstimationUtils::PoseEstimationUtils()
{
    // 初始化关节名称
    joints_name = {"Head_top", "Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "L_Shoulder", 
                  "L_Elbow", "L_Wrist", "R_Hip", "R_Knee", "R_Ankle", "L_Hip", "L_Knee", 
                  "L_Ankle", "Pelvis", "Spine", "Head", "R_Hand", "L_Hand", "R_Toe", "L_Toe"};
    
    joint_num = joints_name.size(); // 21
    
    // 初始化骨架连接
    skeleton = {
        {0, 16}, {16, 1}, {1, 15}, {15, 14}, {14, 8}, {14, 11}, {8, 9}, {9, 10}, {10, 19}, 
        {11, 12}, {12, 13}, {13, 20}, {1, 2}, {2, 3}, {3, 4}, {4, 17}, {1, 5}, {5, 6}, {6, 7}, {7, 18}
    };
    
    // 初始化颜色映射 - 使用彩虹色谱
    // 在C++中我们使用固定的颜色集合来替代Python中的cmap
    colors_cv = {
        cv::Scalar(255, 0, 0), cv::Scalar(255, 63, 0), cv::Scalar(255, 127, 0), 
        cv::Scalar(255, 191, 0), cv::Scalar(255, 255, 0), cv::Scalar(191, 255, 0), 
        cv::Scalar(127, 255, 0), cv::Scalar(63, 255, 0), cv::Scalar(0, 255, 0), 
        cv::Scalar(0, 255, 63), cv::Scalar(0, 255, 127), cv::Scalar(0, 255, 191), 
        cv::Scalar(0, 255, 255), cv::Scalar(0, 191, 255), cv::Scalar(0, 127, 255), 
        cv::Scalar(0, 63, 255), cv::Scalar(0, 0, 255), cv::Scalar(63, 0, 255), 
        cv::Scalar(127, 0, 255), cv::Scalar(191, 0, 255), cv::Scalar(255, 0, 255), 
        cv::Scalar(255, 0, 191), cv::Scalar(255, 0, 127)
    };
}

cv::Mat PoseEstimationUtils::cropImage(const cv::Mat& image, const cv::Vec4i& bbox)
{
    int xmin = bbox[0];
    int ymin = bbox[1];
    int xmax = bbox[2];
    int ymax = bbox[3];
    
    // 确保边界在图像范围内
    xmin = std::max(0, xmin);
    ymin = std::max(0, ymin);
    xmax = std::min(image.cols, xmax);
    ymax = std::min(image.rows, ymax);
    
    return image(cv::Range(ymin, ymax), cv::Range(xmin, xmax)).clone();
}

cv::Mat PoseEstimationUtils::pixel2cam(const cv::Mat& pixel_coord, const cv::Mat& depth, const cv::Vec2f& f, const cv::Vec2f& c)
{
    // 创建输出矩阵
    cv::Mat cam_coord(pixel_coord.rows, 3, CV_32F);
    
    // 复制像素坐标的x和y
    pixel_coord.col(0).copyTo(cam_coord.col(0));
    pixel_coord.col(1).copyTo(cam_coord.col(1));
    
    // 复制深度值
    depth.copyTo(cam_coord.col(2));
    
    // 注意：这里简化了转换，实际应用中可能需要更精确的转换
    // 完整的转换应该是：
    // x = (pixel_x - c_x) / f_x * depth
    // y = (pixel_y - c_y) / f_y * depth
    // z = depth
    
    return cam_coord;
}

cv::Mat PoseEstimationUtils::drawSkeleton(cv::Mat& img, const cv::Mat& keypoints, const cv::Mat& scores, float kp_thres)
{
    cv::Mat result = img.clone();
    
    for (size_t i = 0; i < skeleton.size(); i++)
    {
        int point1_id = skeleton[i].first;
        int point2_id = skeleton[i].second;
        
        cv::Point point1(static_cast<int>(keypoints.at<float>(point1_id, 0)), 
                         static_cast<int>(keypoints.at<float>(point1_id, 1)));
        cv::Point point2(static_cast<int>(keypoints.at<float>(point2_id, 0)), 
                         static_cast<int>(keypoints.at<float>(point2_id, 1)));
        
        // 绘制线段
        cv::line(result, point1, point2, colors_cv[i % colors_cv.size()], 3, cv::LINE_AA);
        
        // 绘制关节点
        cv::circle(result, point1, 5, colors_cv[i % colors_cv.size()], -1, cv::LINE_AA);
        cv::circle(result, point2, 5, colors_cv[i % colors_cv.size()], -1, cv::LINE_AA);
 /*       
        如果需要使用置信度
        if (scores.at<float>(point1_id) > kp_thres && scores.at<float>(point2_id) > kp_thres) {
            cv::line(result, point1, point2, colors_cv[i % colors_cv.size()], 3, cv::LINE_AA);
        }
        
        if (scores.at<float>(point1_id) > kp_thres) {
            cv::circle(result, point1, 5, colors_cv[i % colors_cv.size()], -1, cv::LINE_AA);
        }
        
        if (scores.at<float>(point2_id) > kp_thres) {
            cv::circle(result, point2, 5, colors_cv[i % colors_cv.size()], -1, cv::LINE_AA);
        }
*/
    }
    
    return result;
}

cv::Mat PoseEstimationUtils::drawHeatmap(const cv::Mat& img, const cv::Mat& img_heatmap)
{
    // 归一化热力图
    cv::Mat norm_heatmap;
    double min_val, max_val;
    cv::minMaxLoc(img_heatmap, &min_val, &max_val);
    norm_heatmap = 255.0 * ((img_heatmap - min_val) / (max_val - min_val));
    norm_heatmap.convertTo(norm_heatmap, CV_8UC1);
    
    // 应用颜色映射
    cv::Mat color_heatmap;
    cv::applyColorMap(norm_heatmap, color_heatmap, cv::COLORMAP_MAGMA);
    
    // 混合原图和热力图
    cv::Mat result;
    cv::addWeighted(img, 0.4, color_heatmap, 0.6, 0, result);
    
    return result;
}
/*
cv::Mat PoseEstimationUtils::vis3DMultipleSkeleton(const std::vector<cv::Mat>& kpt_3d, const std::vector<cv::Mat>& kpt_3d_vis, const std::string& filename)
{
    // 创建3D可视化窗口
    cv::viz::Viz3d window("3D Pose Visualization");
    window.setBackgroundColor(cv::viz::Color::white());
    
    // 设置相机位置
    cv::Vec3f cam_pos(0, 0, -3.0), cam_focal_point(0, 0, 0), cam_y_dir(0, -1, 0);
    window.setViewerPose(cv::viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir));
    
    // 为每个人添加骨架
    for (size_t n = 0; n < kpt_3d.size(); n++) {
        for (size_t l = 0; l < skeleton.size(); l++) {
            int i1 = skeleton[l].first;
            int i2 = skeleton[l].second;
            
            // 检查关节点是否可见
            if (kpt_3d_vis[n].at<float>(i1, 0) > 0 && kpt_3d_vis[n].at<float>(i2, 0) > 0) {
                // 创建3D线段
                cv::Point3f pt1(kpt_3d[n].at<float>(i1, 0), kpt_3d[n].at<float>(i1, 1), kpt_3d[n].at<float>(i1, 2));
                cv::Point3f pt2(kpt_3d[n].at<float>(i2, 0), kpt_3d[n].at<float>(i2, 1), kpt_3d[n].at<float>(i2, 2));
                
                // 添加线段到可视化窗口
                std::string line_name = "line_" + std::to_string(n) + "_" + std::to_string(l);
                cv::viz::WLine line(pt1, pt2, cv::viz::Color(colors_cv[l % colors_cv.size()]));
                line.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
                window.showWidget(line_name, line);
            }
            
            // 添加关节点
            if (kpt_3d_vis[n].at<float>(i1, 0) > 0) {
                cv::Point3f pt(kpt_3d[n].at<float>(i1, 0), kpt_3d[n].at<float>(i1, 1), kpt_3d[n].at<float>(i1, 2));
                std::string sphere_name = "sphere_" + std::to_string(n) + "_" + std::to_string(i1);
                cv::viz::WSphere sphere(pt, 0.05, 10, 10, cv::viz::Color(colors_cv[l % colors_cv.size()]));
                window.showWidget(sphere_name, sphere);
            }
            
            if (kpt_3d_vis[n].at<float>(i2, 0) > 0) {
                cv::Point3f pt(kpt_3d[n].at<float>(i2, 0), kpt_3d[n].at<float>(i2, 1), kpt_3d[n].at<float>(i2, 2));
                std::string sphere_name = "sphere_" + std::to_string(n) + "_" + std::to_string(i2);
                cv::viz::WSphere sphere(pt, 0.05, 10, 10, cv::viz::Color(colors_cv[l % colors_cv.size()]));
                window.showWidget(sphere_name, sphere);
            }
        }
    }
    
    // 渲染一帧并捕获图像
    window.spinOnce(1, true);
    cv::Mat result = window.getScreenshot();
    
    // 如果提供了文件名，保存图像
    if (!filename.empty()) {
        cv::imwrite(filename, result);
    }
    
    return result;
}
*/