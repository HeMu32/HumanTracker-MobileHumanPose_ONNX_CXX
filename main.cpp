#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "yolo_fast.h"
#include "mobileHumanPose.h"
/*
int main()
{
	std::vector<cv::Mat> outs;
    cv::Mat frame = cv::imread ("1.png");
    cv::Mat blob = cv::dnn::blobFromImage(
            frame, 
            1.0/255.0, 
            cv::Size(256, 256),
            cv::Scalar(0.5, 0.5, 0.5), 
            true, 
            false
        );

    cv::dnn::Net net = cv::dnn::readNet("mobile_human_pose_working_well_256x256.onnx");
	
	net.setInput(blob);
	net.forward(outs, net.getUnconnectedOutLayersNames());
    
    // 输出每个输出层的形状信息
    for (size_t i = 0; i < outs.size(); i++) {
        std::cout << "输出 " << i << " 的形状: " 
                  << outs[i].dims << " 维, 大小为 [";
        for (int d = 0; d < outs[i].dims; d++) {
            std::cout << outs[i].size[d];
            if (d < outs[i].dims - 1) std::cout << " x ";
        }
        std::cout << "]" << std::endl;
    }
    
    // 如果想查看第一个输出的部分数据
    if (!outs.empty()) {
        std::cout << "第一个输出的前几个值: " << std::endl;
        cv::Mat firstOut = outs[0];
        
        // 处理多维输出 (假设形状为 [1 x 672 x 32 x 32])
        if (firstOut.dims == 4) {
            // 输出前两个通道的热图数据
            for (int c = 0; c < std::min(2, firstOut.size[1]); c++) {
                std::cout << "通道 " << c << " 的热图数据:" << std::endl;
                
                // 创建一个临时Mat来访问该通道的32x32热图
                cv::Mat heatmap(firstOut.size[2], firstOut.size[3], CV_32F);
                
                // 复制数据 - 注意索引顺序：[batch, channel, height, width]
                for (int h = 0; h < firstOut.size[2]; h++) {
                    for (int w = 0; w < firstOut.size[3]; w++) {
                        float* ptr = (float*)firstOut.data + 
                                    (0 * firstOut.size[1] * firstOut.size[2] * firstOut.size[3]) + 
                                    (c * firstOut.size[2] * firstOut.size[3]) + 
                                    (h * firstOut.size[3]) + w;
                        heatmap.at<float>(h, w) = *ptr;
                    }
                }
                
                // 输出热图的部分数据（例如左上角的5x5区域）
                std::cout << "热图左上角5x5区域:" << std::endl;
                for (int h = 0; h < std::min(5, heatmap.rows); h++) {
                    for (int w = 0; w < std::min(5, heatmap.cols); w++) {
                        std::cout << heatmap.at<float>(h, w) << "\t";
                    }
                    std::cout << std::endl;
                }
                
                // 可视化热图（可选）
                cv::Mat heatmapViz;
                cv::normalize(heatmap, heatmapViz, 0, 255, cv::NORM_MINMAX);
                heatmapViz.convertTo(heatmapViz, CV_8U);
                cv::applyColorMap(heatmapViz, heatmapViz, cv::COLORMAP_JET);
                
                // 保存热图
                std::string filename = "heatmap_channel_" + std::to_string(c) + ".jpg";
                cv::imwrite(filename, heatmapViz);
                std::cout << "已保存热图: " << filename << std::endl;
            }
        } else if (firstOut.dims <= 2) {
            std::cout << firstOut.rowRange(0, std::min(3, firstOut.rows)) << std::endl;
        } else {
            std::cout << "输出维度为 " << firstOut.dims << "，请根据形状信息进行适当的访问" << std::endl;
        }
    }

    return 0;
}
*/

int main()
{
	MobileHumanPose pose_estimator("mobile_human_pose_working_well_256x256.onnx");
	// 检测人体
    std::vector<float> 		scores	= {99};

    //std::vector<cv::Vec4i>  boxes = {cv::Vec4i(200, 40, 360, 540)};
    //cv::Mat					image	= cv::imread ("1.png");

    std::vector<cv::Vec4i>  boxes   = {cv::Vec4i(185, 296, 800, 1200)};
    //std::vector<cv::Vec4i>  boxes   = {cv::Vec4i(200, 320, 490, 1187)};
    cv::Mat					image	= cv::imread ("2.jpg");

    
    //std::vector<cv::Vec4i>  boxes = {cv::Vec4i(150, 341, 853, 1190)};
    //cv::Mat					image	= cv::imread ("3.jpg");



	std::string				output_image_path = "dec.jpg";

    // 如果没有检测到人体，退出
    if (boxes.empty()) {
        std::cout << "未检测到人体" << std::endl;
        return 0;
    }
    
    // 模拟深度
    std::vector<float> depths;
    for (const auto& box : boxes) {
        int     width   = box[2] - box[0];
        int     height  = box[3] - box[1];
        float   area    = width * height;
        float   depth   = 500 / (area / (image.rows * image.cols)) + 500;
        depths.push_back(depth);
    }
    
    // 创建结果图像
    cv::Mat pose_img = image.clone();
    cv::Mat heatmap_viz_img = image.clone();
    cv::Mat img_heatmap = cv::Mat::zeros(image.rows, image.cols, CV_32F);
    
    // 姿态估计工具
    PoseEstimationUtils utils;
    
    // 存储3D姿态
    std::vector<cv::Mat> pose_3d_list;
    
    // 处理每个检测到的人体
    for (size_t i = 0; i < boxes.size(); i++) 
    {
        // 估计姿态
        cv::Mat pose_2d, pose_3d, person_heatmap, joint_scores;

        std::tie(pose_2d, pose_3d, person_heatmap, joint_scores) = 
            pose_estimator.estimatePose(image, boxes[i], depths[i]);

        // 使用更高效的2D姿态估计方法
        cv::Mat pose_2d_fast, joint_scores_fast;
        std::tie(pose_2d_fast, joint_scores_fast) = 
            pose_estimator.estimatePose2d(image, boxes[i]);
            
        // 创建裁剪后的图像
        cv::Rect roi(boxes[i][0], boxes[i][1], boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]);
        cv::Mat cropped_image = image(roi).clone();
        
        // 在裁剪图像上绘制关节点
        cv::Mat cropped_pose_img = cropped_image.clone();
        for (int j = 0; j < pose_2d_fast.rows; j++) {
            // 计算关节在裁剪图像中的坐标
            int x = static_cast<int>(pose_2d_fast.at<float>(j, 0));
            int y = static_cast<int>(pose_2d_fast.at<float>(j, 1));
            
            // 确保坐标在裁剪图像范围内
            if (x >= 0 && x < cropped_image.cols && y >= 0 && y < cropped_image.rows) {
                // 根据置信度调整圆的颜色（红色到绿色）
                float confidence = joint_scores_fast.at<float>(j);
                cv::Scalar color(0, 255 * confidence, 255 * (1 - confidence));
                
                // 绘制关节点
                cv::circle(cropped_pose_img, cv::Point(x, y), 5, color, -1);
                
                // 添加关节索引标签
                cv::putText(cropped_pose_img, std::to_string(j), 
                            cv::Point(x + 5, y - 5), cv::FONT_HERSHEY_SIMPLEX, 
                            0.5, cv::Scalar(0, 0, 255), 1);
            }
        }
        //cv::imwrite ("dec2.jpg", cropped_pose_img);
        cv::imshow ("dec", cropped_pose_img);
        cv::waitKey (0);
        true;
        
/*
        // 绘制骨架
        pose_img = utils.drawSkeleton(pose_img, pose_2d, joint_scores);
        
        // 添加热图
        cv::Rect roi(boxes[i][0], boxes[i][1], boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]);
        cv::Mat roi_heatmap = img_heatmap(roi);
        cv::add(roi_heatmap, person_heatmap, roi_heatmap);
        
        // 添加3D姿态
        pose_3d_list.push_back(pose_3d);
*/
    }
    
    // 绘制热图
    heatmap_viz_img = utils.drawHeatmap(heatmap_viz_img, img_heatmap);
    
    
    // 合并图像
    cv::Mat combined_img;
    cv::hconcat(std::vector<cv::Mat>{heatmap_viz_img, pose_img}, combined_img);
    
    // 保存结果
    cv::imwrite(output_image_path, combined_img);
    
    // 显示结果
    cv::namedWindow("Estimated pose", cv::WINDOW_NORMAL);
    cv::imshow("Estimated pose", combined_img);
    cv::waitKey(0);
}

/*
int main()
{
	yolo_fast yolo_model("yolofastv2.onnx", 0.3, 0.3, 0.4);
	printf ("Model Loaded\n");
	std::string imgpath = "E:/Pictures/HMCatalog/Outputs/IMG_5604_Scealdown.jpg";
	cv::Mat srcimg = cv::imread(imgpath);
	yolo_model.detect(srcimg);

	static const std::string kWinName = "Deep learning object detection in OpenCV";
	cv::namedWindow(kWinName, cv::WINDOW_NORMAL);
	cv::imshow(kWinName, srcimg);
	cv::waitKey(0);
	cv::destroyAllWindows();
}
*/
