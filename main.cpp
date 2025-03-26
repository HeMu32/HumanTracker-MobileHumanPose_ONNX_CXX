#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "yolo_fast.h"
#include "mobileHumanPose.h"

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
    
    std::cout << outs[0] << " ";

    return 0;
}

/*
int main()
{
	MobileHumanPose pose_estimator("mobile_human_pose_working_well_256x256.onnx"
							);
	// 检测人体
    std::vector<cv::Vec4i> boxes = {cv::Vec4i(200, 40, 360, 540)};
    std::vector<float> 		scores	= {99};
    cv::Mat					image	= cv::imread ("1.png");

	std::string				output_image_path = "dec.jpg";

    // 如果没有检测到人体，退出
    if (boxes.empty()) {
        std::cout << "未检测到人体" << std::endl;
        return 0;
    }
    
    // 模拟深度
    std::vector<float> depths;
    for (const auto& box : boxes) {
        int width = box[2] - box[0];
        int height = box[3] - box[1];
        float area = width * height;
        float depth = 500 / (area / (image.rows * image.cols)) + 500;
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
    for (size_t i = 0; i < boxes.size(); i++) {
        // 估计姿态
        cv::Mat pose_2d, pose_3d, person_heatmap, joint_scores;
        std::tie(pose_2d, pose_3d, person_heatmap, joint_scores) = 
            pose_estimator.estimatePose(image, boxes[i], depths[i]);
        
        // 绘制骨架
        pose_img = utils.drawSkeleton(pose_img, pose_2d, joint_scores);
        
        // 添加热图
        cv::Rect roi(boxes[i][0], boxes[i][1], boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]);
        cv::Mat roi_heatmap = img_heatmap(roi);
        cv::add(roi_heatmap, person_heatmap, roi_heatmap);
        
        // 添加3D姿态
        pose_3d_list.push_back(pose_3d);
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
*/
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
