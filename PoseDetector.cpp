#include "PoseDetector.h"
#include <iostream>

PoseDetector::PoseDetector(const std::string& poseModelPath, const std::string& yoloModelPath)
    : pose_estimator(poseModelPath)
    , yolo_model(yoloModelPath, 0.3, 0.3, 0.4)
    , detection_done(false)
    , thread_running(false)
    , yolo_thread(nullptr)
{
    // 初始化完成
}

PoseDetector::~PoseDetector()
{
    // 停止线程
    thread_running = false;
    
    // 唤醒线程以便退出
    {
        std::lock_guard<std::mutex> lock(mtxYolo);
        detection_done = true;
    }
    condVarYolo.notify_one();
    
    // 等待线程结束
    if (yolo_thread && yolo_thread->joinable())
    {
        yolo_thread->join();
        delete yolo_thread;
    }
}

void PoseDetector::initDetectionThread()
{
    if (!thread_running)
    {
        thread_running = true;
        yolo_thread = new std::thread(&PoseDetector::yoloDetectionThread, this);
    }
}

void PoseDetector::yoloDetectionThread()
{
    while (thread_running)
    {
        std::unique_lock<std::mutex> lock(mtxYolo);
        condVarYolo.wait(lock, [this]{return !thread_image.empty() || !thread_running;});
        
        // 如果线程被要求退出，则退出循环
        if (!thread_running)
        {
            break;
        }
        
        // 执行检测
        if (detection_done == false)
        {
            yolo_model.detect(thread_image, thread_boxes, 0);
        }
        
        detection_done = true;
        lock.unlock();
        condVarYolo.notify_one();
    }
}

void PoseDetector::processImage(const cv::Mat& image)
{
    if (image.empty())
    {
        std::cout << "输入图像为空" << std::endl;
        return;
    }
    
    std::vector<cv::Vec4i> boxes;
    
    // 添加计时功能
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 准备线程数据
    {
        std::lock_guard<std::mutex> lock(mtxYolo);
        thread_image = image.clone();
        thread_boxes.clear();
        detection_done = false;
    }
    condVarYolo.notify_one();
    
    // 等待检测完成
    std::unique_lock<std::mutex> lock(mtxYolo);
    condVarYolo.wait(lock, [this]{return detection_done;});
    
    boxes = thread_boxes;
    
    auto dec_time = std::chrono::high_resolution_clock::now();
    auto durationDec = std::chrono::duration_cast<std::chrono::milliseconds>(dec_time - start_time);
    printf("找人时间: %ld毫秒  ", durationDec.count());
    
    // 如果没有检测到人体，退出
    if (boxes.empty()) 
    {
        std::cout << "未检测到人体" << std::endl;
        return;
    }
    
    start_time = std::chrono::high_resolution_clock::now();
    
    // 处理每个检测到的人体
    for (size_t i = 0; i < boxes.size(); i++) 
    {
        // 估计姿态
        cv::Mat pose_2d, pose_3d, person_heatmap, joint_scores;
        
        // 使用更高效的2D姿态估计方法
        cv::Mat pose_2d_fast, joint_scores_fast;
        std::tie(pose_2d_fast, joint_scores_fast) = 
            pose_estimator.estimatePose2d(image, boxes[i]);
        
        // 计算并输出执行时间
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        printf("骨骼%d时间: %ld毫秒  ", i, duration.count());
        
        // 在原始图像上绘制指定关节点(9,11,19,20)
        cv::Mat pose_img = image.clone();
        const std::vector<int> target_joints = {9, 11, 19, 20};
        
        for (int j : target_joints) 
        {
            if (j < pose_2d_fast.rows) 
            {
                // 获取关节点坐标
                int x = static_cast<int>(pose_2d_fast.at<float>(j, 0)) + boxes[i][0];
                int y = static_cast<int>(pose_2d_fast.at<float>(j, 1)) + boxes[i][1];
                
                // 确保坐标在图像范围内
                if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) 
                {
                    // 根据置信度调整圆的颜色（红色到绿色）
                    float confidence = joint_scores_fast.at<float>(j);
                    cv::Scalar color(0, 255 * confidence, 255 * (1 - confidence));
                    
                    // 绘制关节点
                    cv::circle(pose_img, cv::Point(x, y), 5, color, -1);
                    
                    // 添加关节索引标签
                    cv::putText(pose_img, std::to_string(j), 
                                cv::Point(x + 5, y - 5), cv::FONT_HERSHEY_SIMPLEX, 
                                0.5, cv::Scalar(0, 0, 255), 1);
                }
            }
        }
        
        // 显示结果图像 - 调整到800像素高
        cv::Mat resized_img;
        float scale = 800.0f / pose_img.rows;
        cv::resize(pose_img, resized_img, cv::Size(), scale, scale, cv::INTER_LINEAR);
        cv::imshow("Pose Estimation", resized_img);
        cv::waitKey(20);
    }
    
    // 为上面的输出换行
    putchar('\n');
}