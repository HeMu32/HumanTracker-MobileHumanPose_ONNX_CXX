#include "HumanTracker.h"
#include <iostream>

HumanTracker::HumanTracker(const std::string& poseModelPath, const std::string& yoloModelPath)
    : pose_estimator(poseModelPath)
    , yolo_model(yoloModelPath, 0.3, 0.3, 0.4)
    , detection_done(false)
    , thread_running(false)
    , yolo_thread(nullptr)
{
    // 初始化完成
}

HumanTracker::~HumanTracker()
{
    // 停止线程
    thread_running = false;
    
    
    {   // 唤醒线程以便退出
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

void HumanTracker::initThreads()
{
    if (!thread_running)
    {
        thread_running = true;
        yolo_thread = new std::thread(&HumanTracker::yoloDetectionThread, this);
    }
}

void HumanTracker::yoloDetectionThread()
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
            if (!thread_boxes.empty() || !thread_image.empty())
                yolo_model.detect(thread_image, thread_boxes, 0);
        }
        
        detection_done = true;
        lock.unlock();
        condVarYolo.notify_one();
    }
}

int HumanTracker::estimate(const cv::Mat& image)
{
    int ret = 0;
    if (image.empty())
    {
#ifdef _DEBUG
        std::cout << "HumanTracker::estimate 输入图像为空" << std::endl;
#endif
        ret = -1;
        return ret;
    }
    
    std::vector<cv::Vec4i> boxes;   // Bound box of detected people by Yolo
    cv::Vec4i indicationBox;        // For visualization, not the bound box by yolo.
    int xCenter;                    // Weighted center of detected person
    int yCenter;                    // Weighted center of detected person
    
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


    if (flagFirstFrame == false)
    {
        /// @todo perform optical flow estimation here
    }
    
    if (boxes.empty()) 
    {
        /// @todo process momentumn-opti flow box generation here
        std::cout << "未检测到人体" << std::endl;
        ret = 1;
        return ret;
    }

    if (boxes.size() > 1)
    {
        /// @todo Process tracking of boxex here

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

        // Calculate detection indication box (h: head to pelvis w: 1/4h)
        // and center (avg of the head, spine and pelvis)
        // For the output of interfrence result of cv::dnn, spine is indexed 9.
        int xSpine  = static_cast<int>(pose_2d_fast.at<float>(9, 0))  + boxes[i][0];
        int ySpine  = static_cast<int>(pose_2d_fast.at<float>(9, 1))  + boxes[i][1];
        // Pelvis is indexed 11
        int xPelvis = static_cast<int>(pose_2d_fast.at<float>(11, 0)) + boxes[i][0];
        int yPelvis = static_cast<int>(pose_2d_fast.at<float>(11, 1)) + boxes[i][1];
        // Have the position of head as avg of joint 19 and 20.
        int xHead   = static_cast<int>(pose_2d_fast.at<float>(19, 0)) + boxes[i][0];
        int yHead   = static_cast<int>(pose_2d_fast.at<float>(19, 1)) + boxes[i][1];
        xHead /= 2.0f;     
        yHead /= 2.0f;
        xHead += (static_cast<int>(pose_2d_fast.at<float>(20, 0)) + boxes[i][0]) / 2.0f;
        yHead += (static_cast<int>(pose_2d_fast.at<float>(20, 1)) + boxes[i][1]) / 2.0f;

        xCenter = (xHead + xSpine + xPelvis) / 3.0f;    // Weighted center of detected person
        yCenter = (yHead + ySpine + yPelvis) / 3.0f;    // Weighted center of detected person

        indicationBox[0] = xHead < xPelvis ? (xHead - (yPelvis - yHead) / 8) : (xPelvis - (yPelvis - yHead) / 10);
        indicationBox[1] = yHead;
        indicationBox[2] = xHead < xPelvis ? (xPelvis + (yPelvis - yHead) / 8) : (xHead + (yPelvis - yHead) / 10);
        indicationBox[3] = yPelvis;

        // Visualization
        cv::Mat dect_img = image.clone();
/*
        // Draw head
        cv::circle (dect_img, cv::Point(xHead, yHead), 5, cv::Scalar(16, 16, 233), -1);     // Red for joints
        cv::putText (dect_img, "Head", cv::Point(xHead + 5, yHead - 5), cv::FONT_HERSHEY_SIMPLEX, 
                    0.5, cv::Scalar(0, 0, 255), 1);
        // Draw spine
        cv::circle (dect_img, cv::Point(xSpine, ySpine), 5, cv::Scalar(16, 16, 233), -1);     // Red for joints
        cv::putText (dect_img, "Spine", cv::Point(xSpine + 5, ySpine - 5), cv::FONT_HERSHEY_SIMPLEX, 
                    0.5, cv::Scalar(0, 0, 255), 1);
        // Draw pelvis
        cv::circle (dect_img, cv::Point(xPelvis, yPelvis), 5, cv::Scalar(16, 16, 233), -1);     // Red for joints
        cv::putText (dect_img, "Pelvis", cv::Point(xPelvis + 5, yPelvis - 5), cv::FONT_HERSHEY_SIMPLEX, 
                    0.5, cv::Scalar(0, 0, 255), 1);
*/
        // Draw indicationBox
        cv::rectangle(dect_img, 
            cv::Point(indicationBox[0], indicationBox[1]), 
            cv::Point(indicationBox[2], indicationBox[3]), 
            cv::Scalar(16, 255, 16), 2);  // Green
/*
        // Weighted center of the person
        cv::circle(dect_img, cv::Point(xCenter, yCenter), 7, cv::Scalar(255, 0, 0), -1);  // Blue dot
        cv::putText(dect_img, "Center", cv::Point(xCenter + 5, yCenter - 5), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 16, 16), 1);
*/
/*
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
                    cv::circle(dect_img, cv::Point(x, y), 5, color, -1);
                    
                    // 添加关节索引标签
                    cv::putText(dect_img, std::to_string(j), 
                                cv::Point(x + 5, y - 5), cv::FONT_HERSHEY_SIMPLEX, 
                                0.5, cv::Scalar(0, 0, 255), 1);
                }
            }
        }
*/        
        // 显示结果图像 - 调整到800像素高
        cv::Mat resized_img;
        float scale = 800.0f / dect_img.rows;
        cv::resize(dect_img, resized_img, cv::Size(), scale, scale, cv::INTER_LINEAR);
        cv::imshow("Pose Estimation", resized_img);
        cv::waitKey(20);
    }
    // 为上面的输出换行
    putchar('\n');

    this->PrevFrame      = image;
    this->flagFirstFrame = true;
    this->xPrevCenter    = xCenter;
    this->yPrevCenter    = yCenter;
    //this->PrevBox        = theTrackedBox;

    return ret;
}