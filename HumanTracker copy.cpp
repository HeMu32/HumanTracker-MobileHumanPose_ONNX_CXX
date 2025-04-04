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
    cv::Vec4i indicationBox;        // For visualization and optical flow calc., not the bound box by yolo.
    int xOptiFlow   = 0;            // x value of optical flow vector
    int yOptiFlow   = 0;            // y value of optical flow vector
    int xCenter     = 0;            // Weighted center of detected person
    int yCenter     = 0;            // Weighted center of detected person
    
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
        // 执行光流估计
        cv::Mat prevGray, currGray;
        
        // 转换为灰度图
        cv::cvtColor(PrevFrame, prevGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(image, currGray, cv::COLOR_BGR2GRAY);
        
        // 使用稠密光流方法
        cv::Mat flow;
        // 计算稠密光流
        cv::calcOpticalFlowFarneback(
            prevGray, currGray, 
            flow, 
            0.5,                // 金字塔缩放因子
            3,                  // 金字塔层数
            4,                 // 窗口大小
            3,                  // 迭代次数
            3,                  // 多项式展开的邻域大小
            1.2,                // 高斯标准差
            0                   // 标志
        );
        
        // 创建掩码，只关注人体区域的光流
        cv::Mat mask = cv::Mat::zeros(flow.size(), CV_8UC1);
        cv::rectangle(mask, 
                     cv::Point(PrevIndiBox[0], PrevIndiBox[1]), 
                     cv::Point(PrevIndiBox[2], PrevIndiBox[3]), 
                     cv::Scalar(255), -1);
        
        // 计算人体区域内的平均光流
        unsigned count = 0;
        float sumDx = 0, sumDy = 0;
        
        // 遍历人体区域内的所有像素
        for (int y = PrevIndiBox[1]; y < PrevIndiBox[3]; y++) 
        {
            for (int x = PrevIndiBox[0]; x < PrevIndiBox[2]; x++) 
            {
                // 确保坐标在图像范围内
                if (x >= 0 && x < flow.cols && y >= 0 && y < flow.rows) {
                    const cv::Vec2f& fxy = flow.at<cv::Vec2f>(y, x);
                    sumDx += fxy[0];
                    sumDy += fxy[1];
                    count++;
                }
            }
        }
        
        // 计算平均光流
        if (count > 0) 
        {
            xOptiFlow = sumDx / count / 10;
            yOptiFlow = sumDy / count / 10;
            printf ("%u  ", count);
        } 
        else 
        {
            xOptiFlow = 0;
            yOptiFlow = 0;
        }

/*      Dude impossible to get that many points
        // 如果没有足够的追踪点，在前一帧的人体中心周围生成新的追踪点
        if (prevPoints.size() < MAX_POINTS / 2)
        {
            prevPoints.clear();
            // 在人体中心周围的区域内生成随机点
            for (int i = 0; i < MAX_POINTS; i++)
            {
                // 生成随机偏移，范围为[-FLOW_RADIUS, FLOW_RADIUS]
                int offsetX = (rand() % (2 * FLOW_RADIUS)) - FLOW_RADIUS;
                int offsetY = (rand() % (2 * FLOW_RADIUS)) - FLOW_RADIUS;
                
                // 计算点的坐标
                int x = xPrevCenter + offsetX;
                int y = yPrevCenter + offsetY;
                
                // 确保点在图像范围内
                if (x >= 0 && x < PrevFrame.cols && y >= 0 && y < PrevFrame.rows)
                {
                    prevPoints.push_back(cv::Point2f(x, y));
                }
            }
        }
*/        
/*
            // 如果有有效的光流点，更新动量和预测的中心位置
            {
                avgDx /= validCount;
                avgDy /= validCount;
                
                // 更新动量 (简单的移动平均)
                momentum[0] = 0.7 * momentum[0] + 0.3 * avgDx;
                momentum[1] = 0.7 * momentum[1] + 0.3 * avgDy;
                
                // 如果没有检测到人体，可以使用光流预测的位置
                if (boxes.empty())
                {
                    // 预测新的中心位置
                    int predictedX = xPrevCenter + momentum[0];
                    int predictedY = yPrevCenter + momentum[1];
                    
                    // 创建一个基于预测位置的检测框
                    cv::Vec4i predictedBox;
                    int boxWidth = PrevBox[2] - PrevBox[0];
                    int boxHeight = PrevBox[3] - PrevBox[1];
                    
                    predictedBox[0] = predictedX - boxWidth / 2;
                    predictedBox[1] = predictedY - boxHeight / 2;
                    predictedBox[2] = predictedBox[0] + boxWidth;
                    predictedBox[3] = predictedBox[1] + boxHeight;
                    
                    // 确保框在图像范围内
                    predictedBox[0] = std::max(0, predictedBox[0]);
                    predictedBox[1] = std::max(0, predictedBox[1]);
                    predictedBox[2] = std::min(image.cols, predictedBox[2]);
                    predictedBox[3] = std::min(image.rows, predictedBox[3]);
                    
                    // 添加预测的框
                    boxes.push_back(predictedBox);
                    
                    // 可视化预测框（调试用）
                    cv::Mat debugImg = image.clone();
                    cv::rectangle(debugImg, 
                        cv::Point(predictedBox[0], predictedBox[1]), 
                        cv::Point(predictedBox[2], predictedBox[3]), 
                        cv::Scalar(0, 0, 255), 2);  // 红色表示预测框
                    
                    cv::imshow("Predicted Box", debugImg);
                    cv::waitKey(20);
                }
            }
*/

    }
    if (boxes.size() > 1)
    {
        /// @todo Process tracking of boxex here

    }
    else if (boxes.empty()) 
    {
        /// @todo process momentumn-opti flow box generation here
        std::cout << "未检测到人体" << std::endl;
        ret = 1;
        return ret;
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
        
        // Draw indicationBox
        cv::rectangle(dect_img, 
            cv::Point(indicationBox[0], indicationBox[1]), 
            cv::Point(indicationBox[2], indicationBox[3]), 
            cv::Scalar(16, 255, 16), 2);  // Green
        
        // 绘制光流向量 - 从前一帧中心点到预测位置的黄色线段
        if (flagFirstFrame == false) 
        {
            // 计算光流向量终点
            cv::Point startPoint(xPrevCenter, yPrevCenter);
            // 放大光流向量以便更好地可视化（乘以10倍）
            cv::Point endPoint(xPrevCenter + xOptiFlow * 10, yPrevCenter + yOptiFlow * 10);
            
            // 绘制黄色线段表示光流方向
            cv::line(dect_img, startPoint, endPoint, cv::Scalar(16, 233, 233), 2);  // Yellow
            
            // Draw center of previous detection
            cv::circle(dect_img, startPoint, 3, cv::Scalar(16, 16, 233), -1);  // Red 
        }

        // Draw center current detection
        cv::circle(dect_img, cv::Point (xCenter, yCenter), 3, cv::Scalar(233, 16, 16), -1);  // Blue


        // 显示结果图像 - 调整到800像素高
        cv::Mat resized_img;
        float scale = 800.0f / dect_img.rows;
        cv::resize(dect_img, resized_img, cv::Size(), scale, scale, cv::INTER_LINEAR);
        cv::imshow("Pose Estimation", resized_img);
        cv::waitKey(20);
    }
    // 为上面的输出换行
    putchar('\n');

    this->flagFirstFrame = false;
    this->PrevFrame      = image;
    this->xPrevCenter    = xCenter;
    this->yPrevCenter    = yCenter;
    this->PrevIndiBox    = indicationBox;
    //this->PrevBox        = theTrackedBox;

    return ret;
}