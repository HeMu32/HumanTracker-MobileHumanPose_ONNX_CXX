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
    
    {   // 唤醒YOLO线程以便退出
        std::lock_guard<std::mutex> lock(mtxYolo);
        detection_done = true;
    }
    condVarYolo.notify_one();
    
    {   // 唤醒光流线程以便退出
        std::lock_guard<std::mutex> lock(mtxOptiFlow);
        optiflow_done = true;
    }
    condVarOptiFlow.notify_one();
    
    // 等待线程结束
    if (yolo_thread && yolo_thread->joinable())
    {
        yolo_thread->join();
        delete yolo_thread;
    }
    
    if (optiflow_thread && optiflow_thread->joinable())
    {
        optiflow_thread->join();
        delete optiflow_thread;
    }
}

void HumanTracker::initThreads()
{
    if (!thread_running)
    {
        thread_running = true;
        yolo_thread = new std::thread(&HumanTracker::yoloDetectionThread, this);
        optiflow_thread = new std::thread(&HumanTracker::optiFlowThread, this);
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

std::pair<int, int> HumanTracker::calculateOpticalFlow(const cv::Mat& prevGray, const cv::Mat& currGray, 
                                                      const cv::Vec4i& box, const cv::Mat& visualImage)
{
    // 每次都在前一帧的人体中心周围生成新的追踪点
    std::vector<cv::Point2f> prevTrackPoints;
    std::vector<cv::Point2f> nextTrackPoints;
    std::vector<uchar> status;
    std::vector<float> err;
    
    // 添加更多的点以增加追踪精度（在框内均匀分布）
    int numPointsX = 5; // 水平方向的点数
    int numPointsY = 12; // 垂直方向的点数
    
    for (int y = 0; y < numPointsY; y++) 
    {
        for (int x = 0; x < numPointsX; x++) 
        {
            float px = box[0] + (box[2] - box[0]) * x / (numPointsX - 1);
            float py = box[1] + (box[3] - box[1]) * y / (numPointsY - 1);
            prevTrackPoints.push_back(cv::Point2f(px, py));
        }
    }
    
    // 如果有追踪点，计算光流
    if (prevTrackPoints.empty())
    {
        return {0, 0}; // 没有追踪点，返回零位移
    }
    
    // 计算窗口大小，确保大于2x2
    int winWidth = std::max(15, (box[2] - box[0]) / 8);
    int winHeight = std::max(15, (box[3] - box[1]) / 8);
    
    // 使用Lucas-Kanade方法计算光流
    cv::calcOpticalFlowPyrLK(
        prevGray, 
        currGray,
        prevTrackPoints, 
        nextTrackPoints,
        status, 
        err,
        cv::Size(winWidth, winHeight), 
        4,  // 增加金字塔层数以处理更大的运动
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.001),  // 提高精度
        cv::OPTFLOW_LK_GET_MIN_EIGENVALS
    );
    
    return processOpticalFlowResults(prevTrackPoints, nextTrackPoints, status, visualImage);
}

std::pair<int, int> HumanTracker::processOpticalFlowResults(
    const std::vector<cv::Point2f>& prevPoints, 
    const std::vector<cv::Point2f>& nextPoints,
    const std::vector<uchar>& status,
    const cv::Mat& visualImage)
{   // Filter out excessive value then calculate average
    int validCount = 0;
    float avgDx = 0, avgDy = 0;
    std::vector<float> dxValues, dyValues;
    
    // Obtain all tracked movements
    for (size_t i = 0; i < status.size(); i++)
    {
        if (status[i])
        {
            float dx = nextPoints[i].x - prevPoints[i].x;
            float dy = nextPoints[i].y - prevPoints[i].y;
            dxValues.push_back(dx);
            dyValues.push_back(dy);
        }
    }
    
    // Want to filter out excessive values, by median and std. dev.
    float medianDx = 0, medianDy = 0;
    float stdDevDx = 0, stdDevDy = 0;
    
    if (!dxValues.empty())
    {
        // Calculate median
        size_t n = dxValues.size() / 2;
        std::nth_element(dxValues.begin(), dxValues.begin() + n, dxValues.end());
        medianDx = dxValues[n];
        
        std::nth_element(dyValues.begin(), dyValues.begin() + n, dyValues.end());
        medianDy = dyValues[n];
        
        // Calculate std. dev.
        for (float dx : dxValues) stdDevDx += (dx - medianDx) * (dx - medianDx);
        for (float dy : dyValues) stdDevDy += (dy - medianDy) * (dy - medianDy);
        
        stdDevDx = std::sqrt(stdDevDx / dxValues.size());
        stdDevDy = std::sqrt(stdDevDy / dyValues.size());
    }
#ifdef _DEBUG_OPTIFLOW
    // Picture for visualization
    cv::Mat flowVis;
    if (!visualImage.empty())
    {
        flowVis = visualImage.clone();
    }
#endif
    
    // Filter out excessive values and obtain average
    std::vector<uchar> filteredStatus = status;
    for (size_t i = 0; i < status.size(); i++)
    {
        if (status[i])
        {
            float dx = nextPoints[i].x - prevPoints[i].x;
            float dy = nextPoints[i].y - prevPoints[i].y;
            
            // Filter out points exceeds 2 std. div.
            if (std::abs(dx - medianDx) <= 2 * stdDevDx && 
                std::abs(dy - medianDy) <= 2 * stdDevDy)
            {
                avgDx += dx;
                avgDy += dy;
                validCount++;
            }
            else
            {
                // Mark as bad prediction manually
                filteredStatus[i] = 0;
            }
        }
    }

    int xFlow = 0, yFlow = 0;
    if (validCount > 0) 
    {
        xFlow = avgDx / validCount;
        yFlow = avgDy / validCount;
    }
#ifdef _DEBUG_OPTIFLOW
    // 可视化光流结果
    if (!visualImage.empty())
    {
        // 绘制所有追踪点
        for (size_t i = 0; i < prevPoints.size(); i++)
        {
            if (filteredStatus[i])
            {
                // 有效点用绿色线表示
                cv::line(flowVis, prevPoints[i], nextPoints[i], 
                         cv::Scalar(0, 255, 0), 1);
                cv::circle(flowVis, nextPoints[i], 2, 
                           cv::Scalar(0, 255, 0), -1);
            }
            else if (i < status.size())
            {
                // 无效点用红色点表示
                cv::circle(flowVis, prevPoints[i], 2, 
                           cv::Scalar(0, 0, 255), -1);
            }
        }
        
        // 显示光流追踪结果
        cv::resize(flowVis, flowVis, cv::Size(), 0.5, 0.5);
        cv::imshow("Optical Flow Tracking", flowVis);
        cv::waitKey(1);
    }
#endif
    return {xFlow, yFlow};
}

void HumanTracker::optiFlowThread()
{
    while (thread_running)
    {
        std::unique_lock<std::mutex> lock(mtxOptiFlow);
        condVarOptiFlow.wait(lock, [this]{return !thread_prevFrame.empty() || !thread_running;});
        
        // Exit loop if a termination is commanded
        if (!thread_running)
        {
            break;
        }
        
        // Optical flow estimation
        if (optiflow_done == false && !thread_prevFrame.empty())
        {
            cv::Mat prevGray, currGray;
            
            // Convert to gray
            cv::cvtColor(thread_prevFrame, prevGray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(thread_image, currGray, cv::COLOR_BGR2GRAY);
            
            // Preform estimation
            std::tie(thread_xOptiFlow, thread_yOptiFlow) = 
                calculateOpticalFlow(prevGray, currGray, thread_prevBox, thread_image);
        }
        
        optiflow_done = true;
        lock.unlock();
        condVarOptiFlow.notify_one();
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
    cv::Vec4i TrackedBox;           // Bound box for the person be tracked, elected from @var boxes 
    cv::Vec4i PredectedBox;         // Predicted bound box, calculated by PrevBox + MoVec
    cv::Vec4i indicationBox;        // For visualization and optical flow calc., not the bound box by yolo.
    int xOptiFlow   = 0;            // x value of optical flow vector
    int yOptiFlow   = 0;            // y value of optical flow vector
    int xCenter     = 0;            // Weighted center of detected person
    int yCenter     = 0;            // Weighted center of detected person
    int xMoVec      = 0;            // Weighted motion vector by combining momentum and optical flow est.
    int yMoVec      = 0;            // Weighted motion vector by combining momentum and optical flow est.
    
    // 添加计时功能
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Start yolo detection for this frame
    {
        std::lock_guard<std::mutex> lock(mtxYolo);
        thread_image = image.clone();
        thread_boxes.clear();
        detection_done = false;
    }
    condVarYolo.notify_one();
    
    // If not the first frame detecting pepole, estimate optical flow 
    if (flagFirstFrame == false)
    {
        /// @todo   add a method to limit the maxium output of optical flow
        ///         under the assertion that a person chould not flash
        std::lock_guard<std::mutex> lock(mtxOptiFlow);
        thread_prevFrame = PrevFrame.clone();
        thread_prevBox = PrevIndiBox;
        optiflow_done = false;
    }
    condVarOptiFlow.notify_one();
    
    // Wait for yolo detection to finish
    {
        std::unique_lock<std::mutex> lock(mtxYolo);
        condVarYolo.wait(lock, [this]{return detection_done;});
        boxes = thread_boxes;
    }
    
    auto dec_time = std::chrono::high_resolution_clock::now();
    auto durationDec = std::chrono::duration_cast<std::chrono::milliseconds>(dec_time - start_time);
    printf("找人时间: %ld毫秒  ", durationDec.count());

    // Wait for optical flow estimation to finish
    if (flagFirstFrame == false)
    {
        std::unique_lock<std::mutex> lock(mtxOptiFlow);
        condVarOptiFlow.wait(lock, [this]{return optiflow_done;});
        xOptiFlow = thread_xOptiFlow;
        yOptiFlow = thread_yOptiFlow;
    }

    // Obtain the weighted movement vector
    xMoVec = momentum[0] * 0.5 + xOptiFlow * 0.5;
    yMoVec = momentum[1] * 0.5 + yOptiFlow * 0.5;

    // Calculate bound box prediction
    PredectedBox[0] = PrevBox[0] + xMoVec;
    PredectedBox[1] = PrevBox[1] + yMoVec;
    PredectedBox[2] = PrevBox[2] + xMoVec;
    PredectedBox[3] = PrevBox[3] + yMoVec;

    /// @todo May introduce a logic in case track lost
    /// @todo Temporal correlation to shallow for now, may keep deeper temporal box info
    if (boxes.size() >= 1)
    {   // Process tracking of boxes here
        /// @todo   Obtain a initial value for data of previous frame,  
        ///         for selection of the person to track.
        /// @todo   Obtain another method that detects person only and returns boxes, 
        ///         in convience of user selectig the person to track in the front end

        // Score is the quare distance of the two diagonal points of detected boxes 
        // to the predicted box generated by the weighted movement vector 
        std::list<float> scoreList;

        for (size_t i = 0; i < boxes.size(); i++) 
        {   // Calculate score
            float fScore = 0;
            for (size_t j = 0; j < 4; j++)
                fScore += powf(PredectedBox[j] - boxes[i][j], 2);
            
            scoreList.push_back(fScore);
        }

        // Find the closest one
        size_t minDistIndex = 0;
        float minDist = std::numeric_limits<float>::max();
        size_t idx = 0;
        
        for (const float& dist : scoreList) 
        {
            if (dist < minDist) 
            {
                minDist = dist;
                minDistIndex = idx;
            }
            idx++;
        }
        
        // Obtain the tracked box
        TrackedBox = boxes[minDistIndex];
    }
    else if (boxes.empty()) 
    {
#ifdef _DEBUG
        std::cout << "未检测到人体  ";
#endif
        ret = 1;

        // In case Yolo fails: just use the prediction
        TrackedBox = PredectedBox;
        xCenter = xPrevCenter + xMoVec;
        yCenter = yPrevCenter + yMoVec;
    }
    
    start_time = std::chrono::high_resolution_clock::now();
    
    // Estimate pose for the tracked box
    // 估计姿态
    cv::Mat pose_2d_fast, joint_scores_fast;
    std::tie(pose_2d_fast, joint_scores_fast) = 
        pose_estimator.estimatePose2d(image, TrackedBox);
    
    // 计算并输出执行时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    printf("骨骼时间: %ld毫秒  ", duration.count());

    // Calculate detection indication box (h: head to pelvis w: 1/4h)
    // and center (avg of the head, spine and pelvis)
    // For the output of interfrence result of cv::dnn, spine is indexed 9.
    int xSpine  = static_cast<int>(pose_2d_fast.at<float>(9, 0))  + TrackedBox[0];
    int ySpine  = static_cast<int>(pose_2d_fast.at<float>(9, 1))  + TrackedBox[1];
    // Pelvis is indexed 11
    int xPelvis = static_cast<int>(pose_2d_fast.at<float>(11, 0)) + TrackedBox[0];
    int yPelvis = static_cast<int>(pose_2d_fast.at<float>(11, 1)) + TrackedBox[1];
    // Have the position of head as avg of joint 19 and 20.
    int xHead   = static_cast<int>(pose_2d_fast.at<float>(19, 0)) + TrackedBox[0];
    int yHead   = static_cast<int>(pose_2d_fast.at<float>(19, 1)) + TrackedBox[1];
    xHead /= 2.0f;     
    yHead /= 2.0f;
    xHead += (static_cast<int>(pose_2d_fast.at<float>(20, 0)) + TrackedBox[0]) / 2.0f;
    yHead += (static_cast<int>(pose_2d_fast.at<float>(20, 1)) + TrackedBox[1]) / 2.0f;

    xCenter = (xHead + xSpine + xPelvis) / 3.0f;    // Weighted center of detected person
    yCenter = (yHead + ySpine + yPelvis) / 3.0f;    // Weighted center of detected person

    indicationBox[0] = xHead < xPelvis ? (xHead - (yPelvis - yHead) / 8) : (xPelvis - (yPelvis - yHead) / 10);
    indicationBox[1] = yHead;
    indicationBox[2] = xHead < xPelvis ? (xPelvis + (yPelvis - yHead) / 8) : (xHead + (yPelvis - yHead) / 10);
    indicationBox[3] = yPelvis;
#ifdef _DEBUG_VISUALIZATOIN
    // Visualization
    cv::Mat dect_img = image.clone();
    
    // Draw tracked box
    cv::rectangle(dect_img, 
        cv::Point(TrackedBox[0], TrackedBox[1]), 
        cv::Point(TrackedBox[2], TrackedBox[3]), 
        cv::Scalar(233, 16, 16), 2);  // Blue
        
    // Draw indicationBox
    cv::rectangle(dect_img, 
        cv::Point(indicationBox[0], indicationBox[1]), 
        cv::Point(indicationBox[2], indicationBox[3]), 
        cv::Scalar(16, 233, 16), 2);  // Green
    
    // 绘制光流向量 - 从前一帧中心点到预测位置的黄色线段
    if (flagFirstFrame == false) 
    {
        // 计算光流向量终点
        cv::Point startPoint(xPrevCenter, yPrevCenter);
        cv::Point endPoint(xPrevCenter + xOptiFlow, yPrevCenter + yOptiFlow);
        
        // 绘制黄色线段表示光流方向
        cv::line(dect_img, startPoint, endPoint, cv::Scalar(16, 233, 233), 2);  // Yellow
        
        // Draw center of previous detection
        cv::circle(dect_img, startPoint, 3, cv::Scalar(16, 16, 233), -1);  // Red 
    }

    // Draw the center of current detection
    cv::circle(dect_img, cv::Point(xCenter, yCenter), 3, cv::Scalar(233, 16, 16), -1);  // Blue

    // 显示结果图像 - 调整到400像素高
    cv::Mat resized_img;
    float scale = 400.0f / dect_img.rows;
    cv::resize(dect_img, resized_img, cv::Size(), scale, scale, cv::INTER_LINEAR);
    cv::imshow("Pose Estimation", resized_img);
    cv::waitKey(20);
#endif
    // 为上面的输出换行
    putchar('\n');

    this->flagFirstFrame = false;
    this->PrevFrame      = image;
    this->PrevBox        = TrackedBox;
    this->PrevIndiBox    = indicationBox;
    this->xPrevCenter    = xCenter;
    this->yPrevCenter    = yCenter;
    this->momentum[0]    = 0.8 * (xCenter - xPrevCenter) + 0.2 * momentum[0];
    this->momentum[1]    = 0.8 * (yCenter - yPrevCenter) + 0.2 * momentum[1];
    //this->PrevBox        = theTrackedBox;

    return ret;
}
