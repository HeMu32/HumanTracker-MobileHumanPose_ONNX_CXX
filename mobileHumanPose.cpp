#include "mobileHumanPose.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// 假设YoloV5s类已经实现
// #include "yoloV5s.h"

MobileHumanPose::MobileHumanPose(const std::string& model_path, 
                               const cv::Vec2f& focal_length, 
                               const cv::Vec2f& principal_points)
    : focal_length(focal_length), principal_points(principal_points)
{
    initializeModel(model_path);
}

std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> MobileHumanPose::operator()(const cv::Mat& image, const cv::Vec4i& bbox, float abs_depth)
{
    return estimatePose(image, bbox, abs_depth);
}

void MobileHumanPose::initializeModel(const std::string& model_path)
{
    try {
        // 使用OpenCV的DNN模块加载ONNX模型
        net = cv::dnn::readNet(model_path);
        
        if (net.empty()) {
            std::cerr << "无法加载模型: " << model_path << std::endl;
            throw std::runtime_error("模型加载失败");
        }
        
        // 获取模型信息
        getModelInputDetails();
        getModelOutputDetails();
        
        std::cout << "模型加载成功: " << model_path << std::endl;
    }
    catch (const cv::Exception& e) {
        std::cerr << "加载模型时出错: " << e.what() << std::endl;
        throw;
    }
}

std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> MobileHumanPose::estimatePose(const cv::Mat& image, const cv::Vec4i& bbox, float abs_depth)
{
    cv::Mat input_tensor = prepareInput(image, bbox);
    cv::Mat output = inference(input_tensor);
    return processOutput(output, abs_depth, bbox);
}

// 添加2D姿态估计函数 - 更高效的版本，不计算3D信息
std::tuple<cv::Mat, cv::Mat> MobileHumanPose::estimatePose2d(const cv::Mat& image, const cv::Vec4i& bbox)
{
    cv::Mat input_tensor = prepareInput(image, bbox);
    cv::Mat output = inference(input_tensor);
    return processOutput2d(output, bbox);
}

cv::Mat MobileHumanPose::prepareInput(const cv::Mat& image, const cv::Vec4i& bbox)
{
    // 检查边界框是否有效
    int x1 = bbox[0], y1 = bbox[1], x2 = bbox[2], y2 = bbox[3];
    if (x1 >= x2 || y1 >= y2 || x1 < 0 || y1 < 0 || x2 > image.cols || y2 > image.rows) {
        std::cout << "警告：无效的边界框 [" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]，将进行调整" << std::endl;
        x1 = std::max(0, x1);
        y1 = std::max(0, y1);
        x2 = std::min(image.cols, x2);
        y2 = std::min(image.rows, y2);
        if (x1 >= x2 || y1 >= y2) {
            // 如果边界框仍然无效，使用整个图像
            x1 = 0;
            y1 = 0;
            x2 = image.cols;
            y2 = image.rows;
        }
    }
    
    cv::Vec4i adjusted_bbox(x1, y1, x2, y2);
    cv::Mat img = utils.cropImage(image, adjusted_bbox);
    
    // 检查裁剪后的图像是否为空
    if (img.empty() || img.rows == 0 || img.cols == 0) {
        std::cout << "警告：裁剪后的图像为空，使用原始图像" << std::endl;
        img = image.clone();
    }
    
    // 转换颜色空间
    cv::Mat rgb_img;
    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    
    img_height = rgb_img.rows;
    img_width = rgb_img.cols;
    img_channels = rgb_img.channels();
    principal_points = cv::Vec2f(img_width/2.0f, img_height/2.0f);
    
    // 调整图像大小
    cv::Mat resized_img;
    cv::resize(rgb_img, resized_img, cv::Size(input_width, input_height), 1, 2);
/*
    cv::imshow ("woc", resized_img);
    cv::waitKey(0);
*/    
/*
    // 创建blob
    cv::Mat blob = cv::dnn::blobFromImage(resized_img, 1.0, 
                                         cv::Size(input_width, input_height),
                                         cv::Scalar(0, 0, 0), true, false);
*/
    // 创建blob - 尝试不同的预处理参数
    cv::Mat blob = cv::dnn::blobFromImage(resized_img, 1.0/255.0, 
                                         cv::Size(256, 256),
                                         cv::Scalar(0.5, 0.5, 0.5), true, false);
    
    return blob;
}

cv::Mat MobileHumanPose::inference(const cv::Mat& input_tensor)
{
    net.setInput(input_tensor);
    std::vector<cv::Mat> outputs;

    net.forward(outputs, this->net.getUnconnectedOutLayersNames());

    return outputs[0];
}

std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> MobileHumanPose::processOutput(const cv::Mat& output, float abs_depth, const cv::Vec4i& bbox)
{
    // 重塑输出为热图
    int batch_size  = output.size[0];
    int channels    = output.size[1];
    int height      = output.size[2];
    int width       = output.size[3];
    
    // 创建热图数组
    cv::Mat heatmaps(channels, height * width, CV_32F);
        
    // 复制数据到热图数组
    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                // 使用ptr方法访问4维数据
                float value = *((float *)output.ptr(0, c, h) + w);
                heatmaps.at<float>(c, h * width + w) = value;
            }
        }
    }

    // 应用softmax
    for (int i = 0; i < joint_num; i++)
    {
        cv::Mat row = heatmaps.row(i);
        double sum = 0;
        for (int j = 0; j < row.cols; j++)
        {
            row.at<float>(j) = std::exp(row.at<float>(j));
            sum += row.at<float>(j);
        }
        for (int j = 0; j < row.cols; j++)
        {
            row.at<float>(j) /= sum;
        }
    }

    // 检查softmax处理后的概率和是否为1
    for (int i = 0; i < joint_num; i++)
    {
        cv::Mat row = heatmaps.row(i);
        double prob_sum = 0;
        for (int j = 0; j < row.cols; j++)
        {
            prob_sum += row.at<float>(j);
        }
        // 由于浮点数精度问题，使用接近1的阈值
        if (std::abs(prob_sum - 1.0) > 1e-5) {
            std::cout << "警告: 关节 " << i << " 的概率和为 " << prob_sum 
                      << "，与预期的1.0有偏差" << std::endl;
        }
#ifdef _DEBUG
        else {
            std::cout << "关节 " << i << " 的概率和正确: " << prob_sum << std::endl;
        }
#endif
    }

    // 计算最大值作为分数
    cv::Mat scores(joint_num, 1, CV_32F);
    for (int i = 0; i < joint_num; i++)
    {
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(heatmaps.row(i), &minVal, &maxVal, &minLoc, &maxLoc);
        scores.at<float>(i) = maxVal;
    }

    // 重塑热图为3D体积
    std::vector<cv::Mat> heatmap_volumes;
    for (int i = 0; i < joint_num; i++)
    {
        cv::Mat joint_heatmap(output_depth, output_height * output_width, CV_32F);
        for (int d = 0; d < output_depth; d++)
        {
            cv::Mat row = heatmaps.row(i * output_depth + d);
            row.copyTo(joint_heatmap.row(d));
        }
        
        // 检查热图中是否有负值并修正
        double minVal, maxVal;
        cv::minMaxLoc(joint_heatmap, &minVal, &maxVal);
        
        if (minVal < 0) {
#ifdef _DEBUG
            std::cout << "修正前 - 关节 " << i << " 的热图中发现负值: 最小值 = " << minVal << ", 最大值 = " << maxVal << std::endl;
#endif
            // 修正负值 - 方法1：将所有负值设为0
            for (int d = 0; d < output_depth; d++) {
                for (int j = 0; j < output_height * output_width; j++) {
                    if (joint_heatmap.at<float>(d, j) < 0) {
                        joint_heatmap.at<float>(d, j) = 0;
                    }
                }
            }
            
#ifdef _DEBUG
            // 重新检查修正后的值
            cv::minMaxLoc(joint_heatmap, &minVal, &maxVal);
            std::cout << "修正后 - 关节 " << i << " 的热图值范围: 最小值 = " << minVal << ", 最大值 = " << maxVal << std::endl;
#endif
        }
        
        heatmap_volumes.push_back(joint_heatmap);
    }
/*
    // 保存热图层以供可视化
    for (int i = 0; i < std::min(joint_num, 1); i++) // 仅保存前1个关节以避免生成过多文件
    {
        for (int d = 0; d < output_depth; d++)
        {
            // 从heatmap_volumes中提取单层热图
            cv::Mat heatmap_layer(output_height, output_width, CV_32F);
            for (int h = 0; h < output_height; h++)
            {
                for (int w = 0; w < output_width; w++)
                {
                    heatmap_layer.at<float>(h, w) = heatmap_volumes[i].at<float>(d, h * output_width + w);
                }
            }
            
            // 归一化热图以便可视化
            cv::Mat normalized_heatmap;
            cv::normalize(heatmap_layer, normalized_heatmap, 0, 255, cv::NORM_MINMAX);
            normalized_heatmap.convertTo(normalized_heatmap, CV_8U);
            
            // 应用伪彩色映射以增强可视化效果
            cv::Mat colored_heatmap;
            cv::applyColorMap(normalized_heatmap, colored_heatmap, cv::COLORMAP_JET);
            
            // 创建文件名并保存
            std::string filename = "heatmap_joint_" + std::to_string(i) + "_depth_" + std::to_string(d) + ".jpg";
            cv::imwrite(filename, colored_heatmap);
        }
    }
*/

    // 将21关节的32层深度热图映射为2D热图并保存
    std::vector<cv::Mat> joint_2d_heatmaps;
    for (int i = 0; i < joint_num; i++)
    {
        // 创建2D热图 - 将深度维度压缩
        cv::Mat joint_2d_heatmap(output_height, output_width, CV_32F, cv::Scalar(0));
        
        // 对每个像素位置，累加所有深度层的值
        for (int h = 0; h < output_height; h++)
        {
            for (int w = 0; w < output_width; w++)
            {
                float pixel_sum = 0;
                for (int d = 0; d < output_depth; d++)
                {
                    pixel_sum += heatmap_volumes[i].at<float>(d, h * output_width + w);
                }
                joint_2d_heatmap.at<float>(h, w) = pixel_sum;
            }
        }
        
        // 保存2D热图以供后续处理
        joint_2d_heatmaps.push_back(joint_2d_heatmap);
/*
        // 归一化热图以便可视化
        cv::Mat normalized_heatmap;
        cv::normalize(joint_2d_heatmap, normalized_heatmap, 0, 255, cv::NORM_MINMAX);
        normalized_heatmap.convertTo(normalized_heatmap, CV_8U);
        
        // 应用伪彩色映射以增强可视化效果
        cv::Mat colored_heatmap;
        cv::applyColorMap(normalized_heatmap, colored_heatmap, cv::COLORMAP_JET);
    
        // 创建文件名并保存
        std::string filename = "joint_" + std::to_string(i) + "_2d_heatmap.jpg";
        cv::imwrite(filename, colored_heatmap);
*/    
    }

    // 从2D热图估算关节坐标
    cv::Mat heatmap_coords(joint_num, 2, CV_32F);
    for (int i = 0; i < joint_num; i++)
    {
        // 找到热图中的最大值位置
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(joint_2d_heatmaps[i], &minVal, &maxVal, &minLoc, &maxLoc);
        
        // 使用最大值位置作为关节坐标
        heatmap_coords.at<float>(i, 0) = maxLoc.x;
        heatmap_coords.at<float>(i, 1) = maxLoc.y;
    }
    
    // 打印从2D热图估算的关节坐标
    std::cout << "从2D热图估算的关节坐标 (x, y):" << std::endl;
    for (int i = 0; i < joint_num; i++)
    {
        std::cout << "关节 " << i << ": (" 
                  << heatmap_coords.at<float>(i, 0) << ", " 
                  << heatmap_coords.at<float>(i, 1) << ")" << std::endl;
    }

    // Calculate coordinate from heatmaps, in accumalative value
    cv::Mat accu_x(joint_num, 1, CV_32F);
    cv::Mat accu_y(joint_num, 1, CV_32F);
    cv::Mat accu_z(joint_num, 1, CV_32F);

    for (int i = 0; i < joint_num; i++)
    {   // Traverse thru all joints
    
        // 计算x坐标
        float x_sum = 0;
        for (int w = 0; w < output_width; w++)
        {
            float col_sum = 0;
            for (int d = 0; d < output_depth; d++)
            {
                for (int h = 0; h < output_height; h++)
                {
                    col_sum += heatmap_volumes[i].at<float>(d, h * output_width + w);

                    if (col_sum < 0) 
                    {
#ifdef _DEBUG
                        std::cout << "发现负值! 关节:" << i << " 深度:" << d << " 高度:" << h << " 宽度:" << w 
                                  << " 值:" << heatmap_volumes[i].at<float>(d, h * output_width + w) 
                                  << " 累积和:" << col_sum << std::endl;
                        // 检查热图中的值
                        double minVal, maxVal;
                        cv::minMaxLoc(heatmap_volumes[i], &minVal, &maxVal);
                        std::cout << "热图体积 " << i << " 的最小值: " << minVal << " 最大值: " << maxVal << std::endl;
#endif
                        
                        // 修正负值而不是退出
                        col_sum = 0;
                    }
                }
            }
            x_sum += col_sum * w;
        }
        accu_x.at<float>(i) = x_sum / output_width;

        // 计算y坐标
        float y_sum = 0;
        for (int h = 0; h < output_height; h++)
        {
            float row_sum = 0;
            for (int d = 0; d < output_depth; d++)
            {
                for (int w = 0; w < output_width; w++)
                {
                    row_sum += heatmap_volumes[i].at<float>(d, h * output_width + w);
                }
            }
            y_sum += row_sum * h;
        }
        accu_y.at<float>(i) = y_sum / output_height;

        // 计算z坐标
        float z_sum = 0;
        for (int d = 0; d < output_depth; d++)
        {
            float depth_sum = 0;
            for (int h = 0; h < output_height; h++)
            {
                for (int w = 0; w < output_width; w++)
                {
                    depth_sum += heatmap_volumes[i].at<float>(d, h * output_width + w);
                }
            }
            z_sum += depth_sum * d;
        }
        accu_z.at<float>(i) = z_sum / output_depth * 2 - 1;
    }

    // 打印每个关节的accu_x, accu_y, accu_z到控制台
    std::cout << "关节累积坐标 (accu_x, accu_y, accu_z):" << std::endl;
    for (int i = 0; i < joint_num; i++)
    {
        std::cout << "关节 " << i << ": (" 
                  << accu_x.at<float>(i) << ", " 
                  << accu_y.at<float>(i) << ", " 
                  << accu_z.at<float>(i) << ")" << std::endl;
    }

    // 创建2D和3D姿态矩阵
    cv::Mat pose_2d(joint_num, 2, CV_32F);
    float xCenter = bbox[0];
    float yCenter = bbox[1];
    for (int i = 0; i < joint_num; i++)
    {
        pose_2d.at<float>(i, 0) = accu_x.at<float>(i) + xCenter;
        pose_2d.at<float>(i, 1) = accu_y.at<float>(i) + yCenter;
    }

    // 打印2D关节位置到控制台
    std::cout << "2D关节位置:" << std::endl;
    for (int i = 0; i < joint_num; i++)
    {
        std::cout << "关节 " << i << ": (" 
                  << pose_2d.at<float>(i, 0) << ", " 
                  << pose_2d.at<float>(i, 1) << ")" << std::endl;
    }

    cv::Mat joint_depth(joint_num, 1, CV_32F);
    for (int i = 0; i < joint_num; i++)
    {
        joint_depth.at<float>(i) = accu_z.at<float>(i) * 1000 + abs_depth;
    }

    cv::Mat pose_3d = utils.pixel2cam(pose_2d, joint_depth, focal_length, principal_points);

    // 计算关节热图
    cv::Mat person_heatmap(output_height, output_width, CV_32F, cv::Scalar(0));
    for (int i = 0; i < joint_num; i++)
    {
        for (int h = 0; h < output_height; h++)
        {
            for (int w = 0; w < output_width; w++)
            {
                float sum = 0;
                for (int d = 0; d < output_depth; d++)
                {
                    sum += heatmap_volumes[i].at<float>(d, h * output_width + w);
                }
                person_heatmap.at<float>(h, w) += std::sqrt(sum);
            }
        }
    }

    // 调整热图大小
    cv::Mat resized_heatmap;
    cv::resize(person_heatmap, resized_heatmap, cv::Size(img_width, img_height));

    return std::make_tuple(pose_2d, pose_3d, resized_heatmap, scores);
}

// 添加一个新的处理函数，专门用于2D姿态估计，不计算热图体积
std::tuple<cv::Mat, cv::Mat> MobileHumanPose::processOutput2d(const cv::Mat& output, const cv::Vec4i& bbox)
{
    // 重塑输出为热图
    int batch_size  = output.size[0];
    int channels    = output.size[1];
    int height      = output.size[2];
    int width       = output.size[3];
    
    // 创建热图数组 - 使用二维结构而不是一维展平结构
    std::vector<cv::Mat> heatmaps;
    for (int i = 0; i < joint_num; i++)
    {
        // 为每个关节创建一个二维热图矩阵
        cv::Mat joint_heatmap(height, width, CV_32F, cv::Scalar(0));
        heatmaps.push_back(joint_heatmap);
    }
    
    // 复制数据到热图数组 - 累加每个关节的所有深度层
    for (int i = 0; i < joint_num; i++)
    {
        for (int d = 0; d < output_depth; d++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    // 使用ptr方法访问4维数据 - 累加所有深度层
                    float value = *((float *)output.ptr(0, i * output_depth + d, h) + w);
                    if (value > 0)
                    {
                        // 直接访问二维矩阵中的元素
                        heatmaps[i].at<float>(h, w) += value;
                    }
                }
            }
        }
    }

    // 对每个关节热图应用Softmax归一化
    for (int i = 0; i < joint_num; i++)
    {
        // 计算指数和总和
        double sum = 0;
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                heatmaps[i].at<float>(h, w) = std::exp(heatmaps[i].at<float>(h, w));
                sum += heatmaps[i].at<float>(h, w);
            }
        }
        
        // 归一化
        if (sum > 0)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    heatmaps[i].at<float>(h, w) /= sum;
                }
            }
        }
    }

    // 创建所有关节的组合热图
    cv::Mat combined_heatmap(height, width, CV_32F, cv::Scalar(0));
    for (int i = 0; i < joint_num; i++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                // 使用新的二维热图数据结构访问方式
                combined_heatmap.at<float>(h, w) += heatmaps[i].at<float>(h, w);
            }
        }
    }
    
    // 归一化组合热图
    cv::Mat normalized_combined;
    cv::normalize(combined_heatmap, normalized_combined, 0, 255, cv::NORM_MINMAX);
    normalized_combined.convertTo(normalized_combined, CV_8U);
    
    // 应用伪彩色映射
    cv::Mat colored_combined;
    cv::applyColorMap(normalized_combined, colored_combined, cv::COLORMAP_JET);
    
    // 保存组合热图
    cv::imwrite("heatmaps_2d/combined_heatmap.jpg", colored_combined);

    // 计算最大值作为分数
    cv::Mat scores(joint_num, 1, CV_32F);
    for (int i = 0; i < joint_num; i++)
    {
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        // 直接在二维热图上查找最大值
        cv::minMaxLoc(heatmaps[i], &minVal, &maxVal, &minLoc, &maxLoc);
        scores.at<float>(i) = maxVal;
    }
    
    // 计算关节热图 - 用于可视化
    cv::Mat person_heatmap(height, width, CV_32F, cv::Scalar(0));
    for (int i = 0; i < joint_num; i++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                person_heatmap.at<float>(h, w) += heatmaps[i].at<float>(h, w);
            }
        }
    }
    // 调整热图大小
    cv::Mat resized_heatmap;
    cv::resize(person_heatmap, resized_heatmap, cv::Size(img_width, img_height));

    // 计算关节坐标 - 使用最大值法而不是加权平均法
    cv::Mat max_x(joint_num, 1, CV_32F);
    cv::Mat max_y(joint_num, 1, CV_32F);
/*
    // 对热图进行高斯模糊预处理，以减少噪声并使关节位置估计更加稳定
    // 直接在原热图上应用高斯模糊
    for (int i = 0; i < joint_num; i++)
    {
        cv::GaussianBlur(heatmaps[i], heatmaps[i], cv::Size(3, 3), 1.5);
    }
*/
    for (int i = 0; i < joint_num; i++)
    {
        // 找到热图中的最大值位置
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        
        // 直接在二维热图上查找最大值
        cv::minMaxLoc(heatmaps[i], &minVal, &maxVal, &minLoc, &maxLoc);
        
        // 使用最大值位置作为关节坐标
        max_x.at<float>(i) = static_cast<float>(maxLoc.x);
        max_y.at<float>(i) = static_cast<float>(maxLoc.y);
        
        // 如果最大值太小，可能是不可靠的检测，使用默认值
        if (maxVal < 0.1) // 可以根据需要调整阈值
        {
            std::cout << "警告: 关节 " << i << " 的最大概率值 " << maxVal << std::endl;
        }
    }

    // 打印每个关节的max_x, max_y到控制台
    std::cout << "2D关节累积坐标 (max_x, max_y):" << std::endl;
    for (int i = 0; i < joint_num; i++)
    {
        std::cout << i << ", "
                  << max_x.at<float>(i) << ", " 
                  << max_y.at<float>(i)
                  << std::endl;
    }

    // 创建2D姿态矩阵
    // 热图位置似乎是左下角对应输入图片的右上角, 不知道为什么...
    // 计算关于框的左上角, 以像素为单位的关节位置
    // 最大值点的y坐标似乎已经翻转了, 不管了
    cv::Mat pose_2d(joint_num, 2, CV_32F);
    int boxW = bbox[2] - bbox[0];
    int boxH = bbox[3] - bbox[1];
    for (int i = 0; i < joint_num; i++)
    {
        pose_2d.at<float>(i, 0) = (1 - max_x.at<float>(i) / 32) * boxW;
        pose_2d.at<float>(i, 1) = (1 - max_y.at<float>(i) / 32) * boxH;
    }

    // 打印2D关节位置到控制台
    std::cout << "2D关节位置:" << std::endl;
    for (int i = 0; i < joint_num; i++)
    {
        std::cout << i << ", " 
                  << pose_2d.at<float>(i, 0) << ", " 
                  << pose_2d.at<float>(i, 1) << std::endl;
    }


    return std::make_tuple(pose_2d, scores);
}

void MobileHumanPose::getModelInputDetails()
{
    // 获取输入层名称
    input_name = net.getLayerNames()[0];
    
    // 获取输入层形状
    // 修复：使用正确的方式获取输入形状 : 存在二次代码生成      ////////////////////////////////
    // 可能在将来考虑直接使用预设模型维度的方法
    std::vector<int> input_shape;
/*
    if (net.getLayerId(input_name) >= 0) {
        // 获取输入层的blob
        const auto& blob = net.getLayer(net.getLayerId(input_name))->blobs[0];
        // 将MatSize转换为vector<int>
        for (int i = 0; i < blob.dims; i++) {
            input_shape.push_back(blob.size[i]);
        }
    } else 
*/
    {
        // 使用默认值
        input_shape = {1, 3, 256, 256};
    }

    for (const auto &c : input_shape) std::cout << c << " ";
    
    channels        = input_shape[1];
    input_height    = input_shape[2];
    input_width     = input_shape[3];
}

void MobileHumanPose::getModelOutputDetails()
{
    // 获取输出层名称
    std::vector<std::string> out_names = net.getUnconnectedOutLayersNames();
    output_names = out_names;
    
    // 假设输出形状为 [1, joint_num*depth, height, width]
    output_depth    = this->output_depth / this->joint_num;  // 64/joint_num-1
    output_height   = this->output_height;  // 假设高度为32
    output_width    = this->output_width;   // 假设宽度为32
}

/*
// 主函数示例实现
void runPoseEstimation(const std::string& pose_model_path, 
                      const std::string& detector_model_path,
                      const std::string& input_image_path,
                      const std::string& output_image_path)
{
    // 相机参数
    cv::Vec2f focal_length(1500, 1500);
    cv::Vec2f principal_points(1280/2, 720/2);
    
    // 初始化姿态估计器
    MobileHumanPose pose_estimator(pose_model_path, focal_length, principal_points);
    
    // 初始化人体检测器
    YoloV5s person_detector(detector_model_path, 0.5, 0.4);
    
    // 读取图像
    cv::Mat image = cv::imread(input_image_path);
    if (image.empty()) {
        std::cerr << "无法读取图像: " << input_image_path << std::endl;
        return;
    }
    
    // 检测人体
    std::vector<cv::Vec4i> boxes;
    std::vector<float> scores;
    person_detector.detect(image, boxes, scores);
    
    // 如果没有检测到人体，退出
    if (boxes.empty()) {
        std::cout << "未检测到人体" << std::endl;
        return;
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
    
    // 绘制3D姿态
    std::vector<cv::Mat> vis_kps_vis(pose_3d_list.size());
    for (size_t i = 0; i < pose_3d_list.size(); i++) {
        vis_kps_vis[i] = cv::Mat::ones(pose_3d_list[i].size(), CV_32F);
    }
    cv::Mat img_3dpos = utils.vis3DMultipleSkeleton(pose_3d_list, vis_kps_vis);
    
    // 调整3D姿态图像大小
    cv::Rect roi(150, 200, img_3dpos.cols - 300, img_3dpos.rows - 400);
    cv::Mat cropped_3dpos = img_3dpos(roi);
    cv::Mat resized_3dpos;
    cv::resize(cropped_3dpos, resized_3dpos, cv::Size(image.cols, image.rows));
    
    // 合并图像
    cv::Mat combined_img;
    cv::hconcat(std::vector<cv::Mat>{heatmap_viz_img, pose_img, resized_3dpos}, combined_img);
    
    // 保存结果
    cv::imwrite(output_image_path, combined_img);
    
    // 显示结果
    cv::namedWindow("Estimated pose", cv::WINDOW_NORMAL);
    cv::imshow("Estimated pose", combined_img);
    cv::waitKey(0);
}
*/

// 主函数
/*
int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cout << "用法: " << argv[0] << " <姿态模型路径> <检测器模型路径> <输入图像路径> [输出图像路径]" << std::endl;
        return -1;
    }
    
    std::string pose_model_path = argv[1];
    std::string detector_model_path = argv[2];
    std::string input_image_path = argv[3];
    std::string output_image_path = argc > 4 ? argv[4] : "output.jpg";
    
    runPoseEstimation(pose_model_path, detector_model_path, input_image_path, output_image_path);
    
    return 0;
}*/