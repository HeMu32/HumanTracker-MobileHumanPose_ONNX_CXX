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
    cv::resize(rgb_img, resized_img, cv::Size(input_width, input_height));
    
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

        std::cout << outputs[0] << " ";
    
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
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                // 使用ptr方法访问4维数据
                float value = *((float*)output.ptr(0, c, h) + w);
                heatmaps.at<float>(c, h * width + w) = value;
            }
        }
    }
    
    // 应用softmax
    for (int i = 0; i < joint_num; i++) {
        cv::Mat row = heatmaps.row(i);
        double sum = 0;
        for (int j = 0; j < row.cols; j++) {
            row.at<float>(j) = std::exp(row.at<float>(j));
            sum += row.at<float>(j);
        }
        for (int j = 0; j < row.cols; j++) {
            row.at<float>(j) /= sum;
        }
    }
    
    // 计算最大值作为分数
    cv::Mat scores(joint_num, 1, CV_32F);
    for (int i = 0; i < joint_num; i++) {
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(heatmaps.row(i), &minVal, &maxVal, &minLoc, &maxLoc);
        scores.at<float>(i) = maxVal;
    }
    
    // 重塑热图为3D体积
    std::vector<cv::Mat> heatmap_volumes;
    for (int i = 0; i < joint_num; i++) {
        cv::Mat joint_heatmap(output_depth, output_height * output_width, CV_32F);
        for (int d = 0; d < output_depth; d++) {
            cv::Mat row = heatmaps.row(i * output_depth + d);
            row.copyTo(joint_heatmap.row(d));
        }
        heatmap_volumes.push_back(joint_heatmap);
    }
    
    // 计算坐标
    cv::Mat accu_x(joint_num, 1, CV_32F);
    cv::Mat accu_y(joint_num, 1, CV_32F);
    cv::Mat accu_z(joint_num, 1, CV_32F);
    
    for (int i = 0; i < joint_num; i++) {
        // 计算x坐标
        float x_sum = 0;
        for (int w = 0; w < output_width; w++) {
            float col_sum = 0;
            for (int d = 0; d < output_depth; d++) {
                for (int h = 0; h < output_height; h++) {
                    col_sum += heatmap_volumes[i].at<float>(d, h * output_width + w);
                }
            }
            x_sum += col_sum * w;
        }
        accu_x.at<float>(i) = x_sum / output_width;
        
        // 计算y坐标
        float y_sum = 0;
        for (int h = 0; h < output_height; h++) {
            float row_sum = 0;
            for (int d = 0; d < output_depth; d++) {
                for (int w = 0; w < output_width; w++) {
                    row_sum += heatmap_volumes[i].at<float>(d, h * output_width + w);
                }
            }
            y_sum += row_sum * h;
        }
        accu_y.at<float>(i) = y_sum / output_height;
        
        // 计算z坐标
        float z_sum = 0;
        for (int d = 0; d < output_depth; d++) {
            float depth_sum = 0;
            for (int h = 0; h < output_height; h++) {
                for (int w = 0; w < output_width; w++) {
                    depth_sum += heatmap_volumes[i].at<float>(d, h * output_width + w);
                }
            }
            z_sum += depth_sum * d;
        }
        accu_z.at<float>(i) = z_sum / output_depth * 2 - 1;
    }
    
    // 创建2D和3D姿态矩阵
    cv::Mat pose_2d(joint_num, 2, CV_32F);
    for (int i = 0; i < joint_num; i++) {
        pose_2d.at<float>(i, 0) = accu_x.at<float>(i) * img_width + bbox[0];
        pose_2d.at<float>(i, 1) = accu_y.at<float>(i) * img_height + bbox[1];
    }
    
    cv::Mat joint_depth(joint_num, 1, CV_32F);
    for (int i = 0; i < joint_num; i++) {
        joint_depth.at<float>(i) = accu_z.at<float>(i) * 1000 + abs_depth;
    }
    
    cv::Mat pose_3d = utils.pixel2cam(pose_2d, joint_depth, focal_length, principal_points);
    
    // 计算关节热图
    cv::Mat person_heatmap(output_height, output_width, CV_32F, cv::Scalar(0));
    for (int i = 0; i < joint_num; i++) {
        for (int h = 0; h < output_height; h++) {
            for (int w = 0; w < output_width; w++) {
                float sum = 0;
                for (int d = 0; d < output_depth; d++) {
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
    output_depth    = 64 / joint_num;  // 假设深度为64/joint_num
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