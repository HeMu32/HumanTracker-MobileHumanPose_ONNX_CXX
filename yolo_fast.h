#pragma once

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class yolo_fast
{
public:
	yolo_fast(std::string modelpath, float objThreshold, float confThreshold, float nmsThreshold);

	/// @brief 				Detect via yolo
	/// @param frame 		Picture
	/// @param boxesResult 	Detected boxes, xyxy
	/// @param filter 		Coco classification tag, 0 for human; or -1 for no filter.
	void detect (cv::Mat &frame, std::vector<cv::Vec4i> &boxesResult, int filter);
	void detect (cv::Mat &frame, std::vector<cv::Vec4i> &boxesResult);

private:
	const float anchors[2][6] 	= { {12.64,19.39, 37.88,51.48, 55.71,138.31}, {126.91,78.23, 131.57,214.55, 279.92,258.87} };
	const float stride[3] 		= { 16.0, 32.0 };
	const int inpWidth 	 = 352;
	const int inpHeight  = 352;
	const int num_stage  = 2;
	const int anchor_num = 3;
	float objThreshold;
	float confThreshold;
	float nmsThreshold;
	std::vector<std::string> classes;
	const std::string classesFile = "coco.names";
	int num_class;
	cv::dnn::Net net;
	void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);
};