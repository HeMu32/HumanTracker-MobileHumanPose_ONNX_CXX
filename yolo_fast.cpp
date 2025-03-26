#include "yolo_fast.h"

yolo_fast::yolo_fast(std::string modelpath, float obj_Threshold, float conf_Threshold, float nms_Threshold)
{
	this->objThreshold = obj_Threshold;
	this->confThreshold = conf_Threshold;
	this->nmsThreshold = nms_Threshold;

	std::ifstream ifs(this->classesFile.c_str());
	std::string line;
	while (std::getline(ifs, line)) this->classes.push_back(line);
	this->num_class = this->classes.size();
	this->net = cv::dnn::readNet(modelpath);
}

void yolo_fast::drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 2);

	//Get the label for the class name and its confidence
	std::string label = cv::format("%.2f", conf);
	label = this->classes[classId] + ":" + label;

	//Display the label at the top of the bounding box
	int baseLine;
	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseLine);
	top = std::max(top, labelSize.height);
	//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 160, 0), 1, cv::LINE_AA);
}

void yolo_fast::detect(cv::Mat& frame)
{
	cv::Mat blob;
	cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(this->inpWidth, this->inpHeight));
	this->net.setInput(blob);
	std::vector<cv::Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	/////generate proposals
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
	int n = 0, q = 0, i = 0, j = 0, nout = this->anchor_num * 5 + this->classes.size(), row_ind = 0;
	float* pdata = (float*)outs[0].data;

	for (n = 0; n < this->num_stage; n++)   ///stage
	{
		int num_grid_x = (int)(this->inpWidth / this->stride[n]);
		int num_grid_y = (int)(this->inpHeight / this->stride[n]);
		for (i = 0; i < num_grid_y; i++)
		{
			for (j = 0; j < num_grid_x; j++)
			{
				cv::Mat scores = outs[0].row(row_ind).colRange(this->anchor_num * 5, outs[0].cols);
				cv::Point classIdPoint;
				double max_class_socre;
				// Get the value and location of the maximum score
				cv::minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
				for (q = 0; q < this->anchor_num; q++)    ///anchor
				{
					const float anchor_w = this->anchors[n][q * 2];
					const float anchor_h = this->anchors[n][q * 2 + 1];
					float box_score = pdata[4 * this->anchor_num + q];
					if (box_score > this->objThreshold && max_class_socre > this->confThreshold)
					{
						float cx = (pdata[4 * q] * 2.f - 0.5f + j) * this->stride[n];  ///cx
						float cy = (pdata[4 * q + 1] * 2.f - 0.5f + i) * this->stride[n];   ///cy
						float w = powf(pdata[4 * q + 2] * 2.f, 2.f) * anchor_w;   ///w
						float h = powf(pdata[4 * q + 3] * 2.f, 2.f) * anchor_h;  ///h

						int left = (cx - 0.5*w)*ratiow;
						int top = (cy - 0.5*h)*ratioh;   ///���껹ԭ��ԭͼ��

						classIds.push_back(classIdPoint.x);
						confidences.push_back(box_score * max_class_socre);
						boxes.push_back(cv::Rect(left, top, (int)(w*ratiow), (int)(h*ratioh)));
					}
				}
				row_ind++;
				pdata += nout;
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);

	
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		this->drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}