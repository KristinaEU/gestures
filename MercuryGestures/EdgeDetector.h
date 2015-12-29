#pragma once
#include "MercuryCore.h"

class EdgeDetector {
public:
	cv::Mat detectedEdges;
	cv::Mat blur;

	EdgeDetector();
	~EdgeDetector();

	void detect(cv::Mat frame);
	void draw();
	cv::Mat getEdges();
};