#pragma once

#include "EdgeDetector.h"

EdgeDetector::EdgeDetector() {}
EdgeDetector::~EdgeDetector() {}

void EdgeDetector::detect(cv::Mat frame) {
	// Reduce noise with a kernel 3x3
	cv::blur(frame, this->blur, cv::Size(3, 3));

	// Canny detector
	int edgeThresh = 1;
	int lowThreshold = 20;
	int const max_lowThreshold = 100;
	int ratio = 3;
	int kernel_size = 3;


	cv::Canny(this->blur, this->detectedEdges, lowThreshold, lowThreshold*ratio, kernel_size);
}

void EdgeDetector::show(std::string windowName) {
	cv::imshow(windowName, this->detectedEdges);
}

cv::Mat EdgeDetector::getEdges() {
	return this->detectedEdges;
}
