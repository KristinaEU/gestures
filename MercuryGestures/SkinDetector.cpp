#pragma once
#include "MercuryCore.h"
#include "SkinDetector.h"

SkinDetector::SkinDetector() {}
SkinDetector::~SkinDetector() {}

/**
* Get the average color of the skin based on the face.
*/
cv::Scalar SkinDetector::getAverageAreaColor(cv::Rect&area, cv::Mat& frame, bool refine, double centerFocus) {
	// we only use the inner area, 0.25 -- 0.75 of both width and height. This is to ensure we only get skin
	int x = area.x + 0.5*area.width - 0.5*area.width  * centerFocus;
	int y = area.y + 0.5*area.height - 0.5*area.height * centerFocus;
	int width = area.width   * centerFocus;
	int height = area.height * centerFocus;

	// setup a mask for the mean operation
	cv::Mat mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8U); // all 0
	mask(cv::Rect(x, y, width, height)) = 255;

	// refining assumes the first skinmap has been created and we will use it to remove the outliers
	if (refine) {
		cv::bitwise_and(this->skinMask, mask, mask);
	}

	// get the mean color
	cv::Scalar bgrColor = cv::mean(frame, mask);
	return bgrColor;
}

/**
* this uses the face area to detect the skin tone of the user (assuming no Burkah). This skin tone is searched for in YCrCb space.
* The aim is to find the hands and/or arms.
*/
void SkinDetector::detect(cv::Rect& face, cv::Mat& frame, bool refine, int noiseRemovalThreshold) {
	cv::Mat mask;
	cv::cvtColor(frame, mask, cv::COLOR_BGR2YCrCb);

	auto color = this->getAverageAreaColor(face, mask, refine);

	// color ranges
	int Y_MIN = rgbBound(color[0] - 100);  // 0
	int Y_MAX = rgbBound(color[0] + 100);  // 255
	int Cr_MIN = rgbBound(color[1] - 10);  // 133
	int Cr_MAX = rgbBound(color[1] + 20);  // 173
	int Cb_MIN = rgbBound(color[2] - 30);  // 77
	int Cb_MAX = rgbBound(color[2] + 10);  // 127

	//filter the image in YCrCb color space
	cv::inRange(mask, cv::Scalar(Y_MIN, Cr_MIN, Cb_MIN), cv::Scalar(Y_MAX, Cr_MAX, Cb_MAX), this->skinMask);
}

void SkinDetector::draw() {
	cv::imshow("skinMask", this->skinMask);
}
