#pragma once

#include "MercuryCore.h"

class SkinDetector {
public:
	cv::Mat skinMask;
	cv::Mat previousSkinMask;


	SkinDetector();
	~SkinDetector();
	cv::Scalar getAverageAreaColor(cv::Rect&area, cv::Mat& frame, bool refine, double centerFocus = 0.6);
	void detect(cv::Rect& face, cv::Mat& frame, bool refine, int noiseRemovalThreshold = 80);
	void show(std::string windowName = "skinMask");
	cv::Mat getMergedMap();
};