#include "MercuryCore.h"
/**
* This will generate a region of interest mask which will bitwise AND-ed with the movment results.
*/
void generateROImask(cv::Mat& mask, BodyRects& maskBody) {
	cv::rectangle(mask, maskBody.lowerTorso, 255, CV_FILLED);
	cv::rectangle(mask, maskBody.lap, 255, CV_FILLED);
	cv::rectangle(mask, maskBody.armLeftUpper, 255, CV_FILLED);
	cv::rectangle(mask, maskBody.armLeftLower, 255, CV_FILLED);
	cv::rectangle(mask, maskBody.armRightUpper, 255, CV_FILLED);
	cv::rectangle(mask, maskBody.armRightLower, 255, CV_FILLED);
}

