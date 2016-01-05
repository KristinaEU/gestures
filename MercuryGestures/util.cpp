#pragma once

#include "MercuryCore.h"

int getCenterX(cv::Rect& face) {
	return face.x + 0.5* face.width;
}

int getCenterY(cv::Rect& face) {
	return face.y + 0.5* face.height;
}

/**
* get the average value of a vector
*/
double getAverage(std::vector<double> data) {
	double sum = 0;
	for (int i = 0; i < data.size(); i++) {
		sum += data[i];
	}
	return sum / data.size();
}


std::string joinString(std::string a, std::string b) {
	std::ostringstream str;
	str << a << b;
	return str.str();
}

std::string joinString(std::string a, int b) {
	std::ostringstream str;
	str << a << b;
	return str.str();
}


std::string joinString(std::string a, double b) {
	std::ostringstream str;
	str << a << b;
	return str.str();
}


std::string joinString(int a, std::string b) {
	std::ostringstream str;
	str << a << b;
	return str.str();
}


/**
 * this method is a preconfigured dilate call
 */
void dilate(cv::Mat& inputFrame, cv::Mat& outputFrame, int kernelSize ) {
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
	cv::dilate(inputFrame, outputFrame, element);
}

/**
 * this method is a preconfigured erode call
 */
void erode(cv::Mat& inputFrame, cv::Mat& outputFrame, int kernelSize ) {
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
	cv::erode(inputFrame, outputFrame, element);
}


/**
 * remove noise by dilating first, then eroding.
 */
void dilateErodeNoiseRemoval(cv::Mat& inputFrame, cv::Mat& outputFrame, int kernelSize) {
	dilate(inputFrame, outputFrame, kernelSize);
	erode(outputFrame, outputFrame, kernelSize);
}



/**
* this method uses calculates the length of the contours of the blobs in the input frame and uses those to remove noise.
*/
void contourNoise(cv::Mat& inputFrameWithBlobs, cv::Mat& outputFrame, int contourThreshold = 80) {
	cv::Mat inputCopy;
	cv::Mat inputCopy2;
	cv::Mat canvas;

	// since the find contours method is destructive to it's input, we copy it
	inputFrameWithBlobs.copyTo(inputCopy);
	inputFrameWithBlobs.copyTo(inputCopy2);
	canvas = cv::Mat::zeros(inputFrameWithBlobs.size(), CV_8U);

	// setup of the contour finding
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(inputCopy, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	// Find the contours and draw them if they are larger than the minimum threshold.
	int number_of_contours = contours.size();
	for (int i = 0; i < number_of_contours; i++) {
		int contour_size = contours[i].size();
		if (contour_size > contourThreshold) {
			cv::drawContours(canvas, contours, i, 255, CV_FILLED, 8, hierarchy, 0, cv::Point());
		}
	}

	// since we use the contours as a method of noise removal, we do not want areas that have been filled by this system.
	// To avoid this, we do the AND operation with the input.
	// The input copy 1 has been ruined by the findContours method, hence we use the second copy. We do not use the original
	// because that could also be used as the output.
	cv::bitwise_and(inputCopy2, canvas, outputFrame);
}

int rgbBound(int color) {
	return std::min(255, std::max(0, color));
}

//TODO: explain
void getSearchSpace(SearchSpace& empty, cv::Mat& inputSpace, cv::Point& focalPoint, int searchSpaceRadius) {
	empty.x = std::max(0, std::min(inputSpace.cols, focalPoint.x - searchSpaceRadius));
	empty.y = std::max(0, std::min(inputSpace.rows, focalPoint.y - searchSpaceRadius));
	int width = std::min(inputSpace.cols, empty.x + 2 * searchSpaceRadius) - empty.x;
	int height = std::min(inputSpace.rows, empty.y + 2 * searchSpaceRadius) - empty.y;
	cv::Rect searchRect(empty.x, empty.y, width, height);
	empty.mat = inputSpace(searchRect);
}
void getSearchSpace(SearchSpace& empty, cv::Mat& inputSpace, cv::Rect& focalRect, int searchSpaceRadius) {
	empty.x = std::max(0, std::min(inputSpace.cols, focalRect.x - searchSpaceRadius));
	empty.y = std::max(0, std::min(inputSpace.rows, focalRect.y - searchSpaceRadius));
	int width = std::min(inputSpace.cols, focalRect.x + focalRect.width + searchSpaceRadius) - empty.x;
	int height = std::min(inputSpace.rows, focalRect.y + focalRect.height + searchSpaceRadius) - empty.y;
	cv::Rect searchRect(empty.x, empty.y, width, height);
	empty.mat = inputSpace(searchRect);
}

void toSearchSpace(SearchSpace& space, cv::Point& point) {
	point.x -= space.x;
	point.y -= space.y;
}

void toSearchSpace(SearchSpace& space, cv::Rect& rect) {
	rect.x -= space.x;
	rect.y -= space.y;
}

void fromSearchSpace(SearchSpace& space, cv::Point& point) {
	point.x += space.x;
	point.y += space.y;
}

void fromSearchSpace(SearchSpace& space, cv::Rect& rect) {
	rect.x += space.x;
	rect.y += space.y;
}

double getDistance(cv::Point& p1, cv::Point& p2) {
	int dx = p1.x - p2.x;
	int dy = p1.y - p2.y;
	return std::sqrt(dx*dx + dy*dy);
}

void rect(cv::Mat& mat, cv::Point point, int radius, cv::Scalar color, int thickness) {
#ifdef DEBUG
	cv::Rect rect(point.x - radius, point.y - radius, 2 * radius, 2 * radius);
	cv::rectangle(mat, rect, color, thickness);
#endif
}