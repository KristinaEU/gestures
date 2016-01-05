#pragma once

#include "MercuryCore.h"
#include "FaceDetector.h"

class MovementDetector {
public:
	cv::Mat movementMap;
	int index = 0;
	int fps = 25;
	std::vector<double> values;
	double maxMovement = 25000.0; // todo: determine this?
	double value;
	double filteredValue;

	MovementDetector(int fps);
	~MovementDetector();

	void detect(cv::Mat& gray, cv::Mat& grayPrev);
	void mask(cv::Mat& mask);
	void calculate(double normalizationFactor);
	void show(std::string windowName = "movementMap");
	void draw(cv::Mat& canvas);
};