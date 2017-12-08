#pragma once

#include "MercuryCore.h"
#include "MovementDetector.h"

MovementDetector::MovementDetector(int fps) {
	this->fps = fps;

	// set size of the filter
	this->values.resize(fps, 0.0);
}

MovementDetector::~MovementDetector() {};

/**
* Calculate the movement and return a double between 0 .. 1
* Thev value is clipped in this range and normalized using face detection.
*/
void MovementDetector::detect(cv::Mat& gray, cv::Mat& grayPrev) {
	// get a value between 0 .. 1 to represent the amount of movement.
	cv::Mat diff;
	// get the amount of movement in this frame
	cv::absdiff(gray, grayPrev, diff);
	cv::threshold(diff, this->movementMap, 25, 255, 0);

	//cv::imshow("unfilteredMovement", diff);
}

void MovementDetector::mask(cv::Mat& mask) {
	cv::bitwise_and(this->movementMap, mask, this->movementMap);
}


void MovementDetector::calculate(double normalizationFactor) {

	this->value = std::min(this->maxMovement, std::max(0.0, (cv::sum(this->movementMap)[0] / 255.0) * normalizationFactor)) / maxMovement;

	values[this->index%this->fps] = this->value;

	this->filteredValue = getAverage(values);

	this->index++;
}

void MovementDetector::show(std::string windowName) {
	cv::imshow(windowName, this->movementMap);
}

void MovementDetector::draw(cv::Mat& canvas) {
	cv::add(canvas, this->movementMap, canvas);
}
