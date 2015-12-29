#pragma once

#include "MercuryCore.h"
#include "ActivityGraph.h"


ActivityGraph::ActivityGraph(int fps) {
	this->fps = fps;
};

ActivityGraph::~ActivityGraph() {};

void ActivityGraph::setVideoProperties(int frameWidth, int frameHeight) {
	this->graph = cv::Mat::zeros(frameHeight, frameWidth, 0);
	this->filteredGraph = cv::Mat::zeros(frameHeight, frameWidth, 0);
	this->frameHeight = frameHeight;
	this->frameWidth = frameWidth;
}

/**
* Draw a graph repesenting the movement over time. This will be drawn on the RGB frame.
*/
void ActivityGraph::draw(
	double value,
	double filteredValue,
	cv::Mat& thresholdFrame, 
	cv::Mat& frame
	) {
	
	// graph settings
	int graphSeconds = 120;
	int graphHeight = 250;
	int distanceFromBottom = 12;

	// Reset the scrollling graph if the graphSeconds are over
	if (this->counter % (graphSeconds*this->fps) == 0) {
		graph = cv::Mat::zeros(this->frameHeight, this->frameWidth, 0);
		filteredGraph = cv::Mat::zeros(this->frameHeight, this->frameWidth, 0);
		this->counter = 0;
		this->yPrev = 0;
	}

	// setup the graph
	int x = (this->counter / double(graphSeconds*fps)) * this->frameWidth;
	int y = this->frameHeight - distanceFromBottom - value * graphHeight;
	int filteredY = this->frameHeight - distanceFromBottom - filteredValue * graphHeight;

	// drawing the time line indicators every 5 seconds
	if (this->counter % (5 * this->fps) == 0) {
		std::ostringstream timeText;
		timeText << graphCounter / this->fps << "s";
		cv::putText(graph, timeText.str(), cv::Point(x, this->frameHeight), CV_FONT_HERSHEY_PLAIN, 0.8, CV_RGB(0, 0, 250));
		cv::line(graph, (cv::Point(x, this->frameHeight)), (cv::Point(x, this->frameHeight - graphHeight)), 50);
	}

	// check if we draw a line or a point. Point is only drawn as the very first item.
	if (this->yPrev > 0) {
		cv::line(graph, (cv::Point(this->xPrev, this->yPrev)), (cv::Point(x, y)), 255);
		cv::line(filteredGraph, (cv::Point(this->xPrev, this->filteredYPrev)), (cv::Point(x, filteredY)), 255);
	}
	else {
		graph.at<uint8_t>(cv::Point(x, y)) = 255;
		filteredGraph.at<uint8_t>(cv::Point(x, filteredY)) = 255;
	}

	// values are now previous values
	this->filteredYPrev = filteredY;
	this->xPrev = x;
	this->yPrev = y;

	// convert the threshold to RGB so we can merge it with the frame image
	cv::Mat thresholdRGB;
	cv::add(thresholdFrame, graph, thresholdFrame);
	cv::cvtColor(thresholdFrame, thresholdRGB, CV_GRAY2BGR);

	// We want the filtered line to be green so we use the composit method.
	std::vector<cv::Mat> composite(3);
	composite[0] = cv::Mat::zeros(frameHeight, frameWidth, 0);
	composite[1] = filteredGraph;
	composite[2] = cv::Mat::zeros(frameHeight, frameWidth, 0);
	cv::Mat colorFilterGraph;
	cv::merge(composite, colorFilterGraph);

	// merge to frame
	cv::add(frame, colorFilterGraph, frame);
	cv::add(frame, thresholdRGB, frame);

	this->counter++;
	this->graphCounter++;
}

