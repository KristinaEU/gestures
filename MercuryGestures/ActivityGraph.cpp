#pragma once

#include "MercuryCore.h"
#include "ActivityGraph.h"


ActivityGraph::ActivityGraph(int fps) {
	this->fps = fps;
};

ActivityGraph::~ActivityGraph() {};

void ActivityGraph::setVideoProperties(int frameWidth, int frameHeight) {
	this->frameHeight = frameHeight;
	this->frameWidth = frameWidth;
}


void ActivityGraph::clearGraph() {
	this->graph = cv::Mat::zeros(this->frameHeight, this->frameWidth, CV_8UC3);
	this->graphMask = cv::Mat::zeros(this->frameHeight, this->frameWidth, CV_8UC3);
	this->counter = 0;
}

void ActivityGraph::addChannel(std::string channelName, cv::Scalar color, double offset) {
	this->channelColors.push_back(color);
	this->channelOffset.push_back(offset);
	this->channelValues.push_back(0);
	this->channelPreviousValues.push_back(0);

	int index = this->channelColors.size() - 1;
	
	this->channelNames[channelName] = index;
}

void ActivityGraph::setValue(std::string channelName, double value) {
	int index = this->channelNames[channelName];
	this->channelValues[index] = value;
}

void ActivityGraph::drawLegend() {
	int legendCounter = 0;
	int legendElementHeight = 30;
	int boxSize = 20;
	for (auto const &iterator : this->channelNames) {
		auto name = iterator.first;
		int index = iterator.second;
		int legendHeight = 10 + legendElementHeight * legendCounter;

		cv::Rect colorArea(10, legendHeight, boxSize, boxSize);
		cv::rectangle(this->graphMask, colorArea, CV_RGB(255, 255, 255), CV_FILLED);
		cv::rectangle(this->graph, colorArea, this->channelColors[index], CV_FILLED);
		cv::putText(this->graph, name, cv::Point(35, legendHeight + 0.75*boxSize), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 255, 255));

		legendCounter++;
	}
}

/**
* Draw a graph repesenting the movement over time. This will be drawn on the RGB frame.
*/
void ActivityGraph::draw(cv::Mat& canvas) {
	// Reset the scrollling graph if the graphSeconds are over
	int graphFrameCount = this->graphSeconds*this->fps;
	if (this->counter % graphFrameCount == 0) {
		this->clearGraph();
		this->drawLegend();
	}

	// x value based on time
	int x = (this->counter / double(graphFrameCount)) * this->frameWidth;
	int prevX = ((this->counter-1) / double(graphFrameCount)) * this->frameWidth;

	// drawing the time line indicators every 5 seconds
	if (this->counter % (5 * this->fps) == 0) {
		cv::putText(this->graph, joinString(this->ticks / this->fps, "s"), cv::Point(x, this->frameHeight), CV_FONT_HERSHEY_PLAIN, 0.9, CV_RGB(255, 255, 255));
		cv::line(this->graph, (cv::Point(x, this->frameHeight)), (cv::Point(x, this->frameHeight - graphHeight)), CV_RGB(255,255,255));
	}

	// draw all channels
	for (int i = 0; i < this->channelValues.size(); i++) {
		double value = this->channelValues[i];
		double previousValue = this->channelPreviousValues[i];

		// get Y value
		int y = this->frameHeight - this->distanceFromBottom - (value + this->channelOffset[i]) * this->graphHeight;
		int prevY = this->frameHeight - this->distanceFromBottom - (previousValue + this->channelOffset[i]) * this->graphHeight;

		int channelOffsetY = this->frameHeight - this->distanceFromBottom - this->channelOffset[i] * this->graphHeight + 2; // 2 is the offset so it does not interfere with value 0

		if (this->counter > 1) {
			// draw the offset line.
			cv::line(this->graph, (cv::Point(0, channelOffsetY)), (cv::Point(x, channelOffsetY)), CV_RGB(255, 255, 255), 1);

			// draw the data lines
			cv::line(this->graph, (cv::Point(prevX, prevY)), (cv::Point(x, y)), this->channelColors[i], 1, CV_AA);
			cv::line(this->graphMask, (cv::Point(prevX, prevY)), (cv::Point(x, y)), CV_RGB(255,255,255), 1, CV_AA);
		}

		// store the previous value.
		this->channelPreviousValues[i] = value;
	}

	// merge to frame
	cv::subtract(canvas, this->graphMask, canvas);
	cv::add(canvas, this->graph, canvas);

	this->counter++;
	this->ticks++;
}

