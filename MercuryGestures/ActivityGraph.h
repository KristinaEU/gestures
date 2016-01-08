#pragma once

#include "MercuryCore.h"

class ActivityGraph {
public: 
	cv::Mat graph;
	cv::Mat graphMask;
	int fps = 25;
	int frameWidth = 0;
	int frameHeight = 0;
	int counter = 0;
	int ticks = 0;

	// graph settings
	int graphSeconds = 60;
	int graphHeight = 250;
	int distanceFromBottom = 15;

	std::map <std::string, int> channelNames;
	std::vector<double> channelValues;
	std::vector<double> channelOffset;
	std::vector<double> channelPreviousValues;
	std::vector<cv::Scalar> channelColors;

	ActivityGraph(int fps, int graphSeconds = 60);
	~ActivityGraph();

	void setVideoProperties(int width, int height);
	
	
	// Draw a graph on the frame based on the output	
	void draw(cv::Mat& canvas);

	void clearGraph();
	void drawLegend();
	void addChannel(std::string channelName, cv::Scalar color = CV_RGB(255, 0, 0), double offset = 0.0);
	void setValue(std::string channelName, double value);

};