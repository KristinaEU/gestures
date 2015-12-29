#pragma once

#include "MercuryCore.h"

class ActivityGraph {
public: 
	cv::Mat graph;
	cv::Mat filteredGraph;
	int xPrev = 0;
	int filteredYPrev = 0;
	int yPrev = 0;
	int fps = 25;
	int frameWidth = 0;
	int frameHeight = 0;
	int counter = 0;
	int graphCounter = 0;

	ActivityGraph(int fps);
	~ActivityGraph();


	void setVideoProperties(int width, int height);
	/*
	* Draw a graph on the frame based on the output
	*/
	void draw(
		double value,
		double filteredValue,
		cv::Mat& thresholdFrame,
		cv::Mat& frame
		);
};