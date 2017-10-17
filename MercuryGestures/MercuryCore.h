#pragma once

#include "opencv2/opencv.hpp"
#include <iostream>
#include <chrono>
#include <ctime>
#include <stdio.h>

#include <dirent.h>

#include <random>
#include <numeric>
#include <vector>

#define _USE_MATH_DEFINES // for C++
#include <cmath>

#define DEBUG
//#define TRAINING
//#define TRAINING_SAVE_DATA
//#define TRAINING_CREATE_NEW_CLASSIFIER

// this videos info tells the last frame where the gesture is finalized
// {"videoName", "lastFrameNumberGesture"}
static std::map<std::string, int> videosInfo = {
            {"RHShake00.mp4", 150},
            {"RHShake01.mp4", 140},
            {"RHShake02.mp4", 120},
            {"RHShake03.mp4", 100},
            {"RHShake04.mp4", 90},
            {"RHShake05.mp4", 115},
            {"RHShake06.mp4", 120},
            {"RHShake07.mp4", 110},
            {"RHShake08.mp4", 110},
            {"RHShake09.mp4", 90},
            {"RHShake10.mp4", 90},
            {"RHShake11.mp4", 170},
            {"RHShake12.mp4", 160},
            {"RHShake13.mp4", 150},
            {"RHShake14.mp4", 125},
            {"RHShake15.mp4", 145},
            {"RHShake16.mp4", 115},
            {"RHShake17.mp4", 160},
            {"RHShake18.mp4", 150},
            {"RHShake19.mp4", 190},
            {"RHShake20.mp4", 120},
            {"RHShake21.mp4", 150},
            {"RHShake22.mp4", 150},
            {"RHShake23.mp4", 175},
            {"RHShake24.mp4", 160},

            {"LHShake00.mp4", 175},
            {"LHShake01.mp4", 150},
            {"LHShake02.mp4", 160},
            {"LHShake03.mp4", 165},
            {"LHShake04.mp4", 145},
            {"LHShake05.mp4", 135},
            {"LHShake06.mp4", 175},
            {"LHShake07.mp4", 160},
            {"LHShake08.mp4", 160},
            {"LHShake09.mp4", 165},
            {"LHShake10.mp4", 150},
            {"LHShake11.mp4", 150},
            {"LHShake12.mp4", 170},
            {"LHShake13.mp4", 130},
            {"LHShake14.mp4", 165},
            {"LHShake15.mp4", 120},
            {"LHShake16.mp4", 130},
            {"LHShake17.mp4", 120},
            {"LHShake18.mp4", 120},
            {"LHShake19.mp4", 125},
            {"LHShake20.mp4", 135},
            {"LHShake21.mp4", 135},
            {"LHShake22.mp4", 145},
            {"LHShake23.mp4", 140},
            {"LHShake24.mp4", 130},

            {"StaticHandsUp00.mp4", 400},
            {"StaticHandsUp01.mp4", 350},
            {"StaticHandsUp02.mp4", 350},
            {"StaticHandsUp03.mp4", 350},
            {"StaticHandsUp04.mp4", 400}
        };

// based on // https://upload.wikimedia.org/wikipedia/commons/6/61/HeadAnthropometry.JPG
const double averageFaceWidth = 15.705; //cm (95th percentile)
const double averageFaceHeight = 23.95; //cm (95th percentile)
const double faceWidthToHeightRatio = averageFaceWidth / averageFaceHeight;

struct SSIMensage {
    double LHLocation;  // Allowed values -> 0.0; 1.0; 2.0; 3.0; 4.0 ;...; 9.0
    double RHLocation;  // Allowed values -> 0.0; 1.0; 2.0; 3.0; 4.0 ;...; 9.0
    double arousal;
    double gesture;
};

struct FaceData {
	cv::Rect rect;
	int count;
};

struct BodyRects {
	cv::Rect face;
	cv::Rect upperTorso;
	cv::Rect lowerTorso;
	cv::Rect lap;
	cv::Rect armRightUpper;
	cv::Rect armRightLower;
	cv::Rect armLeftUpper;
	cv::Rect armLeftLower;
};

struct SearchSpace {
	cv::Mat mat;
	int x;
	int y;
	cv::Rect area;
};

/*
 * Get center x and y for a rect
 */
int getCenterX(cv::Rect& face);
int getCenterY(cv::Rect& face);

/*
 * return a value bounded between 0 and 255.
 */
int rgbBound(int color);

/**
 * get the average value of a vector.
 */
double getAverage(std::vector<double> data);

/*
 * Join two strings together.
 */
std::string joinString(std::string a, std::string b);
std::string joinString(std::string a, int b);
std::string joinString(std::string a, double b);
std::string joinString(int a, std::string b);

/**
 * this method is a preconfigured dilate call
 */
void dilate(cv::Mat& inputFrame, cv::Mat& outputFrame, int kernelSize = 6);

/**
 * this method is a preconfigured erode call
 */
void erode(cv::Mat& inputFrame, cv::Mat& outputFrame, int kernelSize = 6);

/**
 * remove noise by dilating first, then eroding.
 */
void dilateErodeNoiseRemoval(cv::Mat& inputFrame, cv::Mat& outputFrame, int kernelSize = 6);

/**
 * this method uses calculates the length of the contours of the blobs in the input frame and uses those to remove noise.
 */
void contourNoise(cv::Mat& inputFrameWithBlobs, cv::Mat& outputFrame, int contourThreshold);

void getSearchSpace(SearchSpace& empty, cv::Mat& inputSpace, cv::Point& focalPoint, int searchSpaceRadius = 100);
void getSearchSpace(SearchSpace& empty, cv::Mat& inputSpace, cv::Rect&  focalRect,  int searchSpaceRadius = 80);

void toSearchSpace(SearchSpace& space, cv::Point& point);
void toSearchSpace(SearchSpace& space, cv::Rect& rect);

void fromSearchSpace(SearchSpace& space, cv::Point& point);
void fromSearchSpace(SearchSpace& space, cv::Rect& rect);
void fromSearchSpace(SearchSpace& space, cv::Point2f& point);

double getDistance(cv::Point& p1, cv::Point& p2);
void rect(cv::Mat& mat, cv::Point point, int radius, cv::Scalar color, int thickness);

cv::Rect inflateRect(cv::Rect& rectangle, int inflation, cv::Mat& boundary);
