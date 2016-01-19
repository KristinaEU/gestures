#pragma once

#include "MercuryCore.h"

class FaceDetector {
public:
	FaceData face;
	int faceCenterX = 0;
	int faceCenterY = 0;
	int faceReadingThreshold = 5;
	int failureThreshold = 25;
	bool faceLocked = false;
	int goodReadings = 0;
	int badReadings = 0;
	double faceAreaThresholdFactor = 0.08; // 8% of total frame width;
	int frameHeight = 400;
	int frameWidth = 640;
	cv::CascadeClassifier face_cascade;
	std::string face_cascade_name = "./dependencies/haarcascade_frontalface_alt.xml";

	// scaling settings
	double normalizationFactor = 1;
	double normalizationFactorSetup = 0;
	int normalizationIterationsCount = 0;
	int normalizationIterations = 10;
	double pixelSizeInCm = 0.25; // will be refined in the normalization phase
	double normalizationScale = 0.25; // cm per pixel based on face detection.

	FaceDetector();
	~FaceDetector();

	void updateScale();
	void addResultToMask(cv::Mat& mask);
	void draw(cv::Mat& canvas);
	/**
	* This detects faces. It assumes only one face will be in view. It will draw the boundaries of the expected position of
	* user body parts. This can be used to make a mask to sample only important parts of the image for movement. It can also
	* be used to ignore the head movement or upper torso.
	*/
	bool detectFace(cv::Mat& grayscaleImage, FaceData & data);
	bool detect(cv::Mat& gray);
	bool setup();
	void setVideoProperties(int width, int height);
	void reset();
};