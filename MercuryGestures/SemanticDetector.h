#pragma once
#include "MercuryCore.h"

class SemanticDetector {
public:
    int fps = 25; //default
	int frameWidth;
	int frameHeight;
    std::string bodyPart; // body part to be analyzed ("Head" or "Hands")
    double minTimeToDetect = 2.0; // minimum time to detect a gesture (2 sec)
    double interpolationTimeStep = 0.02; // (seconds)
    double normalizationFaceHandDistance = 100; // (cm) it will make the face hand distances be around the range of -1 to +1


	SemanticDetector(int fps, std::string bodyPart);
	~SemanticDetector();

    void detect(cv::Point faceCenterPoint, double pixelSizeInCmTemp, std::vector<cv::Point> positions[]);
	void setVideoProperties(int frameWidth, int frameHeight);
    void logisticsTest();
	//void draw(cv::Mat& canvas);
	//void drawTraces(cv::Mat& canvas);
	//void show(std::string windowName = "debugMapHands");


private:
	double _interpolate( std::vector<double> &xData, std::vector<double> &yData, double x, bool extrapolate );
	/*
	double getFactor(BlobInformation& blob, double upperBound = 0.5);
	void getHandEstimateFromBlob(BlobInformation& blob, Hand& handPosition, bool ignoreIntersection = false);
	void getBothHandPositionsFromBlob(BlobInformation& blob, bool ignoreIntersection = false, Condition condition = NONE);
	void updateFaceMask(cv::Mat& highBlobsMask);
	void updateHandsFromTwoBlobs(BlobInformation& blob1, BlobInformation& blob2, bool ignoreIntersection = false);
	void updateHandsFromNBlobsByPosition(std::vector<BlobInformation>& blobs, bool ignoreIntersection = false);
	void updateHandsFromNBlobsWithAnalysis(std::vector<BlobInformation>& blobs, cv::Mat& edges);
	void handleIntersections();
	*/
};
