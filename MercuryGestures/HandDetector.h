#pragma once
#include "MercuryCore.h"

enum SearchMode {
	FREE_SEARCH, 
	SEARCH_DOWN, 
	SEARCH_STRICT_LEFT, 
	SEARCH_LEFT, 
	SEARCH_STRICT_RIGHT,
	SEARCH_RIGHT,
	SEARCH_UP   
};

enum BlobType {
	BOTTOM_CLIPPED,
	LOW,
	MEDIUM,
	HIGH,
	OTHER
};

enum Condition {
	NONE,
	ONLY_HEAD
};

struct BlobEdgeData {
	int index;
	int edgeCount;
	int size;
};

struct BlobInformation {
	int index;
	cv::Point left;
	cv::Point right;
	cv::Point top;
	cv::Point bottom;
	cv::Point center;
	BlobType type;
	std::vector<cv::Point> contour;
};

class Hand {
public:
	double maxVelocity = 100; // 100 cm / second
	cv::Point position;
	cv::Point blobEstimate;
	cv::Scalar color;
	cv::Mat* rgbSkinMask; // for debug
	bool estimateUpdated = false;
	bool invalidState = false;
	bool ignoreState = false;
	bool leftHand = false;
	int faceCoverageThreshold = 100;

	double cmInPixels;
	int positionIndex = 0;
	int blobIndex = 0;
	int fps = 25;

	// optical flow items
	std::vector<cv::Point2f> opticalFlowPointsPrev, opticalFlowPoints;
	std::vector<uchar> opticalFlowStatus;
	std::vector<bool> opticalFlowSuccess;
	cv::Mat opticalFlowErr;
	cv::Point opticalFlowPoint;

	std::vector<cv::Point> positionHistory;
	std::vector<BlobInformation> blobHistory;

	int historySize = 50;

	Hand();
	~Hand();
	
	// set the estimate based on the blobs. This is a fallback and/or initialization position.
	void setEstimate(cv::Point& estimate, BlobInformation& blob, bool ignoreIntersection = false, Condition condition = NONE);
	
	// handle intersections
	bool isClose(cv::Point& otherHandPosition, bool drawDebug = false);
	bool isIntersecting(cv::Point& otherHandPosition);
	void handleIntersection(cv::Point& otherHandPosition, cv::Mat& skinMask);
	void setInvalideState();
	
	// solve and finalize the positions. The handling of intersections is in between this
	void solve(cv::Mat& gray, cv::Mat& grayPrev, cv::Mat& skinMask, std::vector<BlobInformation>& blobs, cv::Mat& movementMap);
	void finalize(cv::Mat& skinMask, cv::Mat& movementMap);

	// draw on canvas.
	void addResultToMask(cv::Mat& canvas);
	void draw(cv::Mat& canvas);
	void drawTrace(cv::Mat& canvas);
	void drawTrace(cv::Mat& canvas, std::vector<cv::Point>& positions, int startIndex, int r, int g, int b);

private:
	SearchMode getSearchModeFromBlobs(std::vector<BlobInformation>& blobs);
	bool improveByAreaSearch(cv::Mat& skinMask, cv::Mat& movementMap, cv::Point& position);
	void improveByCoverage(cv::Mat& skinMask, SearchMode searchMode, int maxIterations, int colorBase = 255);
	void improveByDirection(cv::Mat& skinMask, SearchMode searchMode, int maxIterations, int colorBase = 255);
	void improveUsingHistory(cv::Mat& skinMask, cv::Mat& movementMap);
	void improvePreviousPoint();

	// prediction
	cv::Point getPredictedPosition(cv::Mat& gray, cv::Mat& grayPrev, cv::Mat& skinMask);
	cv::Point getEstimateByOpticalFlow(cv::Mat& gray, cv::Mat& grayPrev, cv::Mat& skinMask, cv::Point& lastPosition);
	
	// util
	int getNextIndex(int index);
	int getPreviousIndex(int index);
	double getPointQuality(cv::Point& point, cv::Mat& skinMask, int radius = 0);
	double getCoverage(cv::Point& pos, cv::Mat& skinMask, int radius);
	cv::Point lookAround(cv::Point start,
		cv::Mat& skinMask,
		int maxIterations,
		int stepSize,
		int radius, // in cm
		SearchMode searchMode,
		int colorBase = 255
		);
	double shiftPosition(
		cv::Mat& skinMask,
		cv::Point& newPosition,
		cv::Point& basePosition,
		int xOffset,
		int yOffset,
		int radius
	);
	

};

class HandDetector {
public:
	Hand leftHand;
	Hand rightHand;

	int centerX;
	cv::Rect region1;

	int frameWidth;
	int frameHeight;
	int fps = 25;
	double cmInPixels;

	cv::Mat skinMask;
	cv::Mat faceMask;
	cv::Mat rgbSkinMask;
	int faceMaskAverageArea = 0;

	HandDetector(int fps);
	~HandDetector();
	
	void addResultToMask(cv::Mat& canvas);
	void detect(cv::Mat& gray, cv::Mat& grayPrev, cv::Rect& face, cv::Mat& skinMask, cv::Mat& movementMap, cv::Mat& edges, double pixelSizeInCm);
	void draw(cv::Mat& canvas);
	void drawTraces(cv::Mat& canvas);
	void show(std::string windowName = "debugMapHands");
	void setVideoProperties(int frameWidth, int frameHeight);

private:
	double getFactor(BlobInformation& blob, double upperBound = 0.5);
	void getHandEstimateFromBlob(BlobInformation& blob, Hand& handPosition, bool ignoreIntersection = false);
	void getBothHandPositionsFromBlob(BlobInformation& blob, bool ignoreIntersection = false, Condition condition = NONE);
	void updateFaceMask(cv::Mat& highBlobsMask);
	void updateHandsFromTwoBlobs(BlobInformation& blob1, BlobInformation& blob2, bool ignoreIntersection = false);
	void updateHandsFromNBlobsByPosition(std::vector<BlobInformation>& blobs, bool ignoreIntersection = false);
	void updateHandsFromNBlobsWithAnalysis(std::vector<BlobInformation>& blobs, cv::Mat& edges);
	void handleIntersections();
};

double getDistance(cv::Point& p1, cv::Point& p2);