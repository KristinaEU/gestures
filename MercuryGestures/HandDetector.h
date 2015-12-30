#pragma once
#include "MercuryCore.h"

enum SearchMode {
	FREE_SEARCH, // 0
	SEARCH_DOWN, // 1
	SEARCH_LEFT, // 2
	SEARCH_RIGHT,// 3
	SEARCH_UP    // 4
};

enum BlobType {
	BOTTOM_CLIPPED,
	LOW,
	MEDIUM,
	HIGH,
	OTHER
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
	bool intersecting = false;
	bool ignoreIntersect = false;
	bool leftHand = false;

	double cmInPixels;
	int positionIndex = 0;
	int blobIndex = 0;
	int fps = 25;

	std::vector<cv::Point> positionHistory;
	std::vector<cv::Point> rawPositionHistory;
	std::vector<BlobInformation> blobHistory;

	int historySize = 50;

	Hand();
	~Hand();

	void handleIntersection(cv::Point otherHandPosition, cv::Mat& skinMask);
	void setEstimate(cv::Point& estimate, BlobInformation& blob, bool ignoreIntersection = false);
	void solve(cv::Mat& skinMask, std::vector<BlobInformation>& blobs, cv::Mat& movementMap);
	bool improveByAreaSearch(cv::Mat& skinMask, cv::Point& position);
	SearchMode getSearchModeFromBlobs(std::vector<BlobInformation>& blobs);
	void improveByCoverage(cv::Mat& skinMask, SearchMode searchMode, int maxIterations = 20);
	void improveUsingHistory(cv::Mat& skinMask, cv::Mat& movementMap);
	void draw(cv::Mat& canvas);
	void drawTrace(cv::Mat& canvas, std::vector<cv::Point>& positions, int startIndex, int r, int g, int b);
	void updateLastPoint();

	
private:
	cv::Point getPredictedPosition(cv::Mat& skinMask);
	int getNextIndex(int index);
	int getPreviousIndex(int index);
	double getPointQuality(cv::Point& point, cv::Mat& skinMask, int radius = 0);
	cv::Point lookAround(cv::Point start,
		cv::Mat& skinMask,
		int maxIterations = 40,
		int stepSize = 1,
		int radius = 5, // in cm
		SearchMode searchMode = FREE_SEARCH,
		int colorBase = 255
		);
	double shiftPosition(
		cv::Mat& skinMask,
		std::vector<cv::Point>& newPositions,
		cv::Point& basePosition,
		int xOffset,
		int yOffset,
		int radius
	);
	double getCoverage(cv::Point& pos, cv::Mat& skinMask, int radius);
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

	void updateFaceMask(cv::Mat& highBlobsMask);
	void updateHandsFromTwoBlobs(BlobInformation& blob1, BlobInformation& blob2, bool ignoreIntersection = false);
	void updateHandsFromNBlobsByPosition(std::vector<BlobInformation>& blobs, bool ignoreIntersection = false);
	void updateHandsFromNBlobsWithAnalysis(std::vector<BlobInformation>& blobs, cv::Mat& edges);
	void detect(cv::Rect& face, cv::Mat& skinMask, cv::Mat& movementMap, cv::Mat& edges, double pixelSizeInCm);
	void draw(cv::Mat& canvas);
	double getFactor(BlobInformation& blob, double upperBound = 0.5);
	void setVideoProperties(int frameWidth, int frameHeight);
	void getHandEstimateFromBlob(BlobInformation& blob, Hand& handPosition, bool ignoreIntersection = false);
	void getBothHandPositionsFromBlob(BlobInformation& blob, bool ignoreIntersection = false);
};

double getDistance(cv::Point& p1, cv::Point& p2);