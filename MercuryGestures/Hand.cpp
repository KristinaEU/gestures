#pragma once
#include "MercuryCore.h"
#include "HandDetector.h"

/************************************  UTIL  *********************************************/

double getDistance(cv::Point& p1, cv::Point& p2) {
	int dx = p1.x - p2.x;
	int dy = p1.y - p2.y;
	return std::sqrt(dx*dx + dy*dy);
}

void rect(cv::Mat& mat, cv::Point point, int radius, cv::Scalar color, int thickness) {
	cv::Rect rect(point.x - radius, point.y - radius, 2 * radius, 2 * radius);
	cv::rectangle(mat, rect, color, thickness);
}

/************************************ /UTIL  *********************************************/


Hand::Hand() {
	this->position = cv::Point(0,0);
	this->estimate = cv::Point(0, 0);
	this->positionHistory.resize(historySize, cv::Point(0, 0));
	this->blobHistory.resize(historySize);
	this->color = CV_RGB(0, 255, 0);
}
Hand::~Hand() {}


void Hand::setup() {
	this->estimateUpdated = false;
}

int Hand::getNextIndex(int index) {
	return (index + 1) % this->historySize;
}

int Hand::getPreviousIndex(int index) {
	return (index - 1) < 0 ? index - 1 + this->historySize : index - 1;
}

void Hand::setEstimate(cv::Point& estimate, BlobInformation& blob, bool ignoreIntersection) {
	if (blob.type == HIGH || ignoreIntersection == true) {
		this->ignoreIntersect = true;
	}
	this->blobEstimate = estimate;
	this->blobIndex = this->getNextIndex(this->blobIndex);
	this->blobHistory[this->blobIndex] = blob;
	this->estimateUpdated = true;
}


void Hand::draw(cv::Mat& canvas) {
	rect(canvas, this->position, 20, this->color, 2);
	cv::putText(canvas, this->leftHand ? "L" : "R", this->position, 0, 1.0, this->color, 3);
}


void Hand::solve(cv::Mat& skinMask, std::vector<BlobInformation>& blobs) {
	// if the estimate has been updated, update the position. If the improvement algorithms fail, this is the fallback
	if (this->estimateUpdated == true) {
		this->position = this->blobEstimate;
	}
	
	// search for a position based on the last known position
	auto lastPosition = this->positionHistory[this->positionIndex]; // still the last one since we have not yet found the final pos.
	if (lastPosition.x != 0 && lastPosition.y != 0) {
		bool areaSearched = this->improveByAreaSearch(skinMask, lastPosition);
		// if we did not search because of fast moving objects, we try again with a predicted position.
		if (!areaSearched) {
			auto predictedPoint = this->getPredictedPosition(skinMask);
			// if we have a predicted position, use that.
			if (predictedPoint.x != 0 && predictedPoint.y != 0) {
				this->improveByAreaSearch(skinMask, predictedPoint);
			}
		}
	}

	// get a search mode based on the location of the blob
	auto searchMode = this->getSearchModeFromBlobs(blobs);

	// find a good estimate
	this->improveByCoverage(skinMask, searchMode);

	// check if we can use averaging to smooth the result.
	this->improveByAveraging(skinMask);

	this->positionIndex = this->getNextIndex(this->positionIndex);
	this->positionHistory[this->positionIndex] = this->position;
	
}

SearchMode Hand::getSearchModeFromBlobs(std::vector<BlobInformation>& blobs) {
	for (int i = 0; i < blobs.size(); i++) {
		if (blobs[i].left.x   <= this->position.x &&
			blobs[i].right.x  >= this->position.x && 
			blobs[i].top.y    <= this->position.y &&
			blobs[i].bottom.y >= this->position.y) {

			int height = (blobs[i].bottom.y - blobs[i].top.y) / this->cmInPixels;

			if (height < 15) { // height smaller than 15 cm
				return FREE_SEARCH;
			}
			else {
				if (blobs[i].type == LOW) {
					return ONLY_SEARCH_DOWN;
				}
				else if (blobs[i].type == MEDIUM) {
					return FREE_SEARCH;
				}
				else if (blobs[i].type == HIGH) {
					return ONLY_SEARCH_UP;
				}
				else {
					return FREE_SEARCH;
				}
			}
		}
	}
}

void Hand::improveByAveraging(cv::Mat& skinMask) {
	int historyAverage = 3; // must be lower or equal to this->historySize
	int index = this->positionIndex;
	
	double avgX = 0;
	double avgY = 0;
	for (int i = 0; i < historyAverage; i++) {
		avgX += this->positionHistory[index].x;
		avgY += this->positionHistory[index].y;
		index = this->getPreviousIndex(index);
	}
	avgX /= historyAverage;
	avgY /= historyAverage;
	
	if (std::abs(this->position.x - avgX) < 10 && std::abs(this->position.y - avgY) < 10) {
		avgX = (avgX*historyAverage + this->position.x) / (historyAverage + 1);
		avgY = (avgY*historyAverage + this->position.y) / (historyAverage + 1);
		this->position.x = avgX;
		this->position.y = avgY;	
	}	
}

cv::Point Hand::getPredictedPosition(cv::Mat& skinMask) {
	auto position_n_1 = this->positionHistory[this->positionIndex]; // still the last one since we have not yet found the final pos.
	auto position_n_2 = this->positionHistory[this->getPreviousIndex(this->positionIndex)];

	double dt = 1.0 / this->fps;
	int dx1 = position_n_1.x - position_n_2.x;
	int dy1 = position_n_1.y - position_n_2.y;

	double vx1 = dx1 / dt;
	double vy1 = dy1 / dt;
	
	cv::Point newPosition(position_n_1.x + vx1 * dt, position_n_1.y + vy1 * dt);
	
	if (this->getPointQuality(newPosition, skinMask) > 0.7) {
		rect(*this->rgbSkinMask, newPosition, 10, CV_RGB(255, 200, 50), 5);
		return newPosition;
	}
	return cv::Point(0, 0);
}

void Hand::improveByCoverage(cv::Mat& skinMask, SearchMode searchMode) {
	int maxIterations = 20;
	int stepSize = 3;
	int radius = 3.5 * this->cmInPixels; 

	cv::Point maxPos = this->lookAround(this->position, skinMask, maxIterations, stepSize, radius, searchMode);

	// update position with improved one.
	this->coverageEstimate = maxPos;
	this->position = maxPos;
}

bool Hand::improveByAreaSearch(cv::Mat& skinMask, cv::Point& position) {
	double distance = getDistance(position, this->position);
	double maxDistance = 2 * this->maxVelocity / fps * cmInPixels;

	// if a jump or if no data
	//std::cout << "snap distance:" << distance << "\t maxDistance:" << maxDistance << "\t got data:" << this->estimateUpdated << "Int:" << this->intersecting << " ignore:" << this->ignoreIntersect << std::endl;
	if (this->intersecting == false && (distance > maxDistance || this->estimateUpdated == false)) {
		int maxIterations = 10;
		int stepSize = 3;
		int radius = 8.5 * this->cmInPixels;

		double pointQuality = this->getPointQuality(position, skinMask);
		//std::cout << "starting estimation:" << pointQuality << std::endl;
		//cv::circle(*this->rgbSkinMask, position, 9, CV_RGB(255, 100, 255), 3);
		if (pointQuality > 0.4) {
			this->position = this->lookAround(position, skinMask, maxIterations, stepSize, radius, FREE_SEARCH, 100);
			return true;
		}
	}
	return false;
}



/*
* This method will check if there is an intersection with the "other" hand. If so, move it to the side to total overlap is not attained.
*/
void Hand::checkIntersect(cv::Point otherHandPosition) {
	this->intersecting = false;
	int minimalDistance = 30;
	int dx = std::abs(this->position.x - otherHandPosition.x);
	int dy = std::abs(this->position.y - otherHandPosition.y);

	double distance = std::max(1.0,std::sqrt(dx*dx + dy*dy));

	if (distance < minimalDistance) {
		this->intersecting = true;
		if (dx < minimalDistance) {
			this->position.x += (minimalDistance - dx) * (this->leftHand ? -1 : 1);
		}
	}

	if (this->ignoreIntersect) {
		this->intersecting = false;
	}
	this->ignoreIntersect = false;
}



//***************************************** PRIVATE  **********************************************//


double Hand::getMean(cv::Point& pos, cv::Mat& skinMask, int radius) {
	cv::Mat mask = cv::Mat::zeros(skinMask.rows, skinMask.cols, CV_8U); // all 0
	cv::circle(mask, pos, radius, 255, CV_FILLED);
	cv::Mat result;
	cv::bitwise_and(skinMask, mask, result);
	return cv::sum(result)[0] / radius*radius*3.1415*255;
}

double Hand::shiftPosition(
	cv::Mat& skinMask,
	std::vector<cv::Point>& newPositions,
	cv::Point& basePosition, int xOffset, int yOffset, int radius) {

	cv::Point newPos = basePosition;
	newPos.x += xOffset;
	newPos.y += yOffset;
	newPositions.push_back(newPos);
	return this->getMean(newPos, skinMask, radius);
}


/*
* Explore the area around the blob for maximum coverage. This will center a circle within the blob (ideally).
*/
cv::Point Hand::lookAround(
	cv::Point start, 
	cv::Mat& skinMask, 
	int maxIterations,
	int stepSize, 
	int radius, 
	SearchMode searchMode, 
	int colorBase) {

	cv::Point maxPos = start;

	// Setup a search window to greatly speed up the search.
	int searchSpaceRadius = 100;
	int x = std::max(0, std::min(skinMask.cols, maxPos.x - searchSpaceRadius));
	int y = std::max(0, std::min(skinMask.rows, maxPos.y - searchSpaceRadius));
	int width = std::min(skinMask.cols, x + 2*searchSpaceRadius) - x;
	int height = std::min(skinMask.rows, y + 2*searchSpaceRadius) - y;
	cv::Rect searchRect(x, y, width, height);
	cv::Mat searchSpace = skinMask(searchRect);
	
	// DEBUG
	cv::circle(*this->rgbSkinMask, maxPos, radius, CV_RGB(255, colorBase, 0), 2);
	cv::circle(*this->rgbSkinMask, maxPos, 3, CV_RGB(0, 80, 180), 5);

	// Offet the position by the searchwindow
	maxPos.x -= x;
	maxPos.y -= y;

	// get the initial estimate.
	double maxValue = this->getMean(maxPos, searchSpace, radius);

	// search in a box
	std::vector<double> newValues;
	std::vector<cv::Point>  newPositions;
	for (int i = 0; i < maxIterations; i++) {
		// down is POSITIVE y distance
		if (searchMode == ONLY_SEARCH_DOWN || searchMode == FREE_SEARCH) {
			// searching down
			newValues.push_back(this->shiftPosition(searchSpace, newPositions, maxPos, stepSize,  stepSize, radius));
			newValues.push_back(this->shiftPosition(searchSpace, newPositions, maxPos, 0,		  stepSize, radius));
			newValues.push_back(this->shiftPosition(searchSpace, newPositions, maxPos, -stepSize, stepSize, radius));
		}
		if (searchMode == ONLY_SEARCH_UP || searchMode == FREE_SEARCH) {
			// searching up
			newValues.push_back(this->shiftPosition(searchSpace, newPositions, maxPos, stepSize,  -stepSize, radius));
			newValues.push_back(this->shiftPosition(searchSpace, newPositions, maxPos, 0,		  -stepSize, radius));
			newValues.push_back(this->shiftPosition(searchSpace, newPositions, maxPos, -stepSize, -stepSize, radius));
		}
		// search left right
		newValues.push_back(this->shiftPosition(searchSpace, newPositions, maxPos,  stepSize, 0, radius));
		newValues.push_back(this->shiftPosition(searchSpace, newPositions, maxPos, -stepSize, 0, radius));
	
		double newMax = 0;
		int maxIndex = 0;
		for (int j = 0; j < newValues.size(); j++) {
			if (newMax < newValues[j]) {
				newMax = newValues[j];
				maxIndex = j;
			}
		}

		if (newMax >= maxValue) {
			maxValue = newMax;
			cv::Point drawPoint(maxPos.x + x, maxPos.y + y);
			cv::circle(*this->rgbSkinMask, drawPoint, 2, CV_RGB(colorBase, 0, std::min(30 * i, 255)), 5);
			maxPos = newPositions[maxIndex];
		}
		else {
			break;
		}
	}

	// restore the transformation of the coordinates
	maxPos.x += x;
	maxPos.y += y;

	if (searchMode == FREE_SEARCH) {
		cv::circle(*this->rgbSkinMask, maxPos, radius, CV_RGB(255, 0, 0), 2);
	}
	else if (searchMode == ONLY_SEARCH_DOWN) {
		cv::circle(*this->rgbSkinMask, maxPos, radius, CV_RGB(0, 255, 0), 2);
	}
	else {
		cv::circle(*this->rgbSkinMask, maxPos, radius, CV_RGB(0, 0, 255), 2);
	}

	// return best position
	return maxPos;
}

double Hand::getPointQuality(cv::Point& point, cv::Mat& skinMask) {
	int radius = 5 * this->cmInPixels; // cm
	cv::Scalar maxValue = this->getMean(point, skinMask, radius);
	return maxValue[0] / 255.0;
}