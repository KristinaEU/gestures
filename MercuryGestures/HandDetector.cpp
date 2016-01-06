#pragma once
#include "MercuryCore.h"
#include "HandDetector.h"


/************************************  UTIL  *********************************************/

/*
* Get the amount of edges inside of a blob as an integer
*/
BlobEdgeData getEdgeData(BlobInformation& blob, cv::Mat& edges) {
	// the data to fill
	BlobEdgeData data;

	// draw a mask of the blob
	cv::Mat blobMask = cv::Mat::zeros(edges.rows, edges.cols, edges.type());
	std::vector<cv::Vec4i> hierarchy;
	std::vector<std::vector<cv::Point>> contours;
	contours.push_back(blob.contour);
	cv::drawContours(blobMask, contours, 0, 255, CV_FILLED, 8, hierarchy, 0, cv::Point());

	// store the size of the data
	data.size = cv::sum(blobMask)[0] / 255;

	// make the mask a little smaller so we do not use the outer edges of the blob
	erode(blobMask, blobMask, 6);

	// use the AND operation to only get the edges
	cv::Mat combined;
	cv::bitwise_and(edges, blobMask, combined);
	data.edgeCount = cv::sum(combined)[0] / 255;

	return data;
}

/************************************ /UTIL  *********************************************/


/*
* Tell the hands which one is left and right, give them specific colors for drawing, set the fps.
*/
HandDetector::HandDetector(int fps) {
	this->fps = fps;
	this->leftHand.leftHand = true;
	this->leftHand.fps = fps;
	this->leftHand.color = CV_RGB(0, 150, 255);
	this->leftHand.rgbSkinMask = &this->rgbSkinMask;
	
	this->rightHand.fps = fps;
	this->rightHand.color = CV_RGB(0, 255, 0);
	this->rightHand.rgbSkinMask = &this->rgbSkinMask;
}
HandDetector::~HandDetector() {}


/*
* Get the amount of edges inside of a blob as an integer
*/
void HandDetector::setVideoProperties(int frameWidth, int frameHeight) {
	this->frameHeight = frameHeight;
	this->frameWidth = frameWidth;
}


/*
* This is the detector entree point. It does too much at the moment so it is in need of seperation. We get the blobs, filter them on size,
* segment them, judge them, forward them to the specific analysers
*/
void HandDetector::detect(cv::Rect& face, cv::Mat& skinMask, cv::Mat& movementMap, cv::Mat& edges, double pixelSizeInCm) {
	skinMask.copyTo(this->skinMask);
	
	// get an estimate for the center based on the face.
	int centerX = face.x + 0.5 * face.width;
	
	// set the cm->pixel conversion for the hands as well.
	this->cmInPixels = 1.0 / pixelSizeInCm;
	this->leftHand.cmInPixels = this->cmInPixels;
	this->rightHand.cmInPixels = this->cmInPixels;
	
	// important. These are used for vertical segmentation of blobs. 
	int bottomFace = face.y + face.height;
	int lowerBodyHalf = 0.6 * bottomFace + 0.4 * this->frameHeight;

	// give the bottom face information to the hands.
	this->leftHand.faceCoverageThreshold = 0.6 * bottomFace + 0.4 * this->frameHeight;;
	this->rightHand.faceCoverageThreshold = 0.4 * bottomFace + 0.6 * this->frameHeight;;

#ifdef DEBUG
	cv::cvtColor(this->skinMask, this->rgbSkinMask, CV_GRAY2RGB);
	int leftX = centerX - 50 * cmInPixels;
	int rightX = centerX + 50 * cmInPixels;
	cv::line(this->rgbSkinMask, cv::Point(centerX, 0), cv::Point(centerX, this->frameHeight),CV_RGB(255, 0, 0));
	cv::line(this->rgbSkinMask, cv::Point(leftX, 0),   cv::Point(leftX, this->frameHeight),CV_RGB(255, 0, 255));
	cv::line(this->rgbSkinMask, cv::Point(rightX, 0),  cv::Point(rightX, this->frameHeight),CV_RGB(255, 0, 255));
	cv::line(this->rgbSkinMask, cv::Point(0, bottomFace),cv::Point(this->frameWidth, bottomFace),CV_RGB(255, 0, 0));
	cv::line(this->rgbSkinMask, cv::Point(0, lowerBodyHalf), cv::Point(this->frameWidth, lowerBodyHalf), CV_RGB(255, 255, 0));
#endif

	// we detect contours twice, once to draw in full, once to detect. This is done to avoid gaps in contours or contours in contours
	cv::Mat filledContours;
	skinMask.copyTo(filledContours);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	
	// first find.
	cv::findContours(filledContours, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	// 36cm^2 --> decent hand size measurement
	double minContour = 6 * 6 * cmInPixels * cmInPixels; 

	// draw first pass
	for (int i = 0; i < contours.size(); i++) {
		if (cv::contourArea(contours[i]) > minContour) {
			cv::drawContours(filledContours, contours, i, 255, CV_FILLED, 8, hierarchy, 0, cv::Point());
		}
	}

	dilate(filledContours, filledContours);
	erode(filledContours, filledContours);

	// start seconds pass
	contours.clear();
	hierarchy.clear();
	cv::findContours(filledContours, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	// mask to remove faces and other irrelevant blobs from the internal skinmask. Is trained over time.
	cv::Mat highBlobsMask = cv::Mat::zeros(this->skinMask.rows, this->skinMask.cols, this->skinMask.type()); // all 0
	// create a mat for the edgemask.
	cv::Mat edgeMask;
	cv::Mat movementSkinMask;

	// analyze all blobs
	int rangeLeftX = this->frameWidth;
	int rangeRightX = 0;
	std::vector<BlobInformation> lowerBodyBlobs;
	std::vector<BlobInformation> midBodyBlobs;
	std::vector<BlobInformation> highBlobs;
	std::vector<BlobInformation> blobs;
	for (int i = 0; i < contours.size(); i++) {
		if (cv::contourArea(contours[i]) > minContour) {
			cv::Point lowestPoint(0, 0);
			cv::Point leftMostPoint(this->frameWidth, 0);
			cv::Point rightMostPoint(0, 0);
			cv::Point highestPoint(this->frameWidth, this->frameHeight);
			for (int j = 0; j < contours[i].size(); j++) {
				if (contours[i][j].y > lowestPoint.y)
					lowestPoint = contours[i][j];
				if (contours[i][j].y < highestPoint.y)
					highestPoint = contours[i][j];
				if (contours[i][j].x < leftMostPoint.x)
					leftMostPoint = contours[i][j];
				if (contours[i][j].x > rightMostPoint.x)
					rightMostPoint = contours[i][j];
			}
			if (leftMostPoint.x < rangeLeftX)
				rangeLeftX = leftMostPoint.x;
			if (rightMostPoint.x > rangeRightX)
				rangeRightX = rightMostPoint.x;

			auto moment = cv::moments(contours[i], false);
			auto midPoint = cv::Point2f(moment.m10 / moment.m00, moment.m01 / moment.m00);
	
			BlobInformation blob;
			blob.index = i;
			blob.top = highestPoint;
			blob.left = leftMostPoint;
			blob.right = rightMostPoint;
			blob.bottom = lowestPoint;
			blob.center = midPoint;
			blob.type = OTHER;
			blob.contour = contours[i];

			auto color = CV_RGB(255, 0, 0);
			// the highest point is below the chest line
			if (highestPoint.y > lowerBodyHalf) { 
				color = CV_RGB(255, 255, 0);  // yellow
				lowerBodyBlobs.push_back(blob);
				blob.type = LOW;
			}
			// the lowest point is below the chest line and the highest below the chin line
			else if (lowestPoint.y > lowerBodyHalf && highestPoint.y > bottomFace) { 
				color = CV_RGB(255, 100, 50); // orange
				midBodyBlobs.push_back(blob);
				blob.type = MEDIUM;
			}
			// the lowest and highest points are below the chin and above the chest line
			else if (lowestPoint.y > bottomFace && highestPoint.y > bottomFace) {
				color = CV_RGB(150, 50, 150); // dark purple
				midBodyBlobs.push_back(blob);
				blob.type = MEDIUM;
			}
			// only the lowest point is below the chest line, the highest is above the chin line... stupid blob
			else if (lowestPoint.y > lowerBodyHalf && highestPoint.y < bottomFace) {
				color = CV_RGB(255, 0, 0); // red
				blob.type = HIGH;
	  		    highBlobs.push_back(blob);
			}
			// the highest point is above the chin line
			else  if (highestPoint.y < bottomFace) {
				color = CV_RGB(150, 00, 0);  // dark red
				blob.type = HIGH;
				highBlobs.push_back(blob);
				
				cv::drawContours(highBlobsMask, contours, i, 255, CV_FILLED, 8, hierarchy, 0, cv::Point());
			}
#ifdef DEBUG
			// draw the blobs on the rgb skin mask we use for debugging.
			cv::drawContours(this->rgbSkinMask, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
#endif
			blobs.push_back(blob);
		}
	}

	// improve the estimate of the centerX
	centerX = 0.5 * (centerX + 0.5 * (rangeLeftX + rangeRightX));

#ifdef DEBUG
	cv::line(this->rgbSkinMask, cv::Point(centerX, 0), cv::Point(centerX, this->frameHeight), CV_RGB(255, 255, 0));		
#endif

	// update the face mask and process it
	this->updateFaceMask(highBlobsMask);
	cv::subtract(this->skinMask, this->faceMask, this->skinMask);

	// use the updated skin mask to get the edges inside of the blobs
	cv::bitwise_and(edges, this->skinMask, edgeMask);
	cv::bitwise_or(this->skinMask, movementMap, movementSkinMask);
	
	// case 1: 1 blob in the lower segment.
	if (lowerBodyBlobs.size() == 1) {
		// if there is one in the lower range and one in the higher range
		if (midBodyBlobs.size() == 1) { // todo: put back?
			std::vector<BlobInformation> blobContainer;
			blobContainer.push_back(lowerBodyBlobs[0]);
			blobContainer.push_back(midBodyBlobs[0]);
			this->updateHandsFromNBlobsWithAnalysis(blobContainer, edges);
		}
		// if we have one is the lower segment and one in the middle segment, assume these are both hands.
		else {
			this->getBothHandPositionsFromBlob(lowerBodyBlobs[0]);
		}
		
	}
	// case 2: 2 blobs in the lower segment.
	else if (lowerBodyBlobs.size() == 2) {
		this->updateHandsFromTwoBlobs(lowerBodyBlobs[0], lowerBodyBlobs[1]);
	}
	// case 3: more blobs in the lower segment.
	else if (lowerBodyBlobs.size() > 2) {
		this->updateHandsFromNBlobsWithAnalysis(lowerBodyBlobs, edges);
	}
	// case 4: one blob in the middle segment
	else if (midBodyBlobs.size() == 1) {
		this->getBothHandPositionsFromBlob(midBodyBlobs[0]);
	}
	// case 5: two blobs in the middle segment.
	else if (midBodyBlobs.size() == 2) {
		this->updateHandsFromTwoBlobs(midBodyBlobs[0], midBodyBlobs[1]);
	}
	// case 6: mote than 2 blobs in the middle segment
	else if (midBodyBlobs.size() > 2) {
		// case 1:
		// hands on the side, something in between --> segment leftmost and rightmost?
		this->updateHandsFromNBlobsWithAnalysis(midBodyBlobs, edges);
	}
	// case 7: only one blob in the high segment (hands over face? no hands?)
	else if (highBlobs.size() == 1) {
		this->getBothHandPositionsFromBlob(highBlobs[0], false, ONLY_HEAD);
	}
	// case 8: two blobs in the high segment. (one hand over face?)
	else if (highBlobs.size() == 2) {
		// should not be handled, should be taken care of by tracking TODO: check!
	}
	// case 9: multiple blobs in the higher segment	
	else if (highBlobs.size() > 2) {
		this->updateHandsFromNBlobsByPosition(highBlobs);
	}

	// solve for the hands
	this->leftHand.solve( this->skinMask, blobs, movementMap);
	this->rightHand.solve(this->skinMask, blobs, movementMap);

	this->leftHand.handleIntersection(this->rightHand.position, this->skinMask);
	this->rightHand.handleIntersection(this->leftHand.position, this->skinMask);
}



/*
* Get the amount of edges inside of a blob as an integer
*/
void HandDetector::updateFaceMask(cv::Mat& highBlobsMask) {
	cv::Scalar highArea = cv::sum(highBlobsMask);
	double faceMaskThreshold = 0.01;

	// init
	if (this->faceMaskAverageArea == 0) {
		this->faceMaskAverageArea = highArea[0];
	}

	// if it is a close enough match, update the mask. 
	if (std::abs(highArea[0] - faceMaskAverageArea) < faceMaskAverageArea * faceMaskThreshold) {
		this->faceMaskAverageArea = 0.8 *  this->faceMaskAverageArea + 0.2 * highArea[0];
		highBlobsMask.copyTo(this->faceMask);
	}
	// if it is not a match, update the average slowly as a fallback mechanism. Should converge in about 200 frames.
	else {
		this->faceMaskAverageArea = 0.995 *  this->faceMaskAverageArea + 0.005 * highArea[0];
	}
}


/*
* Get the amount of edges inside of a blob as an integer
*/
void HandDetector::draw(cv::Mat& canvas) {
	this->leftHand.draw(canvas);
	this->rightHand.draw(canvas);
}

// show the debug map
void HandDetector::show(std::string windowName) {
	cv::imshow(windowName, this->rgbSkinMask);
}




//***************************************** PRIVATE  **********************************************//


/*
* This will return a factor between 0 and upperbound based on the height of the blob. A tall blob ( >= 25cm ) will give a value closer to upperbound.
* This is a linear curve starting from 0 at 10cm to upperbound at 25cm length.
*/
double HandDetector::getFactor(BlobInformation& blob, double upperBound) {
	double diff = blob.bottom.y - blob.top.y;
	double a10cm = 10 * this->cmInPixels;
	double a25cm = 25 * this->cmInPixels;
	return std::min(upperBound, std::max(0.0, upperBound*((diff - a10cm) / (a25cm - a10cm))));
}


/*
* Here we provide a blob that we think contains a hand. We will try to get a decent estimate of the position here.
* The tracking is done inside the hand object.
*/
void HandDetector::getHandEstimateFromBlob(BlobInformation& blob, Hand& hand, bool ignoreIntersection) {
	// if the blob is long, we pull the hand estimates towards the lowest point.
	double factor = getFactor(blob, 0.5);
	cv::Point estimate;
	estimate.x = factor * blob.bottom.x + (1 - factor) * blob.center.x;
	estimate.y = factor * blob.bottom.y + (1 - factor) * blob.center.y;
	hand.setEstimate(estimate, blob, ignoreIntersection);
}



/*
* Here we provide a blob that we think contains two hands.  We will try to get a decent estimate of the position here.
*/
void HandDetector::getBothHandPositionsFromBlob(BlobInformation& blob, bool ignoreIntersection, Condition condition) {
	auto leftEstimate = blob.center;
	auto rightEstimate = blob.center;

	leftEstimate.x = blob.center.x + 7.5 * this->cmInPixels;
	rightEstimate.x = blob.center.x - 7.5 * this->cmInPixels;

	// if the blob is long, we pull the hand estimates towards the lowest point.
	double factor = getFactor(blob, 0.5);

	leftEstimate.y = factor * blob.bottom.y + (1 - factor) * leftEstimate.y;
	rightEstimate.y = factor * blob.bottom.y + (1 - factor) * rightEstimate.y;

	this->leftHand.setEstimate(leftEstimate, blob, ignoreIntersection, condition);
	this->rightHand.setEstimate(rightEstimate, blob, ignoreIntersection, condition);

	//cv::circle(this->rgbSkinMask, blob.center, 5, CV_RGB(255, 0, 0), 5);
}


/*
* We have 2 blobs, based on left and right we estimate the hand position.
*/
void HandDetector::updateHandsFromTwoBlobs(BlobInformation& blob1, BlobInformation& blob2, bool ignoreIntersection) {
	if (blob1.center.x < blob2.center.x) {
		this->getHandEstimateFromBlob(blob1, this->rightHand, ignoreIntersection);
		this->getHandEstimateFromBlob(blob2, this->leftHand, ignoreIntersection);
	}
	else {
		this->getHandEstimateFromBlob(blob1, this->leftHand, ignoreIntersection);
		this->getHandEstimateFromBlob(blob2, this->rightHand, ignoreIntersection);
	}
}


/*
* We have N blobs, based on leftmost and rightmost we estimate the hand position
*/
void HandDetector::updateHandsFromNBlobsByPosition(std::vector<BlobInformation>& blobs, bool ignoreIntersection) {
	int leftIndex = 0;
	int rightIndex = 0;
	for (int i = 0; i < blobs.size(); i++) {
		if (blobs[leftIndex].center.x < blobs[i].center.x) {
			leftIndex = i;
		}
		if (blobs[rightIndex].center.x > blobs[i].center.x) {
			rightIndex = i;
		}
	}

	this->getHandEstimateFromBlob(blobs[leftIndex], this->leftHand, ignoreIntersection);
	this->getHandEstimateFromBlob(blobs[rightIndex], this->rightHand, ignoreIntersection);
}

/*
* We have N blobs, based on position and edgecount we estimate the location of the hands
*/
void HandDetector::updateHandsFromNBlobsWithAnalysis(std::vector<BlobInformation>& blobs, cv::Mat& edges) {
	std::vector<BlobEdgeData> allBlobs;
	std::vector<BlobEdgeData> possibleHands;

	int maxEdgeCount = 0;
	int handEdgeThreshold = 200; // we assume a hand has at least some edges due to fingers, nails, shadows etc.
	double averageSize = 0;

	// get the data (size & edgecount) for all blobs.
	for (int i = 0; i < blobs.size(); i++) {
		BlobEdgeData data = getEdgeData(blobs[i], edges);
		data.index = i;
		averageSize += data.size;
		maxEdgeCount = std::max(maxEdgeCount, data.edgeCount);

		if (data.edgeCount > handEdgeThreshold) {
			possibleHands.push_back(data);
		}

		allBlobs.push_back(data);
	}
	averageSize /= blobs.size();


	// no blob has enough edges to be a hand.
	if (possibleHands.size() == 0) {
		// 1. some much bigger than others?
		double varSize = 0;
		for (int i = 0; i < allBlobs.size(); i++) {
			varSize += std::pow(allBlobs[i].size - averageSize, 2);
		}
		double stdSize = std::sqrt(varSize);

		// get a new list of possible blobs based on the size.
		for (int i = 0; i < allBlobs.size(); i++) {
			if (allBlobs[i].size - averageSize > stdSize) {
				possibleHands.push_back(allBlobs[i]);
			}
		}

		// 2. if that did not work.. sort by position.
		if (possibleHands.size() == 0) {
			this->updateHandsFromNBlobsByPosition(blobs, true);
		}
	}

	// We have some candidates for hands!
	if (possibleHands.size() == 1) {
		this->getBothHandPositionsFromBlob(blobs[possibleHands[0].index], true);
	}
	else if (possibleHands.size() == 2) {
		// this could be a decent estimate so we disable the ignoreIntersection.
		// TODO: maybe give a score for these blobs?
		this->updateHandsFromTwoBlobs(blobs[possibleHands[0].index], blobs[possibleHands[1].index], false);
	}
	else if (possibleHands.size() > 2) {
		// fall back to sorting by position.
		std::vector<BlobInformation> subset;
		for (int i = 0; i < possibleHands.size(); i++) {
			subset.push_back(blobs[possibleHands[i].index]);
		}
		this->updateHandsFromNBlobsByPosition(subset, true);
	}
}
