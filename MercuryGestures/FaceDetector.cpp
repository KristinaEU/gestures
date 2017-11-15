#pragma once

#include "MercuryCore.h"
#include "FaceDetector.h"

FaceDetector::FaceDetector() {
	this->reset();
}
FaceDetector::~FaceDetector() {}

/**
 * Setting up the cascade classifier(s).
 */
bool FaceDetector::setup() {
	if (!this->face_cascade.load(this->face_cascade_name)) {
		std::cerr << "--(!)Error loading face cascade" << std::endl;
		return false;
	};
	return true;
}

void FaceDetector::updateScale() {
	double pixelSizeInCmTemp = averageFaceHeight / this->face.rect.height;

	// we will normalize the human on screen to 0.25 cm / pixel
	this->normalizationFactorSetup += (pixelSizeInCmTemp / normalizationScale) * (pixelSizeInCmTemp / normalizationScale);
	this->normalizationIterationsCount++;

	if (this->normalizationIterationsCount == this->normalizationIterations) {
		this->normalizationFactor = (normalizationFactorSetup / double(normalizationIterations));
		this->pixelSizeInCm = 0.25 * normalizationFactor; // get the pixel size in centimeters
	}
}

/*
This will be done after the calculation is complete. It will draw a rectangle a bit larger than the face on the canvas
*/
void FaceDetector::addResultToMask(cv::Mat& mask) {
	cv::Rect inflatedHead = inflateRect(this->face.rect, 10, mask);
	cv::rectangle(mask, inflatedHead, 255, CV_FILLED);
}

/*
This will be done after the calculation is complete. It will draw a rectangle a bit larger than the face on the canvas
*/
void FaceDetector::draw(cv::Mat& canvas) {
	cv::rectangle(canvas, this->face.rect, CV_RGB(0,255,0), 2);
}

/**
* This detects faces. It assumes only one face will be in view. It will draw the boundaries of the expected position of
* user body parts. This can be used to make a mask to sample only important parts of the image for movement. It can also
* be used to ignore the head movement or upper torso.
*/
bool FaceDetector::detectFace(cv::Mat& grayscaleImage, FaceData & data) {
	std::vector<cv::Rect> faces;
	int minFaceSize = 0.2 * this->frameHeight;
	this->face_cascade.detectMultiScale(grayscaleImage, faces, 1.3, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(minFaceSize, minFaceSize));
	data.count = faces.size();
	if (faces.size() > 0) {
		data.rect = faces[0];
		return true;
	}
	return false;
}


/*
 * Detect a face in a grayscale image.
 */
bool FaceDetector::detect(cv::Mat& gray) {
//	auto start = std::chrono::high_resolution_clock::now();
	// face detection for normalization
	FaceData newFaces;
	float movementThreshold = this->faceAreaThresholdFactor * this->frameWidth;
	bool detected = false;
	if (this->faceLocked) {
		SearchSpace space;
		getSearchSpace(space, gray, this->face.rect, 50);

		// detect the face in the grayscale image
		detected = this->detectFace(space.mat, newFaces);
		if (detected) {
			// restore original coordinates
			fromSearchSpace(space, newFaces.rect);
		}
		else {
			// fallback
			detected = this->detectFace(gray, newFaces);
		}
	}
	else {
		// detect in full image
		detected = this->detectFace(gray, newFaces);
	}

//    auto elapsed = std::chrono::high_resolution_clock::now() - start;
//	double duration = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / 1000.0;

//	std::cout<<"Facedetector runtime "<<duration<<" ms ("<<detected<<")";

	if (detected) {
		this->badReadings = 0;

		// check if the face is far away from the previous place it was detected
		bool faceInArea = std::abs(getCenterX(newFaces.rect) - this->faceCenterX) < movementThreshold &&
			std::abs(getCenterY(newFaces.rect) - this->faceCenterY) < movementThreshold;

		// need a minimum amount of good readings to accept that this is the location of the face.
		if (faceInArea)
			this->goodReadings += 1;
		else
			this->goodReadings = 0;

		// lock the position if we have enough good readings
		if (this->goodReadings > faceReadingThreshold)
			this->faceLocked = true;

		// if it was a good measurement we store it in the class
		if (newFaces.count == 1 && (this->faceLocked == false || (this->faceLocked && faceInArea))) {
			this->face.count = newFaces.count;
			this->face.rect = newFaces.rect;
		}

		// if we have a face, update the position and return true: we have something to work with.
		if (this->face.count != 0) {
			this->faceCenterX = getCenterX(this->face.rect);
			this->faceCenterY = getCenterY(this->face.rect);

			// update the scale of the face.
			this->updateScale();

			// update history
            this->positionIndex = this->getNextIndex(this->positionIndex);
            cv::Point aux = cv::Point(this->faceCenterX, this->faceCenterY);
            (this->positionHistory)[this->positionIndex] = aux;

			return true;
		}
		return false;
	}
	else {
		// we allow using the old face position for this->failureThreshold frames before reverting to failed state.
		this->goodReadings = 0;
		this->badReadings += 1;
		if (this->badReadings < this->failureThreshold && this->face.count != 0) {
			// this uses the old face
			return true;
		}
		return false;
	}
}

void FaceDetector::setVideoProperties(int frameWidth, int frameHeight) {
	this->frameHeight = frameHeight;
	this->frameWidth = frameWidth;
}

void FaceDetector::reset() {
	this->faceLocked = false;
	this->face.count = 0;
	this->goodReadings = 0;
	this->badReadings = 0;
}

// History is a deck. Get the index:
int FaceDetector::getNextIndex(int index) {
	return (index + 1) % this->historySize;
}

// History is a deck. Get the index:
int FaceDetector::getPreviousIndex(int index) {
	return (index - 1) < 0 ? index - 1 + this->historySize : index - 1;
}

