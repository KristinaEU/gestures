#include "MercuryCore.h"
#include "FaceDetector.h"
#include "ActivityGraph.h"
#include "EdgeDetector.h"
#include "MovementDetector.h"
#include "SkinDetector.h"
#include "HandDetector.h"

/*
* Run the algorithm on this video feed
*/
int run(cv::VideoCapture& cap, int fps) {
	// init the classes
	
	SkinDetector  skinDetector;
	EdgeDetector  edgeDetector;
	HandDetector  handDetector(fps);
	MovementDetector movementDetector(fps);
	MovementDetector maskedMovementDetector(fps);
	ActivityGraph activityGraph(fps);
	FaceDetector  faceDetector(fps);
	if (faceDetector.setup() == false)
		return -1;

	// setup the base collection of cvMats
	cv::Mat rawFrame;
	cv::Mat frame;
	cv::Mat gray;
	cv::Mat grayPrev;

	int frameHeightMax = 400;
	
	BodyRects body;

	int frameIndex = 0;
	bool initialized = false;
	

	// DEBUG
	int waitTime = 2;
	int skip = 0;
	int calcSkip = 0;

	activityGraph.addChannel("movement", CV_RGB(255, 0, 0));
	activityGraph.addChannel("maskedMovement", CV_RGB(0, 255, 0), 0.0);

	for (;;) {
		// debug time elapsed
		auto start = std::chrono::high_resolution_clock::now();

		// get a new video frame
		cap >> rawFrame;

		// check for end of video file.
		if (rawFrame.empty()) { break; }
		int frameWidth  = rawFrame.cols;
		int frameHeight = rawFrame.rows;

		// resize image
		float resizeFactor = frameHeightMax / double(frameHeight);
		cv::Size size(std::round(frameWidth * resizeFactor), frameHeightMax);
		cv::resize(rawFrame, frame, size); 
		frameWidth = frame.cols;
		frameHeight = frame.rows;

		// on the very first frame we initialize the classes
		if (frameIndex == 0) {
			faceDetector.setVideoProperties(frameWidth, frameHeight);
			handDetector.setVideoProperties(frameWidth, frameHeight);
			activityGraph.setVideoProperties(frameWidth, frameHeight);
		}

		// DEBUG
		if (skip > 0) {
			skip--;
			frameIndex++;
			std::cout << "skipping:" << frameIndex << std::endl;
			continue;
		}

		// convert frame to grayscale
		cv::cvtColor(frame, gray, CV_BGR2GRAY);

		// start detection of edges, face and skin
		bool faceDetected = faceDetector.detect(gray);
		double pixelSizeInCm = faceDetector.getScale();
		if (faceDetected) {
			auto face = &(faceDetector.face.rect);

			skinDetector.detect(*face, frame, initialized, (3.0 / pixelSizeInCm) * 4);
			
			//getBodyRect(*face, body);
			//drawBodyRects(body, frame);

			edgeDetector.detect(gray);
			
			if (initialized) {
				movementDetector.detect(gray, grayPrev);
				movementDetector.calculate(faceDetector.normalizationFactor);

				maskedMovementDetector.detect(gray, grayPrev);
				maskedMovementDetector.mask(skinDetector.getMergedMap());
				maskedMovementDetector.calculate(faceDetector.normalizationFactor);

				handDetector.detect(*face, skinDetector.skinMask, movementDetector.movementMap, edgeDetector.detectedEdges, pixelSizeInCm);
				handDetector.draw(frame);

				// draw the graph (optional);
				// activityGraph.setValue("movement", movementDetector.value);
				// activityGraph.setValue("maskedMovement", maskedMovementDetector.value);
				// activityGraph.draw(frame);
				
				//movementDetector.show("originalMovement");
				//maskedMovementDetector.show("maskedSkinMovement");
				//skinDetector.show();
				//edgeDetector.show();
				handDetector.show();
			}
			initialized = true;
		}

		// copy to buffer so we can do a difference check.
		gray.copyTo(grayPrev);

		// DEBUG
		if (calcSkip > 0) {
			calcSkip--;
			frameIndex++;
			std::cout << "hiding:" << frameIndex << std::endl;
			continue;
		}

		// debug time elapsed
		auto elapsed = std::chrono::high_resolution_clock::now() - start;
		long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
		if (frameIndex % 25 == 0) {
			std::cout << "processing time: " << microseconds / 1000.0 << " ms" << std::endl;
		}
	

		// prepare for next loop
		cv::imshow("frame", frame);
		int keystroke = cv::waitKey(waitTime);
		
		if (keystroke == 27) {
			std::cout << keystroke << std::endl;
			return 100;
			break;
		}
		else if (keystroke == 2424832) { // left arrow
			if (waitTime < 100) {
				return -1;
			}
		}
		else if (keystroke == 2555904) { // right arrow
			if (waitTime < 100) {
				return 1;
			}
		}
		else if (keystroke == 32) { // spacebar
			waitTime = 1e6;
		}
		else if (keystroke == 13) {  // return 
			waitTime = 2;
		}
		else if (keystroke == 115) { // s
			skip = 750;
			waitTime = 1e6;
		}
		else if (keystroke == 100) { // d
			calcSkip = 50;
		}
		else if (keystroke == 114) { // r
			return 0;
		}
		else if (keystroke >= 0) {
			std::cout << "key:" << keystroke << std::endl;
		}

		frameIndex += 1;
		//std::cout << frameIndex << std::endl;
	}

	// the camera will be deinitialized automatically in VideoCapture destructor
	return 1;
}



void manage(int movieIndex) {
	int amountOfMovies = 12;
	cv::VideoCapture cap(joinString(joinString("./media/", movieIndex), ".mp4"));

	// initialize video
	if (!cap.isOpened()) {
		std::cout << "Cannot open the video file" << std::endl;
		return;
	}

	// get the fps from the video for the graph time calculation
	int fps = cap.get(CV_CAP_PROP_FPS);
	if (fps <= 0 || fps > 60) {
		fps = 25;
		std::cout << "WARNING: COULD NOT GET FPS; Defaulting to 25fps." << std::endl;
	}

	// run the algorithm 
	int value = run(cap, fps);
	int newIndex = movieIndex;

	if (value == 1)		  // next movie
		newIndex += 1;
	else if (value == -1)  // previous movie
		newIndex -= 1;
	else if (value != 0) {
		return;  // quit
	}
	// if 0, repeat movie

	

	// make sure the cycle of movies is from 0 to amountOfMovies
	newIndex = newIndex % amountOfMovies;
	newIndex = newIndex < 0 ? newIndex + amountOfMovies : newIndex;
	
	cap.release();

	manage(newIndex);
}

int main(int argc, char *argv[]) {
	manage(6);

	return 0;
}