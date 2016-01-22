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
	MovementDetector ROImovementDetector(fps);
	ActivityGraph activityGraph(fps);
	FaceDetector  faceDetector;
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

	activityGraph.addChannel("Skin masked Movement", CV_RGB(0, 255, 0), 0.0);
	activityGraph.addChannel("ROI masked Movement", CV_RGB(255, 0, 0), 0.0);

	for (;;) {
		// get a new video frame
		cap >> rawFrame;

		// debug time elapsed
		auto start = std::chrono::high_resolution_clock::now();

		// check for end of video file.
		if (rawFrame.empty()) { break; }
		int frameWidth  = rawFrame.cols;
		int frameHeight = rawFrame.rows;

		// resize image
		double resizeFactor = frameHeightMax / double(frameHeight);
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
		double pixelSizeInCm = faceDetector.pixelSizeInCm;
		if (faceDetected) {
			auto face = &(faceDetector.face.rect);
			skinDetector.detect(*face, frame, initialized, (3.0 / pixelSizeInCm) * 4);
			edgeDetector.detect(gray);
			
			if (initialized) {
				cv::Mat temporalSkinMask = skinDetector.getMergedMap();
				cv::Mat roiMask = cv::Mat::zeros(temporalSkinMask.rows, temporalSkinMask.cols, temporalSkinMask.type()); // all 0

				// get an initial motion estimate based on the temporal skin mask alone. This is used 
				// in the hand detection
				movementDetector.detect(gray, grayPrev);
				movementDetector.mask(temporalSkinMask);
				movementDetector.calculate(faceDetector.normalizationFactor);

				handDetector.detect(
					gray, grayPrev,
					*face,
					skinDetector.skinMask,
					movementDetector.movementMap,
					edgeDetector.detectedEdges,
					pixelSizeInCm
				);
				handDetector.draw(frame);
				handDetector.drawTraces(frame);
				// faceDetector.draw(frame);

				// create the ROI map with just the hands and the face. This would reduce the difference
				// between long and short sleeves.
				handDetector.addResultToMask(roiMask);
				faceDetector.addResultToMask(roiMask);
				cv::bitwise_and(temporalSkinMask, roiMask, temporalSkinMask);

				// detect movent only within the ROI areas.
				ROImovementDetector.detect(gray, grayPrev);
				ROImovementDetector.mask(temporalSkinMask);
				ROImovementDetector.calculate(faceDetector.normalizationFactor);

				// draw the graph (optional);
				activityGraph.setValue("Skin masked Movement", movementDetector.value);
				activityGraph.setValue("ROI masked Movement", ROImovementDetector.value);
				activityGraph.draw(frame);
				
				//movementDetector.show("maskedSkinMovement");
				//skinDetector.show();
				//edgeDetector.show();
				//handDetector.show();

				// ----------  THIS IS THE VALUE TO PUBLISH TO SSI:  ------------- //
				//																   //
				double publishValue = ROImovementDetector.value;				   //
				//																   //
				// ----------  THIS IS THE VALUE TO PUBLISH TO SSI   ------------- //
			}
			initialized = true;
		}
		else {
			faceDetector.reset();
			handDetector.reset();
			activityGraph.setValue("Skin masked Movement", 0.0);
			activityGraph.setValue("ROI masked Movement", 0.0);
			initialized = false;
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
		double duration = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count() / 1000.0;

		// prepare for next loop
		cv::putText(frame, joinString("f:", frameIndex), cv::Point(frameWidth - 110, 30), 0, 1, CV_RGB(255, 0, 0), 2);
		cv::putText(frame, joinString(joinString("t:", duration)," ms"), cv::Point(frameWidth - 110, 50), 0, 0.5, CV_RGB(255, 0, 0), 1);
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
	}

	// the camera will be deinitialized automatically in VideoCapture destructor
	return 1;
}



void manage(int movieIndex) {
	std::vector<std::string> videoList;
	videoList.push_back("de001_spk02f.mp4");
	videoList.push_back("de003_spk01f.mp4");
	videoList.push_back("de004_spk01f.mp4");
	videoList.push_back("de005_spk03f.mp4");
	videoList.push_back("de007_spk02f.mp4");
	videoList.push_back("de011_spk04f.mp4");
	videoList.push_back("de012_spk03f.mp4");
	videoList.push_back("de014_spk03f.mp4");
	videoList.push_back("de015_spk04f.mp4");
	videoList.push_back("de016_spk03f.mp4");
	videoList.push_back("de017_spk03f.mp4");
	videoList.push_back("de018_spk03f.mp4");
	videoList.push_back("de019_spk01.mp4");
	videoList.push_back("de020_spk01m.mp4");
	videoList.push_back("de021_spk05f.mp4");
	videoList.push_back("de023_spk02f.mp4");
	videoList.push_back("de024_spk02m.mp4");
	videoList.push_back("de028_spk02m.mp4");
	videoList.push_back("de031_spk03f.mp4");
	videoList.push_back("de033_spk02m.mp4");
	videoList.push_back("de033_spk03f.mp4");
	videoList.push_back("es008_spk02f.mp4");
	videoList.push_back("es008_spk03m.mp4");
	videoList.push_back("es012_spk02f.mp4");
	videoList.push_back("es012_spk03m.mp4");
	videoList.push_back("es015_spk02f.mp4");
	videoList.push_back("es017_spk03f.mp4");
	videoList.push_back("es018_spk03f.mp4");
	videoList.push_back("es021_spk04f.mp4");
	videoList.push_back("es024_spk05f.mp4");
	videoList.push_back("es025_spk06f.mp4");
	videoList.push_back("es026_spk06f.mp4");
	videoList.push_back("es028_spk05f.mp4");
	
	int amountOfMovies = videoList.size();
	cv::VideoCapture cap;
	
	cap.open(joinString("./media/", videoList[movieIndex]));
	//cap.open(0);
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
	else {
		// if 0, repeat movie
	}
	

	// make sure the cycle of movies is from 0 to amountOfMovies
	newIndex = newIndex % amountOfMovies;
	newIndex = newIndex < 0 ? newIndex + amountOfMovies : newIndex;
	
	cap.release();

	manage(newIndex);
}

int main(int argc, char *argv[]) {
	manage(20);
	return 0;
}