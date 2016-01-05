#include "MercuryCore.h"
/**
* This will generate a region of interest mask which will bitwise AND-ed with the movment results.
*/
void generateROImask(cv::Mat& mask, BodyRects& maskBody) {
	cv::rectangle(mask, maskBody.lowerTorso, 255, CV_FILLED);
	cv::rectangle(mask, maskBody.lap, 255, CV_FILLED);
	cv::rectangle(mask, maskBody.armLeftUpper, 255, CV_FILLED);
	cv::rectangle(mask, maskBody.armLeftLower, 255, CV_FILLED);
	cv::rectangle(mask, maskBody.armRightUpper, 255, CV_FILLED);
	cv::rectangle(mask, maskBody.armRightLower, 255, CV_FILLED);
}


/**
* This fills a body rect object
*/
void getBodyRect(cv::Rect &detectedFace, BodyRects &body) {
	// the detected face is more or less a square, we will correct the proportions
	body.face = cv::Rect(
		(0.5 * detectedFace.width * (1 - faceWidthToHeightRatio)) + detectedFace.x, // shift rect to correct for decrease in width
		detectedFace.y,
		detectedFace.width * faceWidthToHeightRatio, // correct the width to match human face proportions
		detectedFace.height
		);

	// we use the inverted value to avoid the multiple divisions in the block below.
	double pixelSizeInCmInv = body.face.height / averageFaceHeight;

	// based on http://i1201.photobucket.com/albums/bb342/Nantchev/Human_proportions_by_BenTs_sTock_zps3aa64481.jpg
	double armWidth = 15 * pixelSizeInCmInv; // wider than normal arms. If we want to mark an importance area this can be used
	double armUpperHeight = 35 * pixelSizeInCmInv;
	double armLowerHeight = 35 * pixelSizeInCmInv;
	double torsoWidth = 25 * pixelSizeInCmInv;
	double torsoHeight = 55 * pixelSizeInCmInv;
	double lapHeight = 50 * pixelSizeInCmInv; // to the end of the frame
	double neckHeight = 3 * pixelSizeInCmInv; // smaller than a neck would be because people may look down a bit

	body.upperTorso = cv::Rect(
		(body.face.x + 0.5 * body.face.width) - 0.5 * torsoWidth,
		body.face.y + body.face.height + neckHeight,
		torsoWidth,
		torsoHeight * 0.6
		);
	body.lowerTorso = cv::Rect(
		(body.face.x + 0.5 * body.face.width) - 0.5 * torsoWidth,
		body.face.y + body.face.height + neckHeight + torsoHeight * 0.6,
		torsoWidth,
		torsoHeight * 0.4
		);
	body.lap = cv::Rect(
		(body.face.x + 0.5 * body.face.width) - 0.5 * torsoWidth - armWidth,
		body.face.y + body.face.height + neckHeight + torsoHeight,
		torsoWidth + 2 * armWidth,
		lapHeight
		);
	body.armLeftUpper = cv::Rect(
		(body.face.x + 0.5 * body.face.width) - 0.5 * torsoWidth - armWidth,
		body.face.y + body.face.height,
		armWidth,
		armUpperHeight
		);
	body.armLeftLower = cv::Rect(
		(body.face.x + 0.5 * body.face.width) - 0.5 * torsoWidth - armWidth,
		body.face.y + body.face.height + armUpperHeight,
		armWidth,
		armLowerHeight
		);
	body.armRightUpper = cv::Rect(
		(body.face.x + 0.5 * body.face.width) + 0.5 * torsoWidth,
		body.face.y + body.face.height,
		armWidth,
		armUpperHeight
		);
	body.armRightLower = cv::Rect(
		(body.face.x + 0.5 * body.face.width) + 0.5 * torsoWidth,
		body.face.y + body.face.height + armUpperHeight,
		armWidth,
		armLowerHeight
		);
}
/**
* using the detected face, we draw the expected positions of the user's body. This can be expanded to generate a mask for the
* regions of interest.
*
* @return the pixel size in centimeters
*/
void drawBodyRects(BodyRects &body, cv::Mat &frame) {
	// draw the rects
	cv::rectangle(frame, body.face, CV_RGB(0, 0, 250));
	cv::rectangle(frame, body.upperTorso, CV_RGB(0, 0, 255));
	cv::rectangle(frame, body.lowerTorso, CV_RGB(0, 0, 255));
	cv::rectangle(frame, body.lap, CV_RGB(0, 255, 0));
	cv::rectangle(frame, body.armLeftUpper, CV_RGB(255, 0, 0));
	cv::rectangle(frame, body.armLeftLower, CV_RGB(255, 0, 0));
	cv::rectangle(frame, body.armRightUpper, CV_RGB(255, 0, 0));
	cv::rectangle(frame, body.armRightLower, CV_RGB(255, 0, 0));
}