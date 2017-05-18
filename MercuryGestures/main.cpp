#include "MercuryCore.h"
#include "FaceDetector.h"
#include "ActivityGraph.h"
#include "EdgeDetector.h"
#include "MovementDetector.h"
#include "SkinDetector.h"
#include "HandDetector.h"

bool first = true;
/*
* Run the algorithm on this video feed
*/

int tekenMannequin(cv::Point leftHand, cv::Point rightHand, cv::Rect face, cv::Mat& drawMat, bool extend, int& LH, int& RH ) {
    // init the classes

    double pixelSizeInCm = 0.25;
    double cmInPixels = 1.0 / pixelSizeInCm;
    int frameWidth = drawMat.cols;
    int frameHeight = drawMat.rows;
    int centerX = face.x + 0.5 * face.width;
    int bottomFace = face.y + face.height;
    //int lowerBodyHalf = 0.6 * bottomFace + 0.4 * frameHeight;
    //int hips = lowerBodyHalf + bottomFace;
    int upperBodyHalf = bottomFace + face.height * 0.1;
    int lowerBodyHalf = upperBodyHalf + face.height * 1.1;
    int hips = lowerBodyHalf + face.height * 1.1;
    int leftX = centerX + face.width;
    int rightX = centerX - face.width;
    cv::Point elleboogL = cv::Point(leftX + face.width/2.5, lowerBodyHalf+5 * cmInPixels);
    cv::Point elleboogR = cv::Point(rightX - face.width/2.5, lowerBodyHalf+5 * cmInPixels);

    //cv::Rect rec5 = face;
    cv::Point facePoint(face.x+face.width/2, face.y+face.height/2);

    cv::line(drawMat, cv::Point(centerX, 0), cv::Point(centerX, frameWidth),CV_RGB(100, 124, 50));
    cv::line(drawMat, cv::Point(rightX, 0),   cv::Point(rightX, frameWidth),CV_RGB(255, 0, 255));
    cv::line(drawMat, cv::Point(leftX, 0),  cv::Point(leftX, frameWidth),CV_RGB(255, 0, 255));
    cv::line(drawMat, cv::Point(0, bottomFace),cv::Point(frameWidth, bottomFace),CV_RGB(255, 0, 0));
    cv::line(drawMat, cv::Point(0, lowerBodyHalf), cv::Point(frameWidth, lowerBodyHalf), CV_RGB(255, 152, 0));
    cv::line(drawMat, cv::Point(0, hips), cv::Point(300, hips), CV_RGB(255, 255, 0));

    cv::line(drawMat, cv::Point(rightX, bottomFace), elleboogR, CV_RGB(255, 152, 0));
    cv::line(drawMat, cv::Point(leftX, bottomFace), elleboogL, CV_RGB(255, 255, 0));
    cv::line(drawMat, elleboogL, leftHand, CV_RGB(100, 152, 152));
    cv::line(drawMat, elleboogR, rightHand, CV_RGB(100, 255, 152));
    cv::circle(drawMat, facePoint, face.width/2, CV_RGB(255, 0, 0) ,1,8,0);

    bool dc = false;
    if (dc == true) {
        facePoint = cv::Point(face.x+face.width/2, face.y+face.height/2+face.height);
        cv::circle(drawMat, facePoint, face.width/2, CV_RGB(255, 0, 0) ,1,8,0);
        facePoint = cv::Point(face.x+face.width/2-face.width, face.y+face.height/2+face.height);
        cv::circle(drawMat, facePoint, face.width/2, CV_RGB(255, 0, 0) ,1,8,0);
        facePoint = cv::Point(face.x+face.width/2+face.width, face.y+face.height/2+face.height);
        cv::circle(drawMat, facePoint, face.width/2, CV_RGB(255, 0, 0) ,1,8,0);

        facePoint = cv::Point(face.x+face.width/2, face.y+face.height/2+face.height*2);
        cv::circle(drawMat, facePoint, face.width/2, CV_RGB(255, 0, 0) ,1,8,0);
        facePoint = cv::Point(face.x+face.width/2-face.width, face.y+face.height/2+face.height*2);
        cv::circle(drawMat, facePoint, face.width/2, CV_RGB(255, 0, 0) ,1,8,0);
        facePoint = cv::Point(face.x+face.width/2+face.width, face.y+face.height/2+face.height*2);
        cv::circle(drawMat, facePoint, face.width/2, CV_RGB(255, 0, 0) ,1,8,0);

        facePoint = cv::Point(face.x+face.width/2, face.y+face.height/2+face.height*3);
        cv::circle(drawMat, facePoint, face.width/2, CV_RGB(255, 0, 0) ,1,8,0);
        facePoint = cv::Point(face.x+face.width/2-face.width, face.y+face.height/2+face.height*3);
        cv::circle(drawMat, facePoint, face.width/2, CV_RGB(255, 0, 0) ,1,8,0);
        facePoint = cv::Point(face.x+face.width/2+face.width, face.y+face.height/2+face.height*3);
        cv::circle(drawMat, facePoint, face.width/2, CV_RGB(255, 0, 0) ,1,8,0);
    }


    if (extend) {
        cv::Rect rec1(rightX, upperBodyHalf , centerX-rightX, face.height * 1.1); // Right upperbody
        cv::Rect rec2(centerX, upperBodyHalf , leftX-centerX, face.height *1.1);  // Left upperbody
        cv::Rect rec3(rightX, lowerBodyHalf, centerX-rightX, face.height * 1.1);  // Right lowerbody
        cv::Rect rec4(centerX, lowerBodyHalf, leftX-centerX, face.height * 1.1);  // Left lowerbody

        cv::Rect recRU(rightX-face.width/2, upperBodyHalf , (centerX-rightX)/2, face.height * 1.1); // Right upperarm
        cv::Rect recRL(rightX-face.width/2, lowerBodyHalf , (centerX-rightX)/2, face.height * 1.1); // Right lowerarm

        cv::Rect recLU(leftX, upperBodyHalf , (leftX-centerX)/2, face.height * 1.1); // Left upperarm
        cv::Rect recLL(leftX, lowerBodyHalf , (leftX-centerX)/2, face.height * 1.1); // Left lowerarm

        // adhv al bestaande cirkel hoofd
        // schoot, overwegen deze te gebruiken, namelijk bijna altijd zo

        //cv::rectangle(drawMat, rec1, CV_RGB(255, 0, 0), 2, 8, 0 );
        //cv::rectangle(drawMat, rec2, CV_RGB(255, 255, 0), 2, 8, 0 );
        //cv::rectangle(drawMat, rec3, CV_RGB(255, 0, 255), 2, 8, 0 );
        //cv::rectangle(drawMat, rec4, CV_RGB(125, 125, 125), 2, 8, 0 );

        //cv::rectangle(drawMat, recRU, CV_RGB(0, 125, 125), 2, 8, 0 );
        //cv::rectangle(drawMat, recRL, CV_RGB(65, 65, 125), 2, 8, 0 );

        //cv::rectangle(drawMat, recLU, CV_RGB(125, 0, 125), 2, 8, 0 );
        //cv::rectangle(drawMat, recLL, CV_RGB(125, 65, 65), 2, 8, 0 );

        RH=0; LH=0;
        if (rec1.contains(leftHand)) {
            cv::putText(drawMat, "LeftHand, Right upperbody", cv::Point(rec1.x, rec1.y), 0, 0.4, CV_RGB(255, 0, 0), 1);
            cv::rectangle(drawMat, rec1, CV_RGB(255, 0, 0), 1, 8, 0 );
            LH=1;
        }
        if (rec1.contains(rightHand)) {
            cv::putText(drawMat, "RightHand, Right upperbody", cv::Point(rec1.x, rec1.y), 0, 0.4, CV_RGB(255, 0, 0), 1);
            cv::rectangle(drawMat, rec1, CV_RGB(255, 0, 0), 1, 8, 0 );
            RH=1;
        }

        if (rec2.contains(leftHand)) {
            cv::putText(drawMat, "LeftHand, Left upperbody", cv::Point(rec2.x, rec2.y), 0, 0.4, CV_RGB(255, 255, 0), 1);
            cv::rectangle(drawMat, rec2, CV_RGB(255, 255, 0), 1, 8, 0 );
            LH=2;
        }
        if ( rec2.contains(rightHand)) {
            cv::putText(drawMat, "RightHand, Left upperbody", cv::Point(rec2.x, rec2.y), 0, 0.4, CV_RGB(255, 255, 0), 1);
            cv::rectangle(drawMat, rec2, CV_RGB(255, 255, 0), 1, 8, 0 );
            RH=2;
        }

        if (rec3.contains(leftHand)) {
            cv::putText(drawMat, "LeftHand, Right lowerbody", cv::Point(rec3.x, rec3.y), 0, 0.4, CV_RGB(255, 0, 255), 1);
            cv::rectangle(drawMat, rec3, CV_RGB(255, 0, 255), 1, 8, 0 );
            LH=3;
        }
        if (rec3.contains(rightHand)) {
            cv::putText(drawMat, "RightHand, Right lowerbody", cv::Point(rec3.x, rec3.y), 0, 0.4, CV_RGB(255, 0, 255), 1);
            cv::rectangle(drawMat, rec3, CV_RGB(255, 0, 255), 1, 8, 0 );
            RH=3;
        }

        if (rec4.contains(leftHand)) {
            cv::putText(drawMat, "LeftHand, Left lowerbody", cv::Point(rec4.x, rec4.y), 0, 0.4, CV_RGB(255, 255, 255), 1);
            cv::rectangle(drawMat, rec4, CV_RGB(125, 125, 125), 1, 8, 0 );
            LH=4;
        }
        if (rec4.contains(rightHand)) {
            cv::putText(drawMat, "RightHand, Left lowerbody", cv::Point(rec4.x, rec4.y), 0, 0.4, CV_RGB(255, 255, 255), 1);
            cv::rectangle(drawMat, rec4, CV_RGB(125, 125, 125), 1, 8, 0 );
            RH=4;
        }

        // handen kunnen alleen de andere arm aanduiden
        if (recRU.contains(leftHand)) {
            cv::putText(drawMat, "LeftHand, Right upperarm", cv::Point(recRU.x, recRU.y + 10), 0, 0.4, CV_RGB(255, 255, 255), 1);
            cv::rectangle(drawMat, recRU, CV_RGB(125, 125, 125), 1, 8, 0 );
            LH=6;
        }
        if (recRL.contains(leftHand)) {
            cv::putText(drawMat, "LeftHand, Right lowerarm", cv::Point(recRL.x, recRL.y + 10), 0, 0.4, CV_RGB(255, 255, 255), 1);
            cv::rectangle(drawMat, recRL, CV_RGB(125, 125, 125), 1, 8, 0 );
            LH=7;
        }
        if (recLU.contains(rightHand)) {
            cv::putText(drawMat, "RightHand, Left upperarm", cv::Point(recLU.x, recLU.y + 10), 0, 0.4, CV_RGB(255, 255, 255), 1);
            cv::rectangle(drawMat, recLU, CV_RGB(125, 125, 125), 1, 8, 0 );
            RH=6;
        }
        if (recLL.contains(rightHand)) {
            cv::putText(drawMat, "RightHand, Left lowerarm", cv::Point(recLL.x, recLL.y + 10), 0, 0.4, CV_RGB(255, 255, 255), 1);
            cv::rectangle(drawMat, recLL, CV_RGB(125, 125, 125), 1, 8, 0 );
            RH=7;
        }

        if (face.contains(leftHand)) {
            cv::putText(drawMat, "LeftHand, Face", cv::Point(face.x, face.y), 0, 0.4, CV_RGB(255, 0, 0), 1);
            cv::rectangle(drawMat, face, CV_RGB(255, 0, 0), 1, 8, 0 );
            LH=8;
        }
        if (face.contains(rightHand)) {
            cv::putText(drawMat, "RightHand, Face", cv::Point(face.x, face.y), 0, 0.4, CV_RGB(255, 0, 0), 1);
            cv::rectangle(drawMat, face, CV_RGB(255, 0, 0), 1, 8, 0 );
            RH=8;
        }
    }

    return 1;
}

int zoekArmen(cv::Point leftHand, cv::Point rightHand, cv::Rect face, cv::Mat& baseFrame, cv::Mat& drawMat) {

    cv::Mat canny;
    double pixelSizeInCm = 0.25;
    double cmInPixels = 1.0 / pixelSizeInCm;
    int frameWidth = drawMat.cols;
    int frameHeight = drawMat.rows;
    cv::Point facePoint(face.x+face.width/2, face.y+face.height/2);
    cv::Canny(baseFrame, canny, 100, 100);
    cv::Rect faceRect = face;
    int centerX = faceRect.x + 0.5 * faceRect.width;
    int bottomFace = faceRect.y + faceRect.height;
    int shoulderHeight = bottomFace + faceRect.height/4;
    int lowerBodyHalf = 0.6 * bottomFace + 0.4 * frameHeight;
    int hips = lowerBodyHalf + bottomFace;
    int leftX = centerX + faceRect.width;
    int rightX = centerX - faceRect.width;
    cv::putText(drawMat, joinString(" rightX ", rightX) + joinString(" leftX ", leftX) + joinString(" face width ", faceRect.width) + joinString(" frame/face ", frameHeight/faceRect.width)+ joinString(" headspace ", faceRect.y), cv::Point(0, frameHeight - 20), 0, 0.4, CV_RGB(255, 5, 0), 1);

    cv::Point elleboogL = cv::Point(leftX + faceRect.width/2.5, lowerBodyHalf + faceRect.width/2);
    cv::Point elleboogR = cv::Point(rightX - faceRect.width/2.5, lowerBodyHalf + faceRect.width/2);
    cv::Point shoulderL = cv::Point(leftX, shoulderHeight);
    cv::Point shoulderR = cv::Point(rightX, shoulderHeight);

    unsigned char *input = (unsigned char*)(canny.data);

    // zoek rond linker schouder en bepaal een gemiddeld punt
    // LINKER SCHOUDER
    cv::Rect rec1(shoulderL.x - 10, shoulderL.y - 10, 40, 40);
    cv::rectangle(drawMat, rec1, CV_RGB(255, 0, 0), 1, 8, 0 );

    int avX=0, avY=0, nrX=0, nrY=0;
    int r,g,b;

    for(int j = shoulderL.y-10; j < shoulderL.y + 30; j++){
        for(int i = shoulderL.x + 30; i > shoulderL.x - 10; i--){
            b = input[canny.cols * j + i ] ; //g = input[img.cols * j + i + 1]; //r = input[img.cols * j + i + 2];
            if (b > 0) {
                nrX++; nrY++;
                avX = avX + i;
                avY = avY + j;
                cv::circle(drawMat, cv::Point(i,j), 1, CV_RGB(255, 0, 0),1,8,0);
                break;
            }
        }
    }
    if (nrX == 0 || nrY == 0) {avX = 0; avY = 0;} else {avX = avX / nrX; avY = avY / nrY;}
    cv::Point LST = cvPoint(avX, avY); //Left Shoulder Top
    cv::circle(drawMat, LST, 4, CV_RGB(200, 125, 255) ,1,8,0);

    // zoek rond rechter schouder en bepaal een gemiddeld punt
    // RECHTER SCHOUDER
    rec1 = cv::Rect(shoulderR.x - 30, shoulderR.y - 10, 40, 40);
    cv::rectangle(drawMat, rec1, CV_RGB(255, 0, 0), 1, 8, 0 );

    avX=avY=nrX=nrY=0;
    for(int j = shoulderR.y - 10; j < shoulderR.y + 30; j++){
        for(int i = shoulderR.x - 30; i < shoulderR.x + 10; i++){
            b = input[canny.cols * j + i ] ; //g = input[img.cols * j + i + 1]; //r = input[img.cols * j + i + 2];
            if (b > 0) {
                nrX++; nrY++;
                avX = avX + i;
                avY = avY + j;
                cv::circle(drawMat, cv::Point(i,j), 1, CV_RGB(255, 0, 0) ,1,8,0);
                break;
            }
        }
    }
    if (nrX == 0 || nrY == 0) {avX = 0; avY = 0;} else {avX = avX / nrX; avY = avY / nrY;}
    cv::Point RST = cvPoint(avX, avY); //Left Shoulder Top
    cv::circle(drawMat, RST, 4, CV_RGB(0, 255, 0) ,1,8,0);

    // zoek alle punten in een zw mat rond het gekozen elleboog punt  die wit (groter dan 0) zijn
    // LINKER ELLEBOOG
    rec1 = cv::Rect(elleboogL.x - 20, elleboogL.y - 10, 40, 40);
    cv::rectangle(drawMat, rec1, CV_RGB(255, 0, 0), 1, 8, 0 );

    avX=avY=nrX=nrY=0;
    for(int j = elleboogL.y - 10; j < elleboogL.y + 30; j++){
        for(int i = elleboogL.x + 20; i > elleboogL.x - 20; i--){
            b = input[canny.cols * j + i ] ; //g = input[img.cols * j + i + 1]; //r = input[img.cols * j + i + 2];
            if (b > 0) {
                nrX++; nrY++;
                avX = avX + i;
                avY = avY + j;
                cv::circle(drawMat, cv::Point(i, j), 1, CV_RGB(255, 125, 125),1,8,0);
                break;
            }
        }
    }
    if (nrX == 0 || nrY == 0) {avX = 0; avY = 0;} else {avX = avX / nrX; avY = avY / nrY;}
    cv::Point LE = cvPoint(avX, avY); //Left Elbow
    cv::circle(drawMat, LE, 4, CV_RGB(0, 255, 0) ,1,8,0);

    // zoek alle punten in een zw mat rond het gekozen elleboog punt  die wit (groter dan 0) zijn
    // RECHTER ELLEBOOG
    rec1=cv::Rect(elleboogR.x - 20, elleboogR.y - 10, 40, 40);
    cv::rectangle(drawMat, rec1, CV_RGB(255, 0, 0), 1, 8, 0 );

    avX=avY=nrX=nrY=0;
    for(int j = elleboogR.y - 10; j < elleboogR.y + 30; j++){
        for(int i = elleboogR.x - 20; i < elleboogR.x + 20; i++){
            b = input[canny.cols * j + i ] ; //g = input[img.cols * j + i + 1]; //r = input[img.cols * j + i + 2];
            if (b > 0) {
                nrX++; nrY++;
                avX = avX + i;
                avY = avY + j;
                cv::circle(drawMat, cv::Point(i, j), 1, CV_RGB(255, 125, 125),1,8,0);
                break;
            }
        }
    }
    if (nrX == 0 || nrY == 0) {avX = 0; avY = 0;} else {avX = avX / nrX; avY = avY / nrY;}
    cv::Point RE = cvPoint(avX, avY); //Left Elbow
    cv::circle(drawMat, RE, 4, CV_RGB(255, 255, 0) ,1,8,0);

   if (LE.x == 0 && LE.y == 0) {
        LE = elleboogL;
    }
    if (RE.x == 0 && RE.y == 0) {
        RE = elleboogR;
    }
    if (LST.x == 0 && LST.y == 0) {
        LST = cv::Point(leftX, bottomFace);
    }
    if (RST.x == 0 && RST.y == 0) {
        RST = cv::Point(rightX, bottomFace);
    }
    cv::line(drawMat, LST, LE, CV_RGB(255, 152, 0), 1);
    cv::line(drawMat, RST, RE, CV_RGB(255, 152, 0), 1);
    cv::line(drawMat, LE, cv::Point(leftHand.x, leftHand.y), CV_RGB(255, 152, 0), 1 );
    cv::line(drawMat, RE, cv::Point(rightHand.x, rightHand.y), CV_RGB(255, 152, 0), 1);

    return 0;
}

int zoekOgen(cv::Rect face, cv::Mat& baseFrame, cv::Mat& drawMat, bool extend) {
    // init the classes

    cv::Mat canny;
    cv::Canny(baseFrame, canny, 100, 100);
    unsigned char *input = (unsigned char*)(canny.data);
    cv::Point leftEye, rightEye;
    int stepY = 20;
    int stepX = 30;

    // zoek in de bovenste helft van het gezicht naar de ogen
    // LINKER OOG
    cv::Rect recLeft(face.x + face.width/2, face.y + face.height/2 - stepY, stepX, stepY);
    cv::rectangle(drawMat, recLeft, CV_RGB(255, 0, 0), 1, 8, 0 );

    int avX=0, avY=0, nrX=0, nrY=0;
    int r,g,b;

    for(int j = recLeft.y; j < recLeft.y + stepY; j++){
        for(int i = recLeft.x; i < recLeft.x + stepX; i++){
            b = input[canny.cols * j + i ] ; //g = input[img.cols * j + i + 1]; //r = input[img.cols * j + i + 2];
            if (b > 0) {
                nrX++; nrY++;
                //avX = avX + (i - avX) / nrX;
                //avY = avY + (j - avY ) / nrY;
                avX = avX + i;
                avY = avY + j;
                if (extend) cv::circle(drawMat, cv::Point(i,j), 1, CV_RGB(200, 200, 200),1,8,0);
                //break;
            }
        }
    }
    if (nrX == 0 || nrY == 0) {avX = 0; avY = 0;} else {avX = avX / nrX; avY = avY / nrY;}
    leftEye = cvPoint(avX, avY);
    cv::circle(drawMat, leftEye, 4, CV_RGB(150, 150, 150),2,8,0);


    // zoek in de bovenste helft van het gezicht naar de ogen
    // RECHTER OOG
    cv::Rect recRight(face.x + face.width/2 - stepX, face.y + face.height/2 - stepY, stepX, stepY);
    cv::rectangle(drawMat, recRight, CV_RGB(255, 0, 0), 1, 8, 0 );

    //std::cout << " ------------------------------------------- " << std::endl;
    avX=0; avY=0; nrX=0; nrY=0;
    for(int j = recRight.y; j < recRight.y + stepY; j++){
        for(int i = recRight.x; i < recRight.x + stepX; i++){
            b = input[canny.cols * j + i ] ; //g = input[img.cols * j + i + 1]; //r = input[img.cols * j + i + 2];
            if (b > 0) {
                nrX++; nrY++;
                //avX = avX + (i - avX) / nrX;
                //avY = avY + (j - avY ) / nrY;
                avX = avX + i;
                avY = avY + j;
                if (extend) cv::circle(drawMat, cv::Point(i,j), 1, CV_RGB(200, 200, 200) ,1,8,0);
                //std::cout << " av y " << avY << " y " <<  j << "  nrY " << nrY << " avY " << avY/nrY << std::endl;
                //break;
            }
        }
    }
    if (nrX == 0 || nrY == 0) {avX = 0; avY = 0;} else {avX = avX / nrX; avY = avY / nrY;}
    //std::cout << " av y " << avY << std::endl;
    rightEye = cvPoint(avX, avY);
    cv::circle(drawMat, rightEye, 4, CV_RGB(150, 150, 150),2,8,0);


    // eventueel correctie als ogen niet worden gevonden
    if (rightEye.x == 0 && rightEye.y == 0) {
        //rightEye = elleboogR;
    }
    if (leftEye.x == 0 && leftEye.y == 0) {
        //leftEye = cv::Point(rightX, bottomFace);
    }
    //cv::circle(drawMat, rightEye, 1, CV_RGB(255, 0, 0) ,1,8,0);
    //cv::circle(drawMat, leftEye, 1, CV_RGB(255, 0, 0) ,1,8,0);
    return 0;
}

int hoofdTiltRotate(cv::Rect face, cv::Mat& baseFrame, cv::Mat& drawMat) {
    // init the classes

    cv::Mat canny;
    cv::Canny(baseFrame, canny, 100, 100);
    unsigned char *input = (unsigned char*)(baseFrame.data);
    cv::Point topLeftPart, topRightPart, lowLeftPart, lowRightPart;
    int stepY = face.height/2;
    int stepX = 30;
    bool extend = false;

    // zoek naast de bovenste helft van het gezicht naar randen van het hoofd
    // TOP LINKS
    cv::Rect recTopLeft(face.x + face.width - stepX, face.y, stepX, stepY);
    cv::rectangle(drawMat, recTopLeft, CV_RGB(255, 0, 0), 1, 8, 0 );

    int avX=0, avY=0, nrX=0, nrY=0;
    int r,g,b;

    for(int j = recTopLeft.y; j < recTopLeft.y + face.height/2 ; j++){
        for(int i = recTopLeft.x + stepX; i > recTopLeft.x; i--){
            b = input[canny.cols * j + i ] ; //g = input[img.cols * j + i + 1]; //r = input[img.cols * j + i + 2];
            if (b > 0) {
                nrX++; nrY++;
                //avX = avX + (i - avX) / nrX;
                //avY = avY + (j - avY ) / nrY;
                avX = avX + i;
                avY = avY + j;
                if (extend) cv::circle(drawMat, cv::Point(i,j), 1, CV_RGB(200, 200, 200),1,8,0);
                //break;
            }
        }
    }
    if (nrX == 0 || nrY == 0) {avX = 0; avY = 0;} else {avX = avX / nrX; avY = avY / nrY;}
    topLeftPart = cvPoint(avX, avY);
    cv::circle(drawMat, topLeftPart, 4, CV_RGB(150, 150, 150),2,8,0);


    // zoek naast de bovenste helft van het gezicht naar randen van het hoofd
    // TOP RECHTS
    cv::Rect recTopRight(face.x, face.y, stepX, stepY);
    cv::rectangle(drawMat, recTopRight, CV_RGB(255, 0, 0), 1, 8, 0 );

    //std::cout << " ------------------------------------------- " << std::endl;
    avX=0; avY=0; nrX=0; nrY=0;
    for(int j = recTopRight.y; j < recTopRight.y + stepY; j++){
        for(int i = recTopRight.x; i < recTopRight.x + stepX; i++){
            b = input[canny.cols * j + i ] ; //g = input[img.cols * j + i + 1]; //r = input[img.cols * j + i + 2];
            if (b > 0) {
                nrX++; nrY++;
                //avX = avX + (i - avX) / nrX;
                //avY = avY + (j - avY ) / nrY;
                avX = avX + i;
                avY = avY + j;
                if (extend) cv::circle(drawMat, cv::Point(i,j), 1, CV_RGB(200, 200, 200) ,1,8,0);
                //std::cout << " av y " << avY << " y " <<  j << "  nrY " << nrY << " avY " << avY/nrY << std::endl;
                //break;
            }
        }
    }
    if (nrX == 0 || nrY == 0) {avX = 0; avY = 0;} else {avX = avX / nrX; avY = avY / nrY;}
    //std::cout << " av y " << avY << std::endl;
    topRightPart = cvPoint(avX, avY);
    cv::circle(drawMat, topRightPart, 4, CV_RGB(150, 150, 150),2,8,0);


    // zoek naast de onderste helft van het gezicht naar randen van het hoofd
    // Onder LINKS
    cv::Rect recLowLeft(face.x + face.width - stepX, face.y + face.height/2, stepX, stepY);
    cv::rectangle(drawMat, recLowLeft, CV_RGB(255, 0, 0), 1, 8, 0 );

    avX=0, avY=0, nrX=0, nrY=0;
    for(int j = recLowLeft.y; j < recLowLeft.y + face.height/2 ; j++){
        for(int i = recLowLeft.x + stepX; i > recLowLeft.x; i--){
            b = input[canny.cols * j + i ] ; //g = input[img.cols * j + i + 1]; //r = input[img.cols * j + i + 2];
            if (b > 0) {
                nrX++; nrY++;
                //avX = avX + (i - avX) / nrX;
                //avY = avY + (j - avY ) / nrY;
                avX = avX + i;
                avY = avY + j;
                if (extend) cv::circle(drawMat, cv::Point(i,j), 1, CV_RGB(200, 200, 200),1,8,0);
                //break;
            }
        }
    }
    if (nrX == 0 || nrY == 0) {avX = 0; avY = 0;} else {avX = avX / nrX; avY = avY / nrY;}
    lowLeftPart = cvPoint(avX, avY);
    cv::circle(drawMat, lowLeftPart, 4, CV_RGB(150, 150, 150),2,8,0);


    // zoek naast de onderste helft van het gezicht naar randen van het hoofd
    // Onder RECHTS
    cv::Rect recLowRight(face.x, face.y + face.height/2, stepX, stepY);
    cv::rectangle(drawMat, recLowRight, CV_RGB(255, 0, 0), 1, 8, 0 );

    //std::cout << " ------------------------------------------- " << std::endl;
    avX=0; avY=0; nrX=0; nrY=0;
    for(int j = recLowRight.y; j < recLowRight.y + stepY; j++){
        for(int i = recLowRight.x; i < recLowRight.x + stepX; i++){
            b = input[canny.cols * j + i ] ; //g = input[img.cols * j + i + 1]; //r = input[img.cols * j + i + 2];
            if (b > 0) {
                nrX++; nrY++;
                //avX = avX + (i - avX) / nrX;
                //avY = avY + (j - avY ) / nrY;
                avX = avX + i;
                avY = avY + j;
                if (extend) cv::circle(drawMat, cv::Point(i,j), 1, CV_RGB(200, 200, 200) ,1,8,0);
                //std::cout << " av y " << avY << " y " <<  j << "  nrY " << nrY << " avY " << avY/nrY << std::endl;
                //break;
            }
        }
    }
    if (nrX == 0 || nrY == 0) {avX = 0; avY = 0;} else {avX = avX / nrX; avY = avY / nrY;}
    //std::cout << " av y " << avY << std::endl;
    lowRightPart = cvPoint(avX, avY);
    cv::circle(drawMat, lowRightPart, 4, CV_RGB(150, 150, 150),2,8,0);

    // eventueel correctie als ogen niet worden gevonden
    //if (rightEye.x == 0 && rightEye.y == 0) {
        //rightEye = elleboogR;
    //}
    //if (leftEye.x == 0 && leftEye.y == 0) {
        //leftEye = cv::Point(rightX, bottomFace);
    //}
    //cv::circle(drawMat, rightEye, 1, CV_RGB(255, 0, 0) ,1,8,0);
    //cv::circle(drawMat, leftEye, 1, CV_RGB(255, 0, 0) ,1,8,0);
    return 0;
}


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
    cv::Mat rawFrame, frame, gray, grayPrev, faceMat, roiMask, temporalSkinMask;
    cv::Mat wit, bw, canny;

    int frameHeightMax = 300;
	
	BodyRects body;

	int frameIndex = 0;
	bool initialized = false;
    bool startOk = false;
	

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

		cv::imshow("raw", frame);

		// start detection of edges, face and skin
		bool faceDetected = faceDetector.detect(gray);
		double pixelSizeInCm = faceDetector.pixelSizeInCm;
		if (faceDetected) {
			auto face = &(faceDetector.face.rect);
			skinDetector.detect(*face, frame, initialized, (3.0 / pixelSizeInCm) * 4);
			edgeDetector.detect(gray);
			
			if (initialized) {
                temporalSkinMask = skinDetector.getMergedMap();
                roiMask = cv::Mat::zeros(temporalSkinMask.rows, temporalSkinMask.cols, temporalSkinMask.type()); // all 0

				// get an initial motion estimate based on the temporal skin mask alone. This is used 
				// in the hand detection
				movementDetector.detect(gray, grayPrev);
				movementDetector.mask(temporalSkinMask);
				movementDetector.calculate(faceDetector.normalizationFactor);

                //cv::Mat faceMat;
                frame.copyTo(faceMat);
                //cv::Mat canny = cv::Canny(frame, canny, 100, 100);
                handDetector.detect(
                            gray, grayPrev,
                            *face,
                            skinDetector.skinMask,
                            movementDetector.movementMap,
                            edgeDetector.detectedEdges,
                            pixelSizeInCm,
                            faceMat
                            );

				handDetector.draw(frame);
				handDetector.drawTraces(frame);

				faceDetector.draw(faceMat);
				cv::imshow("face", faceMat);

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
                edgeDetector.show();
                handDetector.show();

                // ADDED for mannequin en handposition labels
                int codeLH=0, codeRH=0, codeGesture=0;
                faceMat.copyTo(wit);
                wit.setTo(cv::Scalar(255,255,255));
                cv::Point leftHand = handDetector.leftHand.position;
                cv::Point rightHand = handDetector.rightHand.position;

                // start schouders en ellebogen zoeken
                cv::Rect faceRect = faceDetector.face.rect;
                zoekArmen(leftHand, rightHand, faceRect, frame, wit);
                zoekOgen(faceRect, frame, wit, true );
                hoofdTiltRotate(faceRect, frame, wit);
                tekenMannequin(leftHand, rightHand, faceRect, wit, true, codeLH, codeRH);
                cv::imshow("wit", wit);
                // ADDED for mannequin en handposition labels

                bool leftHandMissing = false;
                bool rightHandMissing = false;

				// ----------  THIS IS THE VALUE TO PUBLISH TO SSI:  ------------- //
				//																   //
				double publishValue = ROImovementDetector.value;				   //
                if (handDetector.leftHand.position.x == 0 || handDetector.leftHand.position.y ==0) {leftHandMissing = true; }
                if (handDetector.rightHand.position.x == 0 || handDetector.rightHand.position.y ==0) {rightHandMissing = true;}

                // values to grab
                //      publishValue, leftHandMissing, rightHandMissing, codeLH, codeRH, codeGesture (still empty)
                //																   //
				// ----------  THIS IS THE VALUE TO PUBLISH TO SSI   ------------- //

                //if (leftHandMissing || rightHandMissing) std::cout << publishValue << " " << leftHandMissing << " " << rightHandMissing << std::endl;
                //if (codeLH>0||codeRH>0) std::cout << "LABELS " << codeLH << " " << codeRH << std::endl;

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

        // mixed van blobs, canny en lines

        cv::imshow("DebugGUI", frame);
        if ((initialized && first) || startOk) {
            cv::moveWindow("wit", 2000, 0);
            cv::moveWindow("face", 2000, 340);
            cv::moveWindow("DebugGUI", 2450, 340);
            cv::moveWindow("debugMapHands", 2750,0);
            first = false;
        }
        int keystroke = cv::waitKey(waitTime);

        if (keystroke == 27 || keystroke == 1048603 || keystroke == 1048689) {  //ESC or q
            return 100;
            break;
        }
        else if (keystroke == 2424832 || keystroke == 1113937) { // left arrow
            if (waitTime < 100) {
                return -1;
            }
        }
        else if (keystroke == 2555904 || keystroke == 1113939 || keystroke == 83 || keystroke == 54) { // right arrow
            if (waitTime < 100) {
                return 1;
            }
        }
        else if (keystroke == 32 || keystroke == 1048608) { // spacebar
            waitTime = 1e6;
        }
        else if (keystroke == 13 || keystroke == 1048586) {  // return
            waitTime = 2;
        }
        else if (keystroke == 115 || keystroke == 1048694) { // s
            skip = 750;
            waitTime = 1e6;
        }
        else if (keystroke == 100 || keystroke == 1048676) { // d
            calcSkip = 50;
        }
        else if (keystroke == 114 || keystroke == 1048690) { // r
            return 0;
        } else if (keystroke == 97) { // a=vorige
            return 97;
        }
        else if (keystroke >= 0 && keystroke != 255) {
            //std::cout << "key:" << keystroke << std::endl;
        }

        frameIndex += 1;
    }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 1;
}

void manage(int movieIndex) {
    int value;
    std::vector<std::string> videoList;
    //videoList.push_back("tr086_spk13m.mp4");
    videoList.push_back("de001_spk02f.mp4");
    videoList.push_back("de003_spk01f.mp4");
    videoList.push_back("de004_spk01f.mp4");
    videoList.push_back("de005_spk03f.mp4");
    videoList.push_back("de007_spk02f.mp4");
    videoList.push_back("de010_spk03f.mp4");
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
    /*
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
    */

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

    value = run(cap, fps);
    int newIndex = movieIndex;

    if (value == 1)		  // next movie
        newIndex += 1;
    else if (value == -1)  // previous movie
        newIndex -= 1;

    else if (value == 97) {
        cap.open(0);
        fps=24;
        value = run(cap, fps);
        cap.release();
    }
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
    manage(21);
	return 0;
}
