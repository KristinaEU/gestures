/*
* Luis
*/

#pragma once
#include "MercuryCore.h"
#include "SemanticDetector.h"
//#include "cmath"

/*
* Tell the hands which one is left and right, give them specific colors for drawing, set the fps.
*/
SemanticDetector::SemanticDetector(int fps, std::string bodyPart) {
	this->fps = fps;
	this->bodyPart = bodyPart;
}

SemanticDetector::~SemanticDetector() {}


void SemanticDetector::setVideoProperties(int frameWidth, int frameHeight) {
	this->frameHeight = frameHeight;
	this->frameWidth = frameWidth;
}

// took from:
// http://www.cplusplus.com/forum/general/216928/
//
// Returns interpolated value at x from parallel arrays ( xData, yData )
//   Assumes that xData has at least two elements, is sorted and is strictly monotonic increasing
//   boolean argument extrapolate determines behaviour beyond ends of array (if needed)
double SemanticDetector::_interpolate( std::vector<double> &xData, std::vector<double> &yData, double x, bool extrapolate )
{
   int size = xData.size();

   int i = 0;                                                                  // find left end of interval for interpolation
   if ( x >= xData[size - 2] )                                                 // special case: beyond right end
   {
      i = size - 2;
   }
   else
   {
      while ( x > xData[i+1] ) i++;
   }
   double xL = xData[i], yL = yData[i], xR = xData[i+1], yR = yData[i+1];      // points on either side (unless beyond ends)
   if ( !extrapolate )                                                         // if beyond ends of array and not extrapolating
   {
      if ( x < xL ) yR = yL;
      if ( x > xR ) yL = yR;
   }

   double dydx = ( yR - yL ) / ( xR - xL );                                    // gradient

   return yL + dydx * ( x - xL );                                              // linear interpolation
}

void SemanticDetector::detect(cv::Point faceCenterPoint, double pixelSizeInCmTemp, std::vector<cv::Point> positions[]) {

    // checks null parameters
    //if(faceCenterPoint == 0L || !pixelSizeInCmTemp || !positions){
    //    std::cout << "SemanticDetector::detect -> Null detected" << std::endl;
    //    return;
    //}

    double periodFPS = 1.0 / ((double)this->fps); // period of FPS = 1/fps
    //Take left and right hand positions
    std::vector<cv::Point> LHandPositions = positions[0];
    std::vector<cv::Point> RHandPositions = positions[1];
    cv::Point tempPoint;

    // create normalized vector for hand positions
    std::vector<cv::Point2d> LHandNormalized;
    std::vector<cv::Point2d> RHandNormalized;
    //cv::Point2d tempPoint2d; // temporary Point2d

    // check if we have a minimum time to be analyzed (It depends on the fps and history size of the class)
    double LHandPositionsDuraction = ((double)LHandPositions.size()) * periodFPS;
    double RHandPositionsDuraction = ((double)RHandPositions.size()) * periodFPS;
    if( (LHandPositionsDuraction < ( this->minTimeToDetect) )|| (RHandPositionsDuraction < ( this->minTimeToDetect) )){
        std::cout << "SemanticDetector::detect -> No minimum time to analyze gestures! Minimum time:" << ( this->minTimeToDetect) << std::endl;
        std::cout << "\tMinimum time:" << ( this->minTimeToDetect)
        << " || Given time (Left hand): " << LHandPositionsDuraction
        << " || Given time (right hand): " << RHandPositionsDuraction << std::endl;
        return;
    }

    // Take only the newest samples within the minTimeToDectect
    int index = (int) std::ceil( this->minTimeToDetect * periodFPS ); // get the number of newest simples that we want to keep
    LHandPositions.erase(LHandPositions.begin(), LHandPositions.end() - index); // erase oldest samples
    RHandPositions.erase(RHandPositions.begin(), RHandPositions.end() - index); // erase oldest samples

    double x, y;

    std::vector<double> xLHNormalized, yLHNormalized,
                    xRHNormalized,yRHNormalized;
    // Feature scaling and Mean normalization
    // centralize positions according with faceCenterPoint and scale position around values -1 to +1
    for(int i = 0 ; i < LHandPositions.size(); i++){

        // Left hand
        tempPoint = LHandPositions[i] - faceCenterPoint;
        x = ((double)tempPoint.x) * pixelSizeInCmTemp / this->normalizationFaceHandDistance;
        y = ((double)tempPoint.y) * pixelSizeInCmTemp / this->normalizationFaceHandDistance;
        xLHNormalized.push_back(x);
        yLHNormalized.push_back(y);
        //LHandNormalized.insert(LHandNormalized.end(), cv::Point2d(x,y));

        // Right hand
        tempPoint = RHandPositions[i] - faceCenterPoint;
        x = ((double)tempPoint.x) * pixelSizeInCmTemp / this->normalizationFaceHandDistance;
        y = ((double)tempPoint.y) * pixelSizeInCmTemp / this->normalizationFaceHandDistance;
        xRHNormalized.push_back(x);
        yRHNormalized.push_back(y);
        //RHandNormalized.insert(RHandNormalized.end(), cv::Point2d(x,y));
    }

    // --- Linear interpolation ---
    //create time array of the samples at equal to FPS
    std::vector<double> fpsTime, interpolationTime;
    double auxTime;
    for ( auxTime = 0.0 ; auxTime <= this->minTimeToDetect; auxTime+=periodFPS ) {
        fpsTime.push_back( auxTime );
    }

    // create time array for interpolation
    for ( auxTime = 0.0 ; auxTime <= this->minTimeToDetect; auxTime+=this->interpolationTimeStep ) {
        interpolationTime.push_back( auxTime );
    }

    // interpolate normalized movements
    std::vector<double> xLHInterpolated, yLHInterpolated, xRHInterpolated, yRHInterpolated;
    for ( double val : interpolationTime ){
        xLHInterpolated.push_back( _interpolate( fpsTime, xLHNormalized, val, true ) );
        yLHInterpolated.push_back( _interpolate( fpsTime, yLHNormalized, val, true ) );
        xRHInterpolated.push_back( _interpolate( fpsTime, xRHNormalized, val, true ) );
        yRHInterpolated.push_back( _interpolate( fpsTime, yRHNormalized, val, true ) );
    }


    // ANN classifiers or Logistic regression

    // LOGISTICS REGRESSION
    //logisticsTest();



    //create the neural network
    /*Mat_<int> layerSizes(1, 3);
    layerSizes(0, 0) = data.cols;
    layerSizes(0, 1) = 20;
    layerSizes(0, 2) = responses.cols;

    Ptr<ANN_MLP> network = ANN_MLP::create();
    network->setLayerSizes(layerSizes);
    network->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0.1, 0.1);
    network->setTrainMethod(ANN_MLP::BACKPROP, 0.1, 0.1);
    Ptr<TrainData> trainData = TrainData::create(data, ROW_SAMPLE, responses);

    network->train(trainData);
    // Semantic Gesture recognition
    */


/*
    if(this->bodyPart == "Head"){

    }else if(this->bodyPart == "Hands"){

    }else{
        std::cout << "SemanticDetector::detect -> Unknown bodyPart" << std::endl;
    }
    */
}

/*
* Based on:
* https://github.com/opencv/opencv/blob/master/samples/cpp/logistic_regression.cpp
*/

void SemanticDetector::logisticsTest(){

    // simple case with batch gradient
    std::cout << "training...";
    //! [init]
    cv::Ptr<cv::ml::LogisticRegression> lr1 = cv::ml::LogisticRegression::create();
    lr1->setLearningRate(0.001);
    lr1->setIterations(10);
    lr1->setRegularization(cv::ml::LogisticRegression::REG_L2); // sum(w^2)
    lr1->setTrainMethod(cv::ml::LogisticRegression::BATCH);
    lr1->setMiniBatchSize(1); // Is it needed? I want to compute all batch examples

    cv::Mat data_train, data_test;
    cv::Mat labels_train, labels_test;
    //! [init]
    /*lr1->train(data_train, ROW_SAMPLE, labels_train);
    cout << "done!" << endl;

    cout << "predicting...";
    Mat responses;
    lr1->predict(data_test, responses);
    cout << "done!" << endl;

    // show prediction report
    cout << "original vs predicted:" << endl;
    labels_test.convertTo(labels_test, CV_32S);
    cout << labels_test.t() << endl;
    cout << responses.t() << endl;
    cout << "accuracy: " << calculateAccuracyPercent(labels_test, responses) << "%" << endl;

    // save the classifier
    const String saveFilename = "NewLR_Trained.xml";
    cout << "saving the classifier to " << saveFilename << endl;
    lr1->save(saveFilename);

    // load the classifier onto new object
    cout << "loading a new classifier from " << saveFilename << endl;
    Ptr<LogisticRegression> lr2 = StatModel::load<LogisticRegression>(saveFilename);
    */

}





