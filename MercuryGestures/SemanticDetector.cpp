/*
* Copyright 2017 Almende BV
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* Author: Luis F.M. Cunha
*/

#pragma once
#include "MercuryCore.h"
#include "SemanticDetector.h"

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

/*
* took from:
* http://www.cplusplus.com/forum/general/216928/
*
* Returns interpolated value at x from parallel arrays ( xData, yData )
*   Assumes that xData has at least two elements, is sorted and is strictly monotonic increasing
*   boolean argument extrapolate determines behavior beyond ends of array (if needed)
*/
double SemanticDetector::_interpolate( std::vector<double> &xData,
                                      std::vector<double> &yData,
                                      double x,
                                      bool extrapolate ) {
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


/*
* Feature scaling and Mean normalization
* centralize positions according with faceCenterPoint and scale position around values -1 to +1
*/
void SemanticDetector::scaleAndMeanNormalization(cv::Point &faceCenterPoint,
                                double pixelSizeInCmTemp,
                                std::vector<cv::Point> &HandPositions,
                                std::vector<double> &xHNormalizedOutput,
                                std::vector<double> &yHNormalizedOutput){

    double x, y;
    cv::Point tempPoint;
    for(int i = 0 ; i < HandPositions.size(); i++){

        // calculate the distance from hand to face
        tempPoint = HandPositions[i] - faceCenterPoint;
        //Normalize x and y components
        x = ((double)tempPoint.x) * pixelSizeInCmTemp / this->normalizationFaceHandDistance;
        y = ((double)tempPoint.y) * pixelSizeInCmTemp / this->normalizationFaceHandDistance;
        xHNormalizedOutput.push_back(x);
        yHNormalizedOutput.push_back(y);
    }
}


/*
* returns a linear interpolation
*/
void SemanticDetector::getLinearInterpolation(double minTimeToDetect,
                                              double periodFPS,
                                              double interpolationTimeStep,
                                              std::vector<double> &HNormalized,
                                              std::vector<double> &HInterpolated){

    //create vector time of the samples at the same rate as FPS
    std::vector<double> fpsTime, interpolationTime;
    double auxTime;
    int i;
    for ( auxTime = 0.0, i = 0; i < HNormalized.size()/*minTimeToDetect*/; auxTime+=periodFPS, i++ ) {
        fpsTime.push_back( auxTime );
    }
    //std::cout << "fpsTime size: " << fpsTime.size() << "\t\tlast value:" << fpsTime.at(fpsTime.size()-1) << std::endl;

    // create vector time for the interpolation
    for ( auxTime = 0.0 ; auxTime <= minTimeToDetect; auxTime+=interpolationTimeStep ) {
        interpolationTime.push_back( auxTime );
    }
    //std::cout << "interpolationTime size: " << interpolationTime.size() << "\tlast value:" << interpolationTime.at(interpolationTime.size()-1) << std::endl;

    // interpolate normalized movements
    for ( double val : interpolationTime ){
        //std::cout << "val = " << val << std::endl;
        HInterpolated.push_back( _interpolate( fpsTime, HNormalized, val, true ) );
        //std::cout << "interpolated value in val = " << HInterpolated.at(HInterpolated.size()-1) << std::endl;
    }

}

/*
* Get velocities from a vector of positions equally separated by deltaTime temporal space
*/
void SemanticDetector::getVelocity(std::vector<double> &vectorPositions, double deltaTime, std::vector<double> &vectorOutputVelocities){

    double velocity;
    for(int i = 0; i < vectorPositions.size() - 1; i++){
        velocity = (vectorPositions[i+1] - vectorPositions[i]) / deltaTime;
        vectorOutputVelocities.push_back( std::abs(velocity) );
    }
}


void SemanticDetector::getHandFeaturesVector(cv::Point &faceCenterPoint,
                                              double pixelSizeInCmTemp,
                                              double minTimeToDetect,
                                              double fps,
                                              double interpolationTimeStep,
                                              std::vector<cv::Point> &handPositions,
                                              std::vector<double> &featuresOutput){

    double periodFPS = 1.0 / fps;

    // Feature scaling and Mean normalization
    std::vector<double> xHNormalized, yHNormalized;
    scaleAndMeanNormalization(faceCenterPoint, pixelSizeInCmTemp, handPositions, xHNormalized, yHNormalized);

    // --- Linear Interpolation ---
    std::vector<double> xHInterpolated, yHInterpolated;

    getLinearInterpolation(minTimeToDetect, periodFPS, interpolationTimeStep, xHNormalized, xHInterpolated);
    getLinearInterpolation(minTimeToDetect, periodFPS, interpolationTimeStep, yHNormalized, yHInterpolated);

    // Add position Polynomials as features
    /*
        degree = 6;
        out = ones(size(X1(:,1)));
        for i = 1:degree
            for j = 0:i
                out(:, end+1) = (X1.^(i-j)).*(X2.^j);
            end
        end
    */
    std::vector<double> polFeatures;
    polFeatures.push_back(1.0); // bias
    double polVal;
    int degree = 6;

    for(int i = 0; i < xHInterpolated.size(); i++) {
        double valX= xHInterpolated[i], valY = yHInterpolated[i];
        for(int j = 1; j < degree; j++){
            for(int k = 0; k < j; k++){
                polVal = std::pow(valX, j - k) * std::pow(valY,k);
                polFeatures.push_back(polVal);
            }
        }
    }
    //std::cout << "polFeatures => " << polFeatures.size() << std::endl;

    // get velocities
    std::vector<double> xHInterpolatedVelocity, yHInterpolatedVelocity;
    getVelocity(xHInterpolated, periodFPS, xHInterpolatedVelocity);
    getVelocity(yHInterpolated, periodFPS, yHInterpolatedVelocity);


    // ---------------------------------------------|
    // At this point we should have all features    |
    // ---------------------------------------------|

    // Concatenate
    // First velocities x, than velocities y, than position Polynomials
    featuresOutput.insert( featuresOutput.end(), xHInterpolatedVelocity.begin(), xHInterpolatedVelocity.end() );
    featuresOutput.insert( featuresOutput.end(), yHInterpolatedVelocity.begin(), yHInterpolatedVelocity.end() );
    featuresOutput.insert( featuresOutput.end(), polFeatures.begin(), polFeatures.end() );
    //featuresOutput.insert( featuresOutput.end(), polVelocitiesFeatures.begin(), polVelocitiesFeatures.end() );
    //std::cout << "featuresOutput => " << featuresOutput.size() << std::endl;

}




void SemanticDetector::getDataFromFiles(std::string             directoryPath,
                                        std::vector<cv::Point>  &faceCenterPointListOutput,
                                        std::vector<double>     &pixelSizeInCmTempListOutput,
                                        //std::vector<double>     &minTimeToDetectListOutput,
                                        std::vector<double>     &fpsListOutput,
                                        //std::vector<double>     &interpolationTimeStepListOutput,
                                        std::vector<int>        &gestureLabelListOutput,
                                        std::vector< std::vector<cv::Point> > &LHandPositionsListOutput,
                                        std::vector< std::vector<cv::Point> > &RHandPositionsListOutput){

    // Get list of files to take data from
    std::vector<std::string> fileNameList;
    std::string fileName;
    DIR           *dirp;
    struct dirent *directory;
    dirp = opendir(directoryPath.c_str() );
    if (dirp){
        while ((directory = readdir(dirp)) != NULL){
            fileName = std::string( directory->d_name );
            // push back on the list if it is not "." or ".."
            if( (fileName.compare(".") != 0) && (fileName.compare("..") != 0) ){
                //std::cout << fileName << " added to the list" << std::endl;
                fileNameList.push_back(fileName);
            }
        }
        closedir(dirp);
        //std::cout << "fileNameList size = " << fileNameList.size() << std::endl;
    } else {
        std::cout << "Error!! Not possible to open directory: " << directoryPath << std::endl;
        return;
    }

    std::vector<int> LHandPositionsX, LHandPositionsY, RHandPositionsX, RHandPositionsY;

    int faceCenterPointX, faceCenterPointY;
    double pixelSizeInCmTemp;
    cv::Mat data;
    // for each file take the data
    std::string fullPath, videoName;
    int gestureLabel;
    for(int i = 0; i < fileNameList.size(); i++){
        std::vector<cv::Point> LHandPositions, RHandPositions;

        fileName = fileNameList.at(i);
        fullPath = directoryPath + fileName;
        //std::cout << "Reading file: " << fullPath << std::endl;
        cv::FileStorage file;
        if(file.open(fullPath, cv::FileStorage::READ)){
            //std::cout << "file opened!" << std::endl;
            file["LHandPositionsX"] >> LHandPositionsX;
            file["LHandPositionsY"] >> LHandPositionsY;
            file["RHandPositionsX"] >> RHandPositionsX;
            file["RHandPositionsY"] >> RHandPositionsY;

            file["faceCenterPointX"] >> faceCenterPointX;
            file["faceCenterPointY"] >> faceCenterPointY;

            file["pixelSizeInCmTemp"] >> pixelSizeInCmTemp;
            file["fps"] >> fps;

            file["gestureLabel"] >> gestureLabel;
            file["videoName"] >> videoName;

            file.release();
        } else {
            std::cerr << "File can not be opened: " << fullPath << std::endl;
        }

        // Put information together
        double periodFPS = 1.0 / fps;
        cv::Point faceCenterPoint(faceCenterPointX, faceCenterPointY);
        for(int i = 0; i < LHandPositionsY.size(); i++){
            LHandPositions.push_back( cv::Point(LHandPositionsX[i], LHandPositionsY[i]) );
            RHandPositions.push_back( cv::Point(RHandPositionsX[i], RHandPositionsY[i]) );
        }

        //std::cout << " HEREEEE1 i = " << i << std::endl;
        // push data into the lists
        faceCenterPointListOutput.push_back(faceCenterPoint);
        pixelSizeInCmTempListOutput.push_back(pixelSizeInCmTemp);
        fpsListOutput.push_back(fps);
        LHandPositionsListOutput.push_back(LHandPositions);
        RHandPositionsListOutput.push_back(RHandPositions);
        gestureLabelListOutput.push_back(gestureLabel);

    }
}


/*
* Save data into a file.
*/
void SemanticDetector::saveDataInFile(std::string fullPath,
                                         std::string capVideoName,
                                         int gestureLabel,
                                         double pixelSizeInCmTemp,
                                         cv::Point faceCenterPoint,
                                         std::vector<cv::Point> LHandPositions,
                                         std::vector<cv::Point> RHandPositions){
    // We should make some checks like if fullPath exists or null parameters !!!!

    //create a file
    cv::FileStorage file(fullPath, cv::FileStorage::WRITE);

    file << "videoName" << capVideoName;
    file << "gestureLabel" << gestureLabel;
    file << "fps" << this->fps;
    file << "pixelSizeInCmTemp" << pixelSizeInCmTemp;
    file << "faceCenterPointX" << faceCenterPoint.x;
    file << "faceCenterPointY" << faceCenterPoint.y;

    // LH
    // get x and y position for Left hand
    std::vector<int> containerX, containerY;
    for( int i = 0; i < LHandPositions.size(); i++){
        containerX.push_back(LHandPositions[i].x);
        containerY.push_back(LHandPositions[i].y);
    }
    file << "LHandPositionsX" << containerX;
    file << "LHandPositionsY" << containerY;

    containerX.clear();
    containerY.clear();
    // RH
    for( int i = 0; i < RHandPositions.size(); i++){
        containerX.push_back(RHandPositions[i].x);
        containerY.push_back(RHandPositions[i].y);
    }
    file << "RHandPositionsX" << containerX;
    file << "RHandPositionsY" << containerY;

    file.release();

    containerX.clear();
    containerY.clear();
}

/*
* generate new data with some deviation based on original data
*/
void SemanticDetector::generateDataFromOriginal(std::string dataPath,
                                                std::string capVideoName,
                                                int gestureLabel,
                                                double pixelSizeInCmTemp,
                                                cv::Point faceCenterPoint,
                                                std::vector<cv::Point> LHandPositions,
                                                std::vector<cv::Point> RHandPositions){
    // We should make some checks like if dataPath exists or null parameters !!!!

    // generate more data with hands history
    std::vector<cv::Point> containerLH, containerRH;
    cv::Point auxPoint;
    int x ,y;
    std::string fileName, fullPath;

    std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_int_distribution<int> uni(-2,2); // guaranteed unbiased
    int numOfExtraData = 1000;
    for(int i = 0; i < numOfExtraData; i++){

        // clear containers
        containerLH.clear();
        containerRH.clear();

        // LH - create some variation of data based on the original one
        for(int k = 0; k < LHandPositions.size(); k++){
            x = LHandPositions[k].x + uni(rng);
            y = LHandPositions[k].y + uni(rng);
            cv::Point auxPoint(x,y);
            containerLH.push_back(auxPoint);
        }

        // RH - create some variation of data based on the original one
        for(int k = 0; k < RHandPositions.size(); k++){
            x = RHandPositions[k].x + uni(rng);
            y = RHandPositions[k].y + uni(rng);
            cv::Point auxPoint(x,y);
            containerRH.push_back(auxPoint);
        }

        // create file name
        fileName = capVideoName + "Generated" + std::to_string(i);
        fullPath = dataPath + fileName + ".yml";

        // save data into a file
        //(Probably would be better to serialize all data in a single file! Later!)
        saveDataInFile(fullPath,
               capVideoName,
               gestureLabel,
               pixelSizeInCmTemp,
               faceCenterPoint,
               containerLH,
               containerRH);
    }
}

/*
* Generates point [numOfVectors x numOfPoints]
* x                     -> pixel number in x axis
* y                     -> pixel number in y axis
* numOfVectors          -> number of vectors to generate
* numOfPoints           -> number of points to generate in each vector
* positionsListOutput   -> output list
*/
void SemanticDetector::createListOfStaticPositions(int x,
                                                   int y,
                                                   unsigned int numOfVectors,
                                                   unsigned int numOfPoints,
                                                   std::vector< std::vector<cv::Point> > &positionsListOutput){

    for(int i = 0; i < numOfVectors; i++){

        std::vector<cv::Point> auxVector;
        for(int k = 0; k < numOfPoints; k++){
            cv::Point pos(x, y);
            auxVector.push_back(pos);
        }
        positionsListOutput.push_back(auxVector);
    }
    //std::cout << "positionsListOutput size = " << positionsListOutput.size() << std::endl;
}

/*
 * (x, y) = ( c1 + a*cos(2*pi*f*t) , c2 + b*sin(2*pi*f*t) )
 */
void SemanticDetector::createListOfEllipticalPositions(int c1,
                                                     int c2,
                                                     double a,
                                                     double b,
                                                     double f,
                                                     double fps,
                                                     unsigned int numOfVectors,
                                                     unsigned int numOfPoints,
                                                     std::vector< std::vector<cv::Point> > &positionsListOutput){

    int x, y;
    double dt = 1.0 / fps;  // time step
    for(int i = 0; i < numOfVectors; i++){
        std::vector<cv::Point> auxVector;
        double t = 0.0;
        for(int k = 0; k < numOfPoints; k++){
            x = c1 + a * std::cos(2.0 * M_PI * f * t);
            y = c2 + b * std::sin(2.0 * M_PI * f * t);;
            cv::Point pos(x, y);
            auxVector.push_back(pos);
            t = t + dt;
        }
        positionsListOutput.push_back(auxVector);
    }
    //std::cout << "positionsListOutput size = " << positionsListOutput.size() << std::endl;
}

void SemanticDetector::getFeaturesVector(cv::Point &faceCenterPoint,
                          double pixelSizeInCmTemp,
                          double minTimeToDetect,
                          double fps,
                          double interpolationTimeStep,
                          std::vector<cv::Point> &LHandPositions,
                          std::vector<cv::Point> &RHandPositions,
                          std::vector<double> &LHAllConcatOutput,
                          std::vector<double> &RHAllConcatOutput){


    std::vector<double> auxLH, auxRH;

    getHandFeaturesVector(faceCenterPoint,
                          pixelSizeInCmTemp,
                          minTimeToDetect,
                          fps,
                          interpolationTimeStep,
                          LHandPositions,
                          LHAllConcatOutput);

    getHandFeaturesVector(faceCenterPoint,
                          pixelSizeInCmTemp,
                          minTimeToDetect,
                          fps,
                          interpolationTimeStep,
                          RHandPositions,
                          RHAllConcatOutput);
}


template <typename T>
void SemanticDetector::extendVectorRepeting (std::vector<T> &vect, int lenghtVec, std::vector<T> &vectOut) {

   if(lenghtVec <= vect.size()){
        vectOut.insert(vectOut.end(), vect.begin(), vect.begin() + lenghtVec);
   } else {
        // multiples
        int a = lenghtVec / vect.size();
        for(int i = 0; i < a; i++){
            vectOut.insert(vectOut.end(), vect.begin(), vect.begin() + vect.size());
        }
        // last one
        int b = lenghtVec - a * vect.size();
        vectOut.insert(vectOut.end(), vect.begin(), vect.begin() + b);
   }
}

void SemanticDetector::getFeaturedData(std::vector<cv::Point>               &faceCenterPointList,
                                       std::vector<double>                  &pixelSizeInCmTempList,
                                       double                               minTimeToDetect,
                                       std::vector<double>                  &fpsList,
                                       double                               interpolationTimeStep,
                                       std::vector<std::vector<cv::Point>>  &LHandPositionsList,
                                       std::vector<std::vector<cv::Point>>  &RHandPositionsList,
                                       cv::Mat                              &LHFeaturesMatListOutput,
                                       cv::Mat                              &RHFeaturesMatListOutput){

    int numOfData = faceCenterPointList.size();
    for( int i = 0; i < numOfData; i++){

        std::vector<double> LHFeatures, RHFeatures;

        // get single features
        getFeaturesVector(faceCenterPointList[i],
                      pixelSizeInCmTempList[i],
                      minTimeToDetect,
                      fpsList[i],
                      interpolationTimeStep,
                      LHandPositionsList[i],
                      RHandPositionsList[i],
                      LHFeatures,
                      RHFeatures);

        // transform in Mat variables
        cv::Mat LHFeaturesMat(LHFeatures,true);
        cv::Mat RHFeaturesMat(RHFeatures,true);

        // transpose - convert vector into a single row (1xm)
        LHFeaturesMat = LHFeaturesMat.t();
        RHFeaturesMat = RHFeaturesMat.t();

        // add to the list
        LHFeaturesMatListOutput.push_back(LHFeaturesMat);
        RHFeaturesMatListOutput.push_back(RHFeaturesMat);

    }
}


/*
* Took from:
* https://github.com/opencv/opencv/blob/master/samples/cpp/logistic_regression.cpp
*/
float SemanticDetector::calculateAccuracyPercent(const cv::Mat &original, const cv::Mat &predicted){
    return 100 * (float)countNonZero(original == predicted) / predicted.rows;
}

/*
 * precision = true positives / (true positives + false positives)
*/
float SemanticDetector::calculatePrecision(const cv::Mat &originalLabels, const cv::Mat &responses){
    float tp = 0,   // true positives
          fp = 0;   // false positives


    cv::Mat A = (originalLabels > 0);
    cv::Mat B = (responses > 0);

    // Logic function:
    // (A.B = 1)
    cv::Mat AB;
    bitwise_and(A, B, AB);
    tp = (float) countNonZero(AB); // true positives

    // Logic function:
    // (\A.B = 1)
    cv::Mat A_NOT;
    bitwise_not(A, A_NOT);

    cv::Mat A_NOTB;
    bitwise_and(A_NOT, B, A_NOTB);

    fp = (float) countNonZero(A_NOTB);

    return tp / (tp + fp); // precision;
}

/*
 * recall = true positive / (true positives + false negatives)
*/
float SemanticDetector::calculateRecall(const cv::Mat &originalLabels, const cv::Mat &responses){
    float tp = 0,   // true positives
          fn = 0;   // false negatives

    //cv::Mat test = (cv::Mat_<float>(9,1) << 2.5, 0, 1, 1, 1, 0, 0, 1);
    //cv::Mat test2 = (cv::Mat_<float>(9,1) << 0, 1, 1, 0, 1, 0, 1, 1);

    cv::Mat A = (originalLabels > 0);
    cv::Mat B = (responses > 0);

    // Logic function:
    // (A.B = 1)
    cv::Mat AB;
    bitwise_and(A, B, AB);
    tp = (float) countNonZero(AB); // true positives

    // Logic function:
    // (A.\B = 1)
    cv::Mat B_NOT;
    bitwise_not(B, B_NOT);

    cv::Mat AB_NOT;
    bitwise_and(A, B_NOT, AB_NOT);

    fn = (float) countNonZero(AB_NOT); // false negatives

    return tp / (tp + fn);
}

/*
 * F1Score = 2 * precision * recall / (precision + recall)
*/
float SemanticDetector::calculateF1Score(float precision, float recall){
    return 2.0 * precision * recall / (precision + recall);
}

/*
 * sigmoid = 1 ./ (1 + exp(-z) );
 */
void SemanticDetector::sigmoid(cv::Mat &M, cv::Mat &sigmoidOutput){
    cv::Mat val;
    //std::cout << "-M = " << -M << std::endl;
    cv::exp(-M,val);
    //std::cout << " -M val = " << val << std::endl;
    cv::add(val,1,val);
    //std::cout << "add val = " << val << std::endl;
    cv::divide(1,val,sigmoidOutput);
}

/*
* lambda -> regularization value 0.0 or 1.0
*
* h = sigmoid(X * theta);
* J1 = (1 / m) * ( -y' * log(h) - ( 1 - y )' * log(1 - h) );
* J2 = (lambda / (2*m)) * sum(theta(2:end).^2);
*
* J = J1 + J2;
*/
void SemanticDetector::costFunction(cv::Mat &X, cv::Mat y, cv::Mat theta, double lambda, double JOutput){

    double JNorm;

    //std::cout << "X \t" << X.rows   << " x " << X.cols << std::endl;
    //std::cout << "y \t" << y.rows   << " x " << y.cols << std::endl;
    //std::cout << "theta \t" << theta.rows   << " x " << theta.cols << std::endl;

    //std::cout << "lambda = " << lambda << std::endl;


    cv::Mat h;
    cv::Mat calc;

    cv::Mat X_bias = cv::Mat::ones(X.rows, 1, CV_32F);
    cv::hconcat(X_bias, X, X_bias);
    //std::cout << "X_bias " << X_bias.rows   << " x " << X_bias.cols << std::endl;

    calc = X_bias * theta.t();
    //std::cout << "calc " << calc.rows   << " x " << calc.cols << std::endl;

    sigmoid(calc, h);
    //std::cout << "h " << h.rows   << " x " << h.cols << std::endl;

    //cv::Mat h;
    //cv::Mat aux = X * theta;
    //sigmoid(aux, h);
    cv::Mat aux;
    cv::log(h, aux);

    cv::Mat b = -y.t();
    //std::cout << "b \t" << b.rows   << " x " << b.cols << std::endl;
    cv::Mat a1 = b * aux;
    std::cout << "a1 = " << a1 << std::endl;

    cv::Mat aux2;
    cv::log(1.0 - h, aux2);
    cv::Mat a2 = ( ( 1.0 - y ).t() ) * aux2;
    std::cout << "a2 = " << a2 << std::endl;

    cv::Mat J1 = (1.0 / y.rows) * ( a1 - a2 );
    std::cout << "J1 = " << J1 << std::endl;

    cv::Mat sub_theta = theta(cv::Range::all(), cv::Range(1, theta.cols-1));
    cv::Mat sub_thetaPow;
    cv::pow(sub_theta, 2.0, sub_thetaPow);
    double sum = cv::sum(sub_thetaPow)[0];

    double J2double = (lambda / 2.0 * X.rows) * sum;

    cv::Mat J2 = (cv::Mat_<float>(1,1) << J2double);
    std::cout << "J2 = " << J2 << std::endl;

    cv::Mat J(1,1,CV_32F);
    cv::add(J1, J2, J);
    std::cout << "\tJ = " << J << std::endl;
    //std::cout << "JOutput = " << JOutput << std::endl;


}

/*
*   setMat      - set of data to split
*   trainPerc   - [0.0 - 1.0] percentage to take for training data,
*   cvPerc      = [0.0 - 1.0] percentage to take for cross-validation data,
*   testPerc    = [0.0 - 1.0] percentage to take for test data,
*
*/
void SemanticDetector::splitDataInSets(cv::Mat &setMat,
                                          double trainPerc,
                                          double cvPerc,
                                          double testPerc, // not needed
                                          cv::Mat &data_train,
                                          cv::Mat &data_CV,
                                          cv::Mat &data_test){

    int numOfVectors = 1000; // (I should pass it, I guess)
    int firstLimit  = numOfVectors * trainPerc,
        secondLimit = firstLimit + numOfVectors * cvPerc;
    for( int i = 0; i < numOfVectors; i++ ){
        for(int k = 0; k < setMat.rows; k+=numOfVectors){

            int n = k + i; // 0 1000 2000 ... (numOfVectors-1) ...
                           // 1 1001 2001 ...

            cv::Mat row = setMat.row(n).clone();

            // Separate data in train, CV and test sets
            if(i < firstLimit) {
                data_train.push_back(row);
            } else if( i < secondLimit) {
                data_CV.push_back(row);
            } else { // test data
                data_test.push_back(row);
            }
        }
    }
}

void SemanticDetector::createDataLabels(cv::Mat &setMat, bool label, cv::Mat &labelsOutput){

    float value;
    int numOfRows = setMat.rows;
    if(label){
        value = 1.0;
    }else {
        value = 0.0;
    }

    for(int i = 0 ; i < numOfRows; i++){
        labelsOutput.push_back(value);
    }
}

void SemanticDetector::trainClassifier(InfoClassifier &infoClas){
    // create variables to hold the data sets
    cv::Mat data_train,
            data_CV,
            data_test,
            labels_train,
            labels_CV,
            labels_test;

    // get sets of data for training
    getClassifiersTrainData(infoClas,
                            data_train,
                            data_CV,
                            data_test,
                            labels_train,
                            labels_CV,
                            labels_test);

    // Convert data to CV_32F in order to pass to the logisticsTrain function
    data_train.convertTo(data_train, CV_32F);
    labels_train.convertTo(labels_train, CV_32F);

    data_CV.convertTo(data_CV, CV_32F);
    labels_CV.convertTo(labels_CV, CV_32F);

    data_test.convertTo(data_test, CV_32F);
    labels_test.convertTo(labels_test, CV_32F);

    std::cout  << "data_train   => rows: "  << data_train.rows      << ", cols: " << data_train.cols << std::endl;
    std::cout  << "labels_train => rows: "  << labels_train.rows    << ", cols: " << labels_train.cols << std::endl;
    std::cout  << "data_CV      => rows: "  << data_CV.rows         << ", cols: " << data_CV.cols << std::endl;
    std::cout  << "labels_CV    => rows: "  << labels_CV.rows       << ", cols: " << labels_CV.cols << std::endl;
    std::cout  << "data_test    => rows: "  << data_test.rows       << ", cols: " << data_test.cols << std::endl;
    std::cout  << "labels_test  => rows: "  << labels_test.rows     << ", cols: " << labels_test.cols << std::endl;

    // than train
    logisticsTrain(data_train, data_CV, data_test, labels_train, labels_CV, labels_test);

}

void SemanticDetector::getClassifiersTrainData(InfoClassifier &infoClas,
                                               cv::Mat &data_train_output,
                                               cv::Mat &data_CV_output,
                                               cv::Mat &data_test_output,
                                               cv::Mat &labels_train_output,
                                               cv::Mat &labels_CV_output,
                                               cv::Mat &labels_test_output){
    // List of data that is gonna hold all data files information
    std::vector<cv::Point>  faceCenterPointList_positiveData,
                            faceCenterPointList_negativeData;

    std::vector<double>     pixelSizeInCmTempList_positiveData,
                            pixelSizeInCmTempList_negativeData;

    std::vector<double>     fpsList_positiveData,
                            fpsList_negativeData;

    std::vector<int>        gestureLabelList_positiveData,
                            gestureLabelList_negativeData;

    std::vector< std::vector<cv::Point> >  LHandPositionsList_positiveData,
                                           LHandPositionsList_negativeData;

    std::vector< std::vector<cv::Point> >  RHandPositionsList_positiveData,
                                           RHandPositionsList_negativeData;

    //-----------------------------------------------------------
    //---------------------Get Data------------------------------

    std::cout << "Reading positive data..." << std::endl;
    getDataFromFiles(infoClas.pathPositiveData,
                    faceCenterPointList_positiveData,
                    pixelSizeInCmTempList_positiveData,
                    fpsList_positiveData,
                    gestureLabelList_positiveData,
                    LHandPositionsList_positiveData,
                    RHandPositionsList_positiveData);
    std::cout << "\tLHandPositionsList_positiveData size = " << LHandPositionsList_positiveData.size() << std::endl;
    std::cout << "DONE" << std::endl;

    std::cout << "Reading negative data..." << std::endl;
    getDataFromFiles(infoClas.pathNegativeData,
                    faceCenterPointList_negativeData,
                    pixelSizeInCmTempList_negativeData,
                    fpsList_negativeData,
                    gestureLabelList_negativeData,
                    LHandPositionsList_negativeData,
                    RHandPositionsList_negativeData);
    std::cout << "\tLHandPositionsList_negativeData size = " << LHandPositionsList_negativeData.size() << std::endl;
    std::cout << "DONE" << std::endl;

    // If one of the static positions is required for any hand  then generate it
    std::vector< std::vector<cv::Point> >  staticPositions_Generated;
    if(infoClas.use_LH_StaticPos_for_negativeData ||
       infoClas.use_RH_StaticPos_for_negativeData ||
       infoClas.use_LH_StaticPos_for_positiveData ||
       infoClas.use_RH_StaticPos_for_positiveData){

        // generate more Static hands positions for different positions
        std::cout << "Generating static positions data..." << std::endl;

        unsigned int numOfVectors = infoClas.genStaticPosInfo.numOfVectors;
        unsigned int numOfPoints = LHandPositionsList_negativeData[0].size();

        int x_start = infoClas.genStaticPosInfo.x_start,
            x_step  = infoClas.genStaticPosInfo.x_step,
            x_end   = infoClas.genStaticPosInfo.x_end;
        int y_start = infoClas.genStaticPosInfo.y_start,
            y_step  = infoClas.genStaticPosInfo.y_step,
            y_end   = infoClas.genStaticPosInfo.y_end;

        std::cout << "\tx_start =  "   << x_start   << std::endl;
        std::cout << "\tx_step =   "   << x_step    << std::endl;
        std::cout << "\tx_end =    "   << x_end     << std::endl;
        std::cout << "\ty_start =  "   << y_start   << std::endl;
        std::cout << "\ty_step =   "   << y_step    << std::endl;
        std::cout << "\ty_end =    "   << y_end     << std::endl;

        for(int x = x_start; x < x_end; x+=x_step){
            for(int y = y_start; y < y_end; y+=y_step){
                createListOfStaticPositions(x,
                                            y,
                                            numOfVectors,
                                            numOfPoints,
                                            staticPositions_Generated);
            }
        }
        std::cout << "\tstaticPositions_Generated size = " << staticPositions_Generated.size() << std::endl;
        std::cout << "DONE" << std::endl;
    }

    // If one of the Elliptical positions is required for any hand then generate it
    std::vector< std::vector<cv::Point> >  ellipticalPositionsList_Generated;
    if(infoClas.use_LH_EllipticalPos_for_negativeData ||
       infoClas.use_RH_EllipticalPos_for_negativeData ||
       infoClas.use_LH_EllipticalPos_for_positiveData ||
       infoClas.use_RH_EllipticalPos_for_positiveData){

        // Generate elliptical movements (ellipses)
        // (x, y) = ( c1 + a*cos(2*pi*f*t) , c2 + b*sin(2*pi*f*t) )
        std::cout << "Generating elliptical positions data..." << std::endl;

        unsigned int numOfVectors = infoClas.genEllipticalPosInfo.numOfVectors;
        unsigned int numOfPoints = LHandPositionsList_negativeData[0].size();

        int c1_start = infoClas.genEllipticalPosInfo.c1_start,
            c1_step  = infoClas.genEllipticalPosInfo.c1_step,
            c1_end   = infoClas.genEllipticalPosInfo.c1_end,
            c2_start = infoClas.genEllipticalPosInfo.c2_start,
            c2_step  = infoClas.genEllipticalPosInfo.c2_step,
            c2_end   = infoClas.genEllipticalPosInfo.c2_end;

        double a_start  = infoClas.genEllipticalPosInfo.a_start,
               a_step   = infoClas.genEllipticalPosInfo.a_step,
               a_end    = infoClas.genEllipticalPosInfo.a_end,
               b_start  = infoClas.genEllipticalPosInfo.b_start,
               b_step   = infoClas.genEllipticalPosInfo.b_step,
               b_end    = infoClas.genEllipticalPosInfo.b_end,
               f_start  = infoClas.genEllipticalPosInfo.f_start,
               f_step   = infoClas.genEllipticalPosInfo.f_step,
               f_end    = infoClas.genEllipticalPosInfo.f_end;

        std::cout << "\tc1_start =  "   << c1_start << std::endl;
        std::cout << "\tc1_step =   "   << c1_step  << std::endl;
        std::cout << "\tc1_end =    "   << c1_end   << std::endl;
        std::cout << "\tc2_start =  "   << c2_start << std::endl;
        std::cout << "\tc2_step =   "   << c2_step  << std::endl;
        std::cout << "\tc2_end =    "   << c2_end   << std::endl;
        std::cout << "\ta_start =   "   << a_start  << std::endl;
        std::cout << "\ta_step =    "   << a_step   << std::endl;
        std::cout << "\ta_end =     "   << a_end    << std::endl;
        std::cout << "\tb_start =   "   << b_start  << std::endl;
        std::cout << "\tb_step =    "   << b_step   << std::endl;
        std::cout << "\tb_end =     "   << b_end    << std::endl;
        std::cout << "\tf_start =   "   << f_start  << std::endl;
        std::cout << "\tf_step =    "   << f_step   << std::endl;
        std::cout << "\tf_end =     "   << f_end    << std::endl;

        for(int c1 = c1_start; c1 < c1_end; c1+=c1_step){
            for(int c2 = c2_start; c2 < c2_end; c2+=c2_step){

                for(double a = a_start; a < a_end; a+=a_step){
                    for(double b = b_start; b < b_end; b+=b_step){

                        for(double f = f_start; f < f_end; f+=f_end){

                            createListOfEllipticalPositions( c1, c2, a, b, f, this->fps,
                                                          numOfVectors,
                                                          numOfPoints,
                                                          ellipticalPositionsList_Generated);
                        }

                    }
                }

            }
        }
        std::cout << "\tellipticalPositionsList_Generated size = " << ellipticalPositionsList_Generated.size() << std::endl;
        std::cout << "DONE" << std::endl;
    }

    //-----------------------------------------------------------
    //--------------------------Get Features---------------------

    // Positive Data Features
    cv::Mat LH_PositiveData_MatList, RH_PositiveData_MatList;
    getFeaturedData(faceCenterPointList_positiveData,
                   pixelSizeInCmTempList_positiveData,
                   this->minTimeToDetect,
                   fpsList_positiveData,
                   this->interpolationTimeStep,
                   LHandPositionsList_positiveData,
                   RHandPositionsList_positiveData,

                   LH_PositiveData_MatList,
                   RH_PositiveData_MatList);
    std::cout  << "LH_PositiveData_MatList  => rows: "  << LH_PositiveData_MatList.rows      << ", cols: " << LH_PositiveData_MatList.cols << std::endl;

    // Negative Data Features
    cv::Mat LH_NegativeData_MatList, RH_NegativeData_MatList;
    getFeaturedData(faceCenterPointList_negativeData,
                   pixelSizeInCmTempList_negativeData,
                   this->minTimeToDetect,
                   fpsList_positiveData,
                   this->interpolationTimeStep,
                   LHandPositionsList_negativeData,
                   RHandPositionsList_negativeData,

                   LH_NegativeData_MatList,
                   RH_NegativeData_MatList);
    std::cout  << "LH_NegativeData_MatList  => rows: "  << LH_NegativeData_MatList.rows      << ", cols: " << LH_NegativeData_MatList.cols << std::endl;

    // Static Data Features
    cv::Mat LH_StaticPosGen_MatList, RH_StaticPosGen_MatList;
    if(infoClas.use_LH_StaticPos_for_negativeData ||
       infoClas.use_RH_StaticPos_for_negativeData ||
       infoClas.use_LH_StaticPos_for_positiveData ||
       infoClas.use_RH_StaticPos_for_positiveData){

        // HERE I use all info from the real data of negativeData
        // but I will pass the generated static positions

        // Extending lists in order to have the same size of staticPositions_Generated
        // extend face Point list
        std::vector<cv::Point> extended_faceCenterPointList_negativeData;
        extendVectorRepeting (faceCenterPointList_negativeData,
                              staticPositions_Generated.size(),
                              extended_faceCenterPointList_negativeData);

        std::vector<double> extended_pixelSizeInCmTempList_negativeData;
        extendVectorRepeting (pixelSizeInCmTempList_negativeData,
                              staticPositions_Generated.size(),
                              extended_pixelSizeInCmTempList_negativeData);

        std::vector<double> extended_fpsList_negativeData;
        extendVectorRepeting (fpsList_negativeData,
                              staticPositions_Generated.size(),
                              extended_fpsList_negativeData);

        getFeaturedData(extended_faceCenterPointList_negativeData,
                       extended_pixelSizeInCmTempList_negativeData,
                       this->minTimeToDetect,
                       extended_fpsList_negativeData,
                       this->interpolationTimeStep,
                       staticPositions_Generated, // LHandPositionsList
                       staticPositions_Generated, // RHandPositionsList

                       LH_StaticPosGen_MatList,
                       RH_StaticPosGen_MatList);
        //std::cout  << "LH_StaticPosGen_MatList   => rows: "  << LH_StaticPosGen_MatList.rows      << ", cols: " << LH_StaticPosGen_MatList.cols << std::endl;
    }

    // Elliptical Data Features
    cv::Mat LH_EllipticalPosGen_MatList, RH_EllipticalPosGen_MatList;
    if(infoClas.use_LH_EllipticalPos_for_negativeData ||
       infoClas.use_RH_EllipticalPos_for_negativeData ||
       infoClas.use_LH_EllipticalPos_for_positiveData ||
       infoClas.use_RH_EllipticalPos_for_positiveData){

        // Generated Elliptical Positions
        std::vector<cv::Point> extended_faceCenterPointList_OfEllipticalPositions;
        extendVectorRepeting (faceCenterPointList_negativeData,
                              ellipticalPositionsList_Generated.size(),
                              extended_faceCenterPointList_OfEllipticalPositions);

        std::vector<double> extended_pixelSizeInCmTempList_OfEllipticalPositions;
        extendVectorRepeting (pixelSizeInCmTempList_negativeData,
                              ellipticalPositionsList_Generated.size(),
                              extended_pixelSizeInCmTempList_OfEllipticalPositions);

        std::vector<double> extended_fpsList_OfEllipticalPositions;
        extendVectorRepeting (fpsList_positiveData,
                              ellipticalPositionsList_Generated.size(),
                              extended_fpsList_OfEllipticalPositions);


        getFeaturedData(extended_faceCenterPointList_OfEllipticalPositions,
                       extended_pixelSizeInCmTempList_OfEllipticalPositions,
                       this->minTimeToDetect,
                       extended_fpsList_OfEllipticalPositions,
                       this->interpolationTimeStep,
                       ellipticalPositionsList_Generated,  // LHandPositionsList
                       ellipticalPositionsList_Generated,  // RHandPositionsList

                       LH_EllipticalPosGen_MatList,
                       RH_EllipticalPosGen_MatList);
        //std::cout  << "LH_EllipticalPosGen_MatList   => rows: "  << LH_EllipticalPosGen_MatList.rows      << ", cols: " << LH_StaticPosGen_MatList.cols << std::endl;

    }

    //--------------------------------------------------------------------------
    //-------------------------- Data concatenations ---------------------------

    cv::Mat LH_AllFalsePos = LH_NegativeData_MatList,
            RH_AllFalsePos = RH_NegativeData_MatList,
            LH_AllTruePos = LH_PositiveData_MatList,
            RH_AllTruePos = RH_PositiveData_MatList;

    // LH, Negative data and Static Positions
    if(infoClas.use_LH_StaticPos_for_negativeData == true){
        cv::vconcat(LH_AllFalsePos, LH_StaticPosGen_MatList, LH_AllFalsePos);
    }
    // LH, Negative data and Elliptical Positions
    if(infoClas.use_LH_EllipticalPos_for_negativeData == true){
        cv::vconcat(LH_AllFalsePos, LH_EllipticalPosGen_MatList, LH_AllFalsePos);
    }

    // RH, Negative data and Static Positions
    if(infoClas.use_RH_StaticPos_for_negativeData == true){
        cv::vconcat(RH_AllFalsePos, RH_StaticPosGen_MatList, RH_AllFalsePos);
    }
    // RH, Negative data and Elliptical Positions
    if(infoClas.use_RH_EllipticalPos_for_negativeData == true){
        cv::vconcat(RH_AllFalsePos, RH_EllipticalPosGen_MatList, RH_AllFalsePos);
    }

    // LH and Positive data and Static Positions
    if(infoClas.use_LH_StaticPos_for_positiveData == true){
        cv::vconcat(LH_AllTruePos, LH_StaticPosGen_MatList, LH_AllTruePos);
    }
    // LH and Positive data and Elliptical Positions
    if(infoClas.use_LH_EllipticalPos_for_positiveData == true){
        cv::vconcat(LH_AllTruePos, LH_EllipticalPosGen_MatList, LH_AllTruePos);
    }

    // RH, Positive data and Static Positions
    if(infoClas.use_RH_StaticPos_for_positiveData == true){
        cv::vconcat(RH_AllTruePos, RH_StaticPosGen_MatList, RH_AllTruePos);
    }
    // RH, Positive data and Elliptical Positions
    if(infoClas.use_RH_EllipticalPos_for_positiveData == true){
        cv::vconcat(RH_AllTruePos, RH_EllipticalPosGen_MatList, RH_AllTruePos);
    }


    std::cout  << "LH_AllFalsePos  => rows: "  << LH_AllFalsePos.rows      << ", cols: " << LH_AllFalsePos.cols << std::endl;
    std::cout  << "RH_AllFalsePos  => rows: "  << RH_AllFalsePos.rows      << ", cols: " << RH_AllFalsePos.cols << std::endl;
    std::cout  << "LH_AllTruePos   => rows: "  << LH_AllTruePos.rows       << ", cols: " << LH_AllTruePos.cols << std::endl;
    std::cout  << "RH_AllTruePos   => rows: "  << RH_AllTruePos.rows       << ", cols: " << RH_AllTruePos.cols << std::endl;


    //--------------------------------------------------------------------------
    //---------------------- separate data and Label data ----------------------
    double trainPerc = infoClas.trainingSets.trainPerc,
           cvPerc    = infoClas.trainingSets.cvPerc,
           testPerc  = infoClas.trainingSets.testPerc;

    // LH_AllFalsePos
    cv::Mat auxTrain1,      auxCV1,         auxTest1,
            auxLabelTrain1, auxLabelCV1,    auxLabelTest1;
    splitDataInSets(LH_AllFalsePos, trainPerc, cvPerc, testPerc, auxTrain1, auxCV1, auxTest1);

    createDataLabels(auxTrain1, false, auxLabelTrain1);
    createDataLabels(auxCV1,    false, auxLabelCV1);
    createDataLabels(auxTest1,  false, auxLabelTest1);

    // LH_PositiveData_MatList
    cv::Mat auxTrain2,      auxCV2,         auxTest2,
            auxLabelTrain2, auxLabelCV2,    auxLabelTest2;
    splitDataInSets(LH_AllTruePos, trainPerc, cvPerc, testPerc, auxTrain2, auxCV2, auxTest2);

    createDataLabels(auxTrain2, true, auxLabelTrain2);
    createDataLabels(auxCV2,    true, auxLabelCV2);
    createDataLabels(auxTest2,  true, auxLabelTest2);

    // ? Needed ?
    // merge data and labels in the same way to make sure we have correlated data
    mergeMats(auxTrain1, auxTrain2, data_train_output);
    mergeMats(auxCV1,    auxCV2,    data_CV_output);
    mergeMats(auxTest1,  auxTest2,  data_test_output);

    mergeMats(auxLabelTrain1, auxLabelTrain2, labels_train_output);
    mergeMats(auxLabelCV1,    auxLabelCV2,    labels_CV_output);
    mergeMats(auxLabelTest1,  auxLabelTest2,  labels_test_output);

}

void SemanticDetector::mergeMats(cv::Mat &mat1, cv::Mat &mat2, cv::Mat &matOutput){
    int nRows1 = mat1.rows,
        nRows2 = mat2.rows;
    int total = std::max(mat1.rows, mat2.rows);
    for(int i = 0; i < total; i++){

        cv::Mat row1;
        cv::Mat row2;
        if( (i < nRows1) && (i < nRows2) ){ // i is in the range of both mats
            row1 = mat1.row(i);
            row2 = mat2.row(i);
            matOutput.push_back(row1);
            matOutput.push_back(row2);
        }else if( (i < nRows1) && (i >= nRows2) ){  // mat2 is shorter
            row1 = mat1.row(i);
            matOutput.push_back(row1);
        }else if( (i >= nRows1) && (i < nRows2) ){  // mat1 is shorter
            row2 = mat2.row(i);
            matOutput.push_back(row2);
        }else{
            // something went wrong
        }
    }
}

/*
* Based on:
* https://github.com/opencv/opencv/blob/master/samples/cpp/logistic_regression.cpp
*
* ANOTHER SOURCE
* https://stackoverflow.com/questions/37282275/logistic-regression-on-mnist-dataset
*/
void SemanticDetector::logisticsTrain(cv::Mat &data_train,
                                      cv::Mat &data_CV,
                                      cv::Mat &data_test,
                                      cv::Mat &labels_train,
                                      cv::Mat &labels_CV,
                                      cv::Mat &labels_test){

    double learningRate = 0.03;
    int iterations,
        miniBatchSize = 1;

    std::vector<int> iterationsArray = {72000};
    std::cout << "learningRate  = " << learningRate << std::endl;

    for(int i = 0; i < iterationsArray.size(); i++){

        std::cout << "/-----------------------------------------/" << std::endl;
        iterations = iterationsArray[i];
        std::cout << "iterations    = " << iterations << std::endl;
        // simple case with batch gradient
        std::cout << "setting training parameters..." << std::endl;
        //! [init]
        cv::Ptr<cv::ml::LogisticRegression> lr1 = cv::ml::LogisticRegression::create();
        lr1->setLearningRate(learningRate);
        lr1->setIterations(iterations);
        lr1->setRegularization(cv::ml::LogisticRegression::REG_L2); // sum(w^2)
        lr1->setTrainMethod(cv::ml::LogisticRegression::BATCH);
        lr1->setMiniBatchSize(miniBatchSize); // Is it needed? I want to compute all batch examples
        std::cout << "training..." << std::endl;

        std::chrono::time_point<std::chrono::system_clock> startTime, endTime;
        startTime = std::chrono::system_clock::now();

        //! [init]
        lr1->train(data_train, cv::ml::ROW_SAMPLE, labels_train);

        endTime = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = endTime-startTime;
        std::cout << "\telapsed time: " << elapsed_seconds.count() << "s" << std::endl;;

        std::cout << "done!" << std::endl;

        // save the classifier
        const char saveFilename[] = "LHClassifier.xml";
        std::cout << "saving the classifier to " << saveFilename << std::endl;
        lr1->save(saveFilename);

        // predictions
        cv::Mat responses;
        lr1->predict(data_test, responses);

        // show prediction report
        labels_test.convertTo(labels_test, CV_32S);

        float precision = calculatePrecision(labels_test, responses),
              recall    = calculateRecall(labels_test, responses),
              F1Score   = calculateF1Score(precision, recall);
        std::cout << "Accuracy: "   << calculateAccuracyPercent(labels_test, responses) << "%" << std::endl;
        std::cout << "Precision: "  << precision << std::endl;
        std::cout << "Recall: "     << recall << std::endl;
        std::cout << "F1Score: "    << F1Score << std::endl;


        // cost function
        double JOutput;
        cv::Mat thetas = lr1->get_learnt_thetas();
        labels_test.convertTo(labels_test, CV_32F);


        double regularization = 1.0;
        /*
        if(lr1->getRegularization != cv::ml::LogisticRegression::REG_DISABLE){
            regularization = 1.0;
        }else{
            regularization = 0.0;
        }
        */

        std::cout << "data_train: " << std::endl;
        costFunction(data_train,    labels_train,   thetas, regularization, JOutput);
        std::cout << "data_CV: " << std::endl;
        costFunction(data_CV,       labels_CV,      thetas, regularization, JOutput);
        std::cout << "data_test: " << std::endl;
        costFunction(data_test,     labels_test,    thetas, regularization, JOutput);



    }
}


/*
* faceCenterPoint       ->  Position of the face
* pixelSizeInCmTemp     ->
* positions             ->  This is a array of vectors. Those vectors contain the history positions of the tracked body part.
*                           The positions must have separated within the some interval of time.
*                           The newest data is located at the end of the vector.
*/
void SemanticDetector::detect(cv::Point faceCenterPoint,
                              double pixelSizeInCmTemp,
                              std::vector<cv::Point> positions[],
                              float &gestureOutput,
                              int frameIndex) {
    /*
    // checks null parameters
    if(faceCenterPoint == 0L || !pixelSizeInCmTemp || !positions){
        std::cout << "SemanticDetector::detect -> Null detected" << std::endl;
        return;
    }
    */

    double periodFPS = 1.0 / ((double)this->fps); // period of FPS = 1/fps
    //Take left and right hand positions
    std::vector<cv::Point> LHandPositions = positions[0];
    std::vector<cv::Point> RHandPositions = positions[1];
    cv::Point tempPoint;

    // check if we have a minimum time for being analyzed (It depends on the fps and history size of the class)
    double LHandPositionsDuraction = ((double)LHandPositions.size()) * periodFPS;
    double RHandPositionsDuraction = ((double)RHandPositions.size()) * periodFPS;
    if( (LHandPositionsDuraction < (this->minTimeToDetect)) || (RHandPositionsDuraction < (this->minTimeToDetect))){
        std::cout << "SemanticDetector::detect -> No minimum time to analyze gestures! Minimum time:" << ( this->minTimeToDetect) << std::endl;
        std::cout << "\tMinimum time:" << ( this->minTimeToDetect)
        << " || Given time (Left hand): " << LHandPositionsDuraction
        << " || Given time (right hand): " << RHandPositionsDuraction << std::endl;
        return;
    }

    // Take only the newest samples within the minTimeToDetect
    int index = (int) std::ceil( this->minTimeToDetect / periodFPS ); // get the number of newest samples that we want to keep
    LHandPositions.erase(LHandPositions.begin(), LHandPositions.end() - index - 1); // erase oldest samples
    RHandPositions.erase(RHandPositions.begin(), RHandPositions.end() - index - 1); // erase oldest samples



// this section will be used to predict gestures with classifiers already created
#ifndef TRAINING

    std::vector<double> LHAllConcat, RHAllConcat;
    getFeaturesVector(faceCenterPoint,
                      pixelSizeInCmTemp,
                      this->minTimeToDetect,
                      (double)this->fps,
                      this->interpolationTimeStep,
                      LHandPositions,
                      RHandPositions,
                      LHAllConcat,
                      RHAllConcat);

    // convert data to cv::Mat
    cv::Mat LHAllInfo(LHAllConcat, true);
    cv::Mat RHAllInfo(RHAllConcat, true);

    LHAllInfo.convertTo(LHAllInfo, CV_32F);

    // transpose - convert vector into one single row (1xn)
    LHAllInfo = LHAllInfo.t();

    // load classifier
    const char saveFilename[] = "LHClassifier.xml";
    cv::Ptr<cv::ml::LogisticRegression> lr2 = cv::ml::StatModel::load<cv::ml::LogisticRegression>(saveFilename);

    // predict response (By Opencv)
    cv::Mat responses;
    lr2->predict(LHAllInfo, responses);

    // predict response (By me - no binary)
    cv::Mat thetas = lr2->get_learnt_thetas();
    cv::Mat X_bias = cv::Mat::ones(LHAllInfo.rows, 1, CV_32F);
    cv::hconcat(X_bias, LHAllInfo, X_bias);
    cv::Mat calc = X_bias * thetas.t();
    cv::Mat h;
    sigmoid(calc, h);

    // compare responses
    //std::cout << "response : " << responses << " h : " << h << std::endl;

    // add responses to the filter according with certain level of trust
    float h_float = h.at<float>(0,0);
    if(h_float >= trust){
        LHShake_Filter[rowNum] = 1;
    }else {
        LHShake_Filter[rowNum] = 0;
    }

    // if we have all filter filled with positive detections than
    int sum = std::accumulate(LHShake_Filter.begin(), LHShake_Filter.end(), 0);
    if( sum >= LHFilterLength ){
        flag_LHShake = true;    // turn the flat to true
        gestureOutput = 101.0;
    }else{
        flag_LHShake = false;   // turn the flat to false
        gestureOutput = 100.0;
    }
    //std::cout << " \tflag_LHShake = " << flag_LHShake << std::endl;

    // actualize rowNum
    rowNum++;
    if( (rowNum % LHFilterLength) == 0 ){
        rowNum = 0;
    }

#else // this section will be used to train and create classifiers
    #ifdef TRAINING_SAVE_DATA
        /*
         *  This part will analyze the videos, take data from of specific timestamps,
         *  generate more data and save all in files.
        */
        std::cout << "================================" << std::endl;
        std::cout << "|  TRAINING_SAVE_DATA defined! |" << std::endl;
        std::cout << "================================" << std::endl;

        std::string fileName, fullPath;
        int gestureLabel;

        // If frame is the last one of the gesture
        if(frameIndex == videosInfo.find(capVideoName)->second){
            std::cout << "Taking data from " << capVideoName << " at frame " << frameIndex << std::endl;

            // save history into a file
            fileName = capVideoName;
            fullPath = pathOriginalData + fileName + ".yml";
            gestureLabel = 3; // (REALLY BAD IMPLEMENTATION FOR NOW!!)
                              //    1 - "RHShake"
                              //    2 - "LHShake"
                              //    3 - "StaticHandsUp"
            saveDataInFile(fullPath,
                           capVideoName,
                           gestureLabel,
                           pixelSizeInCmTemp,
                           faceCenterPoint,
                           LHandPositions,
                           RHandPositions);

            // generate data with some deviations of the original one
            generateDataFromOriginal(pathCreatedData,
                                     capVideoName,
                                     gestureLabel,
                                     pixelSizeInCmTemp,
                                     faceCenterPoint,
                                     LHandPositions,
                                     RHandPositions);
        }
    #endif // TRAINING_SAVE_DATA

#endif

}



