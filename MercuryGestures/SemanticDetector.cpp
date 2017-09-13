/*
* Company: Almende BV
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
    /*
    std::cout << "xHNormalizedOutput Size: " << xHNormalizedOutput.size() << std::endl;
    std::cout << "yHNormalizedOutput Size: " << yHNormalizedOutput.size() << std::endl;

    cv::Mat auxMat(xHNormalizedOutput, true);
    std::cout << "xHNormalizedOutput : " << auxMat << std::endl;

    cv::Mat auxMat2(yHNormalizedOutput, true);
    std::cout << "yHNormalizedOutput : " << auxMat2 << std::endl;
    */
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

    //std::cout << "---------------- NEW velocities -------------------------------" << std::endl;
    double velocity;
    for(int i = 0; i < vectorPositions.size() - 1; i++){
        velocity = (vectorPositions[i+1] - vectorPositions[i]) / deltaTime;
        //std::cout << "i = " << i << " --> vectorPositions[i+1]: " << vectorPositions[i+1] << " - vectorPositions[i]: " << vectorPositions[i] << std::endl;
        //std::cout << "\tdeltaTime = " << deltaTime << " --> velocity: " << velocity << std::endl;
        vectorOutputVelocities.push_back( std::abs(velocity) );
    }
    //std::cout << "---------------- END velocities -------------------------------" << std::endl;
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
            // push back on hte list if it is not "." or ".."
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

// ( c1 + a*cos(2*pi*f*t) , c2 + b*sin(2*pi*f*t) )
void SemanticDetector::createListOfCircularPositions(int c1,
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
* Here I would like to have the possibility to load multi classifiers to classify multi gestures!
* Maybe I should load a file with the name of all classifiers and then call one by one...lets see!
*
* Based on:
* https://github.com/opencv/opencv/blob/master/samples/cpp/logistic_regression.cpp
*/
void SemanticDetector::logisticsPredition(std::string info, cv::Mat * HandsInfo){

    cv::Mat LHInfo, RHInfo;

    if( info.compare("L") == 0 ) {
        // get hand
        LHInfo = HandsInfo[0];

        // load all classifiers
        //const char fileName[] = "myClassifier.xml";
        //cv::Ptr<cv::ml::LogisticRegression> lr2 = cv::ml::StatModel::load<cv::ml::LogisticRegression>(fileName);
    } else if( info.compare("R") == 0 ) {
        // get hand
        RHInfo = HandsInfo[0];
        // load all classifiers

    } else if( info.compare("LR") == 0 ) {
        // get hands
        LHInfo = HandsInfo[0];
        RHInfo = HandsInfo[1];

        // load all classifiers

    } else {
        // error
        return;
    }


    // test with classifiers


    // get the predition
    //cv::Mat responses2;
    //lr2->predict(data_test, responses2);

    /*
    cv::Mat data_test;
    const char saveFilename[] = "NewLR_Trained.xml";
    // load the classifier onto new object
    std::cout << "loading a new classifier from " << saveFilename << std::endl;
    cv::Ptr<cv::ml::LogisticRegression> lr2 = cv::ml::StatModel::load<cv::ml::LogisticRegression>(saveFilename);

    // predict using loaded classifier
    std::cout << "predicting the dataset using the loaded classfier...";
    cv::Mat responses2;
    lr2->predict(data_test, responses2);
    std::cout << "done!" << std::endl;
    */
}


/*
* Took from:
* https://github.com/opencv/opencv/blob/master/samples/cpp/logistic_regression.cpp
*/
float SemanticDetector::calculateAccuracyPercent(const cv::Mat &original, const cv::Mat &predicted){
    return 100 * (float)countNonZero(original == predicted) / predicted.rows;
}

// sigmoid = 1 ./ (1 + exp(-z) );
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
* lambda -> regularization value ?? 1.0 ??
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

    // !!!! ATTENTION !!!!
    lambda = 1.0;
    std::cout << "lambda = " << lambda << std::endl;


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
    //double J2double = (1.0 / 2.0 * X.rows) * sum;

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

void SemanticDetector::getClassifiersTrainData(cv::Mat &data_train,
                                               cv::Mat &data_CV,
                                               cv::Mat &data_test,
                                               cv::Mat &labels_train,
                                               cv::Mat &labels_CV,
                                               cv::Mat &labels_test){
    // List of data that is gonna hold all data files information
    std::vector<cv::Point>  faceCenterPointList_OfLHShake,
                            faceCenterPointList_OfStaticHandsUp;

    std::vector<double>     pixelSizeInCmTempList_OfLHShake,
                            pixelSizeInCmTempList_OfStaticHandsUp;

    std::vector<double>     fpsList_OfLHShake,
                            fpsList_OfStaticHandsUp;

    std::vector<int>        gestureLabelList_OfLHShake,
                            gestureLabelList_OfStaticHandsUp;

    std::vector< std::vector<cv::Point> >  LHandPositionsList_OfLHShake,
                                           LHandPositionsList_OfStaticHandsUp;

    std::vector< std::vector<cv::Point> >  RHandPositionsList_OfLHShake,
                                           RHandPositionsList_OfStaticHandsUp;

    //-----------------------------------------------------------
    //---------------------Get Data------------------------------

    std::cout << "Reading LHShake data..." << std::endl;
    getDataFromFiles(pathLHData,
                    faceCenterPointList_OfLHShake,
                    pixelSizeInCmTempList_OfLHShake,
                    fpsList_OfLHShake,
                    gestureLabelList_OfLHShake,
                    LHandPositionsList_OfLHShake,
                    RHandPositionsList_OfLHShake);
    std::cout << "\tLHandPositionsList_OfLHShake size = " << LHandPositionsList_OfLHShake.size() << std::endl;
    std::cout << "DONE" << std::endl;

    std::cout << "Reading StaticHandsUp data..." << std::endl;
    getDataFromFiles(pathStaticHandsUpData,
                    faceCenterPointList_OfStaticHandsUp,
                    pixelSizeInCmTempList_OfStaticHandsUp,
                    fpsList_OfStaticHandsUp,
                    gestureLabelList_OfStaticHandsUp,
                    LHandPositionsList_OfStaticHandsUp,
                    RHandPositionsList_OfStaticHandsUp);
    std::cout << "\tLHandPositionsList_OfStaticHandsUp size = " << LHandPositionsList_OfStaticHandsUp.size() << std::endl;
    std::cout << "DONE" << std::endl;


    // generate more Static hands positions for different positions
    std::cout << "Generating more static hands positions data..." << std::endl;
    std::vector< std::vector<cv::Point> >  staticPositions_Generated;

    unsigned int numOfVectors = 1000;
    unsigned int numOfPoints = LHandPositionsList_OfStaticHandsUp[0].size();
    int XmaxWindow = 500, YmaxWindow = 300;
    for(int x = 60; x < XmaxWindow; x+=60){
        for(int y = 50; y < YmaxWindow; y+=50){
            createListOfStaticPositions(x,
                                        y,
                                        numOfVectors,
                                        numOfPoints,
                                        staticPositions_Generated);
        }
    }
    std::cout << "\tstaticPositions_Generated size = " << staticPositions_Generated.size() << std::endl;
    std::cout << "DONE" << std::endl;


    // Generate circular movements (ellipses)
    // (x, y) = ( c1 + a*cos(2*pi*f*t) , c2 + b*sin(2*pi*f*t) )
    std::cout << "Generating circular hands positions data..." << std::endl;
    std::vector< std::vector<cv::Point> >  circularPositionsList_Generated;

    for(int c1 = 150; c1 < (0.75 * XmaxWindow); c1+=62){
        for(int c2 = 75; c2 < YmaxWindow; c2+=75){

            for(double a = 20.0; a < 30.0; a+=10.0){
                for(double b = 20.0; b < 30.0; b+=10.0){

                    for(double f = 0.5; f < 2.0; f+=0.5){

                        createListOfCircularPositions( c1, c2, a, b, f, this->fps,
                                                      numOfVectors,
                                                      numOfPoints,
                                                      circularPositionsList_Generated);
                    }

                }
            }

        }
    }
    std::cout << "\tcircularPositionsList_Generated size = " << circularPositionsList_Generated.size() << std::endl;
    std::cout << "DONE" << std::endl;

    //-----------------------------------------------------------
    //--------------------------Get Features---------------------

    // LHShake
    cv::Mat LH_Shake_MatList, RH_Shake_MatList;
    getFeaturedData(faceCenterPointList_OfLHShake,
                   pixelSizeInCmTempList_OfLHShake,
                   this->minTimeToDetect,
                   fpsList_OfLHShake,
                   this->interpolationTimeStep,
                   LHandPositionsList_OfLHShake,
                   RHandPositionsList_OfLHShake,

                   LH_Shake_MatList,
                   RH_Shake_MatList);
    std::cout  << "LH_Shake_MatList          => rows: "  << LH_Shake_MatList.rows      << ", cols: " << LH_Shake_MatList.cols << std::endl;

    // StaticHandsUp
    cv::Mat LH_StaticHandsUp_MatList, RH_StaticHandsUp_MatList;
    getFeaturedData(faceCenterPointList_OfStaticHandsUp,
                   pixelSizeInCmTempList_OfStaticHandsUp,
                   this->minTimeToDetect,
                   fpsList_OfLHShake,
                   this->interpolationTimeStep,
                   LHandPositionsList_OfStaticHandsUp,
                   RHandPositionsList_OfStaticHandsUp,

                   LH_StaticHandsUp_MatList,
                   RH_StaticHandsUp_MatList);
    std::cout  << "LH_StaticHandsUp_MatList  => rows: "  << LH_StaticHandsUp_MatList.rows      << ", cols: " << LH_StaticHandsUp_MatList.cols << std::endl;

    // Generated static positions

    // HERE I use all info from the real data of StaticHandsUp
    // but I pass the generated static positions

    /*
    std::vector<int> response;
    std::vector<int> test;
    test.push_back(1);
    test.push_back(2);
    test.push_back(3);
    test.push_back(4);
    test.push_back(5);

    extendVectorRepeting(test, 5, response);

    std::cout << "test size     = " << test.size() << std::endl;
    std::cout << "response size = " << response.size() << std::endl;

    for(int i = 0; i < response.size(); i++){
        std::cout << "response["<<i<<"]    = " << response[i] << std::endl;
    }

    std::cout << "response[size] = " << response[test.size()] << std::endl;
    */

    // Extend lists in order to have the same size of staticPositions_Generated
    // extend face Point list
    std::vector<cv::Point> extended_faceCenterPointList_OfStaticHandsUp;
    extendVectorRepeting (faceCenterPointList_OfStaticHandsUp,
                          staticPositions_Generated.size(),
                          extended_faceCenterPointList_OfStaticHandsUp);

    std::vector<double> extended_pixelSizeInCmTempList_OfStaticHandsUp;
    extendVectorRepeting (pixelSizeInCmTempList_OfStaticHandsUp,
                          staticPositions_Generated.size(),
                          extended_pixelSizeInCmTempList_OfStaticHandsUp);

    std::vector<double> extended_fpsList_OfLHShake;
    extendVectorRepeting (fpsList_OfLHShake,
                          staticPositions_Generated.size(),
                          extended_fpsList_OfLHShake);

    cv::Mat LH_StaticPosGen_MatList, RH_StaticPosGen_MatList;
    getFeaturedData(extended_faceCenterPointList_OfStaticHandsUp,
                   extended_pixelSizeInCmTempList_OfStaticHandsUp,
                   this->minTimeToDetect,
                   extended_fpsList_OfLHShake,
                   this->interpolationTimeStep,
                   staticPositions_Generated,
                   staticPositions_Generated,

                   LH_StaticPosGen_MatList,
                   RH_StaticPosGen_MatList);
    //std::cout  << "LH_StaticPosGen_MatList   => rows: "  << LH_StaticPosGen_MatList.rows      << ", cols: " << LH_StaticPosGen_MatList.cols << std::endl;

    // concatenation of all static positions( real ones with generated ones)
    //cv::Mat LH_AllStaticPos;
    //cv::vconcat(LH_StaticHandsUp_MatList, LH_StaticPosGen_MatList, LH_AllStaticPos);
    //std::cout  << "LH_AllStaticPos   => rows: "  << LH_AllStaticPos.rows      << ", cols: " << LH_AllStaticPos.cols << std::endl;


    // Generated Circular Positions
    std::vector<cv::Point> extended_faceCenterPointList_OfCircularPositions;
    extendVectorRepeting (faceCenterPointList_OfStaticHandsUp,
                          circularPositionsList_Generated.size(),
                          extended_faceCenterPointList_OfCircularPositions);

    std::vector<double> extended_pixelSizeInCmTempList_OfCircularPositions;
    extendVectorRepeting (pixelSizeInCmTempList_OfStaticHandsUp,
                          circularPositionsList_Generated.size(),
                          extended_pixelSizeInCmTempList_OfCircularPositions);

    std::vector<double> extended_fpsList_OfCircularPositions;
    extendVectorRepeting (fpsList_OfLHShake,
                          circularPositionsList_Generated.size(),
                          extended_fpsList_OfCircularPositions);

    cv::Mat LH_CircularPosGen_MatList, RH_CircularPosGen_MatList;
    getFeaturedData(extended_faceCenterPointList_OfCircularPositions,
                   extended_pixelSizeInCmTempList_OfCircularPositions,
                   this->minTimeToDetect,
                   extended_fpsList_OfCircularPositions,
                   this->interpolationTimeStep,
                   circularPositionsList_Generated,
                   circularPositionsList_Generated,

                   LH_CircularPosGen_MatList,
                   RH_CircularPosGen_MatList);
    //std::cout  << "LH_CircularPosGen_MatList   => rows: "  << LH_CircularPosGen_MatList.rows      << ", cols: " << LH_StaticPosGen_MatList.cols << std::endl;

    // concatenation of all static positions( real ones with generated ones)
    //cv::Mat LH_AllCircularPos;// = LH_CircularPosGen_MatList;
    //cv::vconcat(LH_AllStaticPos, LH_CircularPosGen_MatList, LH_AllCircularPos);
    //std::cout  << "LH_AllCircularPos   => rows:


    // all concatenations
    cv::Mat LH_AllStaticPos;

    cv::Mat LH_AllFalsePos;
    cv::vconcat(LH_StaticHandsUp_MatList, LH_StaticPosGen_MatList, LH_AllFalsePos);
    cv::vconcat(LH_AllFalsePos, LH_CircularPosGen_MatList, LH_AllFalsePos);

    cv::Mat LH_AllTruePos = LH_Shake_MatList;

    //--------------------------------------------------------------------------
    //---------------------- separate data and Label data ----------------------
    double trainPerc = 0.8, cvPerc = 0.0, testPerc = 0.2;

    // LH_AllFalsePos
    cv::Mat auxTrain1,      auxCV1,         auxTest1,
            auxLabelTrain1, auxLabelCV1,    auxLabelTest1;
    splitDataInSets(LH_AllFalsePos, trainPerc, cvPerc, testPerc, auxTrain1, auxCV1, auxTest1);

    createDataLabels(auxTrain1, false, auxLabelTrain1);
    createDataLabels(auxCV1,    false, auxLabelCV1);
    createDataLabels(auxTest1,  false, auxLabelTest1);

    // LH_Shake_MatList
    cv::Mat auxTrain2,      auxCV2,         auxTest2,
            auxLabelTrain2, auxLabelCV2,    auxLabelTest2;
    splitDataInSets(LH_AllTruePos, trainPerc, cvPerc, testPerc, auxTrain2, auxCV2, auxTest2);

    createDataLabels(auxTrain2, true, auxLabelTrain2);
    createDataLabels(auxCV2,    true, auxLabelCV2);
    createDataLabels(auxTest2,  true, auxLabelTest2);



    // merge in the same way to make sure we have correlated data
    mergeMats(auxTrain1, auxTrain2, data_train);
    mergeMats(auxCV1,    auxCV2,    data_CV);
    mergeMats(auxTest1,  auxTest2,  data_test);

    mergeMats(auxLabelTrain1, auxLabelTrain2, labels_train);
    mergeMats(auxLabelCV1,    auxLabelCV2,    labels_CV);
    mergeMats(auxLabelTest1,  auxLabelTest2,  labels_test);


    /*
    cv::vconcat(auxTrain1,  auxTrain2,  data_train);
    cv::vconcat(auxCV1,     auxCV2,     data_CV);
    cv::vconcat(auxTest1,   auxTest2,   data_test);

    cv::vconcat(auxLabelTrain1, auxLabelTrain2, labels_train);
    cv::vconcat(auxLabelCV1,    auxLabelCV2,    labels_CV);
    cv::vconcat(auxLabelTest1,  auxLabelTest2,  labels_test);
    */
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
void SemanticDetector::logisticsTrain(cv::Mat &data_train, cv::Mat &data_test, cv::Mat &labels_train, cv::Mat &labels_test){

    double learningRate = 3000.0;
    int iterations,
        miniBatchSize = 1;

    //double scale = (double)(std::numeric_limits<float>::max() / std::numeric_limits<double>::max());
    //labels_train.convertTo(labels_train, CV_32S, scale );
    //data_train.convertTo(data_train, CV_32F, scale);

    std::vector<int> iterationsArray = {10000};
    std::cout << "learningRate  = " << learningRate << std::endl;
    //std::cout << "iterations    = " << iterations << std::endl;
    //std::cout << "miniBatchSize = " << miniBatchSize << std::endl;

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

        //! [init]
        lr1->train(data_train, cv::ml::ROW_SAMPLE, labels_train);
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
        std::cout << "accuracy: " << calculateAccuracyPercent(labels_test, responses) << "%" << std::endl;

        // cost function
        double JOutput;
        cv::Mat thetas = lr1->get_learnt_thetas();
        labels_test.convertTo(labels_test, CV_32F);
        costFunction(data_test, labels_test, thetas, learningRate, JOutput);

    }
}


/*
* faceCenterPoint       ->  Position of the face
* pixelSizeInCmTemp     ->
* positions             ->  This is a array of vectors. Those vectors contain the history positions of the tracked body part.
*                           The positions must have separated within the some interval of time.
*                           The newest data is located at the end of the vector.
*/
void SemanticDetector::detect(cv::Point faceCenterPoint, double pixelSizeInCmTemp, std::vector<cv::Point> positions[], int frameIndex) {
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
 {

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
    // transpose - convert vector into one single row (1x589)
    LHAllInfo = LHAllInfo.t();


    // load the classifier onto new object
    const char saveFilename[] = "LHClassifier.xml";
    //std::cout << "loading a new classifier from " << saveFilename << std::endl;
    cv::Ptr<cv::ml::LogisticRegression> lr2 = cv::ml::StatModel::load<cv::ml::LogisticRegression>(saveFilename);
    //std::cout << "predicting...";
    cv::Mat responses;
    lr2->predict(LHAllInfo, responses);

    cv::Mat thetas = lr2->get_learnt_thetas();

    cv::Mat X_bias = cv::Mat::ones(LHAllInfo.rows, 1, CV_32F);
    cv::hconcat(X_bias, LHAllInfo, X_bias);
    cv::Mat calc = X_bias * thetas.t();
    cv::Mat h;
    sigmoid(calc, h);
    std::cout << "response : " << responses << " h : " << h << std::endl;

    // filter
    // add responses to the filter
    //std::cout << "rowNum        : " << rowNum << std::endl;
    LHShake_Filter[rowNum] = responses.at<int>(0,0);
    //std::cout << "LHShake_Filter: " << std::endl;
    //for(int i = 0; i < LHShake_Filter.size(); i++){
    //    std::cout << LHShake_Filter[i] << std::endl;
    //}

    // if we have 10 positive consecutive detections than
    int sum = std::accumulate(LHShake_Filter.begin(), LHShake_Filter.end(), 0);
    if( sum >= 10 ){
        flag_LHShake = true;    // turn the flat to true
    }else{
        flag_LHShake = false;   // turn the flat to false
    }
    std::cout << " \tflag_LHShake = " << flag_LHShake << std::endl;

    // actualize rowNum
    rowNum++;
    if( (rowNum % 10) == 0 ){
        rowNum = 0;
    }

    //std::cout << "LHAllInfo = " << std::endl << " " << LHAllInfo << std::endl << std::endl;

    //cv::Mat hands[2] = {LHAllInfo, RHAllInfo};

    // LOGISTICS REGRESSION
    //logisticsPredition("L", hands);
    //logisticsPredition("R", [RHAllInfo]);
    //logisticsPredition("LR",[LHAllInfo, RHAllInfo]);

    // ANN classifiers or Logistic regression
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

    #ifdef TRAINING_CREATE_NEW_CLASSIFIER
        std::cout << "=============================================" << std::endl;
        std::cout << "|  TRAINING_CREATE_NEW_CLASSIFIER defined!  |" << std::endl;
        std::cout << "=============================================" << std::endl;

        /*
        // List of data that is gonna hold all data files information
        std::vector<cv::Point>  faceCenterPointList_OfRHShake,
                                faceCenterPointList_OfLHShake,
                                faceCenterPointList_OfStaticHandsUp;
        std::vector<double>     pixelSizeInCmTempList_OfRHShake,
                                pixelSizeInCmTempList_OfLHShake,
                                pixelSizeInCmTempList_OfStaticHandsUp;
        //std::vector<double>     minTimeToDetectList;
        std::vector<double>     fpsList_OfRHShake,
                                fpsList_OfLHShake,
                                fpsList_OfStaticHandsUp;
        //std::vector<double>     interpolationTimeStepList;
        std::vector<int>        gestureLabelList_OfRHShake,
                                gestureLabelList_OfLHShake,
                                gestureLabelList_OfStaticHandsUp;
        std::vector< std::vector<cv::Point> >  LHandPositionsList_OfRHShake,
                                               LHandPositionsList_OfLHShake,
                                               LHandPositionsList_OfStaticHandsUp;
        std::vector< std::vector<cv::Point> >  RHandPositionsList_OfRHShake,
                                               RHandPositionsList_OfLHShake,
                                               RHandPositionsList_OfStaticHandsUp;


        std::cout << "Reading RHShake data..." << std::endl;
        getDataFromFiles(pathRHData,
                        faceCenterPointList_OfRHShake,
                        pixelSizeInCmTempList_OfRHShake,
                        //minTimeToDetectList,
                        fpsList_OfRHShake,
                        //interpolationTimeStepList,
                        gestureLabelList_OfRHShake,
                        LHandPositionsList_OfRHShake,
                        RHandPositionsList_OfRHShake);
        std::cout << "DONE" << std::endl;


        std::cout << "Reading LHShake data..." << std::endl;
        getDataFromFiles(pathLHData,
                        faceCenterPointList_OfLHShake,
                        pixelSizeInCmTempList_OfLHShake,
                        //minTimeToDetectList,
                        fpsList_OfLHShake,
                        //interpolationTimeStepList,
                        gestureLabelList_OfLHShake,
                        LHandPositionsList_OfLHShake,
                        RHandPositionsList_OfLHShake);
        std::cout << "DONE" << std::endl;

        std::cout << "Reading StaticHandsUp data..." << std::endl;
        getDataFromFiles(pathStaticHandsUpData,
                        faceCenterPointList_OfStaticHandsUp,
                        pixelSizeInCmTempList_OfStaticHandsUp,
                        //minTimeToDetectList,
                        fpsList_OfStaticHandsUp,
                        //interpolationTimeStepList,
                        gestureLabelList_OfStaticHandsUp,
                        LHandPositionsList_OfStaticHandsUp,
                        RHandPositionsList_OfStaticHandsUp);
        std::cout << "DONE" << std::endl;


        // generate more Static hands positions
        std::cout << "Generating more static hands positions data..." << std::endl;
        std::vector< std::vector<cv::Point> >  staticPositions_Generated;
        // for different position
        for(int x = 60; x < 500; x+=60){
            for(int y = 50; y < 300; y+=300)){
                createListOfStaticPositions(x,
                                            y,
                                            1000,
                                            staticPositions_Generated);
            }
        }
        //std::cout << "staticPositions_Generated size = " << staticPositions_Generated.size() << std::endl;
        std::cout << "DONE" << std::endl;
        */

        // create variables to hold the data sets
        cv::Mat data_train,
                data_CV,
                data_test,
                labels_train,
                labels_CV,
                labels_test;

        // get sets of data for training
        getClassifiersTrainData(data_train, data_CV, data_test, labels_train, labels_CV, labels_test);

        /*
        for(int i = 0; i < LHandPositionsList_OfStaticHandsUp.size(); i++){

            std::vector<cv::Point> gest = LHandPositionsList_OfStaticHandsUp[i];
            std::vector<cv::Point> auxVector;
            for(int k = 0; k < gest.size(); k++){
                cv::Point auxPos(gest[k].x, gest[k].y - 20);
                //std::cout << "gest: " << gest[k] << " auxPos: " << auxPos << std::endl;
                auxVector.push_back(auxPos);
            }
            LHandPositionsList_OfStaticHandsUp_Generated_01.push_back(auxVector);
        }
        //std::cout << "LHandPositionsList_OfStaticHandsUp_Generated_01 size = " << LHandPositionsList_OfStaticHandsUp_Generated_01.size() << std::endl;


        for(int i = 0; i < LHandPositionsList_OfStaticHandsUp.size(); i++){

            std::vector<cv::Point> gest = LHandPositionsList_OfStaticHandsUp[i];
            std::vector<cv::Point> auxVector;
            for(int k = 0; k < gest.size(); k++){
                cv::Point auxPos(gest[k].x, gest[k].y + 20);
                //std::cout << "gest: " << gest[k] << " auxPos: " << auxPos << std::endl;
                auxVector.push_back(auxPos);
            }
            LHandPositionsList_OfStaticHandsUp_Generated_02.push_back(auxVector);
        }
        //std::cout << "LHandPositionsList_OfStaticHandsUp_Generated_02 size = " << LHandPositionsList_OfStaticHandsUp_Generated_02.size() << std::endl;
        */

        /*

        getFeaturedData(faceCenterPointList[n],
                      pixelSizeInCmTempList[n],
                      this->minTimeToDetect,
                      fpsList_OfStaticHandsUp[n],
                      this->interpolationTimeStep,
                      LHandPositionsList[n],
                      RHandPositionsList[n],);

        separateData();


        int numOfData = faceCenterPointList_OfLHShake.size();// + faceCenterPointList_OfRHShake.size();

        int numOfCreatedDataPerGesture = 1000,
            numOfTrainDataPerGesture = numOfCreatedDataPerGesture * 0.8, // per single gesture
            numOfTestDataPerGesture = numOfCreatedDataPerGesture - numOfCreatedDataPerGesture;

        std::cout << "Get Features of train and test data..." << std::endl;
        // Shake movements
        for( int i = 0; i < numOfCreatedDataPerGesture; i++){

            for(int k = 0; k < numOfData; k+=numOfCreatedDataPerGesture){

                int n = k + i; // 0 1000 2000 3000 ... 1 1001 2001 3001 ...
                //std::cout << "n = " << n << std::endl;
                std::vector<double> LHFeatures_OfLHShake, RHFeatures_OfLHShake,
                                    LHFeatures_OfRHShake, RHFeatures_OfRHShake;


                // RHShake
                getFeaturesVector(faceCenterPointList_OfRHShake[n],
                              pixelSizeInCmTempList_OfRHShake[n],
                              this->minTimeToDetect,
                              fpsList_OfRHShake[n],
                              this->interpolationTimeStep,
                              LHandPositionsList_OfRHShake[n],
                              RHandPositionsList_OfRHShake[n],
                              LHFeatures_OfRHShake,
                              RHFeatures_OfRHShake);


                // LHShake
                getFeaturesVector(faceCenterPointList_OfLHShake[n],
                              pixelSizeInCmTempList_OfLHShake[n],
                              this->minTimeToDetect,
                              fpsList_OfLHShake[n],
                              this->interpolationTimeStep,
                              LHandPositionsList_OfLHShake[n],
                              RHandPositionsList_OfLHShake[n],
                              LHFeatures_OfLHShake,
                              RHFeatures_OfLHShake);


                // transform in Mat variables
                cv::Mat LHFeatures_OfLHShakeMat(LHFeatures_OfLHShake,true);
                //cv::Mat LHFeatures_OfRHShakeMat(LHFeatures_OfRHShake,true);

                // transpose - convert vector into one single row (1x589)
                LHFeatures_OfLHShakeMat = LHFeatures_OfLHShakeMat.t();
                //LHFeatures_OfRHShakeMat = LHFeatures_OfRHShakeMat.t();

                // Separate data in train sets and test sets
                if(i < numOfTrainDataPerGesture){
                    data_train.push_back(LHFeatures_OfLHShakeMat);
                    labels_train.push_back(1.0); // gesture

                    //data_train.push_back(LHFeatures_OfRHShakeMat);
                    //labels_train.push_back(0.0); // unknown
                }else{ // test data

                    data_test.push_back(LHFeatures_OfLHShakeMat);
                    labels_test.push_back(1.0); // gesture

                    //data_test.push_back(LHFeatures_OfRHShakeMat);
                    //labels_test.push_back(0.0); // unknown
                }
            }
        }


        // StaticHandsUp data
        numOfData = faceCenterPointList_OfStaticHandsUp.size();

        for( int i = 0; i < numOfCreatedDataPerGesture; i++){
            for(int k = 0; k < numOfData; k+=numOfCreatedDataPerGesture){

                int n = k + i; // 0 1000 2000 3000 ... 1 1001 2001 3001 ...
                std::vector<double> LHFeatures_OfStaticHandsUp, RHFeatures_OfStaticHandsUp;

                // StaticHandsUp
                getFeaturesVector(faceCenterPointList_OfStaticHandsUp[n],
                              pixelSizeInCmTempList_OfStaticHandsUp[n],
                              this->minTimeToDetect,
                              fpsList_OfStaticHandsUp[n],
                              this->interpolationTimeStep,
                              LHandPositionsList_OfStaticHandsUp[n],
                              RHandPositionsList_OfStaticHandsUp[n],
                              LHFeatures_OfStaticHandsUp,
                              RHFeatures_OfStaticHandsUp);

                // transform in Mat variables
                cv::Mat LHFeatures_OfStaticHandsUpMat(LHFeatures_OfStaticHandsUp,true);
                //cv::Mat RHFeatures_OfStaticHandsUpMat(RHFeatures_OfStaticHandsUp,true);

                // transpose - convert vector into one single row (1x589)
                LHFeatures_OfStaticHandsUpMat = LHFeatures_OfStaticHandsUpMat.t();
                //RHFeatures_OfStaticHandsUpMat = RHFeatures_OfStaticHandsUpMat.t();

                // Separate data in train sets and test sets
                if(i < numOfTrainDataPerGesture) {
                    data_train.push_back(LHFeatures_OfStaticHandsUpMat);
                    labels_train.push_back(0.0); // unknown
                } else { // test data
                    data_test.push_back(LHFeatures_OfStaticHandsUpMat);
                    labels_test.push_back(0.0); // unknown
                }
            }
        }


        // Generated StaticHandsUp data
        numOfData = faceCenterPointList_OfStaticHandsUp.size();
        for( int i = 0; i < numOfCreatedDataPerGesture; i++){

            for(int k = 0; k < numOfData; k+=numOfCreatedDataPerGesture){

                int n = k + i; // 0 1000 2000 3000 ... 1 1001 2001 3001 ...
                std::vector<double> LHFeatures_OfStaticHandsUp, RHFeatures_OfStaticHandsUp;

                // StaticHandsUp
                getFeaturesVector(faceCenterPointList_OfStaticHandsUp[n],
                              pixelSizeInCmTempList_OfStaticHandsUp[n],
                              this->minTimeToDetect,
                              fpsList_OfStaticHandsUp[n],
                              this->interpolationTimeStep,
                              LHandPositionsList_OfStaticHandsUp_Generated_01[n],
                              RHandPositionsList_OfStaticHandsUp[n],
                              LHFeatures_OfStaticHandsUp,
                              RHFeatures_OfStaticHandsUp);

                // transform in Mat variables
                cv::Mat LHFeatures_OfStaticHandsUpMat(LHFeatures_OfStaticHandsUp,true);
                //cv::Mat RHFeatures_OfStaticHandsUpMat(RHFeatures_OfStaticHandsUp,true);

                // transpose - convert vector into one single row (1x589)
                LHFeatures_OfStaticHandsUpMat = LHFeatures_OfStaticHandsUpMat.t();
                //RHFeatures_OfStaticHandsUpMat = RHFeatures_OfStaticHandsUpMat.t();

                // Separate data in train sets and test sets
                if(i < numOfTrainDataPerGesture) {
                    data_train.push_back(LHFeatures_OfStaticHandsUpMat);
                    labels_train.push_back(0.0); // unknown
                } else { // test data
                    data_test.push_back(LHFeatures_OfStaticHandsUpMat);
                    labels_test.push_back(0.0); // unknown
                }
            }
        }

        for( int i = 0; i < numOfCreatedDataPerGesture; i++){

            for(int k = 0; k < numOfData; k+=numOfCreatedDataPerGesture){

                int n = k + i; // 0 1000 2000 3000 ... 1 1001 2001 3001 ...
                std::vector<double> LHFeatures_OfStaticHandsUp, RHFeatures_OfStaticHandsUp;

                // StaticHandsUp
                getFeaturesVector(faceCenterPointList_OfStaticHandsUp[n],
                              pixelSizeInCmTempList_OfStaticHandsUp[n],
                              this->minTimeToDetect,
                              fpsList_OfStaticHandsUp[n],
                              this->interpolationTimeStep,
                              LHandPositionsList_OfStaticHandsUp_Generated_02[n],
                              RHandPositionsList_OfStaticHandsUp[n],
                              LHFeatures_OfStaticHandsUp,
                              RHFeatures_OfStaticHandsUp);

                // transform in Mat variables
                cv::Mat LHFeatures_OfStaticHandsUpMat(LHFeatures_OfStaticHandsUp,true);
                //cv::Mat RHFeatures_OfStaticHandsUpMat(RHFeatures_OfStaticHandsUp,true);

                // transpose - convert vector into one single row (1x589)
                LHFeatures_OfStaticHandsUpMat = LHFeatures_OfStaticHandsUpMat.t();
                //RHFeatures_OfStaticHandsUpMat = RHFeatures_OfStaticHandsUpMat.t();

                // Separate data in train sets and test sets
                if(i < numOfTrainDataPerGesture) {
                    data_train.push_back(LHFeatures_OfStaticHandsUpMat);
                    labels_train.push_back(0.0); // unknown
                } else { // test data
                    data_test.push_back(LHFeatures_OfStaticHandsUpMat);
                    labels_test.push_back(0.0); // unknown
                }
            }
        }



        data_train.convertTo(data_train, CV_32F);
        labels_train.convertTo(labels_train, CV_32F);
        data_test.convertTo(data_test, CV_32F);
        labels_test.convertTo(labels_test, CV_32F);

         std::cout << "DONE!" << std::endl;
        //std::cout  << "data_train   = "  << data_train.row(1)      << std::endl;
        //std::cout  << "labels_train = "  << labels_train    << std::endl;
        //std::cout  << "data_test    = "  << data_test       << std::endl;
        //std::cout  << "labels_test  = "  << labels_test     << std::endl;


        */

        // Convert data to CV_32F in order to pass to the logisticsTrain function
        data_train.convertTo(data_train, CV_32F);
        labels_train.convertTo(labels_train, CV_32F);

        data_CV.convertTo(data_CV, CV_32F);
        labels_CV.convertTo(labels_CV, CV_32F);

        data_test.convertTo(data_test, CV_32F);
        labels_test.convertTo(labels_test, CV_32F);

        /*
        cv::Mat train = cv::Mat::ones(8, 4, CV_32F);
        std::cout << "train = " << std::endl << " " << train << std::endl << std::endl;

        cv::Mat labelTrain = (cv::Mat_<double>(8,1) << 0, 1, 0, 1, 0, 1, 0, 1);
        std::cout << "labelTrain = " << std::endl << " " << labelTrain << std::endl << std::endl;
        labelTrain.convertTo(labelTrain, CV_32F);

        cv::Mat test = cv::Mat::zeros(4,4, CV_32F);
        std::cout << "test = " << std::endl << " " << test << std::endl << std::endl;

        cv::Mat labelTest = cv::Mat::ones(4, 1, CV_32F);
        std::cout << "labelTest = " << std::endl << " " << labelTest << std::endl << std::endl;
        logisticsTrain(train, test, labelTrain, labelTest);
        */

        std::cout  << "data_train   => rows: "  << data_train.rows      << ", cols: " << data_train.cols << std::endl;
        std::cout  << "labels_train => rows: "  << labels_train.rows    << ", cols: " << labels_train.cols << std::endl;
        std::cout  << "data_CV      => rows: "  << data_CV.rows         << ", cols: " << data_CV.cols << std::endl;
        std::cout  << "labels_CV    => rows: "  << labels_CV.rows       << ", cols: " << labels_CV.cols << std::endl;
        std::cout  << "data_test    => rows: "  << data_test.rows       << ", cols: " << data_test.cols << std::endl;
        std::cout  << "labels_test  => rows: "  << labels_test.rows     << ", cols: " << labels_test.cols << std::endl;

        // than train
        logisticsTrain(data_train, data_test, labels_train, labels_test);

    #endif // TRAINING_CREATE_NEW_CLASSIFIER

#endif

}



