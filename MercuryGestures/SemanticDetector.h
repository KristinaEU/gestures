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

extern std::string capVideoName;    // this is used just for training propose

static std::map<std::string, int> gestureLablesList = {
        {"unkown", 0},

        {"rightUpperBody",  1},
        {"leftUpperBody",   2},
        {"rightLowerBody",  3},
        {"leftLowerBody",   4},
        {"rightUpperArm",   5},
        {"rightLowerArm",   6},
        {"leftUpperArm",    7},
        {"leftLowerArm",    8},
        {"face",            9},

        {"RHShake",         101},
        {"LHShake",         102},
        {"staticHandsUp",   103},

        {"headShake",       201},
        {"headNod",         202},
        {"headLookLeft",    203},
        {"headLookRight",   204},
        {"headLookUp",      205},
        {"headLookDown",    206},
        {"resitLeft",       207},
        {"resitRight",      208},
        {"resitUp",         209}
};

struct GenerateStaticPositionInfo{

    unsigned int numOfVectors; // 1000 number of vectors to generate

    int x_start, // pixel number where static position will start in the x Axis
        x_step,  // number of pixels to shift in x axis to generate the next static positions
        x_end,
        y_start, // pixel number where static position will start in the y position
        y_step,
        y_end;  // number of pixels to shift in y axis to generate the next static positions
};

/*
 * Used to generate an array of ellipses
 * (x, y) = ( c1 + a*cos(2*pi*f*t) , c2 + b*sin(2*pi*f*t) )
*/
struct GenerateEllipticalPositionInfo{

    unsigned int numOfVectors; // number of vectors to generate

    int c1_start,
        c1_step,
        c1_end,
        c2_start,
        c2_step,
        c2_end;

    double a_start,
           a_step,
           a_end,
           b_start,
           b_step,
           b_end,
           f_start,
           f_step,
           f_end;
};

struct TrainingSets {
    double trainPerc = 0.6, // training set percentage
           cvPerc    = 0.2, // cross validation set percentage
           testPerc  = 0.2; // test set percentage
};

struct InfoClassifier {
    std::string pathPositiveData;    //= "data/SelectedData/LHShake/createdData/";         // for positive gestures
    std::string pathNegativeData;    //= "data/SelectedData/StaticHandsUp/createdData/";   // for negative gestures
    std::string pathClassifier;      //= "LHClassifier.xml";


    int XmaxWindow; // number of maximum pixels on the x axis of the image
    int YmaxWindow; // number of maximum pixels on the y axis of the image

    bool use_LH_StaticPos_for_negativeData     = false;
    bool use_LH_EllipticalPos_for_negativeData = false;
    bool use_RH_StaticPos_for_negativeData     = false;
    bool use_RH_EllipticalPos_for_negativeData = false;

    bool use_LH_StaticPos_for_positiveData     = false;
    bool use_LH_EllipticalPos_for_positiveData = false;
    bool use_RH_StaticPos_for_positiveData     = false;
    bool use_RH_EllipticalPos_for_positiveData = false;



    bool generateStaticPositions = false; // if false than the other variables will not be used
    GenerateStaticPositionInfo      genStaticPosInfo;

    bool generateEllipticalPositions = false;
    GenerateEllipticalPositionInfo  genEllipticalPosInfo;

    TrainingSets trainingSets;
};

class SemanticDetector {
public:
    int fps = 25; //default
	int frameWidth;
	int frameHeight;
    std::string bodyPart; // body part to be analyzed ("Head" or "Hands")
    double minTimeToDetect = 3.0; // (3 sec) time to detect a gesture
    double interpolationTimeStep = 0.02; // (seconds)
    double normalizationFaceHandDistance = 100; // (cm) it will make the face-hand distances be around the range of -1 to +1

    std::string pathOriginalData        = "data/originalData/";    // all original data taken directly from the videos is stored here
    std::string pathCreatedData         = "data/createdData/";      // all created data with some deviation based on original data is stored here
    //std::string pathMixedData           = "data/mixedData/";          // all mixed data is taken from here
    std::string pathLHData              = "data/SelectedData/LHShake/createdData/";
    std::string pathRHData              = "data/SelectedData/RHShake/createdData/";
    std::string pathStaticHandsUpData   = "data/SelectedData/StaticHandsUp/createdData/";




	SemanticDetector(int fps, std::string bodyPart);
	~SemanticDetector();

    void detect(const char classifierName[],
                cv::Point faceCenterPoint,
                double pixelSizeInCmTemp,
                std::vector<cv::Point> positions[],
                float &gestureOutput,
                int frameIndex = 0);

	void setVideoProperties(int frameWidth, int frameHeight);
    void getVelocity(std::vector<double> &vectorPositions, double deltaTime, std::vector<double> &vectorOutputVelocities);

    float calculateAccuracyPercent(const cv::Mat &original, const cv::Mat &predicted);
    float calculatePrecision(const cv::Mat &original, const cv::Mat &predicted);
    float calculateRecall(const cv::Mat &original, const cv::Mat &predicted);
    float calculateF1Score(float precision, float recall);

    void sigmoid(cv::Mat &M, cv::Mat &sigmoidOutput);
    void costFunction(cv::Mat &X, cv::Mat y, cv::Mat theta, double lambda, double JOutput);
    void logisticsTrain(cv::Mat &data_train,
                        cv::Mat &data_CV,
                        cv::Mat &data_test,
                        cv::Mat &labels_train,
                        cv::Mat &labels_CV,
                        cv::Mat &labels_test);

    void trainClassifier(InfoClassifier &infoClas);

    void storeVideoData(int gestureLabel,
                        cv::Point faceCenterPoint,
                        std::vector<std::vector<cv::Point>> &positions,
                        double pixelSizeInCmTemp,
                        int frameIndex);

private:

    float trust = 0.5; // classifier threshold
    int LHFilterLength = 15;
    std::vector<int> LHShake_Filter{std::vector<int>(LHFilterLength,0)};
    int rowNum = 0; // position number of the filter to be stored some value
    bool flag_LHShake = false;  // LH Shake final flag detection


    template <typename T> void extendVectorRepeting (std::vector<T> &vect, int lenghtVec, std::vector<T> &vectOut);

    void splitDataInSets(cv::Mat &setMat,
                          double trainPerc,
                          double cvPerc,
                          double testPerc,
                          cv::Mat &data_train,
                          cv::Mat &data_CV,
                          cv::Mat &data_test);

    void mergeMats(cv::Mat &mat1, cv::Mat &mat2, cv::Mat &matOutput);

	double _interpolate( std::vector<double> &xData, std::vector<double> &yData, double x, bool extrapolate );

    void scaleAndMeanNormalization(cv::Point &faceCenterPoint,
                                double pixelSizeInCmTemp,
                                std::vector<cv::Point> &HandPositions,
                                std::vector<double> &xHNormalizedOutput,
                                std::vector<double> &yHNormalizedOutput);

    void getLinearInterpolation(double minTimeToDetect,
                                  double periodFPS,
                                  double interpolationTimeStep,
                                  std::vector<double> &HNormalized,
                                  std::vector<double> &HInterpolated);

    void getClassifiersTrainData(InfoClassifier &infoClas,
                                 cv::Mat &data_train_output,
                                 cv::Mat &data_CV_output,
                                 cv::Mat &data_test_output,
                                 cv::Mat &labels_train_output,
                                 cv::Mat &labels_CV_output,
                                 cv::Mat &labels_test_output);

    void getFeaturedData(std::vector<cv::Point>             &faceCenterPointList,
                       std::vector<double>                  &pixelSizeInCmTempList,
                       double                               minTimeToDetect,
                       std::vector<double>                  &fpsList,
                       double                               interpolationTimeStep,
                       std::vector<std::vector<cv::Point>>  &LHandPositionsList,
                       std::vector<std::vector<cv::Point>>  &RHandPositionsList,
                       cv::Mat                              &LHFeaturesMatListOutput,
                       cv::Mat                              &RHFeaturesMatListOutput);

    /*
    void getFeaturesVector(cv::Point &faceCenterPoint,
                          double pixelSizeInCmTemp,
                          double minTimeToDetect,
                          double periodFPS,
                          double interpolationTimeStep,
                          std::vector<cv::Point> &LHandPositions,
                          std::vector<cv::Point> &RHandPositions,
                          std::vector<double> &LHAllConcatOutput,
                          std::vector<double> &RHAllConcatOutput);
    */

    void getFeaturesVector(cv::Point &faceCenterPoint,
                          double pixelSizeInCmTemp,
                          double minTimeToDetect,
                          double fps,
                          double interpolationTimeStep,
                          std::vector<std::vector<cv::Point>> &positions,
                          std::vector<std::vector<double>>    &allConcatOutput);

    void getHeadFeaturesVector(cv::Point &faceCenterPoint,
                              double pixelSizeInCmTemp,
                              double minTimeToDetect,
                              double fps,
                              double interpolationTimeStep,
                              std::vector<cv::Point> &headPositions,
                              std::vector<double> &featuresOutput);

    void getHandFeaturesVector(cv::Point &faceCenterPoint,
                              double pixelSizeInCmTemp,
                              double minTimeToDetect,
                              double fps,
                              double interpolationTimeStep,
                              std::vector<cv::Point> &handPositions,
                              std::vector<double> &featuresOutput);

    void saveDataInFile(std::string fullPath,
                        std::string capVideoName,
                        int gestureLabel,
                        double pixelSizeInCmTemp,
                        cv::Point faceCenterPoint,
                        std::vector<std::vector<cv::Point>> positions);
//                        std::vector<cv::Point> LHandPositions,
//                        std::vector<cv::Point> RHandPositions);

    void generateDataFromOriginal(std::string dataPath,
                                  std::string capVideoName,
                                  int gestureLabel,
                                  double pixelSizeInCmTemp,
                                  cv::Point faceCenterPoint,
                                  std::vector<std::vector<cv::Point>> positions);
//                                std::vector<cv::Point> LHandPositions,
//                                std::vector<cv::Point> RHandPositions);

    void createListOfStaticPositions(int x,
                                     int y,
                                     unsigned int numOfVectors,
                                     unsigned int numOfPoints,
                                     std::vector< std::vector<cv::Point> > &positionsListOutput);

    void createListOfEllipticalPositions(int c1,
                                     int c2,
                                     double a,
                                     double b,
                                     double f,
                                     double fps,
                                     unsigned int numOfVectors,
                                     unsigned int numOfPoints,
                                     std::vector< std::vector<cv::Point> > &positionsListOutput);

    void createDataLabels(cv::Mat &setMat, bool label, cv::Mat &labelsOutput);

    void getDataFromFiles(std::string           directoryPath,
                        std::vector<cv::Point>  &faceCenterPointListOutput,
                        std::vector<double>     &pixelSizeInCmTempListOutput,
                        //std::vector<double>     &minTimeToDetectListOutput,
                        std::vector<double>     &fpsListOutput,
                        //std::vector<double>     &interpolationTimeStepListOutput,
                        std::vector<int>        &gestureLabelListOutput,
                        std::vector< std::vector<cv::Point> > &LHandPositionsListOutput,
                        std::vector< std::vector<cv::Point> > &RHandPositionsListOutput);


};
