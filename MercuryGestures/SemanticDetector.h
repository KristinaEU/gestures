#pragma once
#include "MercuryCore.h"

extern std::string capVideoName;    // this is used just for training propose

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
    std::string pathMixedData           = "data/mixedData/";          // all mixed data it taken from here
    std::string pathLHData              = "data/SelectedData/LHShake/createdData/";
    std::string pathRHData              = "data/SelectedData/RHShake/createdData/";
    std::string pathStaticHandsUpData   = "data/SelectedData/StaticHandsUp/createdData/";

    /*
    std::map<int, std::string> gesturesList = {
        {0, "unknow"},
        {1, "RHShake"},
        {2, "LHShake"}
    };*/

    std::map<std::string, int> gesturesList = {
            {"unknow",          0},
            {"RHShake",         1},
            {"LHShake",         2},
            {"StaticHandsUp",   3},
            {"Clap",            4}
        };

	SemanticDetector(int fps, std::string bodyPart);
	~SemanticDetector();

    void detect(cv::Point faceCenterPoint, double pixelSizeInCmTemp, std::vector<cv::Point> positions[], int frameIndex = 0);
	void setVideoProperties(int frameWidth, int frameHeight);
    void getVelocity(std::vector<double> &vectorPositions, double deltaTime, std::vector<double> &vectorOutputVelocities);

    float calculateAccuracyPercent(const cv::Mat &original, const cv::Mat &predicted);
    void sigmoid(cv::Mat &M, cv::Mat &sigmoidOutput);
    void costFunction(cv::Mat &X, cv::Mat y, cv::Mat theta, double lambda, double JOutput);
    void logisticsTrain(cv::Mat &data_train, cv::Mat &data_test, cv::Mat &labels_train, cv::Mat &labels_test);
    void logisticsPredition(std::string info, cv::Mat HandsInfo[]);



private:

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

    void getClassifiersTrainData(cv::Mat &data_train,
                                 cv::Mat &data_CV,
                                 cv::Mat &data_test,
                                 cv::Mat &labels_train,
                                 cv::Mat &labels_CV,
                                 cv::Mat &labels_test);

    void getFeaturedData(std::vector<cv::Point>             &faceCenterPointList,
                       std::vector<double>                  &pixelSizeInCmTempList,
                       double                               minTimeToDetect,
                       std::vector<double>                  &fpsList,
                       double                               interpolationTimeStep,
                       std::vector<std::vector<cv::Point>>  &LHandPositionsList,
                       std::vector<std::vector<cv::Point>>  &RHandPositionsList,
                       cv::Mat                              &LHFeaturesMatListOutput,
                       cv::Mat                              &RHFeaturesMatListOutput);

    void getFeaturesVector(cv::Point &faceCenterPoint,
                          double pixelSizeInCmTemp,
                          double minTimeToDetect,
                          double periodFPS,
                          double interpolationTimeStep,
                          std::vector<cv::Point> &LHandPositions,
                          std::vector<cv::Point> &RHandPositions,
                          std::vector<double> &LHAllConcatOutput,
                          std::vector<double> &RHAllConcatOutput);

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
                        std::vector<cv::Point> LHandPositions,
                        std::vector<cv::Point> RHandPositions);

    void generateDataFromOriginal(std::string dataPath,
                                  std::string capVideoName,
                                  int gestureLabel,
                                  double pixelSizeInCmTemp,
                                  cv::Point faceCenterPoint,
                                  std::vector<cv::Point> LHandPositions,
                                  std::vector<cv::Point> RHandPositions);

    void createListOfStaticPositions(int x,
                                     int y,
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


    //void getTrainAndTestData();
};
