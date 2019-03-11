#pragma once

#include "_Common.h"

class TrainedColorFireDetection
{
   const int MovementThreshold;
   const uint MaxCandidatesNum;
   const double ProbabilityThreshold;
   const double ResultMapLearningRate;
   const double FireSensitivity;

   Mat PrevBlurredFrame;
   Mat DescriptorHistory;

   vector<double> ColorClassifierWeights;
   vector<Point> FireColorCandidates;

#ifdef SHOW_PROCESS
   Mat ProcessResult;
   void showProcess(const Scalar& box_color);
#endif

   void initialize();

   double calculateProbabilityBasedColor(const int& curr_idx, const Vec3b* upper_ptr, const Vec3b* curr_ptr, const Vec3b* lower_ptr);
   void findMovingPixels(Mat& probability_map, const Mat& blurred_frame);
   bool isLocalMaximum(const int& curr_idx, const double* upper_ptr, const double* curr_ptr, const double* lower_ptr) const;
   void findTopProbabilities(vector<pair<Point, double>>& local_maxima);
   void findTopOfLocalMaxima(const Mat& probability_map);
   void classifyColorAndMotion(const Mat& resized_frame);

   void createBlocksFromCandidates(Mat& result_map, const int& block_size);
   void updateResultMapAndFireRegion(Mat& fire_region);

   void getFirePosition(vector<Rect>& fires, const Mat& fire_region, const Mat& frame) const;


public:
   TrainedColorFireDetection();
   ~TrainedColorFireDetection() = default;

   void detectFire(vector<Rect>& fires, Mat& fire_region, const Mat& frame);
   void informOfSceneChanged();
};