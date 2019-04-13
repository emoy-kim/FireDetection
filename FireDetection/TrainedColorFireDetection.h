/*
 * Author: Emoy Kim
 * E-mail: emoy.kim_AT_gmail.com
 * 
 * This code is a modified implementation of the part of [1].
 * 
 * [1] J. Choi and J. Y. Choi, Patch-based fire detection with online outlier learning, 
 *    2015 12th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS), 
 *    Karlsruhe, 2015, pp. 1-6.
 * 
 */

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
   void classifyColorAndMotion(const Mat& frame);

   void createBlocksFromCandidates(Mat& result_map, const int& block_size);
   void updateResultMapAndFireRegion(Mat& fire_region);

   void getFirePosition(vector<Rect>& fires, const Mat& fire_region) const;


public:
   TrainedColorFireDetection();
   ~TrainedColorFireDetection() = default;

   void detectFire(vector<Rect>& fires, Mat& fire_region, const Mat& frame);
   void informOfSceneChanged();
};