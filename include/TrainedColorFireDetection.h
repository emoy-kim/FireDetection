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
public:
   TrainedColorFireDetection();
   ~TrainedColorFireDetection() = default;

   void detectFire(std::vector<cv::Rect>& fires, cv::Mat& fire_region, const cv::Mat& frame);
   void informOfSceneChanged();

private:
   const int MovementThreshold;
   const uint MaxCandidatesNum;
   const double ProbabilityThreshold;
   const double ResultMapLearningRate;
   const double FireSensitivity;

   cv::Mat PrevBlurredFrame;
   cv::Mat DescriptorHistory;

   std::vector<double> ColorClassifierWeights;
   std::vector<cv::Point> FireColorCandidates;

#ifdef SHOW_PROCESS
   Mat ProcessResult;
   void showProcess(const Scalar& box_color);
#endif

   void initialize();

   [[nodiscard]] double calculateProbabilityBasedColor(
      int curr_idx, 
      const cv::Vec3b* upper_ptr, 
      const cv::Vec3b* curr_ptr, 
      const cv::Vec3b* lower_ptr
   );
   void findMovingPixels(cv::Mat& probability_map, const cv::Mat& blurred_frame);
   [[nodiscard]] static bool isLocalMaximum(int curr_idx, const double* upper_ptr, const double* curr_ptr, const double* lower_ptr);
   void findTopProbabilities(std::vector<std::pair<cv::Point, double>>& local_maxima);
   void findTopOfLocalMaxima(const cv::Mat& probability_map);
   void classifyColorAndMotion(const cv::Mat& frame);

   void createBlocksFromCandidates(cv::Mat& result_map, int block_size);
   void updateResultMapAndFireRegion(cv::Mat& fire_region);

   static void getFirePosition(std::vector<cv::Rect>& fires, const cv::Mat& fire_region);
};