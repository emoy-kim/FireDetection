/*
 * Author: Emoy Kim
 * E-mail: emoy.kim_AT_gmail.com
 * 
 * This code is a free software; it can be freely used, changed and redistributed.
 * If you use any version of the code, please reference the code.
 * 
 * The fire features are from the idea of [1].
 * 
 * [1] Hakan Habiboğlu, Yusuf & Günay, Osman & Cetin, A.,
 *    Covariance matrix-based fire and flame detection method in video. 
 *    Machine Vision and Applications. 23. 1-11. 10.1007/s00138-011-0369-1. 
 * 
 */

#pragma once

#include "_Common.h"

#define TOO_CLOSE_MATCHES 255

class CovarianceFireDetection
{
public:
   CovarianceFireDetection();
   ~CovarianceFireDetection() = default;

   void detectFire(std::vector<cv::Rect>& fires, const cv::Mat& fire_region, const cv::Mat& frame);
   void informOfSceneChanged();

private:
   struct CovarianceCandidate : FireCandidate
   {
      int CandidateIndex;
      double MaxFeatureSimilarity;
      std::pair<int, int> SimilarPairIndices;
      cv::Point2f MinFlowPoint;
      cv::Point2f MaxFlowPoint;
      cv::Mat Deltas;
      std::vector<cv::Mat> FrameHistory;
      std::vector<std::vector<double>> FeatureHistory;

      CovarianceCandidate() : CandidateIndex( -1 ), MaxFeatureSimilarity( -1.0 ),
      MinFlowPoint{ 1e+7f, 1e+7f }, MaxFlowPoint{ -1.0f, -1.0f } {}
      CovarianceCandidate(const cv::Rect& region) : CovarianceCandidate() { Region = region; }
   };

   uint FrameCounter;

   const uint CovarianceAnalysisPeriod;
   const float MoveSensitivity;

   cv::Mat OneStepPrevFrame;
   cv::Mat TwoStepPrevFrame;
   cv::Mat FireRegionMask;
   cv::Mat EigenvalueMap;

   std::vector<cv::Rect> PrevDetectedFires;
   std::vector<CovarianceCandidate> CovFeatureCandidates;

#ifdef SHOW_PROCESS
   Mat ProcessResult;

   void destroyExistingWindows(const std::string& prefix_name) const;
   void displayMatches(const CovarianceCandidate& candidate, const std::vector<cv::Point2f>& query_points, const std::vector<cv::Point2f>& target_points, const std::vector<uchar>& found_matches) const;
   void displayHistory(const CovarianceCandidate& candidate, const cv::Scalar& box_color) const;
#endif

   void initialize();

   bool initializeFireCandidates(const std::vector<cv::Rect>& fires, const cv::Mat& resized_fire_region);

   static void accumulateSumAndSquareCombinational(
      std::vector<double>& sums, 
      std::vector<double>& squares, 
      const std::vector<double>& properties
   );
   static void getCovariance(
      std::vector<double>& covariance, 
      const std::vector<double>& sums, 
      const std::vector<double>& squares, 
      int num
   );
   void getRGBCovariance(std::vector<double>& rgb_covariance, const cv::Mat& fire_area, const cv::Mat& fire_mask) const;
   static void getSpatioTemporalFeatures(
      std::vector<double>& features, 
      const std::vector<const uchar*>& spatio_ptrs, 
      const std::vector<const uchar*>& temporal_ptrs, 
      int center_x
   );
   void getSpatioTemporalCovariance(
      std::vector<double>& st_covariance, 
      const std::vector<cv::Mat>& fire_area_set, 
      const cv::Mat& fire_mask
   ) const;
   void getCovarianceFeature(std::vector<double>& features, const cv::Mat& frame, const cv::Rect& fire_box) const;
   static void updateMaxSimilarityAndIndex(CovarianceCandidate& candidate, const std::vector<double>& features);
   void findMinMaxFlowPoint(
      CovarianceCandidate& candidate, 
      const std::vector<cv::Point2f>& query_points, 
      const std::vector<cv::Point2f>& target_points, 
      const std::vector<uchar>& found_matches
   ) const;
   bool getDeltasFromSparseOpticalFlowMatches(CovarianceCandidate& candidate, const cv::Mat& query, const cv::Mat& target) const;
   void getEigenvaluesOfCovariance(std::vector<float>& eigenvalues, const CovarianceCandidate& candidate) const;
   static bool areEigenvaluesSmallAndSimilar(std::vector<float>& eigenvalues, float threshold);
   float getAdaptiveEigenValueThreshold(const CovarianceCandidate& candidate) const;
   bool isStaticObject(const cv::Mat& query, const cv::Mat& target, const cv::Mat& mask) const;
   bool isFeatureRepeated(CovarianceCandidate& candidate);
   void removeRepeatedRegion();
   void classifyCovariance(std::vector<cv::Rect>& fires, const cv::Mat& frame);
};