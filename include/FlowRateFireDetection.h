/*
 * Author: Emoy Kim
 * E-mail: emoy.kim_AT_gmail.com
 * 
 * This code is a free software; it can be freely used, changed and redistributed.
 * If you use any version of the code, please reference the code.
 * 
 */

#pragma once

#include "_Common.h"

class FlowRateFireDetection
{
public:
   FlowRateFireDetection();
   ~FlowRateFireDetection() = default;

   void detectFire(std::vector<cv::Rect>& fires, const cv::Mat& fire_region, const cv::Mat& frame);
   void informOfSceneChanged();

private:
   struct FlowRateCandidate : FireCandidate
   {
      int CandidateIndex;
      double MaxFlowLength;
      cv::Point2f MinFlowPoint;
      cv::Point2f MaxFlowPoint;
      cv::Mat Deltas;
      std::vector<cv::Mat> FrameHistory;

      FlowRateCandidate() : CandidateIndex( -1 ), MaxFlowLength( -1.0 ), 
      MinFlowPoint{ 1e+7f, 1e+7f }, MaxFlowPoint{ -1.0f, -1.0f } {}
      explicit FlowRateCandidate(const cv::Rect& region) : FlowRateCandidate() { Region = region; }
   };

   uint FrameCounter;

   const uint FlowRateAnalysisPeriod;
   const float PCAOutlierXThreshold;

   cv::Mat PrevFrame;
   std::vector<cv::Rect> PrevDetectedFires;
   std::vector<FlowRateCandidate> FlowRateCandidates;

#ifdef SHOW_PROCESS
   Mat ProcessResult;
   void drawFlow(FlowRateCandidate& candidate, const std::vector<cv::Point2f>& query_points, const std::vector<cv::Point2f>& target_points, const std::vector<uchar>& found_matches) const;
   void displayFlows(const FlowRateCandidate& candidate) const;
   void shutdownOutlierMaps() const;
   void drawMarkings(cv::Mat& outlier_map, const cv::Point2f& scale_factor) const;
   void drawPCAOutlierMap(const cv::PCA& pca, const std::vector<cv::Point2f>& outlier_map_points, int candidate_index) const;
#endif

   void initialize();
   
   bool initializeFireCandidates(const std::vector<cv::Rect>& fires);

   static void updateMaxFlowLength(
      FlowRateCandidate& candidate, 
      const std::vector<cv::Point2f>& query_points, 
      const std::vector<cv::Point2f>& target_points, 
      const std::vector<uchar>& found_matches
   );
   static void updateFlowDeltas(
      FlowRateCandidate& candidate, 
      const std::vector<cv::Point2f>& query_points, 
      const std::vector<cv::Point2f>& target_points, 
      const std::vector<uchar>& found_matches
   );
   static void findMinMaxFlowPoint(
      FlowRateCandidate& candidate, 
      const std::vector<cv::Point2f>& query_points, 
      const std::vector<cv::Point2f>& target_points, 
      const std::vector<uchar>& found_matches
   );
   void calculateFlowRate(FlowRateCandidate& candidate, const cv::Mat& frame, const cv::Mat& fire_region) const;
   [[nodiscard]] static float getPCAOutlierYThreshold(const std::vector<cv::Point2f>& outlier_map_points);
   static void getPCAOutlierMapPoints(std::vector<cv::Point2f>& outlier_map_points, cv::PCA& pca, const FlowRateCandidate& candidate);
   void extractPCAInlierOnly(cv::PCA& pca, cv::Mat& inlier, const FlowRateCandidate& candidate) const;
   void getEigenvalues(std::vector<float>& eigenvalues, const FlowRateCandidate& candidate) const;
   bool isTurbulentEnough(const FlowRateCandidate& candidate) const;
   void removeNonTurbulentRegion();
   void classifyFlowRate(std::vector<cv::Rect>& fires, const cv::Mat& frame, const cv::Mat& fire_region);
};