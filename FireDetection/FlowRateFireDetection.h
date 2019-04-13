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
   struct FlowRateCandidate : FireCandidate
   {
      int CandidateIndex;
      double MaxFlowLength;
      Point2f MinFlowPoint;
      Point2f MaxFlowPoint;
      Mat Deltas;
      vector<Mat> FrameHistory;

      FlowRateCandidate() : CandidateIndex( -1 ), MaxFlowLength( -1.0 ), 
      MinFlowPoint{ 1e+7f, 1e+7f }, MaxFlowPoint{ -1.0f, -1.0f } {}
      FlowRateCandidate(const Rect& region) : FlowRateCandidate() { Region = region; }
   };

   uint FrameCounter;

   const uint FlowRateAnalysisPeriod;
   const float PCAOutlierXThreshold;

   Mat PrevFrame;
   vector<Rect> PrevDetectedFires;
   vector<FlowRateCandidate> FlowRateCandidates;

#ifdef SHOW_PROCESS
   Mat ProcessResult;
   void drawFlow(FlowRateCandidate& candidate, const vector<Point2f>& query_points, const vector<Point2f>& target_points, const vector<uchar>& found_matches) const;
   void displayFlows(const FlowRateCandidate& candidate) const;
   void shutdownOutlierMaps() const;
   void drawMarkings(Mat& outlier_map, const Point2f& scale_factor) const;
   void drawPCAOutlierMap(const PCA& pca, const vector<Point2f>& outlier_map_points, const int& candidate_index) const;
#endif

   void initialize();
   
   bool initializeFireCandidates(const vector<Rect>& fires);

   void updateMaxFlowLength(FlowRateCandidate& candidate, const vector<Point2f>& query_points, const vector<Point2f>& target_points, const vector<uchar>& found_matches) const;
   void updateFlowDeltas(FlowRateCandidate& candidate, const vector<Point2f>& query_points, const vector<Point2f>& target_points, const vector<uchar>& found_matches) const;
   void findMinMaxFlowPoint(FlowRateCandidate& candidate, const vector<Point2f>& query_points, const vector<Point2f>& target_points, const vector<uchar>& found_matches) const;
   void calculateFlowRate(FlowRateCandidate& candidate, const Mat& frame, const Mat& fire_region) const;
   float getPCAOutlierYThreshold(const vector<Point2f>& outlier_map_points) const;
   void getPCAOutlierMapPoints(vector<Point2f>& outlier_map_points, PCA& pca, const FlowRateCandidate& candidate) const;
   void extractPCAInlierOnly(PCA& pca, Mat& inlier, const FlowRateCandidate& candidate) const;
   void getEigenvalues(vector<float>& eigenvalues, const FlowRateCandidate& candidate) const;
   bool isTurbulentEnough(const FlowRateCandidate& candidate) const;
   void removeNonTurbulentRegion();
   void classifyFlowRate(vector<Rect>& fires, const Mat& frame, const Mat& fire_region);


public:
   FlowRateFireDetection();
   ~FlowRateFireDetection() = default;

   void detectFire(vector<Rect>& fires, const Mat& fire_region, const Mat& frame);
   void informOfSceneChanged();
};