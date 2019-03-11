#pragma once

#include "_Common.h"

class FlowRateFireDetection
{
   struct FlowRateCandidate : FireCandidate
   {
      Point2f MinFlowPoint;
      Point2f MaxFlowPoint;
      double MaxFlowLength;
      Mat Deltas;
      vector<Mat> FrameHistory;
#ifdef SHOW_PROCESS
      int CandidateIndex;
#endif
   };

   uint FrameCounter;

   const uint FlowRateAnalysisPeriod;
   const float PCAOutlierXThreshold;

   Mat PrevFrame;
   vector<FlowRateCandidate> FlowRateInfos;
   vector<Rect> FireCandidates;

#ifdef SHOW_PROCESS
   Mat ProcessResult;
   void drawFlow(FlowRateCandidate& candidate, const vector<Point2f>& query_points, const vector<Point2f>& target_points, const vector<uchar>& found_mathces) const;
   void displayFlows(const FlowRateCandidate& candidate) const;
   void shutdownOutlierMaps() const;
   void drawMarkings(Mat& outlier_map, const Point2f& scale_factor) const;
   void drawPCAOutlierMap(const PCA& pca, const vector<Point2f>& outlier_map_points, const int& candidate_index) const;
#endif

   void initialize();
   
   bool initializeFireCandidateInfos(const vector<Rect>& fires, const Mat& frame);

   void updateMaxFlowLength(FlowRateCandidate& candidate, const vector<Point2f>& query_points, const vector<Point2f>& target_points, const vector<uchar>& found_mathces) const;
   void updateFlowDeltas(FlowRateCandidate& candidate, const vector<Point2f>& query_points, const vector<Point2f>& target_points, const vector<uchar>& found_mathces) const;
   void findMinMaxFlowPoint(FlowRateCandidate& candidate, const vector<Point2f>& query_points, const vector<Point2f>& target_points, const vector<uchar>& found_mathces) const;
   void calculateFlowRate(FlowRateCandidate& candidate, const Mat& frame, const Mat& fire_region) const;
   float getPCAOutlierYThreshold(const vector<Point2f>& outlier_map_points) const;
   void getPCAOutlierMapPoints(vector<Point2f>& outlier_map_points, PCA& pca, const FlowRateCandidate& candidate) const;
   Mat removePCAOutlier(PCA& pca, const FlowRateCandidate& candidate) const;
   void getEigenvalues(vector<float>& eigenvalues, const FlowRateCandidate& candidate) const;
   bool isTurbulentEnough(const FlowRateCandidate& candidate) const;
   void removeNonTurbulentRegion();
   void classifyFlowRate(const Mat& frame, const Mat& fire_region);

   void getFirePosition(vector<Rect>& fires, const Mat& frame);


public:
   FlowRateFireDetection();

   void detectFire(vector<Rect>& fires, const Mat& fire_region, const Mat& frame);
   void informOfSceneChanged();
};