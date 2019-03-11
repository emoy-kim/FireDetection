#pragma once

#include "_Common.h"

#define TOO_CLOSE_MATCHES 255

class CovarianceFireDetection
{
private:
   struct CovarianceCandidate : FireCandidate
   {
      int CandidateIndex;
      double MaxFeatureSimilarity;
      pair<int, int> SimilarPairIndices;
      Point2f MinFlowPoint;
      Point2f MaxFlowPoint;
      Mat Deltas;
      vector<Mat> FrameHistory;
      vector<vector<double>> FeatureHistory;

      CovarianceCandidate() : CandidateIndex( -1 ), MaxFeatureSimilarity( -1.0 ),
      MinFlowPoint{ 1e+7f, 1e+7f }, MaxFlowPoint{ -1.0f, -1.0f } {}
      CovarianceCandidate(const Rect& region) : CovarianceCandidate() { Region = region; }
   };

   uint FrameCounter;

   const uint CovarianceAnalysisPeriod;
   const float MoveSensitivity;

   Mat OneStepPrevFrame;
   Mat TwoStepPrevFrame;
   Mat FireRegionMask;
   Mat EigenvalueMap;

   vector<Rect> PrevDetectedFires;
   vector<CovarianceCandidate> CovFeatureCandidates;

#ifdef SHOW_PROCESS
   Mat ProcessResult;

   void destroyExistingWindows(const string& prefix_name) const;
   void displayMatches(const CovarianceCandidate& candidate, const vector<Point2f>& query_points, const vector<Point2f>& target_points, const vector<uchar>& found_matches) const;
   void displayHistory(const CovarianceCandidate& candidate, const Scalar& box_color) const;
#endif

   void initialize();

   bool initializeFireCandidates(const vector<Rect>& fires, const Mat& resized_fire_region);

   void accumulateSumAndSquareCombinational(vector<double>& sums, vector<double>& squares, const vector<double>& properties) const;
   void getCovariance(vector<double>& covariance, const vector<double>& sums, const vector<double>& squares, const int& num) const;
   void getRGBCovariance(vector<double>& rgb_covariance, const Mat& fire_area, const Mat& fire_mask) const;
   void getSpatioTemporalFeatures(vector<double>& features, const vector<const uchar*>& spatio_ptrs, const vector<const uchar*>& temporal_ptrs, int center_x) const;
   void getSpatioTemporalCovariance(vector<double>& st_covariance, const vector<Mat>& fire_area_set, const Mat& fire_mask) const;
   void getCovarianceFeature(vector<double>& features, const Mat& frame, const Rect& fire_box) const;
   void updateMaxSimilarityAndIndex(CovarianceCandidate& candidate, const vector<double>& features) const;
   void findMinMaxFlowPoint(CovarianceCandidate& candidate, const vector<Point2f>& query_points, const vector<Point2f>& target_points, const vector<uchar>& found_matches) const;
   bool getDeltasFromSparseOpticalFlowMatches(CovarianceCandidate& candidate, const Mat& query, const Mat& target) const;
   void getEigenvaluesOfCovariance(vector<float>& eigenvalues, const CovarianceCandidate& candidate) const;
   bool areEigenvaluesSmallAndSimilar(vector<float>& eigenvalues, const float& threshold) const;
   float getAdaptiveEigenValueThreshold(const CovarianceCandidate& candidate) const;
   bool isStaticObject(const Mat& query, const Mat& target, const Mat& mask) const;
   bool isFeatureRepeated(CovarianceCandidate& candidate);
   void removeRepeatedRegion();
   void classifyCovariance(vector<Rect>& fires, const Mat& frame);


public:
   CovarianceFireDetection();
   ~CovarianceFireDetection() = default;

   void detectFire(vector<Rect>& fires, const Mat& fire_region, const Mat& frame);
   void informOfSceneChanged();
};