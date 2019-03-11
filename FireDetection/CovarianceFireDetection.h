#pragma once

#include "_Common.h"

#define TOO_CLOSE_MATCHES 255

class CovarianceFireDetection
{
private:
   struct CovarianceCandidate : FireCandidate
   {
      double MaxFeatureSimilarity;
      pair<int, int> SimilarPairIndices;
      Point2f MinFlowPoint;
      Point2f MaxFlowPoint;
      Mat Deltas;
      vector<Mat> FrameHistory;
      vector<vector<double>> FeatureHistory;
#ifdef SHOW_PROCESS
      int CandidateIndex;
#endif
   };

   uint FrameCounter;

   const uint CovarianceAnalysisPeriod;
   const float MoveSensitivity;

   Mat OneStepPrevFrame;
   Mat TwoStepPrevFrame;
   Mat FireRegionMask;
   Mat EigenvalueMap;

   vector<CovarianceCandidate> CovFeatureInfos;
   vector<Rect> FireCandidates;

#ifdef SHOW_PROCESS
   Mat ProcessResult;

   void destroyExistingWindows(const string& prefix_name) const;
   void displayMatches(const CovarianceCandidate& candidate, const vector<Point2f>& query_points, const vector<Point2f>& target_points, const vector<uchar>& found_mathces) const;
   void displayHistory(const CovarianceCandidate& candidate, const Scalar& box_color) const;
   void shutdownOutlierMaps() const;
   void drawMarkings(Mat& outlier_map, const Point2f& scale_factor) const;
   void drawPCAOutlierMap(const PCA& pca, const vector<Point2f>& outlier_map_points, const int& candidate_index) const;
#endif

   void initialize();

   bool initializeFireCandidateInfos(const vector<Rect>& fires, const Mat& frame, const Mat& resized_fire_region);

   void accumulateSumAndSquareCombinational(vector<double>& sums, vector<double>& squares, const vector<double>& properties) const;
   void getCovariance(vector<double>& covariance, const vector<double>& sums, const vector<double>& squares, const int& num) const;
   void getRGBCovariance(vector<double>& rgb_covariance, const Mat& fire_area, const Mat& fire_mask) const;
   void getSpatioTemporalFeatures(vector<double>& features, const vector<const uchar*>&spatio_ptrs, const vector<const uchar*>& temporal_ptrs, int center_x) const;
   void getSpatioTemporalCovariance(vector<double>& st_covariance, const vector<Mat>& fire_area_set, const Mat& fire_mask) const;
   void getCovarianceFeature(vector<double>& features, const Mat& resized_frame, const Rect& fire_box) const;
   void updateMaxSimilarityAndIndex(CovarianceCandidate& candidate, const vector<double>& features) const;
   void findMinMaxFlowPoint(CovarianceCandidate& candidate, const vector<Point2f>& query_points, const vector<Point2f>& target_points, const vector<uchar>& found_mathces) const;
   bool getDeltasFromSparseOpticalFlowMatches(CovarianceCandidate& candidate, const Mat& query, const Mat& target) const;
   float getPCAOutlierYThreshold(const vector<Point2f>& outlier_map_points) const;
   void getPCAOutlierMapPoints(vector<Point2f>& outlier_map_points, PCA& pca, const CovarianceCandidate& candidate) const;
   Mat removePCAOutlier(PCA& pca, const CovarianceCandidate& candidate) const;
   void getEigenvaluesOfCovariance(vector<float>& eigenvalues, const CovarianceCandidate& candidate) const;
   bool areEigenvaluesSmallAndSimilar(vector<float>& eigenvalues, const float& threshold) const;
   float getAdaptiveEigenValueThreshold(const CovarianceCandidate& candidate) const;
   bool isMoving(const Mat& query, const Mat& target, const Mat& mask) const;
   bool isFeatureRepeated(CovarianceCandidate& candidate);
   void removeRepeatedRegion();
   void classifyCovariance(const Mat& resized_frame);

   void getFirePosition(vector<Rect>& fires, const Mat& frame);


public:
   CovarianceFireDetection();
   ~CovarianceFireDetection() = default;

   void detectFire(vector<Rect>& fires, const Mat& fire_region, const Mat& frame);
   void informOfSceneChanged();
   void setFireRegionNumToFind(const int& fire_num) const;
};