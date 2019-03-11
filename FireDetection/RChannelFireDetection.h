#pragma once

#include "_Common.h"

class RChannelFireDetection
{
   struct RChannelCandidate : FireCandidate
   {
      vector<double> MeanPerturbation;

      RChannelCandidate() = default;
      RChannelCandidate(const Rect& region) { Region = region; }
   };

   uint FrameCounter;

   const uint RChannelAnalysisPeriod;
   const double MinPerturbingThreshold;

   vector<Rect> PrevDetectedFires;
   vector<RChannelCandidate> RChannelCandidates;

#ifdef SHOW_PROCESS
   Mat ProcessResult;
   void displayHistogram(const Mat& histogram, const Mat& fire_area, const Mat& fire_mask, const int& index) const;
   void shutdownHistogram() const;
#endif

   void initialize();

   bool initializeFireCandidates(const vector<Rect>& fires);

   Mat getNormalizedHistogram(const Mat& r_channel, const Mat& mask) const;
   double calculateWeightedMean(const Mat& histogram, const float& min_frequency = 0.0f) const;
   bool isPerturbingEnough(const RChannelCandidate& candidate, const double& min_intensity = 0.0) const;
   void removeNonPerturbedRegion();
   void classifyRChannelHistogram(vector<Rect>& fires, const Mat& frame, const Mat& fire_region);


public:
   RChannelFireDetection();
   ~RChannelFireDetection() = default;

   void detectFire(vector<Rect>& fires, const Mat& fire_region, const Mat& frame);
   void informOfSceneChanged();
};