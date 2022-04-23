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

class RChannelFireDetection
{
public:
   RChannelFireDetection();
   ~RChannelFireDetection() = default;

   void detectFire(std::vector<cv::Rect>& fires, const cv::Mat& fire_region, const cv::Mat& frame);
   void informOfSceneChanged();

private:
   struct RChannelCandidate : FireCandidate
   {
      std::vector<double> MeanPerturbation;

      RChannelCandidate() = default;
      explicit RChannelCandidate(const cv::Rect& region) { Region = region; }
   };

   uint FrameCounter;

   const uint RChannelAnalysisPeriod;
   const double MinPerturbingThreshold;

   std::vector<cv::Rect> PrevDetectedFires;
   std::vector<RChannelCandidate> RChannelCandidates;

#ifdef SHOW_PROCESS
   Mat ProcessResult;
   void displayHistogram(const Mat& histogram, const Mat& fire_area, const Mat& fire_mask, int index) const;
   void shutdownHistogram() const;
#endif

   void initialize();

   bool initializeFireCandidates(const std::vector<cv::Rect>& fires);

   [[nodiscard]] static cv::Mat getNormalizedHistogram(const cv::Mat& r_channel, const cv::Mat& mask);
   [[nodiscard]] static double calculateWeightedMean(const cv::Mat& histogram, float min_frequency = 0.0f);
   [[nodiscard]] bool isPerturbingEnough(const RChannelCandidate& candidate, double min_intensity = 0.0) const;
   void removeNonPerturbedRegion();
   void classifyRChannelHistogram(std::vector<cv::Rect>& fires, const cv::Mat& frame, const cv::Mat& fire_region);
};