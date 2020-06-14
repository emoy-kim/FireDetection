/*
 * Author: Emoy Kim
 * E-mail: emoy.kim_AT_gmail.com
 * 
 * This code is a free software; it can be freely used, changed and redistributed.
 * If you use any version of the code, please reference the code.
 * 
 */

#pragma once

#include "TrainedColorFireDetection.h"
#include "RChannelFireDetection.h"
#include "CovarianceFireDetection.h"
#include "FlowRateFireDetection.h"

class UnitedFireDetection
{
public:
   UnitedFireDetection(const UnitedFireDetection&) = delete;
   UnitedFireDetection(const UnitedFireDetection&&) = delete;
   UnitedFireDetection& operator=(const UnitedFireDetection&) = delete;
   UnitedFireDetection& operator=(const UnitedFireDetection&&) = delete;

   UnitedFireDetection(const uint& max_fire_num_to_find);
   ~UnitedFireDetection() = default;

   void detectFire(std::vector<cv::Rect>& fires, const cv::Mat& frame);
   void informOfSceneChanged() const;

private:
   cv::Mat ProcessResult;
   cv::Mat Background;

   std::unique_ptr<TrainedColorFireDetection> TrainedColorBasedDetector;
   std::unique_ptr<RChannelFireDetection> RedChannelBasedDetector;
   std::unique_ptr<CovarianceFireDetection> CovarianceBasedDetector;
   std::unique_ptr<FlowRateFireDetection> FlowRateBasedDetector;
   
   void balanceColor(cv::Mat& balanced_frame, const cv::Mat& frame) const;
   void extractForeground(cv::Mat& foreground, const cv::Mat& frame);
   void extractFireColorPixelsOnly(cv::Mat& fire_color_region, const cv::Mat& frame, const cv::Mat& mask) const;
   void extractFireColorRegion(std::vector<cv::Rect>& fires, cv::Mat& fire_color_region, const cv::Mat& frame);
   void transformOriginalFirePosition(std::vector<cv::Rect>& fires) const;
   void drawAllCandidates(const std::vector<cv::Rect>& fires, const cv::Scalar& box_color, int extended_size);
   void setFireRegion(cv::Mat& fire_region, const std::vector<cv::Rect>& fires) const;
   void findIntersection(std::vector<cv::Rect>& intersection, const std::vector<std::vector<cv::Rect>>& sets) const;
};