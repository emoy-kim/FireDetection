#include "UnitedFireDetection.h"

UnitedFireDetection::UnitedFireDetection(const uint& max_fire_num_to_find)
{
   FireNumToFind = max_fire_num_to_find;
   AnalysisFrameSize.width = 640;
   AnalysisFrameSize.height = 480;

   TrainedColorBasedDetector = std::make_unique<TrainedColorFireDetection>();
   RedChannelBasedDetector = std::make_unique<RChannelFireDetection>();
   CovarianceBasedDetector = std::make_unique<CovarianceFireDetection>();
   FlowRateBasedDetector = std::make_unique<FlowRateFireDetection>();
}

void UnitedFireDetection::balanceColor(cv::Mat& balanced_frame, const cv::Mat& frame)
{
   balanced_frame = frame.clone();
   return;
      
   cv::Mat frame_d;
   frame.convertTo( frame_d, CV_64FC3 );

   cv::Scalar gray_value = cv::mean( frame_d );
   const double max_value = std::max( std::max( gray_value(0), gray_value(1) ), gray_value(2) );
   gray_value(0) = max_value / gray_value(0);
   gray_value(1) = max_value / gray_value(1);
   gray_value(2) = max_value / gray_value(2);

   std::vector<cv::Mat> channels(3);
   cv::split( frame_d, channels );
   channels[0].convertTo( channels[0], CV_8UC1, gray_value(0) );
   channels[1].convertTo( channels[1], CV_8UC1, gray_value(1) );
   channels[2].convertTo( channels[2], CV_8UC1, gray_value(2) );

   cv::merge( channels, balanced_frame );
}

void UnitedFireDetection::extractForeground(cv::Mat& foreground, const cv::Mat& frame)
{
   foreground = cv::Mat::ones( frame.size(), CV_8UC1 );
   if (Background.empty()) cv::cvtColor( frame, Background, cv::COLOR_BGR2GRAY );
   
   cv::Mat gray_frame;
   cv::cvtColor( frame, gray_frame, cv::COLOR_BGR2GRAY );

   cv::Mat frame_difference;
   cv::absdiff( Background, gray_frame, frame_difference );
   cv::threshold( frame_difference, foreground, 20.0, 255.0, cv::THRESH_BINARY );
}

void UnitedFireDetection::extractFireColorPixelsOnly(
   cv::Mat& fire_color_region, 
   const cv::Mat& frame, 
   const cv::Mat& mask
)
{
   const int min_r = 140;
   const int min_difference_r_and_g = 15;

   fire_color_region = cv::Mat::zeros( frame.size(), CV_8UC1 );
   cv::Mat roi_mask = mask.empty() ? cv::Mat::ones( frame.size(), CV_8UC1 ) : mask.clone();
   for (int j = 0; j < frame.rows; ++j) {
      const auto* frame_ptr = frame.ptr<cv::Vec3b>(j);
      const auto* mask_ptr = roi_mask.ptr<uchar>(j);
      auto* fire_ptr = fire_color_region.ptr<uchar>(j);
      for (int i = 0; i < frame.cols; ++i) {
         if (mask_ptr[i]) {
            const auto b = static_cast<int>(frame_ptr[i][0]);
            const auto g = static_cast<int>(frame_ptr[i][1]);
            const auto r = static_cast<int>(frame_ptr[i][2]);

            if (r > min_r && r - g > min_difference_r_and_g && g > b) {
               fire_ptr[i] = 255;
            }
         }
      }
   }

   const cv::Mat circle_elem = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size(3, 3) );
   cv::morphologyEx( fire_color_region, fire_color_region, cv::MORPH_CLOSE, circle_elem );
}

void UnitedFireDetection::extractFireColorRegion(
   std::vector<cv::Rect>& fires, 
   cv::Mat& fire_color_region, 
   const cv::Mat& frame
)
{
   cv::Mat foreground;
   extractForeground( foreground, frame );
   extractFireColorPixelsOnly( fire_color_region, frame, foreground );

   std::vector<std::vector<cv::Point>> contours;
   cv::Mat contoured = fire_color_region.clone();
   cv::findContours( contoured, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );
   
   std::vector<FireCandidate> candidates;
   for (const auto& contour : contours) {
      FireCandidate candidate;
      candidate.Region = cv::boundingRect( cv::Mat(contour) );
      candidates.emplace_back( candidate );
   }
   extractFromCandidates( fires, candidates );
}

void UnitedFireDetection::transformOriginalFirePosition(std::vector<cv::Rect>& fires) const
{
   const cv::Point2d to_frame(
      ProcessResult.cols / static_cast<double>(AnalysisFrameSize.width),
      ProcessResult.rows / static_cast<double>(AnalysisFrameSize.height)
   );
   for (auto& rect : fires) {
      rect = cv::Rect(
         cv::Point(
            static_cast<int>(round( rect.tl().x * to_frame.x )), 
            static_cast<int>(round( rect.tl().y * to_frame.y ))
         ),
         cv::Point(
            static_cast<int>(round( rect.br().x * to_frame.x )), 
            static_cast<int>(round( rect.br().y * to_frame.y ))
         )
      );
   }
}

void UnitedFireDetection::drawAllCandidates(
   const std::vector<cv::Rect>& fires, 
   const cv::Scalar& box_color, 
   int extended_size
)
{
   for (const auto& candidate : fires) {
      cv::rectangle( 
         ProcessResult, 
         cv::Rect(
            candidate.x - extended_size, 
            candidate.y - extended_size, 
            candidate.width + 2 * extended_size, 
            candidate.height + 2 * extended_size
         ), 
         box_color,
         2 
      );
   }
}

void UnitedFireDetection::setFireRegion(cv::Mat& fire_region, const std::vector<cv::Rect>& fires) const
{
   fire_region = cv::Mat::zeros( ProcessResult.size(), CV_8UC1 );
   for (const auto& rect : fires) {
      fire_region(rect) = 255;
   }
}

void UnitedFireDetection::findIntersection(
   std::vector<cv::Rect>& intersection, 
   const std::vector<std::vector<cv::Rect>>& sets
) const
{
   cv::Mat common;
   setFireRegion( common, sets[0] );
   for (uint i = 1; i < sets.size(); ++i) {
      cv::Mat region;
      setFireRegion( region, sets[i] );
      common &= region;
   }
   
   std::vector<std::vector<cv::Point>> contours;
   cv::findContours( common, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );
   
   intersection.clear();
   for (const auto& contour : contours) {
      intersection.emplace_back( cv::boundingRect( cv::Mat( contour ) ) );
   }
}

void UnitedFireDetection::detectFire(std::vector<cv::Rect>& fires, const cv::Mat& frame)
{
   ProcessResult = frame.clone();

   cv::Mat resized_frame;
   cv::resize( frame, resized_frame, AnalysisFrameSize );
   
   cv::Mat balanced_frame;
   balanceColor( balanced_frame, resized_frame );

   cv::Mat fire_region;
   extractFireColorRegion( fires, fire_region, balanced_frame );

   TrainedColorBasedDetector->detectFire( fires, fire_region, balanced_frame );

   std::vector<cv::Rect> result_from_r_channel(fires);
   RedChannelBasedDetector->detectFire( result_from_r_channel, fire_region, balanced_frame );
   transformOriginalFirePosition( result_from_r_channel );
   drawAllCandidates( result_from_r_channel, MAGENTA_COLOR, 4 );

   std::vector<cv::Rect> result_from_covariance(fires);
   CovarianceBasedDetector->detectFire( result_from_covariance, fire_region, balanced_frame );
   transformOriginalFirePosition( result_from_covariance );
   drawAllCandidates( result_from_covariance, YELLOW_COLOR, 8 );

   std::vector<cv::Rect> result_from_flow_rate(fires);
   FlowRateBasedDetector->detectFire( result_from_flow_rate, fire_region, balanced_frame );
   transformOriginalFirePosition( result_from_flow_rate );
   drawAllCandidates( result_from_flow_rate, CYAN_COLOR, 12 );
 
   const std::vector<std::vector<cv::Rect>> results = { 
      result_from_r_channel, 
      result_from_covariance, 
      result_from_flow_rate 
   };
   findIntersection( fires, results );
   drawAllCandidates( fires, RED_COLOR, 16 );
   
   imshow( "Process Result", ProcessResult );
}

void UnitedFireDetection::informOfSceneChanged() const
{
   TrainedColorBasedDetector->informOfSceneChanged();
   RedChannelBasedDetector->informOfSceneChanged();
   CovarianceBasedDetector->informOfSceneChanged();
   FlowRateBasedDetector->informOfSceneChanged();
}