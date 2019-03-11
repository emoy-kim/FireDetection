#include "UnitedFireDetection.h"

uint FireNumToFind;
Mat ProbabilityMap;
Size AnalysisFrameSize;

UnitedFireDetection::UnitedFireDetection(const uint& max_fire_num_to_find)
{
   FireNumToFind = max_fire_num_to_find;
   AnalysisFrameSize.width = 640;
   AnalysisFrameSize.height = 480;

   TrainedColorBasedDetector = make_unique<TrainedColorFireDetection>();
   RedChannelBasedDetector = make_unique<RChannelFireDetection>();
   CovarianceBasedDetector = make_unique<CovarianceFireDetection>();
   FlowRateBasedDetector = make_unique<FlowRateFireDetection>();
}

void UnitedFireDetection::balanceColor(Mat& balanced_frame, const Mat& frame) const
{
   Mat frame_d;
   frame.convertTo( frame_d, CV_64FC3 );

   Scalar gray_value = mean( frame_d );
   const double max_value = max( max( gray_value(0), gray_value(1) ), gray_value(2) );
   gray_value(0) = max_value / gray_value(0);
   gray_value(1) = max_value / gray_value(1);
   gray_value(2) = max_value / gray_value(2);

   vector<Mat> channels(3);
   split( frame_d, channels );
   channels[0].convertTo( channels[0], CV_8UC1, gray_value(0) );
   channels[1].convertTo( channels[1], CV_8UC1, gray_value(1) );
   channels[2].convertTo( channels[2], CV_8UC1, gray_value(2) );

   merge( channels, balanced_frame );
}

void UnitedFireDetection::extractForeground(Mat& foreground, const Mat& frame)
{
   foreground = Mat::ones( frame.size(), CV_8UC1 );
   if (Background.empty()) cvtColor( frame, Background, CV_BGR2GRAY );
   
   Mat gray_frame;
   cvtColor( frame, gray_frame, CV_BGR2GRAY );

   Mat frame_difference;
   absdiff( Background, gray_frame, frame_difference );
   threshold( frame_difference, foreground, 20.0, 255.0, THRESH_BINARY );
}

void UnitedFireDetection::extractFireColorPixelsOnly(Mat& fire_color_region, const Mat& frame, const Mat& mask) const
{
   const int min_r = 140;
   const int min_difference_r_and_g = 15;

   fire_color_region = Mat::zeros( frame.size(), CV_8UC1 );
   Mat roi_mask = mask.empty() ? Mat::ones( frame.size(), CV_8UC1 ) : mask.clone();
   for (int j = 0; j < frame.rows; ++j) {
      const auto* frame_ptr = frame.ptr<Vec3b>(j);
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

   const Mat circle_elem = getStructuringElement( MORPH_ELLIPSE, Size(3, 3) );
   morphologyEx( fire_color_region, fire_color_region, MORPH_CLOSE, circle_elem );
}

void UnitedFireDetection::extractFireColorRegion(vector<Rect>& fires, Mat& fire_color_region, const Mat& frame)
{
   Mat foreground;
   extractForeground( foreground, frame );
   extractFireColorPixelsOnly( fire_color_region, frame, foreground );

   vector<vector<Point>> contours;
   Mat contoured = fire_color_region.clone();
   findContours( contoured, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
   
   vector<FireCandidate> candidates;
   for (const auto& contour : contours) {
      FireCandidate candidate;
      candidate.Region = boundingRect( Mat(contour) );
      candidates.emplace_back( candidate );
   }
   extractFromCandidates( fires, candidates );
}

void UnitedFireDetection::transformOriginalFirePosition(vector<Rect>& fires) const
{
   const Point2d to_frame(
      ProcessResult.cols / static_cast<double>(AnalysisFrameSize.width),
      ProcessResult.rows / static_cast<double>(AnalysisFrameSize.height)
   );
   for (auto& rect : fires) {
      rect = Rect(
         Point(
            static_cast<int>(round( rect.tl().x * to_frame.x )), 
            static_cast<int>(round( rect.tl().y * to_frame.y ))
         ),
         Point(
            static_cast<int>(round( rect.br().x * to_frame.x )), 
            static_cast<int>(round( rect.br().y * to_frame.y ))
         )
      );
   }
}

void UnitedFireDetection::drawAllCandidates(const vector<Rect>& fires, const Scalar& box_color, const int& extended_size)
{
   for (const auto& candidate : fires) {
      rectangle( 
         ProcessResult, 
         Rect(
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

void UnitedFireDetection::setFireRegion(Mat& fire_region, const vector<Rect>& fires) const
{
   fire_region = Mat::zeros( ProcessResult.size(), CV_8UC1 );
   for (const auto& rect : fires) {
      fire_region(rect) = 255;
   }
}

void UnitedFireDetection::findIntersection(vector<Rect>& intersection, const vector<vector<Rect>>& sets) const
{
   Mat common;
   setFireRegion( common, sets[0] );
   for (uint i = 1; i < sets.size(); ++i) {
      Mat region;
      setFireRegion( region, sets[i] );
      common &= region;
   }
   
   vector<vector<Point>> contours;
   findContours( common, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
   
   intersection.clear();
   for (const auto& contour : contours) {
      intersection.emplace_back( boundingRect( Mat( contour ) ) );
   }
}

void UnitedFireDetection::detectFire(vector<Rect>& fires, const Mat& frame)
{
   ProcessResult = frame.clone();

   Mat resized_frame;
   resize( frame, resized_frame, AnalysisFrameSize );
   
   Mat balanced_frame;
   balanceColor( balanced_frame, resized_frame );

   Mat fire_region;
   extractFireColorRegion( fires, fire_region, balanced_frame );

   TrainedColorBasedDetector->detectFire( fires, fire_region, balanced_frame );

   vector<Rect> result_from_r_channel(fires);
   RedChannelBasedDetector->detectFire( result_from_r_channel, fire_region, balanced_frame );
   transformOriginalFirePosition( result_from_r_channel );
   drawAllCandidates( result_from_r_channel, MAGENTA_COLOR, 4 );

   vector<Rect> result_from_covariance(fires);
   CovarianceBasedDetector->detectFire( result_from_covariance, fire_region, balanced_frame );
   transformOriginalFirePosition( result_from_covariance );
   drawAllCandidates( result_from_covariance, YELLOW_COLOR, 8 );

   vector<Rect> result_from_flow_rate(fires);
   FlowRateBasedDetector->detectFire( result_from_flow_rate, fire_region, balanced_frame );
   transformOriginalFirePosition( result_from_flow_rate );
   drawAllCandidates( result_from_flow_rate, CYAN_COLOR, 12 );
 
   const vector<vector<Rect>> results = { 
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