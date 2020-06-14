#include "TrainedColorFireDetection.h"

TrainedColorFireDetection::TrainedColorFireDetection() : 
   MovementThreshold( 100 ), MaxCandidatesNum( 20 ), 
   ProbabilityThreshold( 0.45 ), ResultMapLearningRate( 0.02 ), FireSensitivity( 70.0 )
{
   ColorClassifierWeights = {
       1.74761770219579,
      -0.566372728291717,
      -1.10376760992002,
       1.44742382734593,
      -0.0838645672747927,
      -0.654342214953116
   };
   initialize();
}

void TrainedColorFireDetection::initialize()
{
   PrevBlurredFrame = cv::Mat::zeros( AnalysisFrameSize, CV_8UC3 );
   ProbabilityMap = cv::Mat::zeros( AnalysisFrameSize, CV_64FC1 );
#ifdef SHOW_PROCESS
   cv::destroyWindow( "Final Result Map" );
#endif
}

#ifdef SHOW_PROCESS
void TrainedColorFireDetection::showProcess(const cv::Scalar& box_color)
{
   for (const auto& candidate : FireColorCandidates) {
      cv::rectangle(
         ProcessResult,
         Rect(
            Point(candidate.x - 16, candidate.y - 16),
            Point(candidate.x + 16, candidate.y + 16)
         ),
         box_color,
         2
      );
   }
   imshow( "Processing Result", ProcessResult );
}
#endif

double TrainedColorFireDetection::calculateProbabilityBasedColor(
   int curr_idx, 
   const cv::Vec3b* upper_ptr, 
   const cv::Vec3b* curr_ptr, 
   const cv::Vec3b* lower_ptr
)
{
   const double left_right_r_diff = abs(curr_ptr[curr_idx + 1][2] - curr_ptr[curr_idx - 1][2]);
   const double left_right_g_diff = abs(curr_ptr[curr_idx + 1][1] - curr_ptr[curr_idx - 1][1]);
   const double left_right_b_diff = abs(curr_ptr[curr_idx + 1][0] - curr_ptr[curr_idx - 1][0]);
   const double up_down_r_diff = abs(upper_ptr[curr_idx][2] - lower_ptr[curr_idx][2]);
   const double up_down_g_diff = abs(upper_ptr[curr_idx][1] - lower_ptr[curr_idx][1]);
   const double up_down_b_diff = abs(upper_ptr[curr_idx][0] - lower_ptr[curr_idx][0]);

   return (
      ColorClassifierWeights[0] * curr_ptr[curr_idx][2] +
      ColorClassifierWeights[1] * curr_ptr[curr_idx][1] +
      ColorClassifierWeights[2] * curr_ptr[curr_idx][0] +
      ColorClassifierWeights[3] * (left_right_r_diff + up_down_r_diff) +
      ColorClassifierWeights[4] * (left_right_g_diff + up_down_g_diff) +
      ColorClassifierWeights[5] * (left_right_b_diff + up_down_b_diff)
      ) / 255.0;
}

void TrainedColorFireDetection::findMovingPixels(cv::Mat& probability_map, const cv::Mat& blurred_frame)
{
   for (int j = 1; j < blurred_frame.rows - 1; ++j) {
      const auto* blurred_upper_ptr = blurred_frame.ptr<cv::Vec3b>(j - 1);
      const auto* blurred_ptr = blurred_frame.ptr<cv::Vec3b>(j);
      const auto* blurred_lower_ptr = blurred_frame.ptr<cv::Vec3b>(j + 1);

      const auto* prev_ptr = PrevBlurredFrame.ptr<cv::Vec3b>(j);
      auto* prob_ptr = probability_map.ptr<double>(j);
      for (int i = 1; i < blurred_frame.cols - 1; ++i) {
         const int amount_of_temporal_movement = 
            abs( blurred_ptr[i][0] - prev_ptr[i][0] ) + 
            abs( blurred_ptr[i][1] - prev_ptr[i][1] ) + 
            abs( blurred_ptr[i][2] - prev_ptr[i][2] );

         if (amount_of_temporal_movement > MovementThreshold) {
            prob_ptr[i] = calculateProbabilityBasedColor( i, blurred_upper_ptr, blurred_ptr, blurred_lower_ptr );
         }
      }
   }
   cv::blur( probability_map, probability_map, cv::Size(5, 5) );
}

bool TrainedColorFireDetection::isLocalMaximum(
   int curr_idx, 
   const double* upper_ptr, 
   const double* curr_ptr, 
   const double* lower_ptr
)
{
   return
      curr_ptr[curr_idx - 1] < curr_ptr[curr_idx] && 
      curr_ptr[curr_idx + 1] < curr_ptr[curr_idx] &&
      upper_ptr[curr_idx] < curr_ptr[curr_idx] && 
      lower_ptr[curr_idx] < curr_ptr[curr_idx];
}

void TrainedColorFireDetection::findTopProbabilities(std::vector<std::pair<cv::Point, double>>& local_maxima)
{
   if (local_maxima.size() <= MaxCandidatesNum) {
      for (const auto& local_maximum : local_maxima) {
         FireColorCandidates.emplace_back( local_maximum.first );
      }
   }
   else {
      while (FireColorCandidates.size() < MaxCandidatesNum) {
         const int max_index = findIndexWithMaxOfSecondValues( local_maxima );
         FireColorCandidates.emplace_back( local_maxima[max_index].first );

         local_maxima.erase( local_maxima.begin() + max_index );
      }
   }
}

void TrainedColorFireDetection::findTopOfLocalMaxima(const cv::Mat& probability_map)
{
   std::vector<std::pair<cv::Point, double>> local_maxima;
   for (int j = 1; j < probability_map.rows - 1; ++j) {
      const auto* prob_upper_ptr = probability_map.ptr<double>(j - 1);
      const auto* prob_ptr = probability_map.ptr<double>(j);
      const auto* prob_lower_ptr = probability_map.ptr<double>(j + 1);
      for (int i = 1; i < probability_map.cols - 1; ++i) {
         if (ProbabilityThreshold < prob_ptr[i] && local_maxima.size() <= 4096 &&
             isLocalMaximum( i, prob_upper_ptr, prob_ptr, prob_lower_ptr )) {
            local_maxima.emplace_back( std::make_pair( cv::Point(i, j), prob_ptr[i] ) );
         }
      }
   }

   FireColorCandidates.clear();
   findTopProbabilities( local_maxima );
}

void TrainedColorFireDetection::classifyColorAndMotion(const cv::Mat& frame)
{
   cv::Mat blurred_frame;
   cv::GaussianBlur( frame, blurred_frame, cv::Size(11, 11), 0.1 );

   cv::Mat probability_map = cv::Mat::zeros( blurred_frame.size(), CV_64FC1 );
   findMovingPixels( probability_map, blurred_frame );
   findTopOfLocalMaxima( probability_map );
   PrevBlurredFrame = blurred_frame.clone();
#ifdef SHOW_PROCESS
   imshow( "Probability Map", probability_map );
   showProcess( GREEN_COLOR );
#endif
}

void TrainedColorFireDetection::createBlocksFromCandidates(cv::Mat& result_map, int block_size)
{
   cv::Point top_left, bottom_right;
   for (const auto& candidate : FireColorCandidates) {
      top_left.x = std::max(0, candidate.x - block_size);
      top_left.y = std::max(0, candidate.y - block_size);
      bottom_right.x = std::min(result_map.cols, candidate.x + block_size);
      bottom_right.y = std::min(result_map.rows, candidate.y + block_size);
      result_map(cv::Rect(top_left, bottom_right)) = 1.0;
   }
}

void TrainedColorFireDetection::updateResultMapAndFireRegion(cv::Mat& fire_region)
{
   const int result_box_size = 10;
   cv::Mat result_map = cv::Mat::zeros( ProbabilityMap.size(), ProbabilityMap.type() );
   createBlocksFromCandidates( result_map, result_box_size );

   cv::addWeighted( result_map, ResultMapLearningRate, ProbabilityMap, 1.0 - ResultMapLearningRate, 0.0, ProbabilityMap );

   fire_region |= ProbabilityMap > (100.0 - FireSensitivity) / 100.0;
   const cv::Mat circle_elem = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size(5, 5) );
   cv::morphologyEx( fire_region, fire_region, cv::MORPH_CLOSE, circle_elem );
#ifdef SHOW_PROCESS
   imshow( "Final Result Map", ProbabilityMap );
   imshow( "Fire Region", fire_region );
#endif
}

void TrainedColorFireDetection::getFirePosition(std::vector<cv::Rect>& fires, const cv::Mat& fire_region) const
{
   std::vector<std::vector<cv::Point>> contours;
   cv::Mat contoured = fire_region.clone();
   cv::findContours( contoured, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE );

   fires.clear();
   if (contours.size() <= FireNumToFind) {
      for (const auto& contour : contours) {
         fires.emplace_back( cv::boundingRect( cv::Mat(contour) ) );
      }
   }
   else {
      std::vector<std::pair<cv::Rect, double>> probabilities(contours.size());
      for (uint i = 0; i < contours.size(); ++i) {
         probabilities[i].first = cv::boundingRect( cv::Mat(contours[i]) );
         cv::minMaxLoc( ProbabilityMap(probabilities[i].first), nullptr, &probabilities[i].second, nullptr, nullptr );
      }

      for (uint i = 0; i < FireNumToFind; ++i) {
         const int max_index = findIndexWithMaxOfSecondValues( probabilities );
         fires.emplace_back( probabilities[max_index].first );

         probabilities.erase( probabilities.begin() + max_index );
      }
   }
}

void TrainedColorFireDetection::detectFire(std::vector<cv::Rect>& fires, cv::Mat& fire_region, const cv::Mat& frame)
{
#ifdef SHOW_PROCESS
   ProcessResult = frame.clone();
#endif

   classifyColorAndMotion( frame );

   if (!FireColorCandidates.empty()) {
      updateResultMapAndFireRegion( fire_region );

      getFirePosition( fires, fire_region );
      cv::resize( fire_region, fire_region, frame.size() );
   }
}

void TrainedColorFireDetection::informOfSceneChanged()
{
   initialize();
}