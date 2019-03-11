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
   PrevBlurredFrame = Mat::zeros( AnalysisFrameSize, CV_8UC3 );
   ProbabilityMap = Mat::zeros( AnalysisFrameSize, CV_64FC1 );
#ifdef SHOW_PROCESS
   destroyWindow( "Final Result Map" );
#endif
}

#ifdef SHOW_PROCESS
void TrainedColorFireDetection::showProcess(const Scalar& box_color)
{
   for (const auto& candidate : FireColorCandidates) {
      rectangle(
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

double TrainedColorFireDetection::calculateProbabilityBasedColor(const int& curr_idx, const Vec3b* upper_ptr, const Vec3b* curr_ptr, const Vec3b* lower_ptr)
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

void TrainedColorFireDetection::findMovingPixels(Mat& probability_map, const Mat& blurred_frame)
{
   for (int j = 1; j < blurred_frame.rows - 1; ++j) {
      const auto* blurred_upper_ptr = blurred_frame.ptr<Vec3b>(j - 1);
      const auto* blurred_ptr = blurred_frame.ptr<Vec3b>(j);
      const auto* blurred_lower_ptr = blurred_frame.ptr<Vec3b>(j + 1);

      const auto* prev_ptr = PrevBlurredFrame.ptr<Vec3b>(j);
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
   blur( probability_map, probability_map, Size(5, 5) );
}

bool TrainedColorFireDetection::isLocalMaximum(const int& curr_idx, const double* upper_ptr, const double* curr_ptr, const double* lower_ptr) const
{
   return
      curr_ptr[curr_idx - 1] < curr_ptr[curr_idx] && 
      curr_ptr[curr_idx + 1] < curr_ptr[curr_idx] &&
      upper_ptr[curr_idx] < curr_ptr[curr_idx] && 
      lower_ptr[curr_idx] < curr_ptr[curr_idx];
}

void TrainedColorFireDetection::findTopProbabilities(vector<pair<Point, double>>& local_maxima)
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

void TrainedColorFireDetection::findTopOfLocalMaxima(const Mat& probability_map)
{
   vector<pair<Point, double>> local_maxima;
   for (int j = 1; j < probability_map.rows - 1; ++j) {
      const auto* prob_upper_ptr = probability_map.ptr<double>(j - 1);
      const auto* prob_ptr = probability_map.ptr<double>(j);
      const auto* prob_lower_ptr = probability_map.ptr<double>(j + 1);
      for (int i = 1; i < probability_map.cols - 1; ++i) {
         if (ProbabilityThreshold < prob_ptr[i] && local_maxima.size() <= 4096 &&
             isLocalMaximum( i, prob_upper_ptr, prob_ptr, prob_lower_ptr )) {
            local_maxima.emplace_back( make_pair( Point(i, j), prob_ptr[i] ) );
         }
      }
   }

   FireColorCandidates.clear();
   findTopProbabilities( local_maxima );
}

void TrainedColorFireDetection::classifyColorAndMotion(const Mat& frame)
{
   Mat blurred_frame;
   GaussianBlur( frame, blurred_frame, Size(11, 11), 0.1 );

   Mat probability_map = Mat::zeros( blurred_frame.size(), CV_64FC1 );
   findMovingPixels( probability_map, blurred_frame );
   findTopOfLocalMaxima( probability_map );
   PrevBlurredFrame = blurred_frame.clone();
#ifdef SHOW_PROCESS
   imshow( "Probability Map", probability_map );
   showProcess( GREEN_COLOR );
#endif
}

void TrainedColorFireDetection::createBlocksFromCandidates(Mat& result_map, const int& block_size)
{
   Point top_left, bottom_right;
   for (const auto& candidate : FireColorCandidates) {
      top_left.x = max(0, candidate.x - block_size);
      top_left.y = max(0, candidate.y - block_size);
      bottom_right.x = min(result_map.cols, candidate.x + block_size);
      bottom_right.y = min(result_map.rows, candidate.y + block_size);
      result_map(Rect(top_left, bottom_right)) = 1.0;
   }
}

void TrainedColorFireDetection::updateResultMapAndFireRegion(Mat& fire_region)
{
   const int result_box_size = 10;
   Mat result_map = Mat::zeros( ProbabilityMap.size(), ProbabilityMap.type() );
   createBlocksFromCandidates( result_map, result_box_size );

   addWeighted( result_map, ResultMapLearningRate, ProbabilityMap, 1.0 - ResultMapLearningRate, 0.0, ProbabilityMap );

   fire_region |= ProbabilityMap > (100.0 - FireSensitivity) / 100.0;
   const Mat circle_elem = getStructuringElement( MORPH_ELLIPSE, Size(5, 5) );
   morphologyEx( fire_region, fire_region, MORPH_CLOSE, circle_elem );
#ifdef SHOW_PROCESS
   imshow( "Final Result Map", ProbabilityMap );
   imshow( "Fire Region", fire_region );
#endif
}

void TrainedColorFireDetection::getFirePosition(vector<Rect>& fires, const Mat& fire_region) const
{
   vector<vector<Point>> contours;
   Mat contoured = fire_region.clone();
   findContours( contoured, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );

   fires.clear();
   if (contours.size() <= FireNumToFind) {
      for (const auto& contour : contours) {
         fires.emplace_back( boundingRect( Mat(contour) ) );
      }
   }
   else {
      vector<pair<Rect, double>> probabilities(contours.size());
      for (uint i = 0; i < contours.size(); ++i) {
         probabilities[i].first = boundingRect( Mat(contours[i]) );
         minMaxLoc( ProbabilityMap(probabilities[i].first), nullptr, &probabilities[i].second, nullptr, nullptr );
      }

      for (uint i = 0; i < FireNumToFind; ++i) {
         const int max_index = findIndexWithMaxOfSecondValues( probabilities );
         fires.emplace_back( probabilities[max_index].first );

         probabilities.erase( probabilities.begin() + max_index );
      }
   }
}

void TrainedColorFireDetection::detectFire(vector<Rect>& fires, Mat& fire_region, const Mat& frame)
{
#ifdef SHOW_PROCESS
   ProcessResult = frame.clone();
#endif

   classifyColorAndMotion( frame );

   if (!FireColorCandidates.empty()) {
      updateResultMapAndFireRegion( fire_region );

      getFirePosition( fires, fire_region );
      resize( fire_region, fire_region, frame.size() );
   }
}

void TrainedColorFireDetection::informOfSceneChanged()
{
   initialize();
}