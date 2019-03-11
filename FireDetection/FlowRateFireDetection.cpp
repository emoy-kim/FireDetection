#include "FlowRateFireDetection.h"

FlowRateFireDetection::FlowRateFireDetection() :
   FrameCounter( 0 ), FlowRateAnalysisPeriod( 14 ), PCAOutlierXThreshold( 5.0f )
{
   initialize();
}

void FlowRateFireDetection::initialize()
{
   FrameCounter = 0;
   PrevFrame.release();
   PrevDetectedFires.clear();
#ifdef SHOW_PROCESS
   destroyAllWindows();
#endif
}

#ifdef SHOW_PROCESS
void FlowRateFireDetection::drawFlow(FlowRateCandidate& candidate, const vector<Point2f>& query_points, const vector<Point2f>& target_points, const vector<uchar>& found_matches) const
{
   for (uint i = 0; i < found_matches.size(); ++i) {
      if (found_matches[i]) {
         circle( candidate.FrameHistory.back(), query_points[i], 1, RED_COLOR, 2 );
         circle( candidate.FrameHistory.back(), target_points[i], 1, BLUE_COLOR, 2 );
         arrowedLine( candidate.FrameHistory.back(), query_points[i], target_points[i], RED_COLOR );
      }
   }
}

void FlowRateFireDetection::displayFlows(const FlowRateCandidate& candidate) const
{
   Mat matches_viewer;
   hconcat( candidate.FrameHistory, matches_viewer );
   imshow( "Flow Rate History" + to_string( candidate.CandidateIndex ), matches_viewer );
}
#endif

bool FlowRateFireDetection::initializeFireCandidates(const vector<Rect>& fires)
{
   if (fires.empty()) return false;

   FlowRateCandidates.clear();
   for (const auto& rect : fires) {
      if (isRegionBigEnough( rect )) {
         FlowRateCandidates.emplace_back( rect );
      }
   }
#ifdef SHOW_PROCESS
   for (uint i = 0; i < FlowRateCandidates.size(); ++i) {
      FlowRateCandidates[i].CandidateIndex = i;
   }
#endif
   return true;
}

void FlowRateFireDetection::updateMaxFlowLength(FlowRateCandidate& candidate, const vector<Point2f>& query_points, const vector<Point2f>& target_points, const vector<uchar>& found_matches) const
{
   for (uint i = 0; i < query_points.size(); ++i) {
      if (found_matches[i]) {
         const double distance = getEuclideanDistance( query_points[i], target_points[i] );
         if (distance > candidate.MaxFlowLength) {
            candidate.MaxFlowLength = distance;
         }
      }
   }
}

void FlowRateFireDetection::updateFlowDeltas(FlowRateCandidate& candidate, const vector<Point2f>& query_points, const vector<Point2f>& target_points, const vector<uchar>& found_matches) const
{
   for (uint i = 0; i < query_points.size(); ++i) {
      if (found_matches[i]) {
         const Point2f delta = query_points[i] - target_points[i];
         candidate.Deltas.push_back( Mat((Mat_<float>(1, 2) << delta.x, delta.y)) );
      }
   }
}

void FlowRateFireDetection::findMinMaxFlowPoint(FlowRateCandidate& candidate, const vector<Point2f>& query_points, const vector<Point2f>& target_points, const vector<uchar>& found_matches) const
{
   for (uint i = 0; i < query_points.size(); ++i) {
      if (found_matches[i]) {
         if (query_points[i].x < candidate.MinFlowPoint.x) candidate.MinFlowPoint.x = query_points[i].x;
         if (query_points[i].x > candidate.MaxFlowPoint.x) candidate.MaxFlowPoint.x = query_points[i].x;
         if (target_points[i].x < candidate.MinFlowPoint.x) candidate.MinFlowPoint.x = target_points[i].x;
         if (target_points[i].x > candidate.MaxFlowPoint.x) candidate.MaxFlowPoint.x = target_points[i].x;

         if (query_points[i].y < candidate.MinFlowPoint.y) candidate.MinFlowPoint.y = query_points[i].y;
         if (query_points[i].y > candidate.MaxFlowPoint.y) candidate.MaxFlowPoint.y = query_points[i].y;
         if (target_points[i].y < candidate.MinFlowPoint.y) candidate.MinFlowPoint.y = target_points[i].y;
         if (target_points[i].y > candidate.MaxFlowPoint.y) candidate.MaxFlowPoint.y = target_points[i].y;
      }
   }
}

void FlowRateFireDetection::calculateFlowRate(FlowRateCandidate& candidate, const Mat& frame, const Mat& fire_region) const
{
   Mat query, target;
   cvtColor( PrevFrame(candidate.Region), query, CV_BGR2GRAY );
   cvtColor( frame(candidate.Region), target, CV_BGR2GRAY );
   GaussianBlur( query, query, Size(5, 5), 7 );
   GaussianBlur( target, target, Size(5, 5), 7 );

   vector<Point2f> query_points, target_points;
   const Mat fire_mask = fire_region(candidate.Region);
   const vector<uchar> found_matches = findMatchesUsingOpticalFlowLK( query_points, target_points, query, target, fire_mask );
   updateMaxFlowLength( candidate, query_points, target_points, found_matches );
   updateFlowDeltas( candidate, query_points, target_points, found_matches );
   findMinMaxFlowPoint( candidate, query_points, target_points, found_matches );
#ifdef SHOW_PROCESS
   drawFlow( candidate, query_points, target_points, found_matches );
#endif
}

#ifdef SHOW_PROCESS
void FlowRateFireDetection::shutdownOutlierMaps() const
{
   for (uint i = 0; i < FlowRateCandidates.size(); ++i) {
      destroyWindow( "[FlowRate] Outlier Map" + to_string( i ) );
   }
}

void FlowRateFireDetection::drawMarkings(Mat& outlier_map, const Point2f& scale_factor) const
{
   for (int i = 1; ; ++i) {
      const auto  x_marking = static_cast<const uint>(round( i * scale_factor.x ));
      if (x_marking >= static_cast<uint>(outlier_map.cols)) break;
      line( outlier_map, Point(x_marking, 0), Point(x_marking, 10), BLACK_COLOR, 1 );
      putText( outlier_map, to_string( i ), Point(x_marking, 15), 2, 0.5, BLACK_COLOR );
   }
   for (int i = 1; ; ++i) {
      const auto y_marking = static_cast<const uint>(round( i * scale_factor.y ));
      if (y_marking >= static_cast<uint>(outlier_map.rows)) break;
      line( outlier_map, Point(0, y_marking), Point(10, y_marking), BLACK_COLOR, 1 );
      putText( outlier_map, to_string( i ), Point(15, y_marking), 2, 0.5, BLACK_COLOR );
   }
}

void FlowRateFireDetection::drawPCAOutlierMap(const PCA& pca, const vector<Point2f>& outlier_map_points, const int& candidate_index) const
{
   Mat outlier_map(400, 400, CV_8UC3, WHITE_COLOR);
   float max_x = -1.0f, max_y = -1.0f;
   for (const auto& point : outlier_map_points) {
      if (point.x > max_x) max_x = point.x;
      if (point.y > max_y) max_y = point.y;
   }
   if (max_x < 1e-7f || max_y < 1e-7f) return;

   const Point2f scale_factor(outlier_map.cols * 0.8f / max_x, outlier_map.rows * 0.8f / max_y);
   drawMarkings( outlier_map, scale_factor );

   for (const auto& point : outlier_map_points) {
      const Point scaled_point(
         static_cast<int>(round( point.x * scale_factor.x )), 
         static_cast<int>(round( point.y * scale_factor.y ))
      );
      if (scaled_point.x < 0 || scaled_point.y < 0 || 
         scaled_point.x >= outlier_map.cols || scaled_point.y >= outlier_map.rows)
         continue;
      circle( outlier_map, scaled_point, 5, BLUE_COLOR );
   }

   const auto x_threshold = static_cast<int>(round( PCAOutlierXThreshold * scale_factor.x ));
   line( outlier_map, Point(x_threshold, 0), Point(x_threshold, outlier_map.rows), RED_COLOR );

   const float pca_y_threshold = getPCAOutlierYThreshold( outlier_map_points );
   const auto y_threshold = static_cast<int>(round( pca_y_threshold * scale_factor.y ));
   line( outlier_map, Point(0, y_threshold), Point(outlier_map.cols, y_threshold), RED_COLOR );

   imshow( "[FlowRate] Outlier Map" + to_string( candidate_index ), outlier_map );
}
#endif

float FlowRateFireDetection::getPCAOutlierYThreshold(const vector<Point2f>& outlier_map_points) const
{
   float y_mean = 0.0f;
   for (const auto& point : outlier_map_points) {
      y_mean += point.y;
   }
   if (!outlier_map_points.empty()) {
      y_mean /= outlier_map_points.size();
   }
   return y_mean * 5.0f;
}

void FlowRateFireDetection::getPCAOutlierMapPoints(vector<Point2f>& outlier_map_points, PCA& pca, const FlowRateCandidate& candidate) const
{
   const int& data_num = candidate.Deltas.rows;
   outlier_map_points.resize( data_num );
   
   const Mat mean_vector = pca.mean.t();
   for (int i = 0; i < data_num; ++i) {
      const Mat direction_vector = candidate.Deltas.rowRange( i, i + 1 ).t() - mean_vector;
      const Mat dot_productions = pca.eigenvectors * direction_vector;
      const auto orthogonal_distance = 
         static_cast<const float>(norm( direction_vector - pca.eigenvectors.t() * dot_productions, NORM_L2 ));
      float score_distance = 0.0f;
      for (int d = 0; d < dot_productions.rows; ++d) {
         const auto projected_length_ptr = dot_productions.ptr<float>(d);
         const auto eigenvalue_ptr = pca.eigenvalues.ptr<float>(d);
         if (eigenvalue_ptr[0] > 1e-7f) {
            //score_distance += projected_length_ptr[0] * projected_length_ptr[0] / eigenvalue_ptr[0];
            score_distance += abs( projected_length_ptr[0] / eigenvalue_ptr[0] );
         }
      }
      //score_distance = sqrt( score_distance );
      outlier_map_points[i].x = score_distance;
      outlier_map_points[i].y = orthogonal_distance;
   }
#ifdef SHOW_PROCESS
   drawPCAOutlierMap( pca, outlier_map_points, candidate.CandidateIndex );
#endif
}

void FlowRateFireDetection::extractPCAInlierOnly(PCA& pca, Mat& inlier, const FlowRateCandidate& candidate) const
{
   vector<Point2f> outlier_map_points;
   getPCAOutlierMapPoints( outlier_map_points, pca, candidate );

   const float pca_y_threshold = getPCAOutlierYThreshold( outlier_map_points );
   for (uint i = 0; i < outlier_map_points.size(); ++i) {
      if (outlier_map_points[i].x < PCAOutlierXThreshold ||
          outlier_map_points[i].y < pca_y_threshold) {
         inlier.push_back( candidate.Deltas.rowRange( i, i + 1 ) );
      }
   }
}

void FlowRateFireDetection::getEigenvalues(vector<float>& eigenvalues, const FlowRateCandidate& candidate) const
{
   eigenvalues.resize( 2, 0.0f );
   if (candidate.Deltas.rows <= 2) return;
   PCA pca(candidate.Deltas, Mat(), CV_PCA_DATA_AS_ROW);
   
   Mat inlier;
   extractPCAInlierOnly( pca, inlier, candidate );
   pca(inlier, Mat(), CV_PCA_DATA_AS_ROW);
  
   for (int i = 0; i < pca.eigenvalues.rows; ++i) {
      eigenvalues[i] = pca.eigenvalues.at<float>(i, 0);
   }
}

bool FlowRateFireDetection::isTurbulentEnough(const FlowRateCandidate& candidate) const
{
   vector<float> eigenvalues;
   getEigenvalues( eigenvalues, candidate );
   
   double max_variance = 1e+7;
   if (candidate.MinFlowPoint.x < candidate.MaxFlowPoint.x) {
      const auto width = static_cast<const double>(candidate.Region.width);
      const auto height = static_cast<const double>(candidate.Region.height);
      max_variance = (width * width + height * height) * 0.25;
   }
   return eigenvalues[0] > max_variance * 0.01;
}

void FlowRateFireDetection::removeNonTurbulentRegion()
{
   for (auto it = FlowRateCandidates.begin(); it != FlowRateCandidates.end();) {
#ifdef SHOW_PROCESS 
      displayFlows( *it );
#endif
      if (isTurbulentEnough( *it )) ++it;
      else {
         it = FlowRateCandidates.erase( it );
      }
   }
}

void FlowRateFireDetection::classifyFlowRate(vector<Rect>& fires, const Mat& frame, const Mat& fire_region)
{
   for (auto& candidate : FlowRateCandidates) {
      Mat capture;
      frame(candidate.Region).copyTo( capture, fire_region(candidate.Region) );
      candidate.FrameHistory.emplace_back( capture );
      calculateFlowRate( candidate, frame, fire_region );
   }

   if (FrameCounter == FlowRateAnalysisPeriod - 1) {
#ifdef SHOW_PROCESS
      shutdownOutlierMaps();
#endif
      removeNonTurbulentRegion();
      extractFromCandidates( PrevDetectedFires, FlowRateCandidates );
      FrameCounter = 0;
   }
   else FrameCounter++;

   fires = PrevDetectedFires;
#ifdef SHOW_PROCESS
   imshow( "Flow Rate Classifier", ProcessResult );
#endif
}

void FlowRateFireDetection::detectFire(vector<Rect>& fires, const Mat& fire_region, const Mat& frame)
{
#ifdef SHOW_PROCESS
   ProcessResult = frame.clone();
#endif
   
   if (PrevFrame.empty()) {
      PrevFrame = frame;
      return;
   }

   if (FrameCounter == 0) {
      const bool keep_going = initializeFireCandidates( fires );
      if (!keep_going) return;
   }

   classifyFlowRate( fires, frame, fire_region );

   PrevFrame = frame.clone();
}

void FlowRateFireDetection::informOfSceneChanged()
{
   initialize();
}