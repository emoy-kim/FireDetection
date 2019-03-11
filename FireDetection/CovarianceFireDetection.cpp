#include "CovarianceFireDetection.h"

CovarianceFireDetection::CovarianceFireDetection() :
   FrameCounter( 0 ), CovarianceAnalysisPeriod( 14 ), MoveSensitivity( 2.5f )
{
   initialize();
}

void CovarianceFireDetection::initialize()
{
   FrameCounter = 0;
   OneStepPrevFrame.release();
   TwoStepPrevFrame.release();
   FireCandidates.clear();
#ifdef SHOW_PROCESS
   destroyAllWindows();
#endif
}

#ifdef SHOW_PROCESS
void CovarianceFireDetection::destroyExistingWindows(const string& prefix_name) const
{
   uint index = 0;
   string window_name = prefix_name + to_string( index );
   while (getWindowProperty( window_name, 0 ) >= 0.0) {
      destroyWindow( window_name );
      window_name = prefix_name + to_string( ++index );
   }
}

void CovarianceFireDetection::displayMatches(const CovarianceCandidate& candidate, const vector<Point2f>& query_points, const vector<Point2f>& target_points, const vector<uchar>& found_mathces) const
{
   const Mat fire_mask = FireRegionMask(candidate.Region);
   Mat query, target;
   candidate.FrameHistory[candidate.SimilarPairIndices.first].copyTo(query, fire_mask);
   candidate.FrameHistory[candidate.SimilarPairIndices.second].copyTo(target, fire_mask);
   query.resize( 256, 256 );
   target.resize( 256, 256 );
   Mat unite = query.clone();

   Scalar query_color, target_color;
   for (uint i = 0; i < query_points.size(); ++i) {
      if (found_mathces[i]) {
         if (found_mathces[i] == TOO_CLOSE_MATCHES) {
            query_color = WHITE_COLOR;//BLACK_COLOR;
            target_color = WHITE_COLOR;//BLACK_COLOR;
         }
         else {
            query_color = RED_COLOR;
            target_color = BLUE_COLOR;
         }
         circle( query, query_points[i], 1, query_color, 2 );
         circle( target, target_points[i], 1, target_color, 2 );
         circle( unite, query_points[i], 1, query_color, 2 );
         circle( unite, target_points[i], 1, target_color, 2 );
      }
   }

   Mat matches_viewer;
   hconcat( query, target, matches_viewer );
   hconcat( matches_viewer, unite, matches_viewer );
   imshow( "Matches" + to_string( candidate.CandidateIndex ), matches_viewer );
}

void CovarianceFireDetection::displayHistory(const CovarianceCandidate& candidate, const Scalar& box_color) const
{
   Mat history_viewer;
   hconcat( candidate.FrameHistory, history_viewer );
   
   rectangle( 
      history_viewer, 
      Point( candidate.FrameHistory[0].cols * candidate.SimilarPairIndices.first, 0 ), 
      Point( candidate.FrameHistory[0].cols * (candidate.SimilarPairIndices.first + 1), candidate.FrameHistory[0].rows ), 
      box_color, 2 
   );
   
   rectangle( 
      history_viewer, 
      Point( candidate.FrameHistory[0].cols * candidate.SimilarPairIndices.second, 0 ), 
      Point( candidate.FrameHistory[0].cols * (candidate.SimilarPairIndices.second + 1), candidate.FrameHistory[0].rows ), 
      box_color, 2 
   );

   cout << "===[CovarianceFireDetection]=======================================" << endl;
   cout << "[" << "candidate.CandidateIndex" << "] " << candidate.SimilarPairIndices.first << "-th image is the most similar to ";
   cout << candidate.SimilarPairIndices.second << "-th one as " << candidate.MaxFeatureSimilarity << "." << endl; 
   
   imshow( "Candidate History" + to_string( candidate.CandidateIndex ), history_viewer );
}
#endif

bool CovarianceFireDetection::initializeFireCandidateInfos(const vector<Rect>& fires, const Mat& frame, const Mat& fire_region)
{
   if (fires.empty()) return false;

   EigenvalueMap = Mat::zeros( ProbabilityMap.size(), CV_64FC1 );
   FireRegionMask = fire_region.clone();

   CovFeatureInfos.clear();
   const Point2d to_analysis_frame(
      AnalysisFrameSize.width / static_cast<double>(frame.cols), 
      AnalysisFrameSize.height / static_cast<double>(frame.rows)
   );
   for (auto const &rect : fires) {
      const Rect region = transformFireBoundingBox( rect, to_analysis_frame );
      if (isRegionBigEnough( region )) {
         CovarianceCandidate candidate;
         candidate.Region = region;
         candidate.MaxFeatureSimilarity = -1.0;
         candidate.MinFlowPoint = Point2f(1e+7f, 1e+7f);
         candidate.MaxFlowPoint = Point2f(-1.0f, -1.0f);
         CovFeatureInfos.emplace_back( candidate );
      }
   }
#ifdef SHOW_PROCESS
   for(uint i = 0; i < CovFeatureInfos.size(); ++i) {
      CovFeatureInfos[i].CandidateIndex = i;
   }
#endif
   return true;
}

void CovarianceFireDetection::accumulateSumAndSquareCombinational(vector<double>& sums, vector<double>& squares, const vector<double>& properties) const
{
   uint case_num = 0;
   for(uint i = 0; i < properties.size(); ++i) {
      sums[i] += properties[i];
      for(uint j = i; j < properties.size(); ++j) {
         squares[case_num] += properties[i] * properties[j];
         case_num++;
      }
   }
}

void CovarianceFireDetection::getCovariance(vector<double>& covariance, const vector<double>& sums, const vector<double>& squares, const int& num) const
{
   covariance.clear();
   if (num <= 1) return;

   uint case_num = 0;
   const double one_over_num = 1.0 / static_cast<double>(num);
   const double one_over_num_minus_one = 1.0 / static_cast<double>(num - 1);
   for(uint i = 0; i < sums.size(); ++i) {
      for(uint j = i; j < sums.size(); ++j) {
         const double square_of_mean = sums[i] * sums[j];
         const double variance = (squares[case_num] - square_of_mean * one_over_num) * one_over_num_minus_one;
         covariance.emplace_back( variance );
         case_num++;
      }
   }
}

void CovarianceFireDetection::getRGBCovariance(vector<double>& rgb_covariance, const Mat& fire_area, const Mat& fire_mask) const
{
   vector<double> sums(3, 0.0), square_of_mean(6, 0.0);
   uint valid_num = 0;
   for (int j = 1; j < fire_mask.rows - 1; ++j) {
      auto const mask_ptr = fire_mask.ptr<uchar>(j);
      auto const fire_ptr = fire_area.ptr<Vec3b>(j);
      for (int i = 1; i < fire_mask.cols - 1; ++i) {
         if (mask_ptr[i]) {
            const vector<double> rgb_feature = {
               static_cast<double>(fire_ptr[i][2]),
               static_cast<double>(fire_ptr[i][1]),
               static_cast<double>(fire_ptr[i][0])
            };
            accumulateSumAndSquareCombinational( sums, square_of_mean, rgb_feature );
            valid_num++;
         }
      }
   }
   getCovariance( rgb_covariance, sums, square_of_mean, valid_num );
}

void CovarianceFireDetection::getSpatioTemporalFeatures(vector<double>& features, const vector<const uchar*>&spatio_ptrs, const vector<const uchar*>& temporal_ptrs, int center_x) const
{
   features = {
      static_cast<double>(spatio_ptrs[1][center_x]), // I
      static_cast<double>(spatio_ptrs[1][center_x + 1]) - static_cast<double>(spatio_ptrs[1][center_x - 1]), // Ix
      static_cast<double>(spatio_ptrs[2][center_x] - static_cast<double>(spatio_ptrs[0][center_x])), // Iy
      static_cast<double>(spatio_ptrs[1][center_x + 1]) - 2.0 * static_cast<double>(spatio_ptrs[1][center_x]) + static_cast<double>(spatio_ptrs[1][center_x - 1]), // Ixx
      static_cast<double>(spatio_ptrs[2][center_x]) - 2.0 * static_cast<double>(spatio_ptrs[1][center_x]) + static_cast<double>(spatio_ptrs[0][center_x]), // Iyy
      //static_cast<double>(temporal_ptrs[2][center_x] - static_cast<double>(temporal_ptrs[0][center_x])), // It
      //static_cast<double>(temporal_ptrs[2][center_x] - 2.0 * static_cast<double>(temporal_ptrs[1][center_x]) + static_cast<double>(temporal_ptrs[0][center_x])) // Itt
   };
}

void CovarianceFireDetection::getSpatioTemporalCovariance(vector<double>& st_covariance, const vector<Mat>& fire_area_set, const Mat& fire_mask) const
{
   vector<double> sums(5, 0.0), square_of_mean(15, 0.0);
   uint valid_num = 0;
   for (int j = 1; j < fire_mask.rows - 1; ++j) {
      auto const mask_ptr = fire_mask.ptr<uchar>(j);
      const vector<const uchar*> spatio_ptrs = {
         fire_area_set[1].ptr<uchar>(j - 1),
         fire_area_set[1].ptr<uchar>(j), 
         fire_area_set[1].ptr<uchar>(j + 1)
      };
      const vector<const uchar*> temporal_ptrs = {
         fire_area_set[0].ptr<uchar>(j),
         fire_area_set[1].ptr<uchar>(j), 
         fire_area_set[2].ptr<uchar>(j)
      };
      for (int i = 1; i < fire_mask.cols - 1; ++i) {
         if (mask_ptr[i]) {
            vector<double> spatio_temporal_feature;
            getSpatioTemporalFeatures( spatio_temporal_feature, spatio_ptrs, temporal_ptrs, i );
            accumulateSumAndSquareCombinational( sums, square_of_mean, spatio_temporal_feature );
            valid_num++;
         }
      }
   }
   getCovariance( st_covariance, sums, square_of_mean, valid_num );
}

void CovarianceFireDetection::getCovarianceFeature(vector<double>& features, const Mat& resized_frame, const Rect& fire_box) const
{
   const Mat fire_mask = FireRegionMask(fire_box);

   vector<double> rgb_covariance;
   getRGBCovariance( rgb_covariance, OneStepPrevFrame(fire_box), fire_mask );

   vector<Mat> area_set(3);
   cvtColor( TwoStepPrevFrame(fire_box), area_set[0], CV_BGR2GRAY );
   cvtColor( OneStepPrevFrame(fire_box), area_set[1], CV_BGR2GRAY );
   cvtColor( resized_frame(fire_box), area_set[2], CV_BGR2GRAY );

   vector<double> st_covariance;
   getSpatioTemporalCovariance( st_covariance, area_set, fire_mask );

   features = rgb_covariance;
   features.insert( features.end(), st_covariance.begin(), st_covariance.end() );
}

void CovarianceFireDetection::updateMaxSimilarityAndIndex(CovarianceCandidate& candidate, const vector<double>& features) const
{
   int max_index = -1;
   double max_similarity = candidate.MaxFeatureSimilarity;
   for (uint i = 0; i < candidate.FeatureHistory.size(); ++i) {
      const double similarity = getNormalizedCrossCorrelation( features, candidate.FeatureHistory[i] );
      if (similarity > max_similarity) {
         max_similarity = similarity;
         max_index = i;
      }
   }
   if (max_index >= 0) {
      candidate.MaxFeatureSimilarity = max_similarity;
      candidate.SimilarPairIndices = make_pair( max_index, static_cast<int>(candidate.FeatureHistory.size()) );
   }
}

void CovarianceFireDetection::findMinMaxFlowPoint(CovarianceCandidate& candidate, const vector<Point2f>& query_points, const vector<Point2f>& target_points, const vector<uchar>& found_mathces) const
{
   for (uint i = 0; i < query_points.size(); ++i) {
      if (found_mathces[i]) {
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


bool CovarianceFireDetection::getDeltasFromSparseOpticalFlowMatches(CovarianceCandidate& candidate, const Mat& query, const Mat& target) const
{
   vector<Point2f> query_points, history_points;
   Mat fire_mask = FireRegionMask(candidate.Region);
   fire_mask.resize( 256, 256 );
   //vector<uchar> found_matches = findMatchesUsingOpticalFlowLK( query_points, history_points, query, target, fire_mask );
   vector<uchar> found_matches = findMatches( query_points, history_points, query, target, fire_mask );

   if (query_points.empty()) return false;

   for (uint i = 0; i < query_points.size(); ++i) {
      if (found_matches[i]) {
         const auto x = static_cast<int>(round( history_points[i].x ));
         const auto y = static_cast<int>(round( history_points[i].y ));
         if (x < 0 || x >= query.cols || y < 0 || y >= query.rows || fire_mask.at<uchar>(y, x) == 0)
            continue;

         const Point2f delta = query_points[i] - history_points[i];
         auto const square_distance = static_cast<const float>(getSquaredL2Norm( delta ));
         if (square_distance < MoveSensitivity) {
            found_matches[i] = TOO_CLOSE_MATCHES;
            continue;
         }
         candidate.Deltas.push_back( Mat((Mat_<float>(1, 2) << delta.x, delta.y)) );
      }
   }

   findMinMaxFlowPoint( candidate, query_points, history_points, found_matches );
#ifdef SHOW_PROCESS
   displayMatches( candidate, query_points, history_points, found_matches );
#endif
   return true;
}

#ifdef SHOW_PROCESS
void CovarianceFireDetection::shutdownOutlierMaps() const
{
   for (uint i = 0; i < CovFeatureInfos.size(); ++i) {
      destroyWindow( "COV Outlier Map" + to_string( i ) );
   }
}

void CovarianceFireDetection::drawMarkings(Mat& outlier_map, const Point2f& scale_factor) const
{
   for (int i = 1; ; ++i) {
      auto const x_marking = static_cast<const uint>(round(i * scale_factor.x));
      if (x_marking >= static_cast<uint>(outlier_map.cols)) break;
      line( outlier_map, Point(x_marking, 0), Point(x_marking, 10), BLACK_COLOR, 1 );
      putText( outlier_map, to_string(i), Point(x_marking, 15), 2, 0.5, BLACK_COLOR );
   }
   for (int i = 1; ; ++i) {
      auto const y_marking = static_cast<const uint>(round(i * scale_factor.y));
      if (y_marking >= static_cast<uint>(outlier_map.rows)) break;
      line( outlier_map, Point(0, y_marking), Point(10, y_marking), BLACK_COLOR, 1 );
      putText( outlier_map, to_string(i), Point(15, y_marking), 2, 0.5, BLACK_COLOR );
   }
}

void CovarianceFireDetection::drawPCAOutlierMap(const PCA& pca, const vector<Point2f>& outlier_map_points, const int& candidate_index) const
{
   Mat outlier_map(400, 400, CV_8UC3, WHITE_COLOR);
   float max_x = -1.0f, max_y = -1.0f;
   for (auto const &point : outlier_map_points) {
      if (point.x > max_x) max_x = point.x;
      if (point.y > max_y) max_y = point.y;
   }
   if (max_x < 1e-7f || max_y < 1e-7f) return;

   const Point2f scale_factor(outlier_map.cols * 0.8f / max_x, outlier_map.rows * 0.8f / max_y);
   drawMarkings( outlier_map, scale_factor );

   for (auto const &point : outlier_map_points) {
      const Point scaled_point(
         static_cast<int>(round(point.x * scale_factor.x)), 
         static_cast<int>(round(point.y * scale_factor.y))
      );
      if (scaled_point.x < 0 || scaled_point.y < 0 || 
         scaled_point.x >= outlier_map.cols || scaled_point.y >= outlier_map.rows)
         continue;
      circle( outlier_map, scaled_point, 5, BLUE_COLOR );
   }

   const auto x_threshold = static_cast<int>(round(2.744f * scale_factor.x));
   line( outlier_map, Point(x_threshold, 0), Point(x_threshold, outlier_map.rows), RED_COLOR );
   imshow( "COV Outlier Map" + to_string( candidate_index ), outlier_map );
}
#endif

float CovarianceFireDetection::getPCAOutlierYThreshold(const vector<Point2f>& outlier_map_points) const
{
   float y_mean = 0.0f;
   for (auto const &point : outlier_map_points) {
      y_mean += point.y;
   }
   if (!outlier_map_points.empty()) {
      y_mean /= outlier_map_points.size();
   }
   return y_mean * 6.0f;
}

void CovarianceFireDetection::getPCAOutlierMapPoints(vector<Point2f>& outlier_map_points, PCA& pca, const CovarianceCandidate& candidate) const
{
   const int& data_num = candidate.Deltas.rows;
   outlier_map_points.resize( data_num );

   const Mat mean_col_major = pca.mean.t();
   for (int i = 0; i < data_num; ++i) {
      const Mat direction_vector = candidate.Deltas.rowRange( i, i + 1 ).t() - mean_col_major;
      const Mat dot_productions = pca.eigenvectors * direction_vector;
      auto const orthogonal_distance = 
         static_cast<const float>(norm( direction_vector - pca.eigenvectors.t() * dot_productions, NORM_L2 ));
      float score_distance = 0.0f;
      for (int d = 0; d < dot_productions.rows; ++d) {
         auto const projected_length_ptr = dot_productions.ptr<float>(d);
         auto const eigenvalue_ptr = pca.eigenvalues.ptr<float>(d);
         if (eigenvalue_ptr[0] > 1e-7f) {
            score_distance += projected_length_ptr[0] * projected_length_ptr[0] / eigenvalue_ptr[0];
         }
      }
      score_distance = sqrt( score_distance );
      outlier_map_points[i].x = score_distance;
      outlier_map_points[i].y = orthogonal_distance;
   }
#ifdef SHOW_PROCESS
   drawPCAOutlierMap( pca, outlier_map_points, candidate.CandidateIndex );
#endif
}

Mat CovarianceFireDetection::removePCAOutlier(PCA& pca, const CovarianceCandidate& candidate) const
{
   vector<Point2f> outlier_map_points;
   getPCAOutlierMapPoints( outlier_map_points, pca, candidate );

   Mat inlier;
   const float pca_y_threshold = getPCAOutlierYThreshold( outlier_map_points );
   for (uint i = 0; i < outlier_map_points.size(); ++i) {
      if (outlier_map_points[i].x < 2.0f ||
          outlier_map_points[i].y < pca_y_threshold) {
         inlier.push_back( candidate.Deltas.rowRange( i, i + 1 ) );
      }
   }
   return inlier;
}

void CovarianceFireDetection::getEigenvaluesOfCovariance(vector<float>& eigenvalues, const CovarianceCandidate& candidate) const
{
   eigenvalues.resize(2, 0.0f);
   if (candidate.Deltas.rows < 2) return;
   PCA pca(candidate.Deltas, Mat(), CV_PCA_DATA_AS_ROW);
   
   //const Mat inlier = removePCAOutlier( pca, candidate );
   //pca(inlier, Mat(), CV_PCA_DATA_AS_ROW);

   for (int i = 0; i < pca.eigenvalues.rows; ++i) {
      eigenvalues[i] = pca.eigenvalues.at<float>(i, 0);
   }
}

bool CovarianceFireDetection::areEigenvaluesSmallAndSimilar(vector<float>& eigenvalues, const float& threshold) const
{
   bool is_similar = false;
   if (eigenvalues[0] < 55.0f) {
      is_similar = eigenvalues[0] < threshold || eigenvalues[0] < eigenvalues[1] * 2.0f;
   }
   return is_similar;
}

float CovarianceFireDetection::getAdaptiveEigenValueThreshold(const CovarianceCandidate& candidate) const
{
   float max_variance = 1e+7f;
   if (candidate.MinFlowPoint.x <= candidate.MaxFlowPoint.x) {
      const float width = candidate.MaxFlowPoint.x - candidate.MinFlowPoint.x;
      const float height = candidate.MaxFlowPoint.y - candidate.MinFlowPoint.y;
      max_variance = (width * width + height * height) * 0.25f;
   }

   float threshold;
   if (candidate.MaxFeatureSimilarity < 0.95) {
      threshold = static_cast<float>(max_variance * candidate.MaxFeatureSimilarity * 0.1);
   }
   else {
      threshold = static_cast<float>(max_variance * candidate.MaxFeatureSimilarity * 0.2);
   }
   return threshold;
}

bool CovarianceFireDetection::isMoving(const Mat& query, const Mat& target, const Mat& mask) const
{
   if (query.cols == 0 || query.rows == 0) return false;
   
   const int valid_num = countNonZero(mask);
   if (valid_num == 0) return false;

   const double psnr = PSNR( query, target );
   const bool is_moving = psnr <= 33.0;
   printf("COV Moving [PSNR]: %f\n", psnr);

   //const double norm_difference = norm( query, target, NORM_L2, mask ) / sqrt( valid_num );
   //const bool is_moving = psnr <= 33.0 && norm_difference >= 5.0;
   //printf("COV Moving [NORM]: %f\n", norm_difference);
   return is_moving;
}

bool CovarianceFireDetection::isFeatureRepeated(CovarianceCandidate& candidate)
{
   Mat query, target;
   cvtColor( candidate.FrameHistory[candidate.SimilarPairIndices.first], query, CV_BGR2GRAY );
   cvtColor( candidate.FrameHistory[candidate.SimilarPairIndices.second], target, CV_BGR2GRAY );

   bool repeated = true;
   if (!isMoving( query, target, FireRegionMask(candidate.Region) )) return repeated;

   query.resize( 256, 256 );
   target.resize( 256, 256 );
   const bool matches_exist = getDeltasFromSparseOpticalFlowMatches( candidate, query, target );

   if (matches_exist) {
      vector<float> eigenvalues;
      getEigenvaluesOfCovariance( eigenvalues, candidate );
      const float threshold = getAdaptiveEigenValueThreshold( candidate );
      repeated = areEigenvaluesSmallAndSimilar( eigenvalues, threshold );
      cout << "===[CovarianceFireDetection]=======================================" << endl;
      cout << "Eigenvalues: " << eigenvalues[0] << ", " << eigenvalues[1] << "(#candidates: " << candidate.Deltas.rows << ")" << endl;
      cout << "Similarity: " << candidate.MaxFeatureSimilarity << "(threshold: " << threshold << ")" << endl;
      EigenvalueMap(candidate.Region).setTo( static_cast<double>(eigenvalues[0]), FireRegionMask(candidate.Region) );
   }

#ifdef SHOW_PROCESS
   rectangle( ProcessResult, candidate.Region, repeated ? BLUE_COLOR : RED_COLOR, 2 );
   if (!candidate.FeatureHistory.empty()) 
      displayHistory( candidate, repeated ? BLUE_COLOR : RED_COLOR );
#endif
   return repeated;
}

void CovarianceFireDetection::removeRepeatedRegion()
{
#ifdef SHOW_PROCESS
   destroyExistingWindows( "Candidate History" );
#endif
   Mat result_map = Mat::zeros( ProbabilityMap.size(), CV_64FC1 );
   for (auto it = CovFeatureInfos.begin(); it != CovFeatureInfos.end();) {
      if (isFeatureRepeated( *it )) {
         it = CovFeatureInfos.erase( it );
      }
      else {
         result_map(it->Region) = 1.0;
         ++it;
      }
   }
   normalizeMap( EigenvalueMap );
   EigenvalueMap *= 0.5;
   ProbabilityMap = result_map.mul( EigenvalueMap ) + ProbabilityMap.mul( 1.0 - EigenvalueMap );
}

void CovarianceFireDetection::classifyCovariance(const Mat& resized_frame)
{
   for (auto &candidate : CovFeatureInfos) {
      vector<double> covariance_features;
      getCovarianceFeature( covariance_features, resized_frame, candidate.Region );
      updateMaxSimilarityAndIndex( candidate, covariance_features );
      candidate.FeatureHistory.emplace_back( covariance_features );
      candidate.FrameHistory.emplace_back( OneStepPrevFrame(candidate.Region) );
   }

   if (FrameCounter == CovarianceAnalysisPeriod - 1) {
#ifdef SHOW_PROCESS
      shutdownOutlierMaps();
#endif
      removeRepeatedRegion();
      extractFromCandidates( FireCandidates, CovFeatureInfos );
      FrameCounter = 0;
   }
   else FrameCounter++;
#ifdef SHOW_PROCESS
   imshow( "Covariance Classifier", ProcessResult );
#endif
}

void CovarianceFireDetection::getFirePosition(vector<Rect>& fires, const Mat& frame)
{
   fires.clear();
   const Point2d to_frame(
      frame.cols / static_cast<double>(AnalysisFrameSize.width),
      frame.rows / static_cast<double>(AnalysisFrameSize.height)
   );
   for (auto const &candidate : FireCandidates) {
      fires.emplace_back( transformFireBoundingBox( candidate, to_frame ) );
   }
}

void CovarianceFireDetection::detectFire(vector<Rect>& fires, const Mat& fire_region, const Mat& frame)
{
#ifdef SHOW_PROCESS
   ProcessResult = frame.clone();
#endif

   if (TwoStepPrevFrame.empty()) { TwoStepPrevFrame = frame.clone(); return; }
   if (OneStepPrevFrame.empty()) { OneStepPrevFrame = frame.clone(); return; }

   if (FrameCounter == 0) {
      const bool keep_going = initializeFireCandidateInfos( fires, frame, fire_region );
      if (!keep_going) return;
   }
   classifyCovariance( frame );

   getFirePosition( fires, frame );
   
   TwoStepPrevFrame = OneStepPrevFrame.clone();
   OneStepPrevFrame = frame.clone();
}

void CovarianceFireDetection::informOfSceneChanged()
{
   initialize();
}

void CovarianceFireDetection::setFireRegionNumToFind(const int& fire_num) const
{
   FireNumToFind = fire_num;
}