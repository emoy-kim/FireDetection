#pragma once

#include <OpenCVLinker.h>

using namespace std;
using namespace cv;

#define RED_COLOR     (Scalar(0, 0, 255))
#define GREEN_COLOR   (Scalar(0, 255, 0))
#define BLUE_COLOR    (Scalar(255, 0, 0))
#define YELLOW_COLOR  (Scalar(0, 255, 255))
#define CYAN_COLOR    (Scalar(255, 255, 0))
#define MAGENTA_COLOR (Scalar(255, 0, 255))
#define BLACK_COLOR   (Scalar(0, 0, 0))
#define WHITE_COLOR   (Scalar(255, 255, 255))

//#define SHOW_PROCESS

extern uint FireNumToFind;
extern Mat ProbabilityMap;
extern Size AnalysisFrameSize;

struct FireCandidate { Rect Region; };

/*
inline int printText(const int& color_code, char const* text, ...)
{
   va_list arg_list;
   va_start( arg_list, text );

   SetConsoleTextAttribute( GetStdHandle(STD_OUTPUT_HANDLE ), color_code );
   const int result = _vfprintf_l( stdout, text, nullptr, arg_list );
   SetConsoleTextAttribute( GetStdHandle( STD_OUTPUT_HANDLE ), 7 );
   va_end( arg_list );
   return result;
}
*/

inline uint countThisValue(const uchar& val, const Mat& region)
{
   uint num = 0;
   for (int j = 0; j < region.rows; ++j) {
      auto const region_ptr = region.ptr<uchar>(j);
      for (int i = 0; i < region.cols; ++i) {
         if (region_ptr[i] == val) num++;
      }
   }
   return num;
}

inline double getEuclideanDistance(const Point2f& point1, const Point2f& point2)
{
   return norm( point1 - point2 );
}

inline double getEuclideanDistance(const vector<double>& point1, const vector<double>& point2)
{
   double distance = 0.0;
   for (uint i = 0; i < point1.size(); ++i) {
      const double difference = point1[i] - point2[i];
      distance += difference * difference;
   }
   return sqrt( distance );
}

inline double getCosineSimilarity(const vector<double>& point1, const vector<double>& point2)
{
   double dot_product = 0.0;
   double point1_squared_length = 0.0, point2_squared_length = 0.0;
   for (uint i = 0; i < point1.size(); ++i) {
      dot_product += point1[i] * point2[i];
      point1_squared_length += point1[i] * point1[i];
      point2_squared_length += point2[i] * point2[i];
   }
   const double product_of_lengths = sqrt( point1_squared_length * point2_squared_length );
   if (product_of_lengths < 1e-7) return 1e+20;
   return dot_product / product_of_lengths;
}

inline void getMeanAndStandardDeviation(double& mean, double& standard_deviation, const vector<double>& vector)
{
   double sum = 0.0, squared_sum = 0.0;
   if (vector.size() <= 1) {
      mean = standard_deviation = 0.0;
      return;
   }
   for (auto const& val : vector) {
      sum += val;
      squared_sum += val * val;
   }
   mean = sum / vector.size();
   standard_deviation = sqrt( (squared_sum - sum * mean) / (vector.size() - 1) );
}

inline double getNormalizedCrossCorrelation(const vector<double>& point1, const vector<double>& point2)
{
   if (point1.empty() || point2.empty()) return 0.0;

   double mean1, standard_deviation1;
   getMeanAndStandardDeviation( mean1, standard_deviation1, point1 );

   double mean2, standard_deviation2;
   getMeanAndStandardDeviation( mean2, standard_deviation2, point2 );

   double ncc = 0.0;
   if (standard_deviation1 > 1e-7 && standard_deviation2 > 1e-7) {
      const double scale_factor = 1.0 / (standard_deviation1 * standard_deviation2);
      for (uint i = 0; i < point1.size(); ++i) {
         ncc += (point1[i] - mean1) * (point2[i] - mean2) * scale_factor;
      }
   }
   return ncc / point1.size();
}

inline double getSquaredL2Norm(const Point2f& vector)
{
   return vector.x * vector.x + vector.y * vector.y;
}

inline double getSquaredL2Norm(const Vec2f& vector)
{
   return vector[0] * vector[0] + vector[1] * vector[1];
}

inline bool isRegionBigEnough(const Rect& region)
{
   return region.width > 20 && region.height > 20;
}

inline void normalizeMap(Mat& map)
{
   double max_val;
   minMaxLoc( map, nullptr, &max_val, nullptr, nullptr );
   if (max_val != 0.0) {
      map /= max_val;
   }
}

template<typename T>
void getUnionPointSet(vector<T>& union_set, const vector<T>& set)
{
   for (auto const& new_elem : set) {
      for (auto elem = union_set.begin(); elem != union_set.end(); ++elem) {
         if (new_elem.x == elem->x && new_elem.y == elem->y) {
            union_set.erase( elem );
            break;
         }
      }
   }
   union_set.insert( union_set.end(), set.begin(), set.end() );
}

template<typename T, typename U>
int findIndexWithMaxOfSecondValues(const vector<pair<T, U>>& pairs)
{
   int max_index = -1;
   double max_value = -1.0;
   for (uint i = 0; i < pairs.size(); ++i) {
      if (max_value < static_cast<double>(pairs[i].second)) {
         max_value = static_cast<double>(pairs[i].second);
         max_index = i;
      }
   }
   return max_index;
}

template<typename T, typename U>
void extractFromCandidates(vector<T>& extracted_fires, const vector<U>& candidates)
{
   extracted_fires.clear();
   if (candidates.size() <= FireNumToFind) {
      for (const auto& candidate : candidates) {
         extracted_fires.emplace_back( candidate.Region );
      }
   }
   else {
      vector<pair<Rect, int>> sizes(candidates.size());
      for (uint i = 0; i < candidates.size(); ++i) {
         sizes[i].first = candidates[i].Region;
         sizes[i].second = sizes[i].first.width * sizes[i].first.height;
      }

      for (uint i = 0; i < FireNumToFind; ++i) {
         const int max_index = findIndexWithMaxOfSecondValues( sizes );
         extracted_fires.emplace_back( sizes[max_index].first );

         sizes.erase( sizes.begin() + max_index );
      }
   }
}

inline void getFeatures(vector<Point2f>& points, const Mat& image, const Mat& mask)
{
   const int max_corners = 20;
   for (int block_size = 5; block_size <= 15; block_size += 10) {
      vector<Point2f> corners(max_corners);
      goodFeaturesToTrack( image, corners, max_corners, 0.01, 0.0, mask, block_size );
      getUnionPointSet( points, corners );
   }
   if (points.size() > max_corners) {
      points.erase( points.begin() + max_corners, points.end() );
   }
}

inline vector<uchar> getMatches(vector<Point2f>& target_points, const vector<Point2f>& query_points, const Mat& query, const Mat& target)
{   
   const int points_num = query_points.size();
   target_points.resize(points_num);
   vector<uchar> found_matches(points_num);
   vector<float> errors(points_num);
   const int window_size = 5;
   const TermCriteria term_criteria(TermCriteria::COUNT + TermCriteria::EPS, 50, 0.01);
   calcOpticalFlowPyrLK(
      query,
      target,
      query_points,
      target_points,
      found_matches, 
      errors,
      Size(window_size, window_size),
      3, term_criteria, 0, 0.0001
   );
   return found_matches;
}

inline vector<uchar> findMatchesUsingOpticalFlowLK(vector<Point2f>& query_points, vector<Point2f>& target_points, const Mat& query, const Mat& target, const Mat& mask = Mat())
{
   query_points.clear();
   target_points.clear();
   getFeatures( query_points, query, mask );
   
   vector<uchar> found_matches;
   if (!query_points.empty()) {
      found_matches = getMatches( target_points, query_points, query, target );
   }
   return found_matches;
}

inline vector<uchar> findMatches(vector<Point2f>& query_points, vector<Point2f>& target_points, const Mat& query, const Mat& target, const Mat& mask = Mat())
{
   vector<KeyPoint> query_keypoints, target_keypoints;
   Ptr<FeatureDetector> detector = FastFeatureDetector::create();
   detector->detect( query, query_keypoints );
   detector->detect( target, target_keypoints );
   
   Mat query_descriptor, target_descriptor;  
   Ptr<DescriptorExtractor> extractor = ORB::create();
   extractor->compute( query, query_keypoints, query_descriptor );
   extractor->compute( target, target_keypoints, target_descriptor );
   
   vector<vector<DMatch>> matches;
   if (query_descriptor.cols == target_descriptor.cols) {
      const Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
      matcher->knnMatch( query_descriptor, target_descriptor, matches, 20 );
   }

   query_points.clear();
   target_points.clear();
   for (const auto& match : matches) { 
      for (const auto& key : match) {
         const Point2f from = query_keypoints[key.queryIdx].pt;
         const Point2f to = target_keypoints[key.trainIdx].pt;

         const double distance = getEuclideanDistance( from, to );

         if (distance < 5.15 && abs(from.y - to.y) < 5) {
            query_points.push_back( from );
            target_points.push_back( to );
         }
      }
   }
   return vector<uchar>(query_points.size(), 1);
}