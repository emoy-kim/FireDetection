#include "RChannelFireDetection.h"

RChannelFireDetection::RChannelFireDetection() :
   FrameCounter( 0 ), RChannelAnalysisPeriod( 14 ),
   MinPerturbingThreshold( 1.5 * static_cast<double>(RChannelAnalysisPeriod - 1) )
{
   initialize();
}

void RChannelFireDetection::initialize()
{
   FrameCounter = 0;
   FireCandidates.clear();
#ifdef SHOW_PROCESS
   destroyAllWindows();
#endif
}

#ifdef SHOW_PROCESS
void RChannelFireDetection::displayHistogram(const Mat& histogram, const Mat& fire_area, const Mat& fire_mask, const int& index) const
{
   const Size histogram_viewer(1024, 400);
   const int bin_size = histogram_viewer.width / histogram.rows;
   Mat histogram_graph = Mat::zeros( histogram_viewer, CV_8UC3 );
   for (int i = 1; i < histogram.rows; ++i) {
      line(
         histogram_graph,
         Point(bin_size * (i - 1), static_cast<int>(histogram_viewer.height * (1.0f - histogram.at<float>(i - 1)))),
         Point(bin_size * i, static_cast<int>(histogram_viewer.height * (1.0f - histogram.at<float>(i)))),
         GREEN_COLOR, 2
      );
   }
   const uint gradation = 200 / RChannelAnalysisPeriod;
   for (uint i = 0; i < RChannelInfos[index].MeanPerturbation.size(); ++i) {
      line(
         histogram_graph, 
         Point(bin_size * static_cast<int>(RChannelInfos[index].MeanPerturbation[i]), 0), 
         Point(bin_size * static_cast<int>(RChannelInfos[index].MeanPerturbation[i]), histogram_viewer.height), 
         BLUE_COLOR - Scalar(gradation * i, 0, 0), 2
      );
   }
   if (fire_area.cols <= histogram_viewer.width - 10 && fire_area.rows <= histogram_viewer.height - 10) {
      fire_area.copyTo( histogram_graph(Rect(10, 10, fire_area.cols, fire_area.rows)), fire_mask );
   }
   imshow( "Histogram Graph" + to_string( index ), histogram_graph );
}

void RChannelFireDetection::shutdownHistogram() const
{
   for (uint i = 0; i < RChannelInfos.size(); ++i) {
      destroyWindow( "Histogram Graph" + to_string( i ) );
   }
}
#endif

bool RChannelFireDetection::initializeFireCandidateInfos(const vector<Rect>& fires, const Mat& frame)
{
   if (fires.empty()) {
      return false;
   }
   
   RChannelInfos.clear();
   const Point2d to_analysis_frame(
      AnalysisFrameSize.width / static_cast<double>(frame.cols), 
      AnalysisFrameSize.height / static_cast<double>(frame.rows)
   );
   for (const auto& rect : fires) {
      const Rect region = transformFireBoundingBox( rect, to_analysis_frame );
      if (isRegionBigEnough( region )) {
         RChannelCandidate candidate;
         candidate.Region = region;
         RChannelInfos.emplace_back( candidate );
      }
   }
   return true;
}

Mat RChannelFireDetection::getNormalizedHistogram(const Mat& rchannel, const Mat& mask) const
{
   Mat histogram;
   const int hist_size = 256;
   const float range[2] = { 0.0f, 256.0f };
   const float* hist_range = range;
   calcHist( &rchannel, 1, nullptr, mask, histogram, 1, &hist_size, &hist_range );
   normalizeMap( histogram );
   return histogram;
}

double RChannelFireDetection::calculateWeightedMean(const Mat& histogram, const float& min_frequency) const
{
   double mean = 0.0;
   float sum_of_frequency_left_out = 0.0f;
   for (int i = 0; i < histogram.rows; ++i) {
      const auto frequency = histogram.at<float>(i);
      if (frequency > min_frequency) {
         mean += static_cast<double>(i) * frequency;
      }
      else sum_of_frequency_left_out += frequency;
   }
   const double sum_of_frequency = sum( histogram )[0] - sum_of_frequency_left_out;
   if (sum_of_frequency > 0.0) {
      mean /= sum_of_frequency;
   }
   return mean;
}

bool RChannelFireDetection::isPerturbingEnough(const RChannelCandidate& candidate, const double& min_intensity) const
{
   double amount_of_perturbing = 0.0;
   for (uint i = 1; i < candidate.MeanPerturbation.size(); ++i) {
      if (candidate.MeanPerturbation[i] < min_intensity) return false;
      amount_of_perturbing += fabs( candidate.MeanPerturbation[i] - candidate.MeanPerturbation[i - 1] );
   }
 
   double mean, standard_deviation;
   getMeanAndStandardDeviation( mean, standard_deviation, candidate.MeanPerturbation );
   const double standard_deviation_threshold = 
      sqrt( candidate.Region.width * candidate.Region.width + candidate.Region.height * candidate.Region.height ) * 0.4;
   
   return 
      amount_of_perturbing > MinPerturbingThreshold && 
      (mean >= 150.0 || standard_deviation > standard_deviation_threshold);
}

void RChannelFireDetection::removeNonPerturbedRegion()
{
   for (auto it = RChannelInfos.begin(); it != RChannelInfos.end();) {
      if (isPerturbingEnough( *it, 0.0 )) {
#ifdef SHOW_PROCESS
         rectangle( ProcessResult, it->Region, RED_COLOR, 2 );
#endif
         ++it;
      }
      else {
#ifdef SHOW_PROCESS
         rectangle( ProcessResult, it->Region, BLUE_COLOR, 2 );
#endif
         it = RChannelInfos.erase( it );
      }
   }
}

void RChannelFireDetection::classifyRChannelHistogram(const Mat& frame, const Mat& fire_region)
{
   vector<Mat> channels(3);
   split( frame, channels );
   Mat& r_channel = channels[2];

   for (uint i = 0; i < RChannelInfos.size(); ++i) {
      const Mat histogram = getNormalizedHistogram( r_channel(RChannelInfos[i].Region), fire_region(RChannelInfos[i].Region) );
      RChannelInfos[i].MeanPerturbation.emplace_back( calculateWeightedMean( histogram, 0.0f ) );
#ifdef SHOW_PROCESS
      displayHistogram( histogram, frame(RChannelInfos[i].Region), fire_region(RChannelInfos[i].Region), i );
#endif
   }

   if (FrameCounter == RChannelAnalysisPeriod - 1) {
#ifdef SHOW_PROCESS
      shutdownHistogram();
#endif
      removeNonPerturbedRegion();
      extractFromCandidates( FireCandidates, RChannelInfos );
      FrameCounter = 0;
   }
   else FrameCounter++;
#ifdef SHOW_PROCESS
   imshow( "Red-Channel Classification Result", ProcessResult );
#endif
}

void RChannelFireDetection::getFirePosition(vector<Rect>& fires, const Mat& frame)
{
   fires.clear();
   const Point2d to_frame(
      frame.cols / static_cast<double>(AnalysisFrameSize.width),
      frame.rows / static_cast<double>(AnalysisFrameSize.height)
   );
   for (const auto& candidate : FireCandidates) {
      fires.emplace_back( transformFireBoundingBox( candidate, to_frame ) );
   }
}

void RChannelFireDetection::detectFire(vector<Rect>& fires, const Mat& fire_region, const Mat& frame)
{
#ifdef SHOW_PROCESS
   ProcessResult = frame.clone();
#endif
   
   if (FrameCounter == 0) {
      const bool keep_going = initializeFireCandidateInfos( fires, frame );
      if (!keep_going) return;
   }

   classifyRChannelHistogram( frame, fire_region );

   getFirePosition( fires, frame );
}

void RChannelFireDetection::informOfSceneChanged()
{
   initialize();
}