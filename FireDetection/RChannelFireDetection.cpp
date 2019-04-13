#include "RChannelFireDetection.h"

RChannelFireDetection::RChannelFireDetection() :
   FrameCounter( 0 ), RChannelAnalysisPeriod( 14 ),
   MinPerturbingThreshold( 2.0 * static_cast<double>(RChannelAnalysisPeriod - 1) )
{
   initialize();
}

void RChannelFireDetection::initialize()
{
   FrameCounter = 0;
   PrevDetectedFires.clear();
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
   for (uint i = 0; i < RChannelCandidates[index].MeanPerturbation.size(); ++i) {
      line(
         histogram_graph, 
         Point(bin_size * static_cast<int>(RChannelCandidates[index].MeanPerturbation[i]), 0), 
         Point(bin_size * static_cast<int>(RChannelCandidates[index].MeanPerturbation[i]), histogram_viewer.height), 
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
   for (uint i = 0; i < RChannelCandidates.size(); ++i) {
      destroyWindow( "Histogram Graph" + to_string( i ) );
   }
}
#endif

bool RChannelFireDetection::initializeFireCandidates(const vector<Rect>& fires)
{
   if (fires.empty()) {
      return false;
   }
   
   RChannelCandidates.clear();
   for (const auto& rect : fires) {
      if (isRegionBigEnough( rect )) {
         RChannelCandidates.emplace_back( rect );
      }
   }
   return true;
}

Mat RChannelFireDetection::getNormalizedHistogram(const Mat& r_channel, const Mat& mask) const
{
   Mat histogram;
   const int hist_size = 256;
   const float range[2] = { 0.0f, 256.0f };
   const float* hist_range = range;
   calcHist( &r_channel, 1, nullptr, mask, histogram, 1, &hist_size, &hist_range );
   normalizeMap( histogram );
   return histogram;
}

double RChannelFireDetection::calculateWeightedMean(const Mat& histogram, const float& min_frequency) const
{
   double mean = 0.0;
   double sum_of_frequency_left_out = 0.0f;
   for (int i = 0; i < histogram.rows; ++i) {
      const auto frequency = static_cast<double>(histogram.at<float>(i));
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
   for (auto it = RChannelCandidates.begin(); it != RChannelCandidates.end();) {
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
         it = RChannelCandidates.erase( it );
      }
   }
}

void RChannelFireDetection::classifyRChannelHistogram(vector<Rect>& fires, const Mat& frame, const Mat& fire_region)
{
   vector<Mat> channels(3);
   split( frame, channels );
   Mat& r_channel = channels[2];

   for (uint i = 0; i < RChannelCandidates.size(); ++i) {
      const Mat histogram = getNormalizedHistogram( r_channel(RChannelCandidates[i].Region), fire_region(RChannelCandidates[i].Region) );
      RChannelCandidates[i].MeanPerturbation.emplace_back( calculateWeightedMean( histogram, 0.0f ) );
#ifdef SHOW_PROCESS
      displayHistogram( histogram, frame(RChannelCandidates[i].Region), fire_region(RChannelCandidates[i].Region), i );
#endif
   }

   if (FrameCounter == RChannelAnalysisPeriod - 1) {
#ifdef SHOW_PROCESS
      shutdownHistogram();
#endif
      removeNonPerturbedRegion();
      extractFromCandidates( PrevDetectedFires, RChannelCandidates );
      FrameCounter = 0;
   }
   else FrameCounter++;

   fires = PrevDetectedFires;
#ifdef SHOW_PROCESS
   imshow( "Red-Channel Classification Result", ProcessResult );
#endif
}

void RChannelFireDetection::detectFire(vector<Rect>& fires, const Mat& fire_region, const Mat& frame)
{
#ifdef SHOW_PROCESS
   ProcessResult = frame.clone();
#endif
   
   if (FrameCounter == 0) {
      const bool keep_going = initializeFireCandidates( fires );
      if (!keep_going) return;
   }

   classifyRChannelHistogram( fires, frame, fire_region );
}

void RChannelFireDetection::informOfSceneChanged()
{
   initialize();
}