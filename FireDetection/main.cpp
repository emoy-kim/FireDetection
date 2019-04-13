#include "UnitedFireDetection.h"
#include <chrono>

using namespace chrono;

void getTestset(vector<string>& testset)
{
   testset = {
      "VideoSamples/test1.avi",
      "VideoSamples/test2.avi",
      "VideoSamples/test3.avi",
      "VideoSamples/test4.avi",
      "VideoSamples/test5.avi",
      "VideoSamples/test6.avi",
      "VideoSamples/test7.avi",
   };
}

#define PAUSE ' '
int displayFireDetected(Mat& frame, const vector<Rect>& fires, const bool& to_pause)
{
   for (const auto& fire : fires) {
      rectangle( frame, fire, Scalar(0, 0, 255), 2 );
   }
   imshow( "Fire", frame );
   
   const int key = waitKey( 1 );
   if (to_pause) return PAUSE;
   return key;
}

#define ESC 27
#define TO_BE_CLOSED true
#define TO_BE_CONTINUED false
bool processKeyPressed(UnitedFireDetection& fire_detector, bool& to_pause, const int& key_pressed)
{
   switch (key_pressed) {
   case PAUSE: {
      int key;
      while ((key = waitKey( 1 )) != PAUSE && key != 'f') {}
      to_pause = key == 'f';
   } break;
   case 'r':
      fire_detector.informOfSceneChanged();
      break;
   case ESC:
      return TO_BE_CLOSED;
   default:
      break;
   }
   return TO_BE_CONTINUED;
}

int getVideoAnalysisPeriod(VideoCapture& cam, const double& analysis_fps)
{
   if (analysis_fps <= 0.0) return 1;
   const double fps = cam.get( CV_CAP_PROP_FPS );
   return fps <= analysis_fps ? 1 : static_cast<int>(round(fps / analysis_fps));
}

void moveNextAnalysisPeriod(Mat& frame, VideoCapture& cam, int& current_time, const int& analysis_period)
{
   for (int i = current_time; i < analysis_period; ++i) {
      cam >> frame;
   }
   current_time = analysis_period;
}

void playVideoAndDetectFire(VideoCapture& cam, UnitedFireDetection& fire_detector, const double& analysis_fps)
{
   const int analysis_period = getVideoAnalysisPeriod( cam, analysis_fps );
   
   uint frame_num = 0;
   int key_pressed = -1;
   bool to_pause = false;
   int current_time = analysis_period;
   
   Mat frame;
   vector<Rect> fires;
   while (true) {
      cam >> frame;
      if (frame.empty()) break;

      if (to_pause) {
         frame_num += analysis_period - current_time;
         moveNextAnalysisPeriod( frame, cam, current_time, analysis_period );
      }

      if (current_time == analysis_period) {
         fires.clear();
         time_point<system_clock> start = system_clock::now();
         fire_detector.detectFire( fires, frame );
         const duration<double> detection_time = (system_clock::now() - start) * 1000.0;
         cout << "PROCESS TIME: " << detection_time.count() << " ms... \r";
         current_time = 0;
      }

      key_pressed = displayFireDetected( frame, fires, to_pause );
      if (processKeyPressed( fire_detector, to_pause, key_pressed ) == TO_BE_CLOSED) break;
      current_time++;
      frame_num++;
   }
}

void runTestSet(const vector<string>& testset, const double& analysis_fps)
{
   VideoCapture cam;
   for (auto const &test_data : testset) {
      cam.open( test_data );
      if (!cam.isOpened()) continue;

      const int width = static_cast<int>(cam.get( CV_CAP_PROP_FRAME_WIDTH ));
      const int height = static_cast<int>(cam.get( CV_CAP_PROP_FRAME_HEIGHT ));
      cout << "*** TEST SET(" << width << " x " << height << "): " << test_data.c_str() << "***" << endl;

      UnitedFireDetection fire_detector( 5 );
      playVideoAndDetectFire( cam, fire_detector, analysis_fps );
      cam.release();
   }
}

int main()
{
   const double analysis_fps = 7.0;
   vector<string> testset;
   getTestset( testset );
   runTestSet( testset, analysis_fps );
   return 0;
}