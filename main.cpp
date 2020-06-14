#include "ProjectPath.h"
#include "UnitedFireDetection.h"
#include <chrono>

void getTestset(std::vector<std::string>& testset)
{
   const std::string video_directory_path = std::string(CMAKE_SOURCE_DIR) + "/samples";
   testset = {
      video_directory_path + "/test1.mp4",
      video_directory_path + "/test2.mp4",
      video_directory_path + "/test3.mp4",
      video_directory_path + "/test4.mp4",
      video_directory_path + "/test5.mp4",
      video_directory_path + "/test6.mp4",
      video_directory_path + "/test7.mp4",
   };
}

#define PAUSE ' '
int displayFireDetected(cv::Mat& frame, const std::vector<cv::Rect>& fires, bool to_pause)
{
   for (const auto& fire : fires) {
      cv::rectangle( frame, fire, cv::Scalar(0, 0, 255), 2 );
   }
   imshow( "Fire", frame );
   
   const int key = cv::waitKey( 1 );
   if (to_pause) return PAUSE;
   return key;
}

#define ESC 27
#define TO_BE_CLOSED true
#define TO_BE_CONTINUED false
bool processKeyPressed(UnitedFireDetection& fire_detector, bool& to_pause, int key_pressed)
{
   switch (key_pressed) {
   case PAUSE: {
      int key;
      while ((key = cv::waitKey( 1 )) != PAUSE && key != 'f') {}
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

int getVideoAnalysisPeriod(cv::VideoCapture& cam, double analysis_fps)
{
   if (analysis_fps <= 0.0) return 1;
   const double fps = cam.get( cv::CAP_PROP_FPS );
   return fps <= analysis_fps ? 1 : static_cast<int>(round(fps / analysis_fps));
}

void moveNextAnalysisPeriod(cv::Mat& frame, cv::VideoCapture& cam, int& current_time, int analysis_period)
{
   for (int i = current_time; i < analysis_period; ++i) {
      cam >> frame;
   }
   current_time = analysis_period;
}

void playVideoAndDetectFire(cv::VideoCapture& cam, UnitedFireDetection& fire_detector, double analysis_fps)
{
   const int analysis_period = getVideoAnalysisPeriod( cam, analysis_fps );
   
   uint frame_num = 0;
   int key_pressed = -1;
   bool to_pause = false;
   int current_time = analysis_period;
   
   cv::Mat frame;
   std::vector<cv::Rect> fires;
   while (true) {
      cam >> frame;
      if (frame.empty()) break;

      if (to_pause) {
         frame_num += analysis_period - current_time;
         moveNextAnalysisPeriod( frame, cam, current_time, analysis_period );
      }

      if (current_time == analysis_period) {
         fires.clear();
         std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
         fire_detector.detectFire( fires, frame );
         const std::chrono::duration<double> detection_time = (std::chrono::system_clock::now() - start) * 1000.0;
         std::cout << "PROCESS TIME: " << detection_time.count() << " ms... \r";
         current_time = 0;
      }

      key_pressed = displayFireDetected( frame, fires, to_pause );
      if (processKeyPressed( fire_detector, to_pause, key_pressed ) == TO_BE_CLOSED) break;
      current_time++;
      frame_num++;
   }
}

void runTestSet(const std::vector<std::string>& testset, double analysis_fps)
{
   cv::VideoCapture cam;
   for (auto const &test_data : testset) {
      cam.open( test_data );
      if (!cam.isOpened()) continue;

      const int width = static_cast<int>(cam.get( cv::CAP_PROP_FRAME_WIDTH ));
      const int height = static_cast<int>(cam.get( cv::CAP_PROP_FRAME_HEIGHT ));
      std::cout << "*** TEST SET(" << width << " x " << height << "): " << test_data.c_str() << "***\n";

      UnitedFireDetection fire_detector( 5 );
      playVideoAndDetectFire( cam, fire_detector, analysis_fps );
      cam.release();
   }
}

int main()
{
   constexpr double analysis_fps = 7.0;
   std::vector<std::string> testset;
   getTestset( testset );
   runTestSet( testset, analysis_fps );
   return 0;
}