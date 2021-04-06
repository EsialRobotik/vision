#include <ctime>
#include <iostream>
#include <raspicam/raspicam_cv.h>
#include <raspicam/raspicam_still_cv.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <unistd.h>
#include <cstdint>
#include <cmath> 
#include <assert.h>
#include <libconfig.h++>

using namespace std; 
#define WINDOWS_NAME "color_threshold"

static cv::Mat* photo;
static int Hue_min = 165;
static int Hue_max = 10;
static int Saturation_min = 100;
static int Saturation_max = 255;
static int Value_min = 70;
static int Value_max = 255;


static void on_trackbar(int, void*)
{
    if( Saturation_min > Saturation_max)
        Saturation_min = Saturation_max;
    if( Value_min > Value_max)
        Value_min = Value_max;
    
    cv::setTrackbarMax("Hue_max", WINDOWS_NAME, 50);
    cv::setTrackbarMin("Hue_min", WINDOWS_NAME, 130);
    cv::setTrackbarMax("Saturation_min", WINDOWS_NAME, Saturation_max);
    cv::setTrackbarMax("Value_min", WINDOWS_NAME, Value_max);

   	cv::Mat photo_hsv;
	cv::Mat hsv_mask, hsv_mask2;
	cv::Mat photo_masked;
    cv::cvtColor( *photo, photo_hsv, cv::COLOR_BGR2HSV);
    cv::inRange( photo_hsv, cv::Scalar(Hue_min, Saturation_min, Value_min), cv::Scalar(180, Saturation_max, Value_max), hsv_mask);
    cv::inRange( photo_hsv, cv::Scalar(0, Saturation_min, Value_min), cv::Scalar(Hue_max, Saturation_max, Value_max), hsv_mask2);
    hsv_mask |= hsv_mask2;


    cv::bitwise_and(*photo,*photo, photo_masked, hsv_mask);
    imshow(WINDOWS_NAME, photo_masked);
}


void start_red_calib(cv::Mat* captured_photo )
{
	photo = captured_photo;

	cv::namedWindow(WINDOWS_NAME, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Hue_min", WINDOWS_NAME, &Hue_min, 180, on_trackbar);
    cv::createTrackbar("Hue_max", WINDOWS_NAME, &Hue_max, 180, on_trackbar);
    cv::createTrackbar("Saturation_min", WINDOWS_NAME, &Saturation_min, 255, on_trackbar);
    cv::createTrackbar("Saturation_max", WINDOWS_NAME, &Saturation_max, 255, on_trackbar);
    cv::createTrackbar("Value_min", WINDOWS_NAME, &Value_min, 255, on_trackbar);
    cv::createTrackbar("Value_max", WINDOWS_NAME, &Value_max, 255, on_trackbar);

    on_trackbar(0, nullptr);
    cv::waitKey(0);

    libconfig::Config cfg;
    libconfig::Setting& root = cfg.getRoot();
    libconfig::Setting &red_config = root.add("red", libconfig::Setting::TypeGroup);

    red_config.add("hue_min", libconfig::Setting::TypeInt) = Hue_min;
    red_config.add("hue_max", libconfig::Setting::TypeInt) = Hue_max;
    red_config.add("saturation_min", libconfig::Setting::TypeInt) = Saturation_min;
    red_config.add("saturation_max", libconfig::Setting::TypeInt) = Saturation_max;
    red_config.add("value_min", libconfig::Setting::TypeInt) = Value_min;
    red_config.add("value_max", libconfig::Setting::TypeInt) = Value_max;
    cfg.writeFile("red.cfg");
    cout << " last red configuration writen to red.cfg file" << endl;
}