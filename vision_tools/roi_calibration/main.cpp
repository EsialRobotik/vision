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

cv::Mat K, D;
cv::Mat fisheye_map1, fisheye_map2;

void generateFisheyeUndistordMap(cv::Mat &map1, cv::Mat &map2)
{

    cv::FileStorage file_kd("fisheye_KD", cv::FileStorage::READ);
    file_kd["K_mat"] >> K;
    file_kd["D_mat"] >> D;
    file_kd.release();

    cv::Size img_size(1920, 1080);
    cv::fisheye::initUndistortRectifyMap(K, D, cv::Mat::eye(3, 3, CV_32F), K, img_size, CV_16SC2, map1, map2);
}

int main ( int argc,char **argv ) 
{
   
    generateFisheyeUndistordMap( fisheye_map1, fisheye_map2);
    if( fisheye_map1.size().width == 0 || fisheye_map1.size().height == 0
        || fisheye_map2.size().width == 0 || fisheye_map2.size().height == 0)
    {
        cerr<<"Error retrieving map file for fisheye undistortion in fisheye_map"<<endl;
        return -1;
    }


    time_t timer_begin,timer_end;
    raspicam::RaspiCam_Still_Cv Camera;
    cv::Mat photo, photo_undistorted;
    
    Camera.set(cv::CAP_PROP_FRAME_WIDTH,  1920);
    Camera.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    if (!Camera.open()) 
    {
        cerr<<"Error opening the camera"<<endl;
        return -1;
    }

    libconfig::Config cfg;
    libconfig::Setting& root = cfg.getRoot();
    
   
    cout<<"Warm up... "<<endl; 
    sleep(3);    
    cout<<"Capturing " <<endl;

    Camera.grab();
    Camera.retrieve ( photo);
    cv::remap(photo, photo_undistorted, fisheye_map1, fisheye_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    

    cv::Rect2d cropZone = cv::selectROI("Select left zone", photo_undistorted);
    libconfig::Setting &left_roi = root.add("left_roi", libconfig::Setting::TypeGroup);
    left_roi.add("x", libconfig::Setting::TypeInt) = (int)cropZone.x ;
    left_roi.add("y", libconfig::Setting::TypeInt) = (int)cropZone.y ;
    left_roi.add("width", libconfig::Setting::TypeInt) = (int)cropZone.width ;
    left_roi.add("height", libconfig::Setting::TypeInt) = (int)cropZone.height ;

    cropZone = cv::selectROI("Select right zone", photo_undistorted);
    libconfig::Setting &right_roi = root.add("right_roi", libconfig::Setting::TypeGroup);
    right_roi.add("x", libconfig::Setting::TypeInt) = (int)cropZone.x ;
    right_roi.add("y", libconfig::Setting::TypeInt) = (int)cropZone.y ;
    right_roi.add("width", libconfig::Setting::TypeInt) = (int)cropZone.width ;
    right_roi.add("height", libconfig::Setting::TypeInt) = (int)cropZone.height ;

    cropZone = cv::selectROI("Select center zone", photo_undistorted);
    libconfig::Setting &center_roi = root.add("center_roi", libconfig::Setting::TypeGroup);
    center_roi.add("x", libconfig::Setting::TypeInt) = (int)cropZone.x ;
    center_roi.add("y", libconfig::Setting::TypeInt) = (int)cropZone.y ;
    center_roi.add("width", libconfig::Setting::TypeInt) = (int)cropZone.width ;
    center_roi.add("height", libconfig::Setting::TypeInt) = (int)cropZone.height ;

    cfg.writeFile("vision.cfg");
    cout << "configuration written to vision.cfg file" << endl;   
    
    Camera.release();
 return 0;
}
