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
#include "predefined_colors.hpp"
#include "dock_zone_detection.hpp"
#include "center_zone_detection.hpp"

using namespace std; 

static cv::Mat K, D;
static cv::Mat fisheye_map1, fisheye_map2;



const int32_t aruco_table_pos_x = 1500;
const int32_t aruco_table_pos_y = 1250;


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

    libconfig::Config cfg;
    cfg.readFile("vision.cfg");
    const libconfig::Setting& root = cfg.getRoot();

    int x, y, width,height;
    root["left_roi"].lookupValue("x", x);
    root["left_roi"].lookupValue("y", y);
    root["left_roi"].lookupValue("width", width);
    root["left_roi"].lookupValue("height", height);
    cv::Rect2d cropZoneLeft(x, y, width,height);
    
    root["right_roi"].lookupValue("x", x);
    root["right_roi"].lookupValue("y", y);
    root["right_roi"].lookupValue("width", width);
    root["right_roi"].lookupValue("height", height);
    cv::Rect2d cropZoneRight(x, y, width,height);

    root["center_roi"].lookupValue("x", x);
    root["center_roi"].lookupValue("y", y);
    root["center_roi"].lookupValue("width", width);
    root["center_roi"].lookupValue("height", height);
    cv::Rect2d cropCenterZone(x, y, width,height);

    int hue_min, hue_max, saturation_min, saturation_max, value_min, value_max;
    root["green"].lookupValue("hue_min", hue_min);
    root["green"].lookupValue("hue_max", hue_max);
    root["green"].lookupValue("saturation_min", saturation_min);
    root["green"].lookupValue("saturation_max", saturation_max);
    root["green"].lookupValue("value_min", value_min);
    root["green"].lookupValue("value_max", value_max);
    cv::Scalar green_hsv_low_threshold(hue_min, saturation_min, value_min);
    cv::Scalar green_hsv_high_threshold(hue_max, saturation_max, value_max);
    
    /**
    * NB: Hue value range is [0;180] and is wrapped.
    *   Red colors are located near 0 ( ie: also 180).
    *  So in the case of red color, the applied threshold is :
    *    [minVal ; 180] or [0 ; maxVal]
    */
    root["red"].lookupValue("hue_min", hue_min);
    root["red"].lookupValue("hue_max", hue_max);
    root["red"].lookupValue("saturation_min", saturation_min);
    root["red"].lookupValue("saturation_max", saturation_max);
    root["red"].lookupValue("value_min", value_min);
    root["red"].lookupValue("value_max", value_max);
    cv::Scalar red_hsv_low_threshold(hue_min, saturation_min, value_min);
    cv::Scalar red_hsv_max_range_threshold(180, saturation_max, value_max);
    cv::Scalar red_hsv_zero_threshold(0, saturation_min, value_min);
    cv::Scalar red_hsv_high_threshold(hue_max, saturation_max, value_max);


    

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
   
    cout<<"Warm up... "<<endl; 
    sleep(3);    
    cout<<"Capturing " <<endl;
    

    while(1)
    {
        clock_t start = clock();
        Camera.grab();
        Camera.retrieve ( photo);
        cout<< "capture duration = " <<  float(clock()-start)/CLOCKS_PER_SEC <<endl;
       


        start = clock();
        cv::remap(photo, photo_undistorted, fisheye_map1, fisheye_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        cout<< "fisheye compensation duration = " <<  float(clock()-start)/CLOCKS_PER_SEC <<endl;
        start = clock(); 


       
        vector<int> markerIds;
        vector<vector<cv::Point2f>> markerCorners, rejectedCandidates;
        cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
        cv::aruco::detectMarkers(photo_undistorted, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

        cout << "markerIds.size()" << markerIds.size() << endl;

        // if at least one marker detected
        if (markerIds.size() > 0)
        {
            cv::aruco::drawDetectedMarkers(photo_undistorted, markerCorners, markerIds);
         
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, K, D, rvecs, tvecs);   

            for(int i=0; i<markerIds.size(); i++)
            {
                cv::aruco::drawAxis(photo_undistorted, K, D, rvecs[i], tvecs[i], 0.1);
                cout<< "rvecs" <<  rvecs[i][0] << " " << rvecs[i][1] << " " << rvecs[i][2] <<endl;
                cout<< "tvecs" <<  tvecs[i][0] << " " << tvecs[i][1] << " " << tvecs[i][2] <<endl;

            }
        }
        cout<< "aruco detect duration = " <<  float(clock()-start)/CLOCKS_PER_SEC <<endl;
    

        start = clock(); 

        // Extract right/left zone 
        cv::Mat rightZone = photo_undistorted(cropZoneRight);
        cv::Mat leftZone = photo_undistorted(cropZoneLeft);

        // extract center zone
        cv::Mat centerZone = photo_undistorted(cropCenterZone);
        
        // Transform to HSV
        cv::Mat rightZone_hsv;
        cv::Mat leftZone_hsv;
        cv::Mat centerZone_hsv;
        cv::cvtColor( rightZone, rightZone_hsv, cv::COLOR_BGR2HSV);
        cv::cvtColor( leftZone, leftZone_hsv, cv::COLOR_BGR2HSV);
        cv::cvtColor( centerZone, centerZone_hsv, cv::COLOR_BGR2HSV);
      
        // right color mask 
        cv::Mat rightMaskRed, rightMaskGreen;  // NB: green mask is also use as 2nd mask for red detection
        cv::inRange(rightZone_hsv, red_hsv_low_threshold, red_hsv_max_range_threshold, rightMaskGreen);
        cv::inRange(rightZone_hsv, red_hsv_zero_threshold, red_hsv_high_threshold, rightMaskRed);
        rightMaskRed |= rightMaskGreen ;
        cv::inRange(rightZone_hsv, green_hsv_low_threshold, green_hsv_high_threshold, rightMaskGreen);

        // left color mask 
        cv::Mat leftMaskRed, leftMaskGreen;
        cv::inRange(leftZone_hsv, red_hsv_low_threshold, red_hsv_max_range_threshold, leftMaskGreen);
        cv::inRange(leftZone_hsv, red_hsv_zero_threshold, red_hsv_high_threshold, leftMaskRed);
        leftMaskRed |= leftMaskGreen;
        cv::inRange(leftZone_hsv, green_hsv_low_threshold, green_hsv_high_threshold, leftMaskGreen);

        // center color mask 
        cv::Mat centerMaskRed, centerMaskGreen;
        cv::inRange(centerZone_hsv, red_hsv_low_threshold, red_hsv_max_range_threshold, centerMaskGreen);
        cv::inRange(centerZone_hsv, red_hsv_zero_threshold, red_hsv_high_threshold, centerMaskRed);
        centerMaskRed |= centerMaskGreen;
        cv::inRange(centerZone_hsv, green_hsv_low_threshold, green_hsv_high_threshold, centerMaskGreen);


        /*
         *  Left / right dock zone 
         */
        // Bound a rectangle around the detected colors
        cv::Rect leftBoundRect = cv::boundingRect(leftMaskRed|leftMaskGreen);
        cv::Rect rightBoundRect = cv::boundingRect(rightMaskRed|rightMaskGreen);

        // call dock zone color detection algorithm
        dockZoneDetection(true, rightMaskRed, rightMaskGreen, rightBoundRect, rightZone);
        dockZoneDetection(false, leftMaskRed, leftMaskGreen, leftBoundRect, leftZone);


        /*
         * Center zone 
         */
        centerZoneDetection(centerMaskGreen, centerZone, cv::Scalar(168, 255, 0) );
        centerZoneDetection(centerMaskRed, centerZone, cv::Scalar(0, 162, 255) );
       
        // Various display
        cv::rectangle( photo_undistorted, cropZoneLeft.tl(), cropZoneLeft.br(), ColorWhite, 1);
        cv::rectangle( photo_undistorted, cropZoneRight.tl(), cropZoneRight.br(), ColorWhite, 1);

        cout<< "compute duration = " <<  float(clock()-start)/CLOCKS_PER_SEC <<endl;
      
        cv::imshow( "photo_undistorted", photo_undistorted );
        cv::waitKey(0);
    }

    cout<<"Stop camera..."<<endl;
    Camera.release();
   
 return 0;
}



