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
#include "position_detection.hpp"

using namespace std; 

static cv::Mat K, D;
static cv::Mat fisheye_map1, fisheye_map2;
static cv::Mat position_detection_rvec, position_detection_tvec;
static cv::Mat rotationMatrix(3,3,cv::DataType<double>::type);

const int32_t aruco_table_pos_x = 1500;
const int32_t aruco_table_pos_y = 1250;

float cups_size[3][3];

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

    /**
     * Read configuration from vision.cfg
     */

    libconfig::Config cfg;
    cfg.readFile("vision.cfg");
    const libconfig::Setting& root = cfg.getRoot();

    const libconfig::Setting &cup_size_settings = root.lookup("cup_size");
    for (int n = 0; n < cup_size_settings.getLength(); ++n)
        cups_size[n/3][n%3] =  cup_size_settings[n];


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


    // ROI zones will be computed in position_detection
    cv::Rect2d cropZoneLeft;
    cv::Rect2d cropZoneRight;
    cv::Rect2d cropCenterZone;
    cv::Rect2d gagZone;
    cv::Rect2d gagZone2;


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


    Camera.grab();
    Camera.retrieve ( photo);
    cv::remap(photo, photo_undistorted, fisheye_map1, fisheye_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    bool position_detection_ok = detectArucoAndComputeRotVecMatrixes(photo_undistorted, K, D, position_detection_rvec, position_detection_tvec, rotationMatrix);
    if( position_detection_ok)
    {
        cropZoneRight  = localizeZone(K, D,position_detection_rvec, position_detection_tvec, 0.0, 1059.0, -134.0, 640.0);
        cropZoneLeft   = localizeZone(K, D,position_detection_rvec, position_detection_tvec, 0.0, 2359.0, -134.0, 1940.0);
        cropCenterZone = localizeZone(K, D,position_detection_rvec, position_detection_tvec, 500.0, 2000.0, 0.0, 1000.0);

        gagZone = localizeZone(K, D,position_detection_rvec, position_detection_tvec, 100.0, 100.0, 0.0, 0.0);
        gagZone2 = localizeZone(K, D,position_detection_rvec, position_detection_tvec, 1200.0, 3000.0, 1000.0, 0.0);

    }


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

        // Do an open/close on the right color mask to remove noise
        int kernel_open_size= 3;
        cv::Mat kernel_open = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_open_size*2+1, kernel_open_size*2+1));
        int kernel_close_size= 2;
        cv::Mat kernel_close = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_close_size*2+1, kernel_close_size*2+1));

        cv::morphologyEx(rightMaskGreen, rightMaskGreen, cv::MORPH_OPEN,kernel_open, cv::Point(-1,-1),  2);
        cv::morphologyEx(rightMaskGreen, rightMaskGreen, cv::MORPH_CLOSE,kernel_close, cv::Point(-1,-1),  2);
        cv::morphologyEx(rightMaskRed, rightMaskRed, cv::MORPH_OPEN,kernel_open, cv::Point(-1,-1),  2);
        cv::morphologyEx(rightMaskRed, rightMaskRed, cv::MORPH_CLOSE,kernel_close, cv::Point(-1,-1),  2);
        

        // left color mask 
        cv::Mat leftMaskRed, leftMaskGreen;
        cv::inRange(leftZone_hsv, red_hsv_low_threshold, red_hsv_max_range_threshold, leftMaskGreen);
        cv::inRange(leftZone_hsv, red_hsv_zero_threshold, red_hsv_high_threshold, leftMaskRed);
        leftMaskRed |= leftMaskGreen;
        cv::inRange(leftZone_hsv, green_hsv_low_threshold, green_hsv_high_threshold, leftMaskGreen);

        // Do an open/close on the left color mask to remove noise
        cv::morphologyEx(leftMaskGreen, leftMaskGreen, cv::MORPH_OPEN,kernel_open, cv::Point(-1,-1),  2);
        cv::morphologyEx(leftMaskGreen, leftMaskGreen, cv::MORPH_CLOSE,kernel_close, cv::Point(-1,-1),  2);
        cv::morphologyEx(leftMaskRed, leftMaskRed, cv::MORPH_OPEN,kernel_open, cv::Point(-1,-1),  2);
        cv::morphologyEx(leftMaskRed, leftMaskRed, cv::MORPH_CLOSE,kernel_close, cv::Point(-1,-1),  2);


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
        vector<cv::Point> redCupList, greenCupList;
        // centerZoneDetection(centerMaskGreen, centerZone, cv::Scalar(168, 255, 0), greenCupList );
        centerZoneDetection(centerMaskRed, centerZone, cv::Scalar(0, 162, 255), redCupList );

       
        // Various display
        cv::rectangle( photo_undistorted, cropCenterZone.tl(), cropCenterZone.br(), ColorWhite, 1);
        cv::rectangle( photo_undistorted, cropZoneRight.tl(), cropZoneRight.br(), ColorWhite, 1);
        cv::rectangle( photo_undistorted, cropZoneLeft.tl(), cropZoneLeft.br(), ColorWhite, 1);
        cv::rectangle( photo_undistorted, gagZone.tl(), gagZone.br(), ColorWhite, 1);
        cv::rectangle( photo_undistorted, gagZone2.tl(), gagZone2.br(), ColorWhite, 1);


        cv::Rect2d preciseRightBoundRec(rightBoundRect.x + cropZoneRight.x , rightBoundRect.y + cropZoneRight.y , rightBoundRect.width, rightBoundRect.height);
        cv::rectangle( photo_undistorted, preciseRightBoundRec.tl(), preciseRightBoundRec.br(), ColorWhite, 1);

        cv::Rect2d preciseLeftBoundRec(leftBoundRect.x + cropZoneLeft.x , leftBoundRect.y + cropZoneLeft.y , leftBoundRect.width, leftBoundRect.height);
        cv::rectangle( photo_undistorted, preciseLeftBoundRec.tl(), preciseLeftBoundRec.br(), ColorWhite, 1);


        for(int cup=0; cup < redCupList.size(); cup++)
        {
            cv::Point pointInImage(redCupList[cup].x + cropCenterZone.x, redCupList[cup].y + cropCenterZone.y );
            cv::Point pointOnTable = positionOnTableFromPointInImage(pointInImage, K, rotationMatrix, position_detection_tvec);
            cout << "image " << pointInImage << " <=> table " << pointOnTable << endl;
        }
        cout<< "compute duration = " <<  float(clock()-start)/CLOCKS_PER_SEC <<endl;
      
        cv::namedWindow("photo_undistorted",cv::WINDOW_NORMAL|cv::WINDOW_KEEPRATIO);
        cv::imshow( "photo_undistorted", photo_undistorted );         
        cv::namedWindow("centerMaskRed",cv::WINDOW_NORMAL|cv::WINDOW_KEEPRATIO);
        cv::imshow( "centerMaskRed", centerMaskRed );     
        

        cv::waitKey(0);
    }

    cout<<"Stop camera..."<<endl;
    Camera.release();
   
 return 0;
}



