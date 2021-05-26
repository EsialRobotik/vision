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

cv::Scalar ColorWhite (255, 255, 255, 0);
cv::Scalar Colorblue (255, 0, 0);


#define DISPLAY_NAME "photo"
bool new_photo = true;
bool retry = false;

void on_compute(int state,void *zob)
{
    new_photo = false;
    cv::destroyWindow (DISPLAY_NAME);
}

void on_new_photo(int state,void *zob)
{
    new_photo = true;
    cv::destroyWindow (DISPLAY_NAME);
}

void on_OK(int state,void *zob)
{
    cv::destroyWindow (DISPLAY_NAME);
}

void on_retry(int state,void *zob)
{
    retry = true;
    cv::destroyWindow (DISPLAY_NAME);
}


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
    root["center_roi"].lookupValue("x", x);
    root["center_roi"].lookupValue("y", y);
    root["center_roi"].lookupValue("width", width);
    root["center_roi"].lookupValue("height", height);
    cv::Rect2d centerZoneRect(x, y, width,height);

    int hue_min, hue_max, saturation_min, saturation_max, value_min, value_max;
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

    float cups_size[3][3];

    for(int line=0; line<3; line++)
    {
        for(int col=0; col<3; col++)
        {

retry:
            retry = false;
            while( new_photo)
            {
                Camera.grab();
                Camera.retrieve ( photo);
                cv::remap(photo, photo_undistorted, fisheye_map1, fisheye_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

                
                // Draw center zone on image
                cv::rectangle( photo_undistorted, centerZoneRect.tl(), centerZoneRect.br(), ColorWhite, 3);

                // Then show the zone where the cup must be places
                int radius = ((centerZoneRect.height + centerZoneRect.width)/2)*0.05;
                cv::circle( photo_undistorted,cv::Point(centerZoneRect.x+ (col*0.5)*centerZoneRect.width, centerZoneRect.y+ (line*0.5)*centerZoneRect.height), radius, Colorblue, 4);

                // Show the image
                new_photo = true;
                cv::namedWindow(DISPLAY_NAME,cv::WINDOW_NORMAL|cv::WINDOW_KEEPRATIO);
                cv::imshow( DISPLAY_NAME, photo_undistorted );
                cv::displayOverlay(DISPLAY_NAME, "Put a RED cup in the rectangle, near to the circle \n Press CTRL+P to access panel and see control buttons");
                cv::createButton("Capture an another photo", on_new_photo);
                cv::createButton("Compute area", on_compute);
                cv::waitKey(0);
                
            }
            new_photo = true;

            // grab a another image to compute area zone
            Camera.grab();
            Camera.retrieve ( photo);
            cv::remap(photo, photo_undistorted, fisheye_map1, fisheye_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
            
            cv::Mat centerZone = photo_undistorted(centerZoneRect);
            cv::Mat centerZone_hsv;
            cv::cvtColor( centerZone, centerZone_hsv, cv::COLOR_BGR2HSV);

            cv::Mat centerMaskRed, centerMaskRed2;
            cv::inRange(centerZone_hsv, red_hsv_low_threshold, red_hsv_max_range_threshold, centerMaskRed2);
            cv::inRange(centerZone_hsv, red_hsv_zero_threshold, red_hsv_high_threshold, centerMaskRed);
            centerMaskRed |= centerMaskRed2 ;

            int kernel_close_size= 2;
            cv::Mat kernel_close = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_close_size*2+1, kernel_close_size*2+1));

            cv::Mat centerMaskRed_closed;
            cv::morphologyEx(centerMaskRed, centerMaskRed_closed, cv::MORPH_CLOSE,kernel_close, cv::Point(-1,-1),  2);

            vector<vector<cv::Point> > contours;
            cv::findContours(centerMaskRed_closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            double maxArea=0;
            int maxAreaIdx=-1;
            for(int i=0; i<contours.size(); i++)
            {
                double area = contourArea(contours[i]);
                if( area > maxArea )
                {
                    maxArea = area;
                    maxAreaIdx = i;
                }
            }
            if( maxAreaIdx == -1)
                goto retry;

            cv::Mat colored_display;
            cv::cvtColor( centerMaskRed_closed, colored_display,  cv::COLOR_GRAY2BGR);
            cv::drawContours(colored_display, contours, maxAreaIdx, Colorblue, 5);
            

            cv::namedWindow(DISPLAY_NAME,cv::WINDOW_NORMAL);
            cv::createButton("OK", on_OK);
            cv::createButton("retry", on_retry);
            cv::imshow( DISPLAY_NAME, colored_display );
            cv::waitKey(0);

            if(retry)
                goto retry;

            cups_size[line][col] = maxArea;
        }
    }

    libconfig::Config cfg_cup_size;
    libconfig::Setting& cup_size_root = cfg_cup_size.getRoot();
    libconfig::Setting &cup_size_array = cup_size_root.add("cup_size", libconfig::Setting::TypeArray);

    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            cup_size_array.add(libconfig::Setting::TypeFloat) = cups_size[i][j];
    
    cfg_cup_size.writeFile("cup_size.cfg");
    cout << "configuration writen to cup_size.cfg file, merge config in vision.cfg" << endl;

    Camera.release();
 return 0;
}
