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

void on_change(int state,void *zob)
{

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
    root["green"].lookupValue("hue_min", hue_min);
    root["green"].lookupValue("hue_max", hue_max);
    root["green"].lookupValue("saturation_min", saturation_min);
    root["green"].lookupValue("saturation_max", saturation_max);
    root["green"].lookupValue("value_min", value_min);
    root["green"].lookupValue("value_max", value_max);
    cv::Scalar green_hsv_low_threshold(hue_min, saturation_min, value_min);
    cv::Scalar green_hsv_high_threshold(hue_max, saturation_max, value_max);


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

    for(int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
        {
            Camera.grab();
            Camera.retrieve ( photo);
            cv::remap(photo, photo_undistorted, fisheye_map1, fisheye_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

            
            // Draw center zone on image
            cv::rectangle( photo_undistorted, centerZoneRect.tl(), centerZoneRect.br(), ColorWhite, 3);

            // Then show the zone where the cup must be places
            int radius = ((centerZoneRect.height + centerZoneRect.width)/2)*0.05;
            cv::circle( photo_undistorted,cv::Point(centerZoneRect.x+ (i*0.5)*centerZoneRect.width, centerZoneRect.y+ (j*0.5)*centerZoneRect.height), radius, Colorblue, 4);

            // Show the image

            cv::namedWindow("photo_undistorted",cv::WINDOW_NORMAL|cv::WINDOW_KEEPRATIO);
            cv::imshow( "photo_undistorted", photo_undistorted );
            cv::displayOverlay("photo_undistorted", "Put a green cup in the rectangle, near to the circle \n Press CTRL+P to access panel and see control buttons");
            cv::createButton("Get Another image", on_change);
            cv::createButton("Compute area", on_change);
            cv::waitKey(0);
            cv::destroyWindow ("photo_undistorted");

            // grab a another image to compute area zone
            Camera.grab();
            Camera.retrieve ( photo);
            cv::remap(photo, photo_undistorted, fisheye_map1, fisheye_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
            
            cv::Mat centerZone = photo_undistorted(centerZoneRect);
            cv::Mat centerZone_hsv;
            cv::cvtColor( centerZone, centerZone_hsv, cv::COLOR_BGR2HSV);

            cv::Mat centerMaskGreen;
            cv::inRange(centerZone_hsv, green_hsv_low_threshold, green_hsv_high_threshold, centerMaskGreen);

            int kernel_close_size= 2;
            cv::Mat kernel_close = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_close_size*2+1, kernel_close_size*2+1));

            cv::Mat centerMaskGreen_closed;
            cv::morphologyEx(centerMaskGreen, centerMaskGreen_closed, cv::MORPH_CLOSE,kernel_close, cv::Point(-1,-1),  2);

            vector<vector<cv::Point> > contours;
            cv::findContours(centerMaskGreen_closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if( contours.size() != 1)
            {
                // Error, retry ?
            }

            double area = contourArea(contours[0]);
            cout << "area "  << area << endl;

            cv::namedWindow("centerZone",cv::WINDOW_NORMAL);
            cv::createButton("btn", on_change);
            cv::imshow( "centerZone", centerMaskGreen_closed );
            cv::waitKey(0);
            cv::destroyWindow ("centerZone");
        }
    }
    
    Camera.release();
 return 0;
}
