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
#include <vector>
#include <cmath> 
#include <assert.h>
#include <thread>
#include <libconfig.h++>
#include <pigpiod_if2.h>
#include "predefined_colors.hpp"
#include "dock_zone_detection.hpp"
#include "center_zone_detection.hpp"
#include "position_detection.hpp"
#include "unistd.h"
#include <iostream>
#include <boost/asio.hpp>

using namespace std; 
using namespace boost::asio;
using ip::udp;

void send_rack_infos(udp::socket & socket, udp::endpoint &remote_endpoint, bool isRightDock, vector<t_cup> &dockCupList);

cv::Rect2d extractTrapeziumZoneFromPointsInImage(cv::Mat &photo_undistorted, t_trapezium rightZoneTrapeziumInImage, cv::Mat &extract);
void drawTrapezium(cv::Mat &photo_undistorted, t_trapezium &rightZoneTrapeziumInImage);
bool keep_alive_running=true;
void keep_alive(int pigpioID);
bool is_capturing = false;
void capturing(int pigpioID);


bool accept_calibration = false;
void accept_calibration_btn_watch(int pigpioID);

bool do_reset = false;
void do_reset_btn_watch(int pigpioID);

int pigpioID;
int GPIO_aruco_42 = 12;
int GPIO_aruco_51 = 16;
int GPIO_aruco_69 = 19;


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
    pigpioID = pigpio_start(0, 0);
    if ( pigpioID < 0)
    {
        cerr<<"Unable to start pigpio "<<endl;
        return -1;
    }
    std::thread keep_alive_thread(keep_alive, pigpioID);
    std::thread capturing_thread(capturing, pigpioID);
    std::thread accept_calibration_btn_thread(accept_calibration_btn_watch, pigpioID);
    std::thread do_reset_btn_thread(do_reset_btn_watch, pigpioID);

    set_mode(pigpioID, GPIO_aruco_42, PI_OUTPUT);
    set_mode(pigpioID, GPIO_aruco_51, PI_OUTPUT);
    set_mode(pigpioID, GPIO_aruco_69, PI_OUTPUT);



    io_service io_service;

    udp::socket socket(io_service);
    socket.open(udp::v4());
    udp::endpoint remote_endpoint = udp::endpoint(ip::address::from_string("127.0.0.1"), 4269);
    boost::system::error_code err;
    socket.send_to(boost::asio::buffer("robot", sizeof("robot")), remote_endpoint, 0, err);
    


    /*
     * Retrieve fishey calibration from Camera & distortion matrix in configuration file.
     */ 
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


    t_trapezium centerZoneTrapeziumInImage;
    t_trapezium rightZoneTrapeziumInImage;
    t_trapezium leftZoneTrapeziumInImage;

reset_GOTO:
    do_reset = false;
    accept_calibration = false;
    is_capturing = false;
    bool position_detection_ok = false;
    while( !accept_calibration || !position_detection_ok)
    {

        //Camera.grab();
        //Camera.retrieve ( photo);
        photo = cv::imread("29_blurred.jpg");


        cv::remap(photo, photo_undistorted, fisheye_map1, fisheye_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);


        position_detection_ok = detectArucoAndComputeRotVecMatrixes(photo_undistorted, K, D, position_detection_rvec, position_detection_tvec, rotationMatrix);
        if( position_detection_ok)
        {
            t_trapezium right_trapezium_in_table;
            right_trapezium_in_table.top_left = cv::Point(-22, 1059);
            right_trapezium_in_table.top_right = cv::Point(-22, 640);
            right_trapezium_in_table.bottom_right = cv::Point(-114, 640);
            right_trapezium_in_table.bottom_left = cv::Point(-114, 1059);
            rightZoneTrapeziumInImage = localizeTrapezium(K, D,position_detection_rvec, position_detection_tvec, right_trapezium_in_table, 115);


            t_trapezium left_trapezium_in_table;
            left_trapezium_in_table.top_left = cv::Point(-22, 2359);
            left_trapezium_in_table.top_right = cv::Point(-22, 1940);
            left_trapezium_in_table.bottom_right = cv::Point(-114, 1940);
            left_trapezium_in_table.bottom_left = cv::Point(-114, 2359);
            leftZoneTrapeziumInImage = localizeTrapezium(K, D,position_detection_rvec, position_detection_tvec, left_trapezium_in_table, 115);

            t_trapezium center_trapezium_in_table;
            center_trapezium_in_table.top_left = cv::Point(500,2000);
            center_trapezium_in_table.top_right = cv::Point(500,1000);
            center_trapezium_in_table.bottom_right = cv::Point(0,1000);
            center_trapezium_in_table.bottom_left = cv::Point(0,2000);
            centerZoneTrapeziumInImage = localizeTrapezium(K, D,position_detection_rvec, position_detection_tvec, center_trapezium_in_table);
        }

        if(do_reset)
            goto reset_GOTO;

    }

    is_capturing = true;
    while(1)
    {
        clock_t start = clock();
        // Camera.grab();
        // Camera.retrieve ( photo);
        

        cv::remap(photo, photo_undistorted, fisheye_map1, fisheye_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);


        // Extract right/left zone 
        cv::Mat rightZone, leftZone;
        extractTrapeziumZoneFromPointsInImage(photo_undistorted, rightZoneTrapeziumInImage, rightZone  );
        extractTrapeziumZoneFromPointsInImage(photo_undistorted, leftZoneTrapeziumInImage, leftZone  );



        // extract center zone
        cv::Mat centerZone;
        cv::Rect2d centerZoneRoi = extractTrapeziumZoneFromPointsInImage(photo_undistorted, centerZoneTrapeziumInImage, centerZone  );
        
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
        vector<t_cup> rightDockCupList, leftDockCupList;
        dockZoneDetection(true, rightMaskRed, rightMaskGreen, rightBoundRect, rightZone, rightDockCupList);
        dockZoneDetection(false, leftMaskRed, leftMaskGreen, leftBoundRect, leftZone, leftDockCupList);


        /*
         * Center zone 
         */
        vector<cv::Point> redCupList, greenCupList;
        centerZoneDetection(centerMaskGreen, centerZone, cv::Scalar(168, 255, 0), greenCupList );
        centerZoneDetection(centerMaskRed, centerZone, cv::Scalar(0, 162, 255), redCupList );

       
        // Various display
        drawTrapezium(photo_undistorted, centerZoneTrapeziumInImage);
        drawTrapezium(photo_undistorted, rightZoneTrapeziumInImage);
        drawTrapezium(photo_undistorted, leftZoneTrapeziumInImage);


#ifdef CONSOLE_DISP
        for(int cup=0; cup < redCupList.size(); cup++)
        {
            cv::Point pointInImage(redCupList[cup].x + centerZoneRoi.x, redCupList[cup].y + centerZoneRoi.y );
            cv::Point pointOnTable = positionOnTableFromPointInImage(pointInImage, K, rotationMatrix, position_detection_tvec, 115/2);
            cout << "image " << pointInImage << " <=> table " << pointOnTable << endl;
        }

        for(int cup=0; cup < greenCupList.size(); cup++)
        {
            cv::Point pointInImage(greenCupList[cup].x + centerZoneRoi.x, greenCupList[cup].y + centerZoneRoi.y );
            cv::Point pointOnTable = positionOnTableFromPointInImage(pointInImage, K, rotationMatrix, position_detection_tvec, 115/2);
            cout << "image " << pointInImage << " <=> table " << pointOnTable << endl;
        }

#endif
        
        send_rack_infos(socket, remote_endpoint, true, rightDockCupList);
        send_rack_infos(socket, remote_endpoint, false, leftDockCupList);

      
        cv::namedWindow("photo_undistorted",cv::WINDOW_NORMAL|cv::WINDOW_KEEPRATIO);
        cv::imshow( "photo_undistorted", photo_undistorted ); 
        cv::namedWindow("rightZone",cv::WINDOW_NORMAL|cv::WINDOW_KEEPRATIO);
        cv::imshow( "rightZone", rightZone );   
        cv::namedWindow("leftZone",cv::WINDOW_NORMAL|cv::WINDOW_KEEPRATIO);
        cv::imshow( "leftZone", leftZone );         
        cv::waitKey(0);

#ifdef CONSOLE_DISP
        cout<< "total duration = " <<  float(clock()-start)/CLOCKS_PER_SEC <<endl;
#endif

        if(do_reset)
            goto reset_GOTO;
    }

    cout<<"Stop camera..."<<endl;
    Camera.release();
    keep_alive_running = false;
    keep_alive_thread.join();
    pigpio_stop(pigpioID);
    socket.close();

 return 0;
}

void send_rack_infos(udp::socket & socket, udp::endpoint &remote_endpoint, bool isRightDock, vector<t_cup> &dockCupList)
{
    char buffer[13];
    sprintf(buffer, "rack%c#", isRightDock ? 'D' : 'G');
    for(int cup=0; cup < dockCupList.size(); cup++)
    {
        char cupChar = '?';
        if (dockCupList[cup] == cup_red)
            cupChar = 'r';
        else if (dockCupList[cup] == cup_green)
            cupChar = 'v';

        buffer[6+cup] = cupChar;
    }
    buffer[11] = '\n';
    buffer[12] = 0;

    boost::system::error_code err;
    socket.send_to(boost::asio::buffer(buffer, sizeof(buffer)), remote_endpoint, 0, err);
}


cv::Rect2d extractTrapeziumZoneFromPointsInImage(cv::Mat &photo_undistorted, t_trapezium rightZoneTrapeziumInImage, cv::Mat &extract  )
{
    int minX = std::min(rightZoneTrapeziumInImage.top_left.x, rightZoneTrapeziumInImage.bottom_left.x);
    int minY = std::min(rightZoneTrapeziumInImage.top_left.y, rightZoneTrapeziumInImage.top_right.y);
    int maxX = std::max(rightZoneTrapeziumInImage.top_right.x, rightZoneTrapeziumInImage.bottom_right.x);
    int maxY = std::max(rightZoneTrapeziumInImage.bottom_right.y, rightZoneTrapeziumInImage.bottom_left.y);

    cv::Rect2d roi(minX, minY, maxX-minX, maxY-minY);
    cv::Mat cropedZone = photo_undistorted(roi);

    rightZoneTrapeziumInImage.top_left.x     -= minX;
    rightZoneTrapeziumInImage.top_right.x    -= minX;
    rightZoneTrapeziumInImage.bottom_right.x -= minX;
    rightZoneTrapeziumInImage.bottom_left.x  -= minX;
    rightZoneTrapeziumInImage.top_left.y     -= minY;
    rightZoneTrapeziumInImage.top_right.y    -= minY;
    rightZoneTrapeziumInImage.bottom_right.y -= minY;
    rightZoneTrapeziumInImage.bottom_left.y  -= minY;


    cv::Mat mask (cropedZone.rows, cropedZone.cols, CV_8UC1, cv::Scalar(0));
    vector< vector<cv::Point> >  co_ordinates;
    co_ordinates.push_back(vector<cv::Point>());
    co_ordinates[0].push_back(rightZoneTrapeziumInImage.top_left);
    co_ordinates[0].push_back(rightZoneTrapeziumInImage.top_right);
    co_ordinates[0].push_back(rightZoneTrapeziumInImage.bottom_right);
    co_ordinates[0].push_back(rightZoneTrapeziumInImage.bottom_left);
    drawContours( mask,co_ordinates,0, cv::Scalar(255),-1, 8 );

    cropedZone.copyTo(extract, mask);
    return roi;
}


void drawTrapezium(cv::Mat &photo_undistorted, t_trapezium &rightZoneTrapeziumInImage)
{
    vector< vector<cv::Point> >  co_ordinates;
    co_ordinates.push_back(vector<cv::Point>());
    co_ordinates[0].push_back(rightZoneTrapeziumInImage.top_left);
    co_ordinates[0].push_back(rightZoneTrapeziumInImage.top_right);
    co_ordinates[0].push_back(rightZoneTrapeziumInImage.bottom_right);
    co_ordinates[0].push_back(rightZoneTrapeziumInImage.bottom_left);
    drawContours( photo_undistorted,co_ordinates,0, ColorWhite);
}

void keep_alive(int pigpioID)
{
    const int gpio_num = 15;
    set_mode(pigpioID, gpio_num, PI_OUTPUT);

    while(keep_alive_running)
    {

        usleep(1000*200);
        gpio_write(pigpioID, gpio_num, 1);
        usleep(1000*200);
        gpio_write(pigpioID, gpio_num, 0);
    }
}

void capturing(int pigpioID)
{
    const int gpio_num = 6;
    set_mode(pigpioID, gpio_num, PI_OUTPUT);
    while(keep_alive_running)
    {
        if( !is_capturing )
        {
            usleep(1000*200);
            gpio_write(pigpioID, gpio_num, is_capturing);
        }
        else
        {
            usleep(1000*200);
            gpio_write(pigpioID, gpio_num, 1);
            usleep(1000*200);
            gpio_write(pigpioID, gpio_num, 0);
        }
        
    }
}

void accept_calibration_btn_watch(int pigpioID)
{
    const int gpio_num = 10;
    set_mode(pigpioID, gpio_num, PI_INPUT);
    set_pull_up_down(pigpioID, gpio_num, PI_PUD_UP);
    

    while(1)
    {
        wait_for_edge(pigpioID, gpio_num, FALLING_EDGE, 86400);
        accept_calibration = true;
        usleep(1000*500);
    }
}

void do_reset_btn_watch(int pigpioID)
{
    const int gpio_num = 11;
    set_mode(pigpioID, gpio_num, PI_INPUT);
    set_pull_up_down(pigpioID, gpio_num, PI_PUD_UP);
    
    while(1)
    {
        wait_for_edge(pigpioID, gpio_num, FALLING_EDGE, 86400);    
        do_reset = true;
        usleep(1000*500);
    }
}