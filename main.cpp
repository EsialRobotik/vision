#include <ctime>
#include <iostream>
#include <raspicam/raspicam_cv.h>
//#include <raspicam/raspicamtypes.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace std; 
#include <unistd.h>


int main ( int argc,char **argv ) {
   
    time_t timer_begin,timer_end;
    raspicam::RaspiCam_Cv Camera;
    cv::Mat image;
    cv::Mat image_hsv;
    cv::Mat mask, mask2, mask_erode;
    cv::Mat final;

        int dilation_size = 5;
      cv::Mat kernel_erose = getStructuringElement(  cv::MORPH_RECT,
                       cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                       cv::Point( dilation_size, dilation_size ) );

    //Open camera
    cout<<"Opening Camera..."<<endl;
    if (!Camera.open()) {cerr<<"Error opening the camera"<<endl;return -1;}
   
    cout<<"Warm up... "<<endl; 
    sleep(3);
    cout<<"Capturing "<<endl;
      
    

    clock_t time_begin = clock(); 
    Camera.grab();
    Camera.retrieve ( image);

    // cv::Rect2d cropZone = cv::selectROI(image);
    // cv::Rect2d cropZoneLeft(678, 617, 60, 136);
    cv::Rect2d cropZoneRight(648, 203, 70, 154);
    image = image(cropZoneRight);
     cout<<"selection " << cropZoneRight <<endl; 




    cv::cvtColor( image, image_hsv, cv::COLOR_BGR2HSV);

    
    cv::inRange(image_hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), mask);
    cv::inRange(image_hsv, cv::Scalar(170, 100, 100), cv::Scalar(180, 255, 255), mask2);
    mask |= mask2;

     cv::erode(mask, mask_erode, kernel_erose);

    cv::bitwise_and(image,image, final, mask= mask);

    

    clock_t time_end = clock();
    

    cout<<"Stop camera..."<<endl;
    Camera.release();
   
    
    double secondsElapsed = difftime ( timer_end,timer_begin );
    cout<< " duration = " <<  float(time_end-time_begin)/CLOCKS_PER_SEC <<endl;

    cv::namedWindow( "image" );
    cv::namedWindow( "mask" );
    cv::namedWindow( "mask_erode" );
    cv::namedWindow( "final" );
    
    cv::imshow( "image", image );
    cv::imshow( "mask", mask );
    cv::imshow( "mask_erode", mask_erode );
    cv::imshow( "final", final );
    

    cv::waitKey(0);
    return 0;
}
