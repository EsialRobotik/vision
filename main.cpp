#include <ctime>
#include <iostream>
#include <raspicam/raspicam_cv.h>
#include <raspicam/raspicam_still_cv.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace std; 
#include <unistd.h>
#include <cstdint>

static void dockZoneDetection(bool isRightZone, cv::Mat &redMask, cv::Mat & greenMask, cv::Rect &boundRect,  cv::Mat &zone);


int main ( int argc,char **argv ) {
   
    time_t timer_begin,timer_end;
    raspicam::RaspiCam_Still_Cv Camera;
    cv::Mat photo;

    
    
    cv::Mat final;

        int dilation_size = 5;
      cv::Mat kernel_erose = getStructuringElement(  cv::MORPH_RECT,
                       cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                       cv::Point( dilation_size, dilation_size ) );

    
    Camera.set(cv::CAP_PROP_FRAME_WIDTH,  1920);
    Camera.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    Camera.setRotation(90);


    cout<<"Opening Camera..."<<endl;
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
 
        clock_t capture_start = clock();
        Camera.grab();
        Camera.retrieve ( photo);
        
        cout<< "capture duration = " <<  float(clock()-capture_start)/CLOCKS_PER_SEC <<endl;
        cout<<"size row:" << photo.rows << " cols:" << photo.cols <<endl;


        clock_t time_begin = clock(); 

        // cv::Rect2d cropZone = cv::selectROI(photo);
        //  cout<<"selection " << cropZone <<endl; 

        // Extract right/left zone 
        cv::Rect2d cropZoneLeft(253, 665, 293, 103);

        cv::Rect2d cropZoneRight(1100, 680, 303 , 80);
        cv::Mat rightZone = photo(cropZoneRight);
        cv::Mat leftZone = photo(cropZoneLeft);
        
        // Transform to HSV
        cv::Mat rightZone_hsv;
        cv::Mat leftZone_hsv;
        cv::cvtColor( rightZone, rightZone_hsv, cv::COLOR_BGR2HSV);
        cv::cvtColor( leftZone, leftZone_hsv, cv::COLOR_BGR2HSV);

        
        // right color mask 
        cv::Mat rightMaskRed, rightMaskGreen;  // NB: green mask is also use as 2nd mask for red detection
        cv::inRange(rightZone_hsv, cv::Scalar(0, 100, 70), cv::Scalar(8, 255, 255), rightMaskRed);
        cv::inRange(rightZone_hsv, cv::Scalar(170, 100, 70), cv::Scalar(180, 255, 255), rightMaskGreen);
        rightMaskRed |= rightMaskGreen ;
        cv::inRange(rightZone_hsv, cv::Scalar(40, 50, 30), cv::Scalar(80, 255, 255), rightMaskGreen);

        // left color mask 
        cv::Mat leftMaskRed, leftMaskGreen;
        cv::inRange(leftZone_hsv, cv::Scalar(0, 100, 70), cv::Scalar(8, 255, 255), leftMaskRed);
        cv::inRange(leftZone_hsv, cv::Scalar(170, 100, 70), cv::Scalar(180, 255, 255), leftMaskGreen);
        leftMaskRed |= leftMaskGreen;
        cv::inRange(leftZone_hsv, cv::Scalar(40, 50, 30), cv::Scalar(80, 255, 255), leftMaskGreen);



        // Bound a rectangle around the detected colors
        cv::Rect leftBoundRect = cv::boundingRect(leftMaskRed|leftMaskGreen);
        cv::Rect rightBoundRect = cv::boundingRect(rightMaskRed|rightMaskGreen);


        dockZoneDetection(true, rightMaskRed, rightMaskGreen, rightBoundRect, rightZone);
        dockZoneDetection(false, leftMaskRed, leftMaskGreen, leftBoundRect, leftZone);

        clock_t time_end = clock();

        cv::Scalar ColorWhite(255, 255, 255, 0);
        cv::rectangle(   leftZone, leftBoundRect.tl(), leftBoundRect.br(), ColorWhite   );    
        cv::rectangle(   rightZone, rightBoundRect.tl(), rightBoundRect.br(), ColorWhite   );    

      
        
        double secondsElapsed = difftime ( timer_end,timer_begin );
        cout<< " duration = " <<  float(time_end-time_begin)/CLOCKS_PER_SEC <<endl;

        cv::namedWindow( "photo" );
        // cv::namedWindow( "rightZone" );
        // cv::namedWindow( "rightZonemMask" );
        // cv::namedWindow( "leftZone" );
        // cv::namedWindow( "leftZonemMaskRed" );
        
        cv::imshow( "photo", photo );
        // cv::imshow( "rightZone", rightZone );
        // cv::imshow( "rightZonemMask", rightMaskRed | rightMaskGreen );
        // cv::imshow( "leftZone", leftZone);
        // cv::imshow( "leftZonemMaskRed",  leftMaskRed );
        
        cv::waitKey(1);
   
    }

    cout<<"Stop camera..."<<endl;
    Camera.release();
   
 return 0;
}



static void dockZoneDetection(bool isRightZone ,cv::Mat &redMask, cv::Mat & greenMask, cv::Rect &boundRect,  cv::Mat &zone)
{
    constexpr int nbCup = 5;
    constexpr int nbChunk = 6; // the image is divided in 6 chunk and the 1st cup is in the 2 first chunk 
    int chunkWidth = boundRect.width / nbChunk;

    cv::Point origin = boundRect.tl();
    cv::Point originBr = boundRect.br();

    cv::Scalar ColorGreen(0, 255, 0);
    cv::Scalar ColorRed  (0, 0, 255);
    cv::Scalar Colorblack  (0, 0, 0);
        

    for(int cup=0; cup<nbCup; cup++)
    {
        int cupWidth;
        if(isRightZone)
            cupWidth = (cup == 0) ? 2*chunkWidth : chunkWidth;
        else
            cupWidth = (cup == nbCup-1) ? 2*chunkWidth : chunkWidth;

        int currentZoneXstart;
        if(isRightZone)
            currentZoneXstart =  (cup == 0) ? 0 : (cup+1)*chunkWidth;
        else
            currentZoneXstart =  cup*chunkWidth;


        cv::Point tl(origin.x + currentZoneXstart, origin.y);
        cv::Point br(tl.x + cupWidth, tl.y + boundRect.height);
        cv::Rect roi(tl, br);

        cv::Mat subGreen = greenMask(roi);
        int nbGreen = 0;
        for (cv::MatIterator_<uint8_t> it = subGreen.begin<uint8_t>(); it != subGreen.end<uint8_t>(); ++it)
        {
            nbGreen += *it;
        }

        cv::Mat subRed = redMask(roi);
        int nbRed = 0;
        for (cv::MatIterator_<uint8_t> it = subRed.begin<uint8_t>(); it != subRed.end<uint8_t>(); ++it)
        {
            nbRed += *it;
        }
        
        if( nbRed > nbGreen)
        {
            cv::rectangle(   zone, tl, br, ColorRed, 10);
            cout << "cup :" << cup << " Red" << endl;    
        }
        else
        {
            cv::rectangle(   zone, tl, br, ColorGreen, 10);
            cout << "cup :" << cup << " Green"<< endl;    
        }


        cv::rectangle(   zone, tl, br, Colorblack);
    }
}