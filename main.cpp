#include <ctime>
#include <iostream>
#include <raspicam/raspicam_cv.h>
#include <raspicam/raspicam_still_cv.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace std; 
#include <unistd.h>
#include <cstdint>

static void dockZoneDetection(bool isRightZone, cv::Mat &redMask, cv::Mat & greenMask, cv::Rect &boundRect,  cv::Mat &zone);
static void centerZoneDetection(cv::Mat &centerZoneMask, cv::Mat &centerZoneImage, cv::Scalar circleColor );


cv::Scalar ColorWhite(255, 255, 255, 0);

int main ( int argc,char **argv ) {
   
    time_t timer_begin,timer_end;
    raspicam::RaspiCam_Still_Cv Camera;
    cv::Mat photo, photo_undistorted;


    /* Undistort stuff */
    float dataK[] = { 637.2385378800869, 0.0, 863.5527311953624, 0.0, 638.578239541455, 504.03676841955485, 0.0, 0.0, 1.0 };
    cv::Mat K = cv::Mat(3, 3, CV_32F, dataK);

    float dataD[] = { 0.2214277948742857, -0.1187171592448915, 0.10125203296576267, -0.044765584282516376};
    cv::Mat D = cv::Mat(1, 4, CV_32F, dataD);

    cv::Size image_size(1920, 1080);

    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(K, D, cv::Mat(), K, image_size, CV_16SC2, map1, map2);

    cout << "K=" << K << endl;
    cout << "D=" << D << endl;
    cout << "image_size=" << image_size << endl;
    
    

        int dilation_size = 4;
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
        cout<<"size row:" << photo.rows << " cols:" << photo.cols << "photo.size()" << photo.size()<<endl;


        clock_t time_begin = clock(); 

        
        // cv::remap(photo, photo_undistorted, map1, map2, cv::INTER_LINEAR);

        // cv::Rect2d cropZone = cv::selectROI(photo);
        //  cout<<"selection " << cropZone <<endl; 

        // Extract right/left zone 
        cv::Rect2d cropZoneLeft(181, 659, 254, 104);
        cv::Rect2d cropZoneRight(1178, 698, 241, 70);
        cv::Mat rightZone = photo(cropZoneRight);
        cv::Mat leftZone = photo(cropZoneLeft);

        // extract center zone
        cv::Rect2d cropCenterZone(499, 339, 667, 355);
        cv::Mat centerZone = photo(cropCenterZone);
        
        // Transform to HSV
        cv::Mat rightZone_hsv;
        cv::Mat leftZone_hsv;
        cv::Mat centerZone_hsv;
        cv::cvtColor( rightZone, rightZone_hsv, cv::COLOR_BGR2HSV);
        cv::cvtColor( leftZone, leftZone_hsv, cv::COLOR_BGR2HSV);
        cv::cvtColor( centerZone, centerZone_hsv, cv::COLOR_BGR2HSV);
        
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

        // center color mask 
        cv::Mat centerMaskRed, centerMaskGreen;
        cv::inRange(centerZone_hsv, cv::Scalar(0, 100, 70), cv::Scalar(8, 255, 255), centerMaskRed);
        cv::inRange(centerZone_hsv, cv::Scalar(170, 100, 70), cv::Scalar(180, 255, 255), centerMaskGreen);
        centerMaskRed |= centerMaskGreen;
        cv::inRange(centerZone_hsv, cv::Scalar(40, 50, 30), cv::Scalar(80, 255, 255), centerMaskGreen);


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
        cv::rectangle( photo, cropZoneLeft.tl(), cropZoneLeft.br(), ColorWhite, 1);
        cv::rectangle( photo, cropZoneRight.tl(), cropZoneRight.br(), ColorWhite, 1);



        clock_t time_end = clock();
        double secondsElapsed = difftime ( timer_end,timer_begin );
        cout<< "Compute duration = " <<  float(time_end-time_begin)/CLOCKS_PER_SEC <<endl;
      
        cv::imshow( "photo", photo );
        cv::waitKey(0);
    }

    cout<<"Stop camera..."<<endl;
    Camera.release();
   
 return 0;
}


static void centerZoneDetection(cv::Mat &centerZoneMask, cv::Mat &centerZoneImage, cv::Scalar circleColor )
{
    cv::Mat kernel = cv::Mat::ones(3,3,  CV_8U);
    cv::morphologyEx(centerZoneMask, centerZoneMask, cv::MORPH_OPEN,kernel, cv::Point(-1,-1),  2);

    cv::Mat sure_bg;
    cv::dilate(centerZoneMask, sure_bg, kernel, cv::Point(-1,-1), 3);
    
    cv::Mat centerMaskDist;
    cv::distanceTransform(centerZoneMask, centerMaskDist, cv::DIST_L2, 5);


    double maxVal;     
    cv::minMaxLoc( centerMaskDist, 0, &maxVal, 0, 0 );
    cv::Mat sure_fg;
    cv::threshold(centerMaskDist,sure_fg, 0.5*maxVal,255,cv::THRESH_BINARY);

    // Pour pouvoir afficher le watershed
    cv::normalize(centerMaskDist,centerMaskDist,0,1,cv::NORM_MINMAX);

    sure_fg.convertTo(sure_fg, CV_8U, 10);

    vector<vector<cv::Point> > fg_contours;
    cv::findContours(sure_fg, fg_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);


    cv::Mat unknow = sure_bg-sure_fg;

    cv::Mat markers;
    cv::connectedComponents(sure_fg, markers);
    markers +=1;
    markers.setTo(0, unknow==255);

    cv::watershed(centerZoneImage, markers);


    for (size_t i = 0; i < fg_contours.size(); i++)
    {
        cv::Moments moment = cv::moments(fg_contours[i]);
        cv::Point center(moment.m10 / (moment.m00+1e-5), moment.m01 / (moment.m00+1e-5));
        cv::circle( centerZoneImage, center, 5, circleColor, -1 );
    }

    centerZoneImage.setTo( cv::Scalar(255, 255, 255), markers==-1);
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