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
#include "predefined_colors.hpp"
#include "dock_zone_detection.hpp"

using namespace std; 

void dockZoneDetection(bool isRightZone ,cv::Mat &redMask, cv::Mat & greenMask, cv::Rect &boundRect,  cv::Mat &zone)
{
    constexpr int nbCup = 5;
    constexpr int nbChunk = 6; // the image is divided in 6 chunk and the 1st cup is in the 2 first chunk 
    int chunkWidth = boundRect.width / nbChunk;

    cv::Point origin = boundRect.tl();
    cv::Point originBr = boundRect.br();

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

        int nb_pixel_in_zone = subRed.rows * subRed.cols;
        
        if( nbRed < 0.1*nb_pixel_in_zone  && nbGreen < 0.1*nb_pixel_in_zone )
        {
           cv::rectangle(   zone, tl, br, ColorWhite, 10);
            cout << "cup :" << cup << "  unknown" << endl;   
        }
        else if( nbRed > nbGreen)
        {
            cv::rectangle(   zone, tl, br, ColorRed, 10);
            cout << "cup :" << cup << " Red" << endl;    
        }
        else
        {
            cv::rectangle(   zone, tl, br, ColorGreen, 10);
            cout << "cup :" << cup << " Green"<< endl;    
        }


        cv::rectangle( zone, tl, br, Colorblack);
    }
}