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
#include "center_zone_detection.hpp"


using namespace std; 


static void trySeparateOverlappingElement(cv::Mat &centerZoneMask, cv::Mat &centerZoneMask_closed, cv::Mat &centerZoneMask_separated    )
{
    vector<vector<cv::Point> > contours;
    cv::findContours(centerZoneMask_closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat detectedObjectContourMask = cv::Mat::ones(centerZoneMask_closed.rows, centerZoneMask_closed.cols, CV_8UC1);
    detectedObjectContourMask.setTo(255);
    float oneElementAreaSize = 2000; // TODO : depends de la hauteur de l'objet dans l'image !!
    for (size_t contour = 0; contour < contours.size(); contour++)
    {
        float area = contourArea(contours[contour]);
        int nbOjectInArea = round(area/oneElementAreaSize);
        cout << "area " << area << "/" << oneElementAreaSize << "=" << nbOjectInArea << endl;

        if( nbOjectInArea < 2)
            continue;

        cv::Mat areaMat = cv::Mat::zeros(centerZoneMask_closed.rows, centerZoneMask_closed.cols, CV_8UC1);
        cv::drawContours(areaMat, contours, contour, cv::Scalar(255), -1);


        int count=0;
        for (cv::MatIterator_<uint8_t> it = areaMat.begin<uint8_t>(); it != areaMat.end<uint8_t>(); ++it)
            if( *it > 0 ) count++;

        cv::Mat kmeanPoints(count, 1 , CV_32FC2);
        int currentIdx=0;
        for( int x = 0; x < areaMat.rows; x++ ) 
        {
           for( int y = 0; y < areaMat.cols; y++ ) 
           {
                if( areaMat.at<uint8_t>(x,y) > 0 )
                {
                    kmeanPoints.at<cv::Vec2f>(currentIdx++) = cv::Vec2f((float)y,(float)x);
                }
           }
        }

        cv::Mat labels;
        std::vector<cv::Point2f> centers;
        cv::kmeans( kmeanPoints, nbOjectInArea, labels,
                    cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 10, 1.0),
                    3, cv::KMEANS_PP_CENTERS, centers);

        for( int clusterNum=0; clusterNum<nbOjectInArea; clusterNum++)
        {
            cv::Mat currentClusterShape = cv::Mat::zeros(centerZoneMask_closed.rows, centerZoneMask_closed.cols, CV_8UC1);
            for(int  i = 0; i < count; i++ )
            {
                int clusterIdx = labels.at<int>(i);
                if( clusterNum != clusterIdx)
                    continue;

                cv::Point ipt = kmeanPoints.at<cv::Point2f>(i);
                currentClusterShape.at<uint8_t>(ipt) = 255;
            }

            vector<vector<cv::Point> > cluster_contours;
            cv::findContours(currentClusterShape, cluster_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            assert(cluster_contours.size() < 2);
            cv::drawContours(detectedObjectContourMask, cluster_contours, -1, cv::Scalar(0), 2);
        }
    } 


    cv::bitwise_and(centerZoneMask,centerZoneMask, centerZoneMask_separated, detectedObjectContourMask);

}


void centerZoneDetection(cv::Mat &centerZoneMask, cv::Mat &centerZoneImage, cv::Scalar circleColor )
{

    int kernel_close_size= 2;
    cv::Mat kernel_close = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_close_size*2+1, kernel_close_size*2+1));

    cv::Mat centerZoneMask_closed;
    cv::morphologyEx(centerZoneMask, centerZoneMask_closed, cv::MORPH_CLOSE,kernel_close, cv::Point(-1,-1),  2);


    cv::Mat centerZoneMask_separated;
    trySeparateOverlappingElement(centerZoneMask, centerZoneMask_closed, centerZoneMask_separated);

   
    cv::Mat centerZoneMask_opened;
    int kernel_open_size= 3;
    cv::Mat kernel_open = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_open_size*2+1, kernel_open_size*2+1));
    cv::morphologyEx(centerZoneMask_separated, centerZoneMask_opened, cv::MORPH_OPEN,kernel_open, cv::Point(-1,-1),  5);


    cv::Mat sure_bg;
    cv::dilate(centerZoneMask, sure_bg, kernel_open, cv::Point(-1,-1), 3);

    
    cv::Mat centerMaskDist;
    cv::distanceTransform(centerZoneMask_opened, centerMaskDist, cv::DIST_L2, 5);


    double maxVal;     
    cv::minMaxLoc( centerMaskDist, 0, &maxVal, 0, 0 );
    cv::Mat sure_fg;
    cv::threshold(centerMaskDist,sure_fg, 0.5*maxVal,255,cv::THRESH_BINARY);

    // Pour pouvoir afficher le dist transform
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
        double area = contourArea(fg_contours[i]);
        cv::Moments moment = cv::moments(fg_contours[i]);
        cv::Point center(moment.m10 / (moment.m00+1e-5), moment.m01 / (moment.m00+1e-5));
        cv::circle( centerZoneImage, center, 5, circleColor, -1 );
    }

    centerZoneImage.setTo( cv::Scalar(255, 255, 255), markers==-1);
}



