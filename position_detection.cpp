#include "position_detection.hpp"
#include <ctime>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <unistd.h>
#include <cstdint>
#include <cmath> 
#include <assert.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <pigpiod_if2.h>


using namespace std; 


extern int pigpioID;
extern int GPIO_aruco_42;
extern int GPIO_aruco_51;
extern int GPIO_aruco_69;

cv::Point positionOnTableFromPointInImage(cv::Point &pointInImage, cv::Mat &cameraMatrix, cv::Mat &rotationMatrix, cv::Mat &tvec, int up_offset )
{
    cv::Mat uvPoint = (cv::Mat_<double>(3,1) << pointInImage.x, pointInImage.y, 1);
    cv::Mat leftSideMat  = rotationMatrix.inv() * cameraMatrix.inv() * uvPoint;
    cv::Mat rightSideMat = rotationMatrix.inv() * tvec;

    double s = up_offset  + rightSideMat.at<double>(2,0)/leftSideMat.at<double>(2,0); 
    cv::Mat positionIn3d = rotationMatrix.inv() * (s * cameraMatrix.inv() * uvPoint - tvec);

    cv::Point positionOnTable(positionIn3d.at<double>(0), positionIn3d.at<double>(1));
    return positionOnTable;
}


bool detectArucoAndComputeRotVecMatrixes(cv::Mat const &photo_undistorted, cv::Mat const  &K, cv::Mat const  &D, cv::Mat &rvec, cv::Mat &tvec, cv::Mat &rotationMatrix)
{
    vector<int> markerIds;
    vector<vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
    cv::aruco::detectMarkers(photo_undistorted, dictionary, markerCorners, markerIds, parameters, rejectedCandidates, K, D);


    // Reset display
    gpio_write(pigpioID, GPIO_aruco_42, 0);
    gpio_write(pigpioID, GPIO_aruco_51, 0);
    gpio_write(pigpioID, GPIO_aruco_69, 0);

    // if at least one marker detected
    if (markerIds.size() > 0)
    {
#if 0
       cv::aruco::drawDetectedMarkers(photo_undistorted, markerCorners, markerIds);
#endif
    }
    else
    {
        return false;
    }


   // Position of the corner of the aruco tag on the eurobot table
    vector<cv::Point3f> objectPosition;
    vector<cv::Point2f> objetImagePosition;

    for (int marker_id = 0 ; marker_id < markerIds.size(); marker_id++)
    {
        if( markerIds[marker_id] == 42)
        {
            cout << "Aruco 42 detected " << endl;
            gpio_write(pigpioID, GPIO_aruco_42, 1);

            objectPosition.push_back(cv::Point3f(1200.0, 1450.0, 0.0));
            objetImagePosition.push_back(markerCorners[marker_id][0]);
            
            objectPosition.push_back(cv::Point3f(1200.0, 1550.0, 0.0));
            objetImagePosition.push_back(markerCorners[marker_id][1]);

            objectPosition.push_back(cv::Point3f(1300.0, 1550.0, 0.0));
            objetImagePosition.push_back(markerCorners[marker_id][2]);

            objectPosition.push_back(cv::Point3f(1300.0, 1450.0, 0.0));
            objetImagePosition.push_back(markerCorners[marker_id][3]);
        }
        else if( markerIds[marker_id] == 69)
        {
            cout << "Aruco 69 detected " << endl;
            gpio_write(pigpioID, GPIO_aruco_69, 1);

            objectPosition.push_back(cv::Point3f(273.0, 834.0-18.0, 0.0));
            objetImagePosition.push_back(markerCorners[marker_id][0]);
            
            objectPosition.push_back(cv::Point3f(273.0, 834.0-201.0, 0.0));
            objetImagePosition.push_back(markerCorners[marker_id][1]);

            objectPosition.push_back(cv::Point3f(91.0, 834.0-201.0, 0.0));
            objetImagePosition.push_back(markerCorners[marker_id][2]);

            objectPosition.push_back(cv::Point3f(91.0, 834.0-18.0, 0.0));
            objetImagePosition.push_back(markerCorners[marker_id][3]);

        }
        else if (markerIds[marker_id] == 51 )
        {
            cout << "Aruco 51 detected " << endl;
            gpio_write(pigpioID, GPIO_aruco_51, 1);

            objectPosition.push_back(cv::Point3f(18.0, 1462.0, 0.0));
            objetImagePosition.push_back(markerCorners[marker_id][0]);
            
            objectPosition.push_back(cv::Point3f(200.0, 1462.0, 0.0));
            objetImagePosition.push_back(markerCorners[marker_id][1]);

            objectPosition.push_back(cv::Point3f(200.0, 1280.0, 0.0));
            objetImagePosition.push_back(markerCorners[marker_id][2]);

            objectPosition.push_back(cv::Point3f(18.0, 1280.0, 0.0));
            objetImagePosition.push_back(markerCorners[marker_id][3]);

        }
    }

#if 1
    cout << " Injected point in PnP solve " << endl;
    for(int i=0 ; i<objectPosition.size(); i++)
    {
        cout << "corner:" <<objetImagePosition[i] << " <=> img:" << objectPosition[i] << endl;
    }
#endif

    cv::Mat emptyDist;
    if( objectPosition.size() > 0)
    {
        bool res = cv::solvePnP(objectPosition, objetImagePosition, K, emptyDist, rvec, tvec, false);
        if(res)
             cv::Rodrigues(rvec,rotationMatrix);
         return res;
    }
    else
    {
        return false;
    }
}


cv::Rect2d localizeZone(cv::Mat const &K, cv::Mat const &D, cv::Mat const &rvec, cv::Mat const &tvec,
                        float TL_x, float TL_y, float BR_x, float BR_y)
{
    vector<cv::Point3f> objectPoint;
    objectPoint.push_back(cv::Point3f(TL_x, TL_y, 0.0)); //TL
    objectPoint.push_back(cv::Point3f(BR_x, BR_y, 0.0)); // BR

    vector<cv::Point2f> imagePoint;
    cv::Mat emptyDist;

    cv::projectPoints(objectPoint, rvec, tvec, K, emptyDist, imagePoint);       
  
    return cv::Rect2d(imagePoint[0].x, imagePoint[0].y, imagePoint[1].x-imagePoint[0].x, imagePoint[1].y-imagePoint[0].y);
}


t_trapezium localizeTrapezium(cv::Mat const &K, cv::Mat const &D, cv::Mat const &rvec, cv::Mat const &tvec, t_trapezium & trapezium_in_table, int up_offset )
{
    vector<cv::Point3f> objectPoint;
    objectPoint.push_back(cv::Point3f(trapezium_in_table.top_left.x, trapezium_in_table.top_left.y, up_offset));
    objectPoint.push_back(cv::Point3f(trapezium_in_table.top_right.x, trapezium_in_table.top_right.y, up_offset));
    objectPoint.push_back(cv::Point3f(trapezium_in_table.bottom_right.x, trapezium_in_table.bottom_right.y, up_offset));
    objectPoint.push_back(cv::Point3f(trapezium_in_table.bottom_left.x, trapezium_in_table.bottom_left.y, up_offset));

    vector<cv::Point2f> imagePoint;
    cv::Mat emptyDist;

    cv::projectPoints(objectPoint, rvec, tvec, K, emptyDist, imagePoint);

    t_trapezium trapezium_in_image;
    trapezium_in_image.top_left = cv::Point(imagePoint[0].x, imagePoint[0].y);
    trapezium_in_image.top_right = cv::Point(imagePoint[1].x, imagePoint[1].y);
    trapezium_in_image.bottom_right = cv::Point(imagePoint[2].x, imagePoint[2].y);
    trapezium_in_image.bottom_left = cv::Point(imagePoint[3].x, imagePoint[3].y);

    return trapezium_in_image;
}

