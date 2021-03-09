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
cv::Scalar ColorWhite(255, 255, 255, 0);

static void dockZoneDetection(bool isRightZone ,cv::Mat &redMask, cv::Mat & greenMask, cv::Rect &boundRect,  cv::Mat &zone);
static void trySeparateOverlappingElement(cv::Mat &centerZoneMask, cv::Mat &centerZoneMask_closed);
static void centerZoneDetection(cv::Mat &centerZoneMask, cv::Mat &centerZoneImage, cv::Scalar circleColor );

const int32_t aruco_table_pos_x = 1500;
const int32_t aruco_table_pos_y = 1250;


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
    root["left_roi"].lookupValue("x", x);
    root["left_roi"].lookupValue("y", y);
    root["left_roi"].lookupValue("width", width);
    root["left_roi"].lookupValue("height", height);
    cv::Rect2d cropZoneLeft(x, y, width,height);
    
    root["right_roi"].lookupValue("x", x);
    root["right_roi"].lookupValue("y", y);
    root["right_roi"].lookupValue("width", width);
    root["right_roi"].lookupValue("height", height);
    cv::Rect2d cropZoneRight(x, y, width,height);

    root["center_roi"].lookupValue("x", x);
    root["center_roi"].lookupValue("y", y);
    root["center_roi"].lookupValue("width", width);
    root["center_roi"].lookupValue("height", height);
    cv::Rect2d cropCenterZone(x, y, width,height);

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
    

    while(1)
    {
 
        clock_t start = clock();
        Camera.grab();
        Camera.retrieve ( photo);


      
        cv::imshow( "photo", photo );
        cv::waitKey(0);
        
        cout<< "capture duration = " <<  float(clock()-start)/CLOCKS_PER_SEC <<endl;
       


        start = clock();
        cv::remap(photo, photo_undistorted, fisheye_map1, fisheye_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

        cout<< "fisheye compensation duration = " <<  float(clock()-start)/CLOCKS_PER_SEC <<endl;
        start = clock(); 


       
        vector<int> markerIds;
        vector<vector<cv::Point2f>> markerCorners, rejectedCandidates;
        cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
        cv::aruco::detectMarkers(photo_undistorted, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

        cout << "markerIds.size()" << markerIds.size() << endl;

    // if at least one marker detected
        if (markerIds.size() > 0)
        {
            cout << "markerIds[0]" << markerIds[0] << endl;
            cv::aruco::drawDetectedMarkers(photo_undistorted, markerCorners, markerIds);
         
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, K, D, rvecs, tvecs);   

            for(int i=0; i<markerIds.size(); i++)
            {
                cv::aruco::drawAxis(photo_undistorted, K, D, rvecs[i], tvecs[i], 0.1);
                cout<< "rvecs" <<  rvecs[i][0] << " " << rvecs[i][1] << " " << rvecs[i][2] <<endl;
                cout<< "tvecs" <<  tvecs[i][0] << " " << tvecs[i][1] << " " << tvecs[i][2] <<endl;

            }
        }
        cout<< "aruco detect duration = " <<  float(clock()-start)/CLOCKS_PER_SEC <<endl;
    

    // cv::imwrite("photo_undistorted.jpg", photo_undistorted);
        // cv::Rect2d cropZone = cv::selectROI(photo);
        //  cout<<"selection " << cropZone <<endl; 

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
        cv::rectangle( photo_undistorted, cropZoneLeft.tl(), cropZoneLeft.br(), ColorWhite, 1);
        cv::rectangle( photo_undistorted, cropZoneRight.tl(), cropZoneRight.br(), ColorWhite, 1);

        cout<< "compute duration = " <<  float(clock()-start)/CLOCKS_PER_SEC <<endl;
      
        cv::imshow( "photo_undistorted", photo_undistorted );
        cv::waitKey(0);
    }

    cout<<"Stop camera..."<<endl;
    Camera.release();
   
 return 0;
}



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


static void centerZoneDetection(cv::Mat &centerZoneMask, cv::Mat &centerZoneImage, cv::Scalar circleColor )
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


        cv::rectangle(   zone, tl, br, Colorblack);
    }
}