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
cv::Scalar ColorRed (0, 0, 255);


void generateFisheyeUndistordMap(cv::Mat &map1, cv::Mat &map2)
{
    cv::FileStorage file_kd("fisheye_KD", cv::FileStorage::READ);
    file_kd["K_mat"] >> K;
    file_kd["D_mat"] >> D;
    file_kd.release();

    cout << "K=>" << K << endl;

    cv::Size img_size(1920, 1080);
    cv::fisheye::initUndistortRectifyMap(K, D, cv::Mat::eye(3, 3, CV_32F), K, img_size, CV_16SC2, map1, map2);
}

cv::Point positionOnTableFromPointInImage(cv::Point &pointInImage, cv::Mat &cameraMatrix, cv::Mat &rotationMatrix, cv::Mat &tvec)
{
    cv::Mat uvPoint = (cv::Mat_<double>(3,1) << pointInImage.x, pointInImage.y, 1);
    cv::Mat leftSideMat  = rotationMatrix.inv() * cameraMatrix.inv() * uvPoint;
    cv::Mat rightSideMat = rotationMatrix.inv() * tvec;

    double s = rightSideMat.at<double>(2,0)/leftSideMat.at<double>(2,0); 
    cv::Mat positionIn3d = rotationMatrix.inv() * (s * cameraMatrix.inv() * uvPoint - tvec);

    cv::Point positionOnTable(positionIn3d.at<double>(0), positionIn3d.at<double>(1));
    return positionOnTable;
}


bool detectArucoAndComputeRotVecMatrixes(cv::Mat const &photo_undistorted, cv::Mat const  &K, cv::Mat const  &D, cv::Mat &rvec, cv::Mat &tvec )
{
    vector<int> markerIds;
    vector<vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
    cv::aruco::detectMarkers(photo_undistorted, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

    cout << " markerIds.size()" << markerIds.size() << endl;

    // if at least one marker detected
    if (markerIds.size() > 0)
        cv::aruco::drawDetectedMarkers(photo_undistorted, markerCorners, markerIds);
    else
        return false;

   // Position of the corner of the aruco tag on the eurobot table
    vector<cv::Point3f> objectPosition;
    vector<cv::Point2f> objetImagePosition;

    for (int marker_id = 0 ; marker_id < markerIds.size(); marker_id++)
    {
        cout << " markerCorners=>" << markerCorners[marker_id] << endl;

        if( markerIds[marker_id] == 42)
        {
            objectPosition.push_back(cv::Point3f(1550.0, 1300.0, 0.0));
            objetImagePosition.push_back(markerCorners[marker_id][0]);
            
            objectPosition.push_back(cv::Point3f(1450.0, 1300.0, 0.0));
            objetImagePosition.push_back(markerCorners[marker_id][1]);

            objectPosition.push_back(cv::Point3f(1450.0, 1200.0, 0.0));
            objetImagePosition.push_back(markerCorners[marker_id][2]);

            objectPosition.push_back(cv::Point3f(1550.0, 1200.0, 0.0));
            objetImagePosition.push_back(markerCorners[marker_id][3]);
        }
        else if( markerIds[marker_id] == 69)
        {
            // TODO !
            // objectPosition.push_back(cv::Point3f(1550.0, 1300.0, 0.0));
            // objetImagePosition.push_back(markerCorners[marker_id][0]);
            
            // objectPosition.push_back(cv::Point3f(1450.0, 1300.0, 0.0));
            // objetImagePosition.push_back(markerCorners[marker_id][1]);

            // objectPosition.push_back(cv::Point3f(1450.0, 1200.0, 0.0));
            // objetImagePosition.push_back(markerCorners[marker_id][2]);

            // objectPosition.push_back(cv::Point3f(1550.0, 1200.0, 0.0));
            // objetImagePosition.push_back(markerCorners[marker_id][3]);
        }
    }

    cout << " Injected point in PnP solve " << endl;
    for(int i=0 ; i<objectPosition.size(); i++)
    {
        cout << "corner:" <<objetImagePosition[i] << " <=> img:" << objectPosition[i] << endl;
    }

    cv::Mat emptyDist;
    if( objectPosition.size() > 0)
        return cv::solvePnP(objectPosition, markerCorners[0], K, emptyDist, rvec, tvec);
    
    return false;
}


cv::Rect2d localizeZone(cv::Mat const &K, cv::Mat const &D, cv::Mat const &rvec, cv::Mat const &tvec,
                        float TL_x, float TL_y, float BR_x, float BR_y)
{
    vector<cv::Point3f> objectPoint;
    objectPoint.push_back(cv::Point3f(TL_x, TL_y, 0.0)); //TL
    objectPoint.push_back(cv::Point3f(BR_x, BR_y, 0.0)); // BR

     vector<cv::Point2f> imagePoint;
    cv::projectPoints(objectPoint, rvec, tvec, K, D, imagePoint);       
    cout << "objectPoint = " << objectPoint << endl;
    cout << "imagePoint = " << imagePoint << endl;

    return cv::Rect2d(imagePoint[0].x, imagePoint[0].y, imagePoint[1].x-imagePoint[0].x, imagePoint[1].y-imagePoint[0].y);
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


    raspicam::RaspiCam_Still_Cv Camera;
    cv::Mat photo, photo_undistorted;
    cv::Mat rotationMatrix(3,3,cv::DataType<double>::type);
    
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



    // grab a another image to compute area zone
    Camera.grab();
    Camera.retrieve ( photo);
    cv::remap(photo, photo_undistorted, fisheye_map1, fisheye_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    
    cv::Mat rvec;
    cv::Mat tvec;
    bool detection_ok = detectArucoAndComputeRotVecMatrixes(photo_undistorted, K, D, rvec, tvec);
    if( detection_ok)
    {
        cv::Rodrigues(rvec,rotationMatrix);
        cv::Point pointInImage(1074,626);
        cv::Point positionOnTable = positionOnTableFromPointInImage(pointInImage, K, rotationMatrix, tvec);
            std::cout << "pointInImage= " << pointInImage <<  " <=>  positionOnTable=" << positionOnTable << std::endl;


        cv::Mat emptyDist;
        cv::Rect2d dockRight = localizeZone(K, D, rvec, tvec, 1060.0, 20.0, 620.0, -150.0);

        cout << "dockRight" << dockRight << endl;

        cv::rectangle( photo_undistorted, dockRight.tl(), dockRight.br(), ColorWhite, 1);
        // cv::rectangle( photo_undistorted, cropZoneRight.tl(), cropZoneRight.br(), ColorRed, 1);
    }

    cv::namedWindow("photo_undistorted",cv::WINDOW_NORMAL|cv::WINDOW_KEEPRATIO);     
    cv::imshow( "photo_undistorted", photo_undistorted );
    cv::waitKey(0);

    Camera.release();
 return 0;
}
