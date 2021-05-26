#include <opencv2/opencv.hpp>

cv::Point positionOnTableFromPointInImage(cv::Point &pointInImage, cv::Mat &cameraMatrix, cv::Mat &rotationMatrix, cv::Mat &tvec);

bool detectArucoAndComputeRotVecMatrixes(cv::Mat const &photo_undistorted, cv::Mat const  &K, cv::Mat const  &D, 
	cv::Mat &rvec, cv::Mat &tvec, cv::Mat &rotationMatrix );

cv::Rect2d localizeZone(cv::Mat const &K,cv::Mat const &D, cv::Mat const &rvec, cv::Mat const &tvec,
    float TL_x, float TL_y, float BR_x, float BR_y);