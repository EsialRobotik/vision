#include <opencv2/opencv.hpp>

typedef enum
{
	cup_red,
	cup_green,
	cup_unknown
} t_cup;

void dockZoneDetection(bool isRightZone ,cv::Mat &redMask, cv::Mat & greenMask, cv::Rect &boundRect,  cv::Mat &zone, std::vector<t_cup> &cupList);