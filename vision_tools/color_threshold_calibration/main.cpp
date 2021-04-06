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

cv::Mat photo_main;
void start_green_calib(cv::Mat* captured_photo );
void start_red_calib(cv::Mat* captured_photo );

const string keys =
{
    "{help h usage ?| | color calibration.  >> One color must be choosen << }"
    "{red          | | red calibration}"
    "{green        | | green calibration}"
};

int main ( int argc,char **argv ) 
{
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("color calibration util: \n\t ## One color must be choosen ##");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 1;
    }
    if( ! (parser.has("red") xor parser.has("green")))
    {
        cout << "Warning: ONE color must be choosen" << endl;
        parser.printMessage();
        return 2;   
    }

    time_t timer_begin,timer_end;
    raspicam::RaspiCam_Still_Cv Camera;
    
    
    Camera.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
    Camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    if (!Camera.open()) 
    {
        cerr<<"Error opening the camera"<<endl;
        return -1;
    }

    cout<<"Warm up... "<<endl; 
    sleep(3);    
    cout<<"Capturing " <<endl;
    

 
    Camera.grab();
    Camera.retrieve ( photo_main);
    
    if( parser.has("green"))
    {
        start_green_calib(&photo_main);
    }
    else
    {
        start_red_calib(&photo_main);
    }

    
   Camera.release();
 return 0;
}
