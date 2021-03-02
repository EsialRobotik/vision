
static void tryToSeparateMore(cv::Mat &centerZoneMask, cv::Mat &centerZoneImage, cv::Mat &centerZoneMask_separated)
{
    // Transform to gray scale
    cv::Mat centerZoneImage_gray;
    cv::cvtColor( centerZoneImage, centerZoneImage_gray, cv::COLOR_BGR2GRAY);

    // Use bilateralFilter to remove noise while keeping sharpness
    cv::Mat centerZoneImage_gray_blur;
    cv::bilateralFilter( centerZoneImage_gray, centerZoneImage_gray_blur, 5, 50, 50);

    // Apply laplacian to detect edge
    cv::Mat centerZoneImage_lap;
    cv::Laplacian(centerZoneImage_gray_blur, centerZoneImage_lap, CV_32F, 5);
    cv::convertScaleAbs(centerZoneImage_lap, centerZoneImage_lap);


    // Just keep interesting part of the image
    cv::Mat centerZoneImage_lap_masked;
    cv::bitwise_and(centerZoneImage_lap,centerZoneImage_lap, centerZoneImage_lap_masked, centerZoneMask);

    // Apply thresholding to keep edges 
    cv::Mat centerZoneImage_mask;    
    cv::threshold(centerZoneImage_lap_masked, centerZoneImage_mask, 0.7*255, 255,cv::THRESH_BINARY_INV);

    // Apply the computed mask to the source mask
    cv::bitwise_and(centerZoneMask,centerZoneMask, centerZoneMask_separated, centerZoneImage_mask);
}

static void lineDetectionToSeparate(cv::Mat &centerZoneMask, cv::Mat &centerZoneImage, cv::Mat &centerZoneMask_separated)
{
    // Transform to gray scale
    cv::Mat centerZoneImage_gray;
    cv::cvtColor( centerZoneImage, centerZoneImage_gray, cv::COLOR_BGR2GRAY);
  
    // Blur, then apply canny to detect edge
    cv::Mat centerZoneImage_gray_blurred;
    cv::bilateralFilter( centerZoneImage_gray, centerZoneImage_gray_blurred, 5, 50, 50);
    int edgeThresh = 20;
    cv::Mat centerZoneImage_canny;
    cv::Canny( centerZoneImage_gray_blurred, centerZoneImage_canny, edgeThresh, edgeThresh*3, 3 );

    cv::Mat centerZoneImage_canny_masked;
    cv::bitwise_and(centerZoneImage_canny,centerZoneImage_canny, centerZoneImage_canny_masked, centerZoneMask);


    cv::imshow( "centerZoneImage_canny", centerZoneImage_canny );
    cv::imshow( "centerZoneImage_canny_masked", centerZoneImage_canny_masked );
    cv::imshow( "centerZoneImage_gray_blurred", centerZoneImage_gray_blurred );
    cv::imshow( "centerZoneMask", centerZoneMask );
    cv::waitKey(0);
}
