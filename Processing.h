#ifndef PROCESSING_H
#define PROCESSING_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>

class Processing
{
    public:
        Processing();

        cv::Mat readImage(std::string path);
        void displayImage(cv::Mat imgI, std::string nom, int window, int factor=1);
        void bgrToHsv(cv::Mat imgI, cv::Mat &imgO, cv::Mat &H, cv::Mat &S, cv::Mat &V);
        void thresholding(cv::Mat imgI, cv::Mat &imgO, int thresh);

    private:

};
#endif