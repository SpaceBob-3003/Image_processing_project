#include "Processing.h"

Processing::Processing(){}

cv::Mat Processing::readImage(std::string path)
{
    cv::Mat image = cv::imread(path);
    if (image.empty()) 
    {
    std::cerr << "Erreur : Impossible de charger l'image !" << std::endl;
    }
    return image;
}

void Processing::bgrToHsv(cv::Mat imgI, cv::Mat& imgO, cv::Mat& H, cv::Mat& S, cv::Mat& V)
{
    std::vector<cv::Mat> hsv_planes;
    cv::cvtColor(imgI, imgO, cv::COLOR_BGR2HSV);
    cv::split(imgO, hsv_planes);
    H = hsv_planes[0];
    S = hsv_planes[1];
    V = hsv_planes[2];
}

void Processing::displayImage(cv::Mat imgI, std::string nom, int window, int factor)
{
    cv::Mat imgr;
    cv::resize(imgI, imgr, cv::Size(static_cast<int>(imgI.cols / factor), static_cast<int>(imgI.rows / factor)));
    cv::namedWindow(nom, window);
    cv::imshow(nom, imgr);
}

void Processing::thresholding(cv::Mat imgI, cv::Mat& imgO, int thresh)
{
    cv::threshold(imgI, imgO, thresh, 255, cv::THRESH_BINARY);
}