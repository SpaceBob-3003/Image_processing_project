#include "Processing.h"

int main()
{
    Processing proc;
    cv::Mat img = proc.readImage("../image/20251015_142145.jpg");
    cv::Mat imgHsv;
    cv::Mat H;
    cv::Mat S;
    cv::Mat V;
    cv::Mat imgB;

    cv::GaussianBlur(img, imgB, cv::Size(35,35), 0);

    proc.bgrToHsv(imgB, imgHsv, H, S, V);

    cv::Mat imgThresh;
    proc.thresholding(S, imgThresh, 110);

    cv::Mat kernel = cv::Mat::ones(35, 35, CV_8U);
    cv::Mat kernel2 = cv::Mat::ones(31, 31, CV_8U);

    cv::morphologyEx(imgThresh, imgThresh, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(imgThresh, imgThresh, cv::MORPH_CLOSE, kernel2);

    

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(imgThresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> all_points;
    for (const auto& c : contours){
        all_points.insert(all_points.end(), c.begin(), c.end());
    }

    cv::RotatedRect minRect = cv::minAreaRect(all_points);

    cv::Point2f rectPoints[4];
    minRect.points(rectPoints);

    std::vector<cv::Point> rectContour;
    for (int j = 0; j < 4; j++) {
        rectContour.push_back(cv::Point(cvRound(rectPoints[j].x), cvRound(rectPoints[j].y)));
    }

    drawContours(img, std::vector<std::vector<cv::Point>>{rectContour}, 0, cv::Scalar(0, 255, 0), 2);


    std::cout << "Number of contours found: " << contours.size() << std::endl;

    //drawing of contours
    cv::Mat imgContours = cv::Mat::zeros(img.size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); i++)
    {
        cv::Scalar color = cv::Scalar(0, 255, 0); // Green color for contours
        cv::drawContours(imgContours, contours, static_cast<int>(i), color, 2);
    }

    proc.displayImage(imgThresh, "Image seuillée", cv::WINDOW_AUTOSIZE, 4);

    proc.displayImage(imgB, "Image floutée", cv::WINDOW_AUTOSIZE, 4);

    proc.displayImage(imgContours, "Contours", cv::WINDOW_AUTOSIZE, 4);

    proc.displayImage(img, "Image avec BoundingBox", cv::WINDOW_AUTOSIZE, 4);


    cv::waitKey(0);
    return 0;
}