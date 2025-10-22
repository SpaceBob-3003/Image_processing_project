#include "Processing.h"

int main()
{
    Processing proc;
    cv::Mat img = proc.readImage("../image/20251015_142012.jpg");
    if (img.empty()) {
        std::cerr << "Error: Image failed to load. Check the path '../image/20251015_141945.jpg'." << std::endl;
        return 1;
    }
    cv::Mat imgHsv;
    cv::Mat H;
    cv::Mat S;
    cv::Mat V;
    cv::Mat imgB;

    cv::GaussianBlur(img, imgB, cv::Size(21,21), 0);

    proc.bgrToHsv(imgB, imgHsv, H, S, V);

    cv::Mat imgThresh;
    proc.thresholding(S, imgThresh, 125);

    cv::Mat kernel = cv::Mat::ones(35, 35, CV_8U);
    cv::Mat kernel2 = cv::Mat::ones(31, 31, CV_8U);

    cv::morphologyEx(imgThresh, imgThresh, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(imgThresh, imgThresh, cv::MORPH_CLOSE, kernel2);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(imgThresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        std::cerr << "Warning: No contours found after thresholding. Adjust your threshold or preprocessing." << std::endl;
        proc.displayImage(imgThresh, "Image seuillée", cv::WINDOW_AUTOSIZE, 4);
        proc.displayImage(imgB, "Image floutée", cv::WINDOW_AUTOSIZE, 4);
        cv::waitKey(0);
        return 0;
    }

    // Gather all points as Point2f to ensure correct depth for downstream ops
    std::vector<cv::Point2f> all_points;
    all_points.reserve(1024);
    for (const auto& c : contours){
        for (const auto& p : c) {
            all_points.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
        }
    }

    if (all_points.empty()) {
        std::cerr << "Warning: Accumulated points are empty; cannot compute minAreaRect." << std::endl;
        proc.displayImage(imgThresh, "Image seuillée", cv::WINDOW_AUTOSIZE, 4);
        proc.displayImage(imgB, "Image floutée", cv::WINDOW_AUTOSIZE, 4);
        cv::waitKey(0);
        return 0;
    }

    cv::RotatedRect minRect = cv::minAreaRect(all_points);

    cv::Point2f rectPoints[4];
    minRect.points(rectPoints);


    // Get the rotation angle and adjust if needed
    float angle = minRect.angle;
    if (minRect.size.width < minRect.size.height) {
        angle += 90.0;
    }

    // Get the rotation matrix
    cv::Point2f center = minRect.center;
    cv::Mat rotMatrix = cv::getRotationMatrix2D(center, angle, 1.0);

    // Rotate the original image
    cv::Mat imgRotated;
    cv::warpAffine(img, imgRotated, rotMatrix, img.size(), cv::INTER_LINEAR);

    // Calculate the new bounding rectangle size and make it 50% larger
    cv::Size rectSize = minRect.size;
    if (minRect.size.width < minRect.size.height) {
        std::swap(rectSize.width, rectSize.height);
    }
    
    // Increase size by 50%
    rectSize.width *= 1.5;
    rectSize.height *= 1.5;

    // Extract the rotated rectangle region
    cv::Mat imgCropped;
    cv::getRectSubPix(imgRotated, rectSize, center, imgCropped);

    proc.displayImage(imgCropped, "Image Recadrée", cv::WINDOW_AUTOSIZE, 4);

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
