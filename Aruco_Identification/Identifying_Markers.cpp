#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/core/cvstd.hpp>

//Following along here: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html

void generate_int_png(const int& num, cv::aruco::Dictionary& D) {
//Generates a single PNG of the choosen marker
    cv::Mat markerImage;
    cv::aruco::generateImageMarker(D, num, 200, markerImage, 1);
    cv::imwrite("marker23.png", markerImage);
}

void cameraCalibration() {
    //https://learnopencv.com/camera-calibration-using-opencv/
    /*
        Steps to calibration:
        - Define a real world coordinate of 3D points using checkerboard pattern of known size
        - Capture the images of the checkerboard from different viewpoints
        - Use findChessboardCorners method in OpenCV to find the pixel coordinates (u, v) for each 3D point in different images
        - Find camera parameters using calibrateCamera method in OpenCV, the 3D points, and the pixel coordinates
    */
    int CHECKERBOARD[2] {6,9};   //Stating the size of the checkerboard
    std::vector<std::vector< cv::Point3f> > objPoints;    //Creating a vector that will store the 3D points of each checkerboard image
    std::vector<std::vector<cv::Point2f> > imgPoints; //Creating a vector to store vectors of 2D points for each checkerboard image

    //Defining the world coordinates for 3D points
    std::vector<cv::Point3f> obj_P; //Creating an empty vector that stores tuples of (x, y, z)
    for(int i= 0; i < CHECKERBOARD[1]; i++) {   //For every square (x2) on the checkerboard create a point (i, j, 0)
        for(int j= 0; j < CHECKERBOARD[0]; j++) {
            obj_P.push_back(cv::Point3f(j, i, 0));
            //{(0,0,0), (1,0,0), (2,0,0)... (5,0,0), (0,1,0), (1,1,0)... (5,5,0)}
        }
    }
    
    //Individual path of images
    std::vector<cv::String> images; //Empty vector used to store the paths to jpg's
    //Path of folder containing said images
    std::string path = "./calibrationImages/*.jpg";
    cv::glob(path, images); //Uses the path to collect the directory of every .jpg in the ./images folder. Storing the directory path in the vector images

    cv::Mat frame, gray;    //Creates two matrices one called "frame" (used to store the image) and one called "gray" (to store a gray scaled copy of the image)

    std::vector<cv::Point2f> corner_pts;    //Where the corners points are stored in a 2D format
    bool success;   //Bool variable that tracks the success of finding corners

    for(int i= 0; i < images.size(); i++) { //Itterates through every image found in the path above
        frame = cv::imread(images[i]);  //Stores the image as a matrix in frame
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);  //Gray scales the image (original, gray scale copy (output), gray scale)

        //Marked as true IF findChessboardCorners finds corners in the gray scaled image (should be 6x2 total (12)). Considered complete when adaptive thresh, fast check, or normalization completes
        success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        if(success) {   //If corners were found
            //Create a criteria that is considered complete at either 30 iterations or when accuracy reaches <=0.001 degree of difference
            cv::TermCriteria criteria (cv::TermCriteria::Type::EPS | cv::TermCriteria::Type::MAX_ITER, 30, 0.001);
            //Find the corners of the gray scaled image using the corner_pts vector (of tuples) and the size of the checkerboard. Completes when one of the criterias are met
            cv::cornerSubPix(gray, corner_pts, cv::Size(11,11), cv::Size(-1,-1), criteria);
            //Draws lines over the original image, marking the corner points found above
            cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
            objPoints.push_back(obj_P); //Keeps track of the 3D coordinates of each point on the image
            imgPoints.push_back(corner_pts);    //Keeps track of the 2D coordinates of each point on the image
        }
        //cv::imshow("Image", frame); //Show the image generated with the corners and lines, naming the window "Image"
        //cv::waitKey(0); //Not continuing until a key is pressed
    }
    cv::destroyAllWindows();    //Destroys all windows if they weren't closed when the waitKey was pressed
    //Creates empty matrices that will store the cameras intrinsic parameters (focal length, lens distortion, etc.), 
    //  the distortion coeffecients (describing lens distortion), rotation vectors for each image, and translation vectors for each image
    cv::Mat cameraMatrix, distCoeffs, R, T;
    //Calibrates the camera, returning the following alterations required to remove lense distortion effects
    cv::calibrateCamera(objPoints, imgPoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);
}

void markerDetection() {
    //https://stackoverflow.com/questions/76365598/having-problems-with-opencv-4-7-and-aruco-module
    const cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::VideoCapture inputVideo(0);
    cv::Mat inputImage;
    while(true) {
        inputVideo >> inputImage;
        std::vector<std::vector<cv::Point2f> > markerCorners;
        std::vector<int> markerIds;
        //cv::aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds);
        cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
        cv::aruco::ArucoDetector detector(dictionary, detectorParams);
        detector.detectMarkers(inputImage, markerCorners, markerIds);

        if(markerIds.size() > 0) {
            cv::aruco::drawDetectedMarkers(inputImage, markerCorners, markerIds);
        }
        cv::imshow("Image", inputImage);
        cv::waitKey(1);
    }
}

int main() {
    //https://learnopencv.com/geometry-of-image-formation/
    //cameraCalibration();
    markerDetection();

    
    return 0;
}
