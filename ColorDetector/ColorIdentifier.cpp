#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tuple>
#include <map>

/*
Algorithm:
    - Read in frames from web cam
    - Transform it into a matrix that holds the HSV
    - Create a range of the hue's to determine each color/ shade
    - Remove noise from the image (necessary?)
    - Detect color in image mask
    - Create display for each color
    - Output the detected colors overlayed on the image mask
*/

std::vector<cv::Mat> cameraCalibration() {
    //https://learnopencv.com/camera-calibration-using-opencv/
    //https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
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
    cv::Mat cameraMatrix = (cv::Mat1d(3,3)<<1,0,0,0,1,0,0,0,1);
    cv::Mat distCoeffs, R, T;
    //Calibrates the camera, returning the following alterations required to remove lense distortion effects
    cv::calibrateCamera(objPoints, imgPoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);

    return std::vector<cv::Mat> {cameraMatrix, distCoeffs, R, T};
    
}

void colorDetection(std::vector<cv::Mat> cm_Calib) {
    cv::VideoCapture inputVideo(0);
    if(!inputVideo.isOpened()) {
        std::cout << "Error: Could not open camera." << std::endl; throw std::exception();
    }
    cv::Mat inputImage;
    while(true) {
        //Pulled from the table here: https://stackoverflow.com/questions/12357732/hsv-color-ranges-table
        //Formated as degree/2 (openCV range of H is 0-179), percent, percent
        //std::string colors_string [16] = {"Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Purple", "Magenta/ Pink", "Black", "White"};
        //std::tuple<double, double, double> colors_HSV [16] = {{0,0,1}, {0,0,0.75}, {0,0,0.50}, {0,0,0}, {0,1,1}, {0,1,0.50}, {30,1,1}, {30,1,0.50}, {60,1,1}, {60,1,0.50}, {90,1,0.50}, {120,1,1}, {120,1,0.50}, {150,1,0.50}, {150,1,1}, {90,1,1}};

        
        //Red is added twice due to the circular nature of HSV (0-10 and 350-360)
        //https://stackoverflow.com/questions/36817133/identifying-the-range-of-a-color-in-hsv-using-opencv
        std::string colors_string [10] = {"Black", "White", "Red", "Red", "Green", "Blue", "Yellow", "Purple", "Orange", "Gray"};
        std::tuple<double, double, double> colors_HSV_min [10] = {{0,0,0}, {0,0,231}, {159,50,70}, {0,50,70}, {36,50,70}, {90,50,70}, {25,50,70}, {129,50,70}, {10,50,70}, {0,0,40}};
        std::tuple<double, double, double> colors_HSV_max [10] = {{180, 255,30}, {180,18,255}, {180,255,255}, {9,255,255}, {89,255,255}, {128,255,255}, {35,255,255}, {158,255,255}, {24,255,255}, {180,18,230}};

        inputVideo >> inputImage;   //Take an input frame from the video source and turn it into a matrix
        cv::Mat outputImage;    //Creating a matrix for the output image
        cv::undistort(inputImage, outputImage, cm_Calib[0], cm_Calib[1]);   //Undistort the image using the cameraMatrix and distCoeffs acquired from cameraCalibration
        cv::Mat HSV_Image;  //Creating a new empty matrix to store the HSV colors
        cv::cvtColor(outputImage, HSV_Image, cv::COLOR_HSV2BGR);   //Change the output image into a matrix of HSV colors

        //std::cout << "Height: " << (outputImage.size()).height << std::endl; //720 / 2 = 360
        //std::cout << "Width: " << (outputImage.size()).width;  //1280 / 2 = 640
        //Middle of the camera is 640,360 so a rectangle to check for the color could be 600,400 to 680,320
        cv::Size outputImage_Size = outputImage.size();
        cv::Point pt1 ((outputImage_Size.width/2)-60, (outputImage_Size.height/2)+60);   //Top left point of the rectangle
        cv::Point pt2 ((outputImage_Size.width/2)+60, (outputImage_Size.height/2)-60);   //Bottom right point of the rectangle
        cv::Mat center_rectangle = outputImage(cv::Rect(pt1, pt2));   //Creating a submatrix of the center rectangle HSV (which will be used to analyze the colors)
        cv::rectangle(outputImage, pt1, pt2, cv::Scalar(0,255,0));  //Display the green rectangle where colors will be read in
        
        //Loop through the matrix taking the average of each pixels scalar value (0,1,2). Don't forget 1 is closer to 360 in the first scalar than say 350 is to 360
        //double H_total; double S_total; double V_total;
        std::map<std::tuple<double,double,double>, int> colors_Map;
        int Dominant_Color_Count = -1;
        std::tuple<double,double,double> Dominant_Color;
        for(int i= 0; i < 120; i++) {
            for(int j= 0; j < 120; j++) {
                cv::Vec3b hsv = center_rectangle.at<cv::Vec3b>(i,j);    //Taking the pixel value
                std::tuple<double, double, double> hsv_Sub = {hsv.val[0], hsv.val[1], hsv.val[2]};
                if(colors_Map.find(hsv_Sub) == colors_Map.end()) {
                    colors_Map.insert(std::pair<std::tuple<double,double,double>, int>(hsv_Sub, 0));
                }
                else {
                    colors_Map[hsv_Sub] += 1;
                    if(colors_Map[hsv_Sub] >= Dominant_Color_Count) {
                        Dominant_Color_Count = colors_Map[hsv_Sub];
                        Dominant_Color = hsv_Sub;
                    }
                }
            }
        }
        double centerRectangle_size = center_rectangle.size().height * center_rectangle.size().width;
        //         H       S       V
        //         360-360 0-255   0-255
        
        int Dom_Color_Max_Count [10] = {0,0,0,0,0,0,0,0,0,0};
        for(int i= 0; i < 10; i++) {
            if(std::get<0>(Dominant_Color) >= std::get<0>(colors_HSV_min[i]) && std::get<0>(Dominant_Color) <= std::get<0>(colors_HSV_max[i])) {
                Dom_Color_Max_Count[i] += 1;
            }
            if(std::get<1>(Dominant_Color) >= std::get<1>(colors_HSV_min[i]) && std::get<1>(Dominant_Color) <= std::get<1>(colors_HSV_max[i])) {
                Dom_Color_Max_Count[i] += 1;
            }
            if(std::get<2>(Dominant_Color) >= std::get<2>(colors_HSV_min[i]) && std::get<2>(Dominant_Color) <= std::get<2>(colors_HSV_max[i])) {
                Dom_Color_Max_Count[i] += 1;
            }
        }

        int maxColorCount = -1;
        int maxColorLocation = 0;
        for(int i= 0; i < 10; i++) {
            if(Dom_Color_Max_Count[i] >= maxColorCount) {
                maxColorCount = Dom_Color_Max_Count[i];
                maxColorLocation = i;
            }
        }
        

        //https://stackoverflow.com/questions/35113979/calculate-distance-between-colors-in-hsv-space
        
        std::string HSVScale = "H: " + std::to_string(std::get<0>(Dominant_Color)) + "\tS: " + std::to_string(std::get<1>(Dominant_Color)) + "\tV: " + std::to_string(std::get<2>(Dominant_Color));
        std::cout << HSVScale << std::endl;
        std::cout << std::endl;
        cv::putText(outputImage, colors_string[maxColorLocation], cv::Point((outputImage_Size.width/2)-60, (outputImage_Size.height/2)-150), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0,255,0), 1, cv::LINE_AA);
        cv::putText(outputImage, HSVScale, cv::Point(50,50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0,255,0), 1, cv::LINE_AA);
        cv::imshow("Output", outputImage);  //Displaying the filled in output image
        cv::waitKey(20); //Display the image for 20ms
    }
}

int main() {

    std::vector<cv::Mat> cam_Calibration = cameraCalibration(); //cameraMatrix, distCoeffs, R, T
    colorDetection(cam_Calibration);


    return 0;
}