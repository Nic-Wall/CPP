#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include "matrices.hpp"
#include <thread>
#include <chrono>

/*
Had to build CMakeLists.txt.
1. Create CMakeLists.txt
2. Create build directory
3. Enter build directory
4. Fill in CMakeLists.txt
5. run "cmake .." in build dir
6. run "make" in build dir
7. Run "./NameOfProject" in build dir
Set "-std=c++11" in tasks.json under "args" and set the C/C++ Standard to C++11 in C/C++Confirgurations
Also added paths to custom library and OpenCV in intellisense
*/
//Look into adjacency matrices
//https://stackoverflow.com/questions/36823037/why-use-a-matrix-for-3d-projection
//https://gamedev.stackexchange.com/questions/72044/why-do-we-use-4x4-matrices-to-transform-things-in-3d
//http://www.malinc.se/math/linalg/rotatecubeen.php

matrix Trans_3D_into_2D(matrix M, double x_angle, double y_angle, double z_angle) {
    matrix P (std::vector<std::vector<double> > {{1,0}, {0,1}, {0,0}});

    x_angle = x_angle / 100; y_angle = y_angle /100; z_angle = z_angle / 100;
    matrix X_Rotation (std::vector<std::vector<double> > {{1,0,0}, {0,cos(x_angle),sin(x_angle)}, {0, -sin(x_angle), cos(x_angle)}});
    matrix Y_Rotation (std::vector<std::vector<double> > {{cos(y_angle), 0, -sin(y_angle)}, {0, 1, 0}, {sin(y_angle), 0, cos(y_angle)}});
    matrix Z_Rotation (std::vector<std::vector<double> > {{cos(z_angle), sin(z_angle), 0}, {-sin(z_angle), cos(z_angle), 0}, {0, 0, 1}});

    std::cout << "P: " << P.shape() << "\tM: " << M.shape() << "\tx_R: " << x_angle << "\ty_R: " << y_angle << "\tz_R: " << z_angle << std::endl;

    matrix rotatedMatrix = (((P*X_Rotation)*Y_Rotation)*Z_Rotation)*M;
    rotatedMatrix.print();
    return rotatedMatrix;
    //w=Pv to w=P*xR*yR*zR*v
    /*
        Turning a 3D point into a 2D point is done with the following calculation...
            {a}         {a}
        v = {b}     w = {b}     P = {1 0 0}
            {c}                     {0 1 0}

        with w = Pv
    */
}

int main() {
    //Creating 3D points
    std::vector<matrix> point_list_3D;  //A vector of matrices
    //Adding all three dimensional points to the vector (as matrices)  x  , y  , z
    point_list_3D.push_back(matrix(std::vector<std::vector<double> > {{300, 300, 300}}));
    point_list_3D.push_back(matrix(std::vector<std::vector<double> > {{300, 700, 300}}));
    point_list_3D.push_back(matrix(std::vector<std::vector<double> > {{700, 300, 300}}));
    point_list_3D.push_back(matrix(std::vector<std::vector<double> > {{700, 700, 300}}));
    point_list_3D.push_back(matrix(std::vector<std::vector<double> > {{300, 300, 700}}));
    point_list_3D.push_back(matrix(std::vector<std::vector<double> > {{300, 700, 700}}));
    point_list_3D.push_back(matrix(std::vector<std::vector<double> > {{700, 300, 700}}));
    point_list_3D.push_back(matrix(std::vector<std::vector<double> > {{700, 700, 700}}));

    double x = 15; double y = 0; double z = 0;  //Declaring the starting axis rotations
    
    int rotationWatcher = 0;   //Declaring a watcher that keeps track of how many an axis rotation was made (in degree increments)
    int imageWidth = 2600; int imageHeight = 1920;  //Declaring the window width and height
    double* incrementalRotationAxis = &z;   //Declaring the alias for which axis is currently being rotated
    bool negativePoint = false; //Creating a bool that tracks if a point is moving off the screen (behind point 0,0)
    
    while(true) {   //Continue forever until manually stopped via the terminal
        //Line drawing order (cube with no fill in colors, just lines)
        int line_order [16] = {7,5,1,0,4,6,2,3,1,5,4,0,2,6,7,3};    //Each point was built and added to the matrix above, this references the order at which they should be generated
        int colorPicker = 0;    //Starting the color picker as 0 (used to index a vector full of Scalar(R,G,B) hexes)
        //Generating image
        cv::Mat image(imageHeight, imageWidth, CV_8UC3, cv::Scalar(0, 0, 0));   //Creating the black background image (on which the cube floats)
        for(int i= 0; i < (sizeof(line_order)/sizeof(*line_order))-1; i++) {    //Itterating through every point in the line_order list (-1 because it's pointstart(i) <> pointend(i+1))
            //Setting Color Rotation
            cv::Scalar colorChart [3] = {cv::Scalar(45,255,0), cv::Scalar(255,45,0), cv::Scalar(0,45,255)}; //Declaring color array

            std::cout << "i: " << i << "\ti+1: " << i+1 << std::endl;   //Output for terminal reference

            matrix lineStart = Trans_3D_into_2D(point_list_3D[line_order[i]], x, y, z); //Transforming the 3D point in a 2D point with the rotation stated on by x, y, and z (axis)
            matrix lineEnd = Trans_3D_into_2D(point_list_3D[line_order[i+1]], x, y, z); //Same as above

            cv::Point pointStart (lineStart.locateValue(0,0)+900, lineStart.locateValue(1,0)+300);  //Creating a point with the point class (adding +300/+600 to keep it near the middle of the image rather than starting at 0,0)
            cv::Point pointEnd (lineEnd.locateValue(0,0)+900, lineEnd.locateValue(1,0)+300);    //Same as above
            cv::line(image, pointStart, pointEnd, colorChart[colorPicker], 3);  //Creating a line using the two points and a color choosen by the colorPicker
            colorPicker = (colorPicker == 2) ? 0 : colorPicker + 1; //Iterates through the colorPicker per iteration in the for loop (0,1,2,0,1,2,0,1,2,0...etc.)
            std::cout << std::endl; //Output for terminal reference
            if(pointStart.x <= 1 || pointEnd.x <= 1 || pointStart.x >= imageWidth || pointEnd.x >= imageWidth || pointStart.y <= 1 || pointEnd.y <= 1 || pointStart.y >= imageHeight || pointEnd.y >= imageHeight) {  //Need to account for them rotating off the other side of the window
            //If the points are near the edges of the screen rotate the cube the opposite direction for the remainder of the rotationWatcher
                negativePoint = true;   //Notify the alias needs to be rotated the other way
            }
        }
        if(negativePoint == true && rotationWatcher < 90) {
        //If a negativePoint is found and the rotationWatcher is below 90...
            (*incrementalRotationAxis) -= 1;    //Rotate the opposite direction (away from the edge of the screen)
        }
        else {  //If the cube isn't near the edge of the screen
            //Displaying image
            (*incrementalRotationAxis) += 1;    //Rotate one degree further on the next loop
            if(rotationWatcher >= 90) { //If the cube as rotated 90+ degrees continually on one axis
                int changingDirections = rand() % 3; //Generates a random number: 0, 1, or 2
                if(changingDirections == 0) {incrementalRotationAxis = &x;}  //0 = x
                else if(changingDirections == 1) {incrementalRotationAxis = &y;} //1 = y
                else {incrementalRotationAxis = &z;} //2 == z
                //randomly pick a new axis to rotate on
                rotationWatcher = 0;    //Set the rotation watcher to 0 (so it can travel back up to 90)
                negativePoint = false;  //Set negativePoint to false so the cube rotates in a positive manner once again
            }
        }
        cv::imshow("Cube", image);  //Display the image created
        cv::waitKey(1); //Wait one microsecond
        rotationWatcher += 1;   //Incremenet rotationWatcher by one
    }
    
    

    /*
    //Filled in rectangles
    //A bit more difficult because one has to take into account what is "in front"
    while(true) {
        int point_order [8] = {0,5,6,0,3,6,5,3};    //Only need two opposing points for a rectangle
                                    //Blue                Red                  Green                Yellow                 Magenta                Red                  White
        cv::Scalar colorChart [7] = {cv::Scalar(0,0,255), cv::Scalar(255,0,0), cv::Scalar(0,255,0), cv::Scalar(255,255,0), cv::Scalar(150,0,150), cv::Scalar(255,0,0), cv::Scalar(255,255,255)};
        cv::Mat image(1920, 1920, CV_8UC3, cv::Scalar(0,0,0));
        for(int i= 0; i < (sizeof(point_order)/sizeof(*point_order))-1; i++) {
            std::cout << "i: " << i << "\ti+1: " << i+1 << std::endl;
            matrix First3DSquarePoint = Trans_3D_into_2D(point_list_3D[point_order[i]], x, y, z);
            matrix Second3DSquarePoint = Trans_3D_into_2D(point_list_3D[point_order[i+1]], x, y, z);

            cv::Point F2DSqrPoint (First3DSquarePoint.locateValue(0,0)+900, First3DSquarePoint.locateValue(1,0)+300);
            cv::Point S2DSqrPoint (Second3DSquarePoint.locateValue(0,0)+900, Second3DSquarePoint.locateValue(1,0)+300);
            
            std::cout << std::endl;

            cv::rectangle(image, F2DSqrPoint, S2DSqrPoint, colorChart[i], -1);
        }
        //Displaying the image
        cv::imshow("Filled In Cube", image);
        cv::waitKey(1);
        y += 1;
    }
    */
    


    //https://docs.opencv.org/4.x/d3/d96/tutorial_basic_geometric_drawing.html

    //How to display a new image without destroying the window: https://stackoverflow.com/questions/13439750/opencv-how-to-display-a-series-of-images-without-reinitializing-a-window
    //Remove previously drawn lines: https://answers.opencv.org/question/209955/how-to-remove-drawings-from-mat/
    //https://www.mathsisfun.com/sine-cosine-tangent.html

    //Needs to rotate on it's axis (i.e. two static points) not around an invisible pole
    
    return 0;
}