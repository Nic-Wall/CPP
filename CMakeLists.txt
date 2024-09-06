#https://stackoverflow.com/questions/12529119/may-i-include-in-hpp-files

#https://stackoverflow.com/questions/30004850/cmake-does-not-find-opencv-libraries
#Adding bare minimums
cmake_minimum_required(VERSION 3.29)
project(RotatingShapes)

#Setting CMake standard (receive errors when building otherwise)
set (CMAKE_CXX_STANDARD 11)

#Adding the extra include directories
#OpenCV
find_package(OpenCV REQUIRED PATHS "/opt/homebrew/Cellar/opencv/4.9.0_7/include/opencv4")
include_directories("/opt/homebrew/Cellar/opencv/4.9.0_7/include/opencv4")
#MatriceLA
include_directories("/Users/nwallace/Desktop/MiscCoding/CPPStuff/CustomLibs/MatricesLA")

#Adding the extra lib directories
link_directories("/opt/homebrew/Cellar/opencv/4.9.0_8/lib")

#Adding the executable
add_executable(3DGeometry "RotatingCube.cpp")

#Linking the libraries to my project
target_link_libraries(3DGeometry ${OpenCV_LIBS})