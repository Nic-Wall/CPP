cmake_minimum_required(VERSION 3.29)
project(ColorDetection)

set (CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED PATHS "/opt/homebrew/Cellar/opencv/4.9.0_8/include/opencv4")
include_directories("/opt/homebrew/Cellar/opencv/4.9.0_8/include/opencv4")

link_directories("/opt/homebrew/Cellar/opencv/4.9.0_8/lib")

add_executable(ColorIdentification "ColorIdentifier.cpp")

target_link_libraries(ColorIdentification ${OpenCV_LIBS})