CMakeLists.txt: Points the compiler to the correct path of the libraries when creating the executable ColorIdentification
ColorIdentification: Executable of ColorIdentifier.cpp: Opens a new window with a copy of the camera's view, a square, and text displaying the prominent color displayed in the square.
ColorIdentifier.cpp: Uses OpenCV to attempt to identify colors based on pixels HSV and RGB data read in from the camera. It doesn't work perfectly and I'm still working to find out why.
