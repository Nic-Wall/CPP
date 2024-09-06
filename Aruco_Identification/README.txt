ArUco Marker Test Sheet.docx: A white piece of paper with ArUco markers pasted on in a nonsensical way. Should be held up to the camera so the ID of the markers can be displayed when read.
CheckerboardForCalibration.docx: A checkerboard (9x6 squares) picture on a word doc used to acquire the calibration measurements from the algorithm built into the openCV library.
CMakeLists.txt: Used to point the compiler to the proper library locations and determine the name of the executable created when running Identifying_Markers.cpp
Identifying_Markers.cpp: Readable code used to acquire calibration for the camera (using photos taken of the CheckerboardForCalibration at different angles (10 recommended)) and identification of ArUco Markers on the ArUco Marker Test Sheet.
ArUcoMarkerDetection: Executable of Identifying_Markers.cpp (will not run properly without supplying your own ten photos for calibration)
