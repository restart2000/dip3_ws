cmake_minimum_required(VERSION 3.0.2)
project(image_pkg)

Set(OpenCV_DIR /opt/ros/kinetic/share/OpenCV-3.3.1-dev)
find_package(catkin REQUIRED roscpp OpenCV)
catkin_package()
include_directories(${catkin_INCLUDE_DIRS}   ${OpenCV_INCLUDE_DIRS})
add_executable(EdgeDetection src/EdgeDetection.cpp)
target_link_libraries(EdgeDetection  ${catkin_LIBRARIES} ${OpenCV_LIBS})
