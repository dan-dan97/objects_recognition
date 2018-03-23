cmake_minimum_required(VERSION 2.8)
project(pcl_test)

set(CMAKE_CXX_STANDARD 11)

find_package(PCL REQUIRED)
find_package(OpenCV 2 REQUIRED)
find_library(LIBREALSENSE2_LIB realsense2 REQUIRED)
find_library(KEYBOARDLIB_LIB keyboard_lib REQUIRED)

include_directories(
    include
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

set(PLC_TEST_LIB_HEADERS )
set(PLC_TEST_LIB_SOURCES src/pcl_test.cpp)

add_executable(pcl_test ${PLC_TEST_LIB_SOURCES} ${PLC_TEST_LIB_HEADERS})
target_link_libraries(pcl_test ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${LIBREALSENSE2_LIB} ${KEYBOARDLIB_LIB})
