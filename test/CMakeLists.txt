set(OpenCV_DIR /opt/ros/kinetic/share/OpenCV-3.3.1-dev)
set( CMAKE_CXX_FLAGS "-std=c++11" )
find_package( OpenCV REQUIRED)

#添加头文件
find_package( PCL REQUIRED COMPONENTS commo  io visualization)
ADD_DEFINITIONS( ${PCL_DEFINTIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS})
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

#自己写的库
#2dTo3d点转换与图片转换为点云的库
ADD_LIBRARY( pt2Topt3 SHARED ${PROJECT_SOURCE_DIR}/src/point2dTo3d_lib.cpp)
TARGET_LINK_LIBRARIES( pt2Topt3 ${OpenCV_LIBS} ${PCL_LIBRARIES})

#特征检测、匹配、求解PnP
ADD_LIBRARY( featurematch SHARED ${PROJECT_SOURCE_DIR}/src/featurematch_lib.cpp)
TARGET_LINK_LIBRARIES( featurematch pt2Topt3_lib ${OpenCV_LIBS} ${PCL_LIBRARIES})


ADD_EXECUTABLE( test test.cpp)
TARGET_LINK_LIBRARIES( test featurematch_lib pt2Topt3_lib)
