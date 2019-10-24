#ADD_EXECUTABLE( hello_slam hello.cpp)

set(OpenCV_DIR /opt/ros/kinetic/share/OpenCV-3.3.1-dev)
set( CMAKE_CXX_FLAGS "-std=c++11" )
find_package( OpenCV REQUIRED)

#添加头文件
include_directories( "/usr/include/eigen3" )
find_package( PCL REQUIRED COMPONENTS commo  io visualization filters)
ADD_DEFINITIONS( ${PCL_DEFINTIONS} )
INCLUDE_DIRECTORIES( ${PCL_INCLUDE_DIRS})
LINK_LIBRARIES( ${PCL_LIBRARY_DIRS} )

#自己写的库
#2dTo3d点转换与图片转换为点云的库
ADD_LIBRARY( pt2Topt3_lib SHARED point2dTo3d_lib.cpp)
TARGET_LINK_LIBRARIES( pt2Topt3_lib ${OpenCV_LIBS} ${PCL_LIBRARIES})

#特征检测、匹配、求解PnP
ADD_LIBRARY( featurematch_lib SHARED featurematch_lib.cpp)
TARGET_LINK_LIBRARIES( featurematch_lib pt2Topt3_lib ${OpenCV_LIBS} ${PCL_LIBRARIES})

#新帧加入到点云中
ADD_LIBRARY( joinPointCloud_lib SHARED joinPointCloud_lib.cpp)
TARGET_LINK_LIBRARIES( joinPointCloud_lib ${OpenCV_LIBS} ${PCL_LIBRARIES} pt2Topt3_lib featurematch_lib)

ADD_EXECUTABLE( generate_pointcloud generatePointCloud.cpp)
TARGET_LINK_LIBRARIES( generate_pointcloud ${OpenCV_LIBS} ${PCL_LIBRARIES})

ADD_EXECUTABLE( feature_match featurematch.cpp)
TARGET_LINK_LIBRARIES( feature_match ${OpenCV_LIBS} ${PCL_LIBRARIES} pt2Topt3_lib)

ADD_EXECUTABLE( joinPointCloud joinPointCloud.cpp)
TARGET_LINK_LIBRARIES( joinPointCloud ${OpenCV_LIBS} ${PCL_LIBRARIES} pt2Topt3_lib featurematch_lib)

ADD_EXECUTABLE( sampleVO visualOdometry.cpp)
TARGET_LINK_LIBRARIES( sampleVO ${OpenCV_LIBS} ${PCL_LIBRARIES} pt2Topt3_lib featurematch_lib joinPointCloud_lib)


#添加g2o的编译信息
LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
FIND_PACKAGE( G2O REQUIRED)

#CSparse
FIND_PACKAGE( CSparse REQUIRED)
INCLUDE_DIRECTORIES( ${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})
ADD_EXECUTABLE( optimizeVO optimizeVo.cpp)
TARGET_LINK_LIBRARIES( optimizeVO ${OpenCV_LIBS} ${PCL_LIBRARIES} pt2Topt3_lib
                                                featurematch_lib joinPointCloud_lib
                                                g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff
                                                g2o_csparse_extension ${CSPARSE_LIBRARY})
