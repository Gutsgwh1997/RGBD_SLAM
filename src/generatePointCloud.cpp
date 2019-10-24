#include<iostream>
#include<string>
#include "slamBase.h"

#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>

#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<pcl/visualization/cloud_viewer.h>

//定义点云的类型
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

 int main(int argc, char *argv[])
{
    cv::Mat rgb_1,depth_1;
    rgb_1 = cv::imread("./data/rgb_1.png");
    cv::imshow("rgb",rgb_1);
    cv::waitKey(0);
    depth_1 = cv::imread("./data/depth_1.png",-1);
    cv::imshow("depth",depth_1);
    cv::waitKey(0);

    //读取相机的内参
    ParameterReader PR;
    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.cx = atof( PR.getData("camera.cx").c_str());
    camera.cy = atof( PR.getData("camera.cy").c_str());
    camera.fx = atof( PR.getData("camera.fx").c_str());
    camera.fy = atof( PR.getData("camera.fy").c_str());
    camera.scale = atof( PR.getData("camera.scale").c_str());

    PointCloud::Ptr cloud( new PointCloud);
    //遍历深度图
    for ( int m=0; m<depth_1.rows; m++)
        for (int n=0; n<depth_1.cols; n++)
        {
            //获取深渡值
            ushort d = depth_1.ptr<ushort>(m)[n];
            if (d ==0)
                continue;
            PointT p;
            p.z = double(d) / camera.scale;
            p.x = (n - camera.cx)*p.z / camera.fx;
            p.y = (m - camera.cy)*p.z / camera.fy;

            p.b = rgb_1.ptr<uchar>(m)[n*3];
            p.g = rgb_1.ptr<uchar>(m)[n*3+1];
            p.r = rgb_1.ptr<uchar>(m)[n*3+2];

            //把点加入到点云中
            cloud->points.push_back(p);
        }
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;
    pcl::visualization::CloudViewer viewer( "viewer" );
    viewer.showCloud(cloud);
    while( !viewer.wasStopped())
    {

    }
    cloud->points.clear();
    return 0;
}
