#include "slamBase.h"
#include<pcl/visualization/cloud_viewer.h>
#include<pcl/filters/statistical_outlier_removal.h>

//将RGBD相机的一帧转换为点云
PointCloud::Ptr image2PointCloud (cv::Mat& rgb_img, cv::Mat& depth_img, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    PointCloud::Ptr cloud( new PointCloud);
    //遍历深度图
    for ( int m=0; m<depth_img.rows; m++)
        for (int n=0; n<depth_img.cols; n++)
        {
            //获取深渡值
            ushort d = depth_img.ptr<ushort>(m)[n];
            if (d ==0)
                continue;
//            if (d >= 10000)
//                continue;
            PointT p;
            p.z = double(d) / camera.scale;
            p.x = (n - camera.cx)*p.z / camera.fx;
            p.y = (m - camera.cy)*p.z / camera.fy;

            p.b = rgb_img.ptr<uchar>(m)[n*3];
            p.g = rgb_img.ptr<uchar>(m)[n*3+1];
            p.r = rgb_img.ptr<uchar>(m)[n*3+2];

            //把点加入到点云中
            cloud->points.push_back(p);
        }
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;

//    //depth filter and statistical removal
//    ParameterReader pd;
//    double meank= atof( pd.getData("meank").c_str());
//    double stddevmulthresh = atof(pd.getData("stddevmulthresh").c_str());
//    PointCloud::Ptr tmp ( new PointCloud());
//    pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
//    statistical_filter.setMeanK(meank);                     //设置进行统计时，考虑查询点相邻点的数量
//    statistical_filter.setStddevMulThresh(stddevmulthresh);   //是否为离群点的阈值
//    statistical_filter.setInputCloud(cloud);
//    statistical_filter.filter(*tmp);
//    return tmp;
      return cloud;
//    pcl::visualization::CloudViewer viewer( "viewer of one frame" );
//    viewer.showCloud(cloud);
//    while( !viewer.wasStopped())
//    {

//    }
//    return cloud;
}

//将RGBD相机得到的像素坐标和深度坐标转化为3D坐标
cv::Point3f point2dTo3d(cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    cv::Point3f p;
    p.z = double(point.z) / camera.scale;
    p.x = (point.x - camera.cx)*p.z / camera.fx;
    p.y = (point.y - camera.cy)*p.z / camera.fy;
    return p;
}
