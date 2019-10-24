/**
* @projectName   RGBD_SLAM
* @brief          实现两两帧之间的visual odometry
* @author       GWH_HIT
* @date          2019-10-16
**/

#include "slamBase.h"
#include <fstream>
#include <sstream>
#include <cmath>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include<pcl/filters/voxel_grid.h>
#include<pcl/filters/statistical_outlier_removal.h>
#include <sstream>

bool readTumDataset(ParameterReader& pd, vector<string>& rgb_files, vector<string>& depth_files);
double normofTransform( cv::Mat& rvec, cv::Mat& tvec);
void readPictures(int index, vector<string>& rgb_files, vector<string>& depth_files);

 int main(int argc, char *argv[])
{
     //我这里使用TUM的数据集,读取数据集
     vector<string> rgb_files;
     vector<string> depth_files;
     ParameterReader pd;
     CAMERA_INTRINSIC_PARAMETERS camera;
     camera.cx = atof( pd.getData("camera.cx").c_str());
     camera.cy = atof( pd.getData("camera.cy").c_str());
     camera.fx = atof( pd.getData("camera.fx").c_str());
     camera.fy = atof( pd.getData("camera.fy").c_str());
     camera.scale = atof( pd.getData("camera.scale").c_str());
     //读取Tum数据集
     //readTumDataset(pd, rgb_files, depth_files);

     for (int index = 1; index < 780; index++)
     {
         readPictures(index, rgb_files, depth_files);
     }
//     //自己给的图像
//     string datapath = "/home/gwh/catkin_vins/src/RGBD_SLAM/data/";
//     for (int i = 1; i<6; i++)
//     {
//         rgb_files.push_back(datapath+"color/"+to_string(i)+".png");
//         depth_files.push_back(datapath+"depth/"+to_string(i)+".pgm");
//     }

//     测试图像路径保存正确与否
//     for (int i = 0; i<5; i++)
//     {
//         cv::Mat image,image_depth;
//         image = cv::imread(rgb_files[i]);
//         image_depth = cv::imread(depth_files[i]);
//         cv::imshow("ceshi",image);
//         cv::imshow("depth",image_depth);
//         cv::waitKey(0);
//     }
     //对TUM数据集的每一张图片进行处理
     int start_index = atoi(pd.getData("start_index").c_str());
     int end_index = atoi(pd.getData("end_index").c_str());
     //声明数据结构
     FRAME frame1,frame2;
     RESULT_OF_PNP relative_motion;
     PointCloud::Ptr cloud1(new PointCloud());
     PointCloud::Ptr result_cloud(new PointCloud());
     //初始帧
     frame1.rgb = cv::imread(rgb_files[start_index]);
     frame1.depth = cv::imread(depth_files[start_index], -1);
     computeKeypointsAndDescriptors(frame1);
     cloud1 = image2PointCloud(frame1.rgb, frame1.depth,camera);

     pcl::visualization::CloudViewer viewer("viewer of frames");
     string visualize_pointcloud = pd.getData("visualize_pointcloud");
     bool visualize = visualize_pointcloud == string("yes");
     int min_inliers = atoi(pd.getData("min_inliers").c_str());
     double max_norm = atof(pd.getData("max_norm").c_str());
     for ( int i = start_index+1; i < end_index+1; i++)
     {
         frame2.rgb=cv::imread(rgb_files[i]);
         cv::imshow("yuantu",frame2.rgb);
         cv::waitKey(20);
         frame2.depth=cv::imread(depth_files[i], -1);
         computeKeypointsAndDescriptors(frame2);
         bool pnp = estimateMotion2(frame1, frame2, camera, relative_motion);
         if ( pnp == false)
         {
             cout<<"两帧间的配对点过少，PNP 失败！"<<endl;
             continue;
         }
         if ( relative_motion.inliers < min_inliers )
         {
             cout<<"RANSAC_PNP后内点过少!"<<endl;
             continue;
         }
         if ( normofTransform( relative_motion.rvec, relative_motion.tvec) > max_norm)
         {
             cout<<"帧间运动过大: "<<normofTransform( relative_motion.rvec, relative_motion.tvec)<<endl;
             continue;
         }
         cout<<"第 "<<i<<" 帧相对的旋转为："<<relative_motion.rvec.t();
         cout<<"相对的平移为: "<<relative_motion.tvec.t()<<endl;
         Eigen::Isometry3d T_21;
         T_21 = cvMat2Eigen(relative_motion.rvec,relative_motion.tvec);
         result_cloud = joinPointCloud( cloud1, frame2,  T_21, camera);
         frame1 = frame2;
         cloud1 = result_cloud;
         if (visualize)
             viewer.showCloud( cloud1 );
       }
    //去除离群点
     double meank= atof( pd.getData("meank").c_str());
     double stddevmulthresh = atof(pd.getData("stddevmulthresh").c_str());
     PointCloud::Ptr after_filter ( new PointCloud());
     pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
     statistical_filter.setMeanK(meank);                     //设置进行统计时，考虑查询点相邻点的数量
     statistical_filter.setStddevMulThresh(stddevmulthresh);   //是否为离群点的阈值
     statistical_filter.setInputCloud(cloud1);
     statistical_filter.filter(*after_filter);

     //体素滤波
     static pcl::VoxelGrid<PointT> voxel;
     double gridsize = atof(pd.getData("voxel_grid").c_str());
     voxel.setLeafSize(gridsize,gridsize,gridsize);
     voxel.setInputCloud(after_filter);
     PointCloud::Ptr tmp(new PointCloud());
     voxel.filter(*tmp);
     pcl::io::savePCDFile("data/after_filter.pcd",*tmp);
}




 /*********************************************************************************
  * Description:  计算两帧之间的运动量
  *Calls:              无
  *Called By:      main
  *Input:             两帧之间的相对旋转、平移
  *Output:
  *Others:           运动量大小,是位移与旋转的范数加和
**********************************************************************************/
double normofTransform( cv::Mat& rvec, cv::Mat& tvec)
{
    return fabs(min(cv::norm(rvec), 2*M_PI - cv::norm(rvec))) + fabs(cv::norm(tvec));
}

/*********************************************************************************
  * Description:    读取TUM数据集对应的associate.txt文件，将文件中的路径存储到vector
  *Calls:                无
  *Called By:         main.cpp
  *Input:                slambase.h中的数据读取类，用于保留rgb与depth图像的vector
  *Output:             文件不存在，false
**********************************************************************************/
bool readTumDataset(ParameterReader& pd, vector<string>& rgb_files, vector<string>& depth_files)
{
    string associate_root = pd.getData("associate_root").c_str();
    ifstream fin( associate_root+"associate.txt");
    if (!fin.is_open())
    {
        cerr<<"associate.txt doesn't exist."<<endl;
        return false;
    }
    while( !fin.eof())
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_files.push_back( associate_root + rgb_file);
        depth_files.push_back( associate_root + depth_file);

        if( fin.good() == false)
            break;
    }
}

/*********************************************************************************
  * Description:   按照index索引读取xx.png图片存储到vector
  *Calls:               无
  *Called By:        main
  *Input:               索引值、两个vector
  *Output:            无
  *Others:
**********************************************************************************/
void readPictures(int index, vector<string>& rgb_files, vector<string>& depth_files)
{
    string rgb_path = "/home/gwh/catkin_vins/src/RGBD_SLAM/data/GX_Datasets/rgb_png/";
    string depth_path = "/home/gwh/catkin_vins/src/RGBD_SLAM/data/GX_Datasets/depth_png/";

    stringstream ss;
    ss<<rgb_path<<index<<".png";
    string rgb_file, depth_file;
    ss>>rgb_file;

    ss.clear();
    ss<<depth_path<<index<<".png";
    ss>>depth_file;

    rgb_files.push_back(rgb_file);
    depth_files.push_back(depth_file);

}
