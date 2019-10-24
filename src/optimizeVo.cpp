/**
* @projectName   RGBD_SLAM
* @brief          带有位姿图优化的VO
* @author       GWH_HIT
* @date          2019-10-18
**/

#include "slamBase.h"
#include <fstream>
#include <sstream>
#include <cmath>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <sstream>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>

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

     //读取数据集的全部数据
     for (int index = 1; index < 780; index++)
     {
         readPictures(index, rgb_files, depth_files);
     }
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

     string visualize_pointcloud = pd.getData("visualize_pointcloud");
     bool visualize = visualize_pointcloud == string("yes");
     int min_inliers = atoi(pd.getData("min_inliers").c_str());
     double max_norm = atof(pd.getData("max_norm").c_str());

     //新增g2o的优化
     //选择优化方法
     //每个误差项优化变量维度为6，误差维度为3
     //对应g2o的结构图就清晰了
     typedef g2o::BlockSolver_6_3  SlamBlockSolver;
     typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

     //初始化求解器
     SlamLinearSolver* linearSolver = new SlamLinearSolver();
     linearSolver->setBlockOrdering(false);
     SlamBlockSolver* blockSolver = new SlamBlockSolver(linearSolver);
     g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(blockSolver);

     //g2o的重点就是围绕这个编程
     g2o::SparseOptimizer globalOptimizer;
     globalOptimizer.setAlgorithm(solver);
     globalOptimizer.setVerbose(false);

     //向globalOptimizer增加一个顶点
     g2o::VertexSE3* v = new g2o::VertexSE3();
     v->setId(start_index);
     v->setEstimate(Eigen::Isometry3d::Identity());
     v->setFixed(true);
     globalOptimizer.addVertex(v);

     int num = 0;
     for ( int i = start_index+1; i < end_index+1; i++)
     {
         frame2.rgb=cv::imread(rgb_files[i]);
         frame2.depth=cv::imread(depth_files[i], -1);
         computeKeypointsAndDescriptors(frame2);
         bool pnp = estimateMotion2(frame1, frame2, camera, relative_motion);
         if ( pnp == false)
         {
             cout<<"两帧间的配对点过少，PNP 失败！"<<endl;
             num++;
             continue;
         }
         if ( relative_motion.inliers < min_inliers )
         {
             cout<<"RANSAC_PNP后内点过少!"<<endl;
             num++;
             continue;
         }
         if ( normofTransform( relative_motion.rvec, relative_motion.tvec) > max_norm)
         {
             cout<<"帧间运动过大: "<<normofTransform( relative_motion.rvec, relative_motion.tvec)<<endl;
             num++;
             continue;
         }
         cout<<"第 "<<i<<" 帧相对的旋转为："<<relative_motion.rvec.t();
         cout<<"相对的平移为: "<<relative_motion.tvec.t()<<endl;
         Eigen::Isometry3d T_21;
         T_21 = cvMat2Eigen(relative_motion.rvec,relative_motion.tvec);

         //向g2o中增加这个顶点与上一帧联系的边
         //1、顶点只需要设定id
         g2o::VertexSE3* v = new g2o::VertexSE3();
         v->setId(i);
         v->setEstimate(Eigen::Isometry3d::Identity());
         globalOptimizer.addVertex(v);

         //2、边部分，设定ID，和观测值
         g2o::EdgeSE3* edge = new g2o::EdgeSE3();
         //链接此边的两个顶点的id
         edge->vertices()[0] = globalOptimizer.vertex(i -1 - num);
         num = 0;
         edge->vertices()[1] = globalOptimizer.vertex(i);
         //信息矩阵
         Eigen::Matrix<double,6,6> information = Eigen::Matrix<double,6,6>::Identity();
         // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
         // 因为pose为6D的，信息矩阵是6*6的阵，假设位置和角度的估计精度均为0.1且互相独立
         // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
         information(0,0) = information(1,1) = information(2,2) = 100;
         information(3,3) = information(4,4) = information(5,5) = 100;
         edge->setInformation(information);
         edge->setMeasurement(T_21);
         globalOptimizer.addEdge(edge);
//         result_cloud = joinPointCloud( cloud1, frame2,  T_21, camera);
         frame1 = frame2;
         cloud1 = result_cloud;
//         if (visualize)
//             viewer.showCloud( cloud1 );
       }
      // 优化所有边
      cout<<"optimizing pose graph, vertices: "<<globalOptimizer.vertices().size()<<endl;
      globalOptimizer.save("./data/result_before.g2o");
      globalOptimizer.initializeOptimization();
      globalOptimizer.optimize( 100 ); //可以指定优化步数
      globalOptimizer.save( "./data/result_after.g2o" );
      cout<<"Optimization done."<<endl;

      globalOptimizer.clear();
//      //去除离群点
//     double meank= atof( pd.getData("meank").c_str());
//     double stddevmulthresh = atof(pd.getData("stddevmulthresh").c_str());
//     PointCloud::Ptr after_filter ( new PointCloud());
//     pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
//     statistical_filter.setMeanK(meank);                     //设置进行统计时，考虑查询点相邻点的数量
//     statistical_filter.setStddevMulThresh(stddevmulthresh);   //是否为离群点的阈值
//     statistical_filter.setInputCloud(cloud1);
//     statistical_filter.filter(*after_filter);

//     //体素滤波
//     static pcl::VoxelGrid<PointT> voxel;
//     double gridsize = atof(pd.getData("voxel_grid").c_str());
//     voxel.setLeafSize(gridsize,gridsize,gridsize);
//     voxel.setInputCloud(after_filter);
//     PointCloud::Ptr tmp(new PointCloud());
//     voxel.filter(*tmp);
//     pcl::io::savePCDFile("data/after_filter.pcd",*tmp);
      return 0;
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
