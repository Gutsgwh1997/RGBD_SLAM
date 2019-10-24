#include "slamBase.h"

#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

/*********************************************************************************
  * Description: 将一副FRAME拼接到已有点云
  *Calls:             image2PointCloud
  *Called By:
  *Input:            T_21
  *Output:         拼接后的PointCloud::Ptr
  *Others:
**********************************************************************************/
PointCloud::Ptr joinPointCloud( PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d& T,
                                CAMERA_INTRINSIC_PARAMETERS& camera)
{
    //转换点云
    std::cout<<"Converting image to clouds ......"<<std::endl;
    PointCloud::Ptr newcloud = image2PointCloud(newFrame.rgb, newFrame.depth, camera);

    //合并点云
    std::cout<<"Combining clouds ......"<<std::endl;
    PointCloud::Ptr output(new PointCloud());
    pcl::transformPointCloud(*newcloud, *output, T.inverse().matrix());
    *output+=*original;

    //体素滤波
    static pcl::VoxelGrid<PointT> voxel;
    static ParameterReader pd;
    double gridsize = atof(pd.getData("voxel_grid").c_str());
    voxel.setLeafSize(gridsize, gridsize, gridsize);
    voxel.setInputCloud(output);
    PointCloud::Ptr tmp(new PointCloud());
    voxel.filter(*tmp);
    return tmp;
}

Eigen::Isometry3d cvMat2Eigen(cv::Mat& rvec, cv::Mat& tvec)
{
    //将旋转向量转化为矩阵
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    Eigen::Matrix3d R_eigen;
    cv::cv2eigen(R, R_eigen);

    //将平移向量和旋转向量合成变换矩阵
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd rotation_vector(R_eigen);
    Eigen::Vector3d trans(tvec.at<double>(0,0),
                                        tvec.at<double>(0,1),
                                        tvec.at<double>(0,2));
//    Eigen::Translation<double,3> trans(tvec.at<double>(0,0),
//                                                               tvec.at<double>(0,1),
//                                                               tvec.at<double>(0,2));
//    T = rotation_vector;
//    T(0,3) = trans.x();
//    T(1,3) = trans.y();
//    T(2,3) = trans.z();
    //第二种实现方式
    T.rotate(rotation_vector);
    T.pretranslate(trans);
    return T;
}
