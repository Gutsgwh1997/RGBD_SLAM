#include"slamBase.h"

#include<opencv2/core/eigen.hpp>
#include<opencv2/calib3d/calib3d.hpp>   //罗德里格斯公式

#include<pcl/common/transforms.h>
#include<pcl/visualization/cloud_viewer.h>

#include<Eigen/Core>
#include<Eigen/Geometry>

 int main(int argc, char *argv[])
{
     cv::Mat rgb_1,rgb_2;
     rgb_1 = cv::imread("./data/rgb_1.png");
     rgb_2 = cv::imread("./data/rgb_2.png");
     cv::Mat depth_1, depth_2;
     depth_1 = cv::imread("./data/depth_1.png");
     depth_2 = cv::imread("./data/depth_2.png");

     FRAME frame1,frame2;
     frame1.rgb = rgb_1;
     frame1.depth = depth_1;
     frame2.rgb = rgb_2;
     frame2.depth = depth_2;

     ParameterReader PR;
     CAMERA_INTRINSIC_PARAMETERS camera;
     camera.cx = atof( PR.getData("camera.cx").c_str());
     camera.cy = atof( PR.getData("camera.cy").c_str());
     camera.fx = atof( PR.getData("camera.fx").c_str());
     camera.fy = atof( PR.getData("camera.fy").c_str());
     camera.scale = atof( PR.getData("camera.scale").c_str());

     computeKeypointsAndDescriptors(frame1);
     computeKeypointsAndDescriptors(frame2);

    RESULT_OF_PNP relativemontion;
    relativemontion = estimateMotion(frame1,frame2,camera);
    std::cout<<"Relative rotation is \n"<<relativemontion.rvec<<std::endl;
    std::cout<<"Relative translation is \n"<<relativemontion.tvec<<std::endl;

    //将旋转向量转化为矩阵
    cv::Mat R;
    cv::Rodrigues(relativemontion.rvec, R);
    Eigen::Matrix3d R_eigen;
    cv::cv2eigen(R, R_eigen);

    //将平移向量和旋转向量合成变换矩阵
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(R_eigen);
    std::cout<<"Translation"<<std::endl;
    Eigen::Translation<double,3> trans(relativemontion.tvec.at<double>(0,0),
                                                               relativemontion.tvec.at<double>(0,1),
                                                               relativemontion.tvec.at<double>(0,2));
    T = angle;
    T(0,3) = trans.x();
    T(1,3) = trans.y();
    T(2,3) = trans.z();

    //转换点云
    std::cout<<"Converting image to clouds"<<std::endl;
    PointCloud::Ptr cloud1 = image2PointCloud(frame1.rgb,frame1.depth,camera);
    PointCloud::Ptr cloud2 = image2PointCloud(frame2.rgb,frame2.depth,camera);

    //转换点云
    std::cout<<"Combining clouds"<<std::endl;
    PointCloud::Ptr output(new PointCloud());
    pcl::transformPointCloud(*cloud1,*output,T.matrix());
    *output += *cloud2;
    pcl::visualization::CloudViewer viewer("viewer of two frames");
    viewer.showCloud(output);
    while( !viewer.wasStopped())
    {

    }

    return 0;
}
