#include<string>
#include<iostream>
#include<fstream>
#include<map>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>

//定义点云的类型
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

using namespace std;
# pragma once

// 参数读取类
//相机内参的结构体
struct CAMERA_INTRINSIC_PARAMETERS
{
    double cx,cy,fx,fy,scale;
};

//帧结构
struct FRAME
{
    cv::Mat rgb,depth;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

//PnP结果
struct RESULT_OF_PNP
{
    cv::Mat rvec, tvec;
    //内点的数量
    int inliers;
};

class ParameterReader
{
public:
     ParameterReader ( string filename = "./parameter.txt")
    {
        ifstream fin( filename.c_str());
        if (!fin)
        {
            cerr<<"parameter file does not exist."<<endl;
            return ;
        }
        while(!fin.eof())
        {
            string str;
            getline(fin,str);
            if(str[0] == '#')
                continue;
            int pos;
            pos = str.find("=");
            if (pos == -1)
                continue;
            string key = str.substr(0,pos);
            string value = str.substr(pos+1, str.length());
            data[key] = value;
            if (!fin.good())
                break;
        }
    }



    string getData( string key)
    {
        map<string, string>::iterator iter = data.find(key);
        if (iter == data.end())
        {
            cerr<<"Parameter name "<<key<<" not found!"<<std::endl;
            return string("Not found!");
        }

        return iter->second;
    }


public:
    map<string, string> data;
};

//将RGBD相机的一帧转换为点云
PointCloud::Ptr image2PointCloud (cv::Mat& rgb_img, cv::Mat& depth_img, CAMERA_INTRINSIC_PARAMETERS& camera);

//将RGBD相机得到的像素坐标和深度坐标转化为3D坐标
cv::Point3f point2dTo3d(cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera);

//提取特征点并计算描述子(ORB)
void computeKeypointsAndDescriptors(FRAME & frame);

//PnP求解两帧之间的运动R_2_1,t_2_1
RESULT_OF_PNP estimateMotion(FRAME & frame_1, FRAME & frame_2, CAMERA_INTRINSIC_PARAMETERS& camera);
bool estimateMotion2(FRAME & frame_1, FRAME & frame_2, CAMERA_INTRINSIC_PARAMETERS& camera, RESULT_OF_PNP& result);
//将Opencv格式的旋转向量和平移向量转化为Eigen中的变换矩阵
Eigen::Isometry3d cvMat2Eigen(cv::Mat& rvec, cv::Mat& tvec);

//将新来的帧加入到原始点云,运动T为T_21
PointCloud::Ptr joinPointCloud( PointCloud::Ptr original, FRAME& newFrame, Eigen::Isometry3d& T,
                                CAMERA_INTRINSIC_PARAMETERS& camera);
