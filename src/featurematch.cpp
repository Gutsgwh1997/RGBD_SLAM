#include "slamBase.h"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

int main(int argc, char* argv[])
{
    cv::Mat rgb_1,rgb_2;
    rgb_1 = cv::imread("./data/rgb_1.png");
    cv::Mat depth_1 = cv::imread("./data/depth_1.png");
    rgb_2 = cv::imread("./data/rgb_2.png");

    //读取相机的内参
    ParameterReader PR;
    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.cx = atof( PR.getData("camera.cx").c_str());
    camera.cy = atof( PR.getData("camera.cy").c_str());
    camera.fx = atof( PR.getData("camera.fx").c_str());
    camera.fy = atof( PR.getData("camera.fy").c_str());
    camera.scale = atof( PR.getData("camera.scale").c_str());

    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1,descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    detector->detect (rgb_1, keypoints_1);
    detector->detect (rgb_2, keypoints_2);

    descriptor->compute(rgb_1, keypoints_1, descriptors_1);
    descriptor->compute(rgb_2, keypoints_2, descriptors_2);

//    cv::Mat outimg1;
//    cv::drawKeypoints( rgb_1, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
//    cv::imshow("ORBfeatures",outimg1);
//    cv::waitKey(0);

    vector<cv::DMatch> matches;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, matches );

    vector<cv::DMatch> goodmatches;
    vector<cv::Point3f> pts_3d;
    vector<cv::Point2f> pts_2d;
    double minDis = 9999;
    for (cv::DMatch match:matches)
    {
        if(match.distance!=0 && match.distance<minDis)
            minDis = match.distance;
    }
    for (cv::DMatch match:matches)
    {
        if (match.distance<4*minDis)
        {
            goodmatches.push_back(match);
            cv::Point2f  p_1 = keypoints_1[match.queryIdx].pt;
            ushort d_1 = depth_1.ptr<ushort>(int(p_1.y))[int(p_1.x)];
            if (d_1 == 0)
                continue;
            cv::Point3f p_1_ ;
            p_1_.x = p_1.x;
            p_1_.y = p_1.y;
            p_1_.z = d_1;
            p_1_ = point2dTo3d(p_1_,camera);
            pts_3d.push_back(p_1_);

            cv::Point2f p_2 = keypoints_2[match.trainIdx].pt;
            pts_2d.push_back(p_2);
        }
    }
    cv::Mat goodImgMatch;
    cv::drawMatches(rgb_1, keypoints_1, rgb_2, keypoints_2, goodmatches, goodImgMatch);
    cv::imshow("GoodMatch"+std::to_string(goodmatches.size()), goodImgMatch);

    //求解pnp
    double camera_intrinsic_data[3][3] = {{camera.fx, 0 ,camera.cx},
                                                                   {0,camera.fy, camera.cy},
                                                                   {0,0,1}};
    cv::Mat camera_K(3,3,CV_64F,camera_intrinsic_data);
    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac(pts_3d, pts_2d, camera_K, cv::Mat(), rvec, tvec, false, 100, 1.0f, 0.9, inliers);
    //cv::solvePnP ( pts_3d, pts_2d, camera_K, cv::Mat(), rvec, tvec, false );
    cv::Mat R;
    cv::Rodrigues ( rvec, R ); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    cout<<"R="<<endl<<R<<endl;
    cout<<"t="<<endl<<tvec<<endl;

    //画出Ransac之后的匹配点对
    vector<cv::DMatch> matchesShow;
    for ( int i = 0; i<inliers.rows; i++)
    {
        matchesShow.push_back(goodmatches[inliers.ptr<int>(i)[0]]);
    }
    cv::Mat matchespicture;
    cv::drawMatches(rgb_1, keypoints_1, rgb_2, keypoints_2, matchesShow, matchespicture);
    cv::imshow("After Ransac"+std::to_string(matchesShow.size()), matchespicture);
    cv::waitKey(0);
    return 0;
}
