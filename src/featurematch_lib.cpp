#include"slamBase.h"
#include<opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>

//提取特征点并计算描述子(ORB)
void computeKeypointsAndDescriptors(FRAME & frame)
{
    cv::Mat img;
    img = frame.rgb;

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptor;

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor_extra = cv::ORB::create();

    detector->detect(img,keypoints);
    descriptor_extra->compute(img,keypoints,descriptor);

    frame.keypoints = keypoints;
    frame.descriptors = descriptor;

}

//PnP求解两帧之间的运动R_2_1,t_2_1
RESULT_OF_PNP estimateMotion(FRAME & frame_1, FRAME & frame_2, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    //计算特征点与描述子
    computeKeypointsAndDescriptors(frame_1);
    computeKeypointsAndDescriptors(frame_2);
    //匹配(汉明距离，暴力匹配)
    cv::Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    vector<cv::DMatch> matches;
    matcher->match ( frame_1.descriptors , frame_2.descriptors, matches );
    //初步筛选匹配结果
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
            cv::Point2f  p_1 = frame_1.keypoints[match.queryIdx].pt;
            ushort d_1 = frame_1.depth.ptr<ushort>(int(p_1.y))[int(p_1.x)];
            if (d_1 == 0)
                continue;
            cv::Point3f p_1_ ;
            p_1_.x = p_1.x;
            p_1_.y = p_1.y;
            p_1_.z = d_1;
            p_1_ = point2dTo3d(p_1_, camera);
            pts_3d.push_back(p_1_);

            cv::Point2f p_2 = frame_2.keypoints[match.trainIdx].pt;
            pts_2d.push_back(p_2);
        }
    }
//    cv::Mat goodImgMatch;
//    cv::drawMatches(frame_1.rgb, frame_1.keypoints, frame_2.rgb, frame_2.keypoints, goodmatches, goodImgMatch);
//    cv::imshow("GoodMatch"+std::to_string(goodmatches.size()), goodImgMatch);
//    cv::waitKey(0);

    //求解PnP
    double camera_intrinsic_data[3][3] = {{camera.fx, 0 ,camera.cx},
                                                                   {0,camera.fy, camera.cy},
                                                                   {0,0,1}};
    cv::Mat camera_K(3,3,CV_64F,camera_intrinsic_data);
    cv::Mat rvec, tvec, inliers;
    cv::solvePnPRansac(pts_3d, pts_2d, camera_K, cv::Mat(), rvec, tvec, false, 100, 1.0f, 0.9, inliers);
//   cv::solvePnP ( pts_3d, pts_2d, camera_K, cv::Mat(), rvec, tvec, false );
//    cv::Mat R;
//    cv::Rodrigues ( rvec, R ); // r为旋转向量形式，用Rodrigues公式转换为矩阵
//    cout<<"R="<<endl<<R<<endl;
//    cout<<"t="<<endl<<tvec<<endl;

    //画出Ransac之后的匹配点对
//    vector<cv::DMatch> matchesShow;
//    for ( int i = 0; i<inliers.rows; i++)
//    {
//        matchesShow.push_back(goodmatches[inliers.ptr<int>(i)[0]]);
//    }
//    cv::Mat matchespicture;
//    cv::drawMatches(frame_1.rgb, frame_1.keypoints, frame_2.rgb, frame_2.keypoints, matchesShow, matchespicture);
//    cv::imshow("After Ransac"+std::to_string(matchesShow.size()), matchespicture);
//    cv::waitKey(0);
    RESULT_OF_PNP result;
    result.rvec = rvec;
    result.tvec = tvec;
    result.inliers = inliers.rows;
    return result;
}

//PnP求解两帧之间的运动R_2_1,t_2_1
bool estimateMotion2(FRAME & frame_1, FRAME & frame_2, CAMERA_INTRINSIC_PARAMETERS& camera, RESULT_OF_PNP& result)
{
    //计算特征点与描述子
    computeKeypointsAndDescriptors(frame_1);
    computeKeypointsAndDescriptors(frame_2);
    //匹配(汉明距离，暴力匹配)
    cv::Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    vector<cv::DMatch> matches;
    matcher->match ( frame_1.descriptors , frame_2.descriptors, matches );
    //初步筛选匹配结果
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
            cv::Point2f  p_1 = frame_1.keypoints[match.queryIdx].pt;
            ushort d_1 = frame_1.depth.ptr<ushort>(int(p_1.y))[int(p_1.x)];
            if (d_1 == 0)
                continue;
            cv::Point3f p_1_ ;
            p_1_.x = p_1.x;
            p_1_.y = p_1.y;
            p_1_.z = d_1;
            p_1_ = point2dTo3d(p_1_, camera);
            pts_3d.push_back(p_1_);

            cv::Point2f p_2 = frame_2.keypoints[match.trainIdx].pt;
            pts_2d.push_back(p_2);
        }
    }
//    cv::Mat goodImgMatch;
//    cv::drawMatches(frame_1.rgb, frame_1.keypoints, frame_2.rgb, frame_2.keypoints, goodmatches, goodImgMatch);
//    cv::imshow("GoodMatch"+std::to_string(goodmatches.size()), goodImgMatch);
//    cv::waitKey(0);

    //求解PnP
    ParameterReader pd;
    int min_good_match = atoi(pd.getData("min_good_match").c_str());
    if ( pts_3d.size() < min_good_match)    //EPNP求解需要四对不共面的匹配点，共面需要3对
        return false;
    else
    {
        double camera_intrinsic_data[3][3] = {{camera.fx, 0 ,camera.cx},
                                                                       {0,camera.fy, camera.cy},
                                                                       {0,0,1}};
        cv::Mat camera_K(3,3,CV_64F,camera_intrinsic_data);
        cv::Mat rvec, tvec, inliers;
        cv::solvePnPRansac(pts_3d, pts_2d, camera_K, cv::Mat(), rvec, tvec, false, 100, 1.0f, 0.9, inliers);
        result.rvec = rvec;
        result.tvec = tvec;
        result.inliers = inliers.rows;
        vector<cv::DMatch> matchesShow;
        for ( int i = 0; i<inliers.rows; i++)
        {
            matchesShow.push_back(goodmatches[inliers.ptr<int>(i)[0]]);
        }
        cv::Mat matchespicture;
        cv::drawMatches(frame_1.rgb, frame_1.keypoints, frame_2.rgb, frame_2.keypoints, matchesShow, matchespicture);
        cv::Mat notation_1(50, matchespicture.cols, CV_8UC3, cv::Scalar(255,255,255));
        cv::putText(notation_1,"After Ransac there are "+std::to_string(matchesShow.size())+" features.", cv::Point2f(20,30),CV_FONT_HERSHEY_SIMPLEX,1,cv::Scalar(255),3);
        cv::vconcat(notation_1, matchespicture, matchespicture);
        cv::imshow("After Ransac", matchespicture);
        cv::waitKey(5);
        return true;
    }

}
