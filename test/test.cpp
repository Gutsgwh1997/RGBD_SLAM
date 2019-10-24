#include"slamBase.h"

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
    std::cout<<"Relative rotation is "<<relativemontion.rvec<<std::endl;
    std::cout<<"Relative translation is "<<relativemontion.tvec<<std::endl;

    return 0;
}
