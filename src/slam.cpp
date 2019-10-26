/**
* @projectName   RGBD_SLAM
* @brief          一个相对完整的RGBDSLAM,包括关键帧的选取，回环检测．
* @author       GWH_HIT
* @date          2019-10-25
**/
#include "slamBase.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

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

using namespace std;

//设置表示结果的枚举量
enum CHECK_RESULT {NOT_MATCHED = 0, TOO_FAR_AWAY, TOO_CLOSE, KEYFRAME};

void readPictures(int index, vector<string>& rgb_files, vector<string>& depth_files);

CHECK_RESULT checkKeyframes( FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opt, bool is_loops,
                             CAMERA_INTRINSIC_PARAMETERS& camera);

void checkNearbyLoops( vector<FRAME>& frames, FRAME& frame, g2o::SparseOptimizer& opt,
                            CAMERA_INTRINSIC_PARAMETERS& camera);

void checkRandomLoops( vector<FRAME>& frames, FRAME& frame, g2o::SparseOptimizer& opt,
                            CAMERA_INTRINSIC_PARAMETERS& camera);


//g2o的定义放在前面，定义的过程看g2o的框架图
typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

int main(int argc, char *argv[])
{
    //参数读取
    ParameterReader pd;
    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.cx = atoi( pd.getData("camera.cx").c_str());
    camera.cy = atof( pd.getData("camera.cy").c_str());
    camera.fx = atof( pd.getData("camera.fx").c_str());
    camera.fy = atof( pd.getData("camera.fy").c_str());
    camera.scale = atof( pd.getData("camera.scale").c_str());

    //读取数据
    vector<string> rgb_files;
    vector<string> depth_files;
    int start_index = atoi(pd.getData("start_index").c_str());
    int end_index = atoi(pd.getData("end_index").c_str());
    for ( int index = 1; index < 780; index++)
    {
        readPictures(index, rgb_files, depth_files);
    }

    //所有的关键帧在这
    vector<FRAME> keyframes;
    FRAME frame1, frame2;
    frame1.rgb = cv::imread(rgb_files[start_index]);
    frame1.depth = cv::imread(depth_files[start_index], -1); //注意灰度图，加参数-1!!!
    frame1.frameID = start_index;  //frameID是新添加的成员变量
    computeKeypointsAndDescriptors(frame1);
    PointCloud::Ptr cloud_raw = image2PointCloud(frame1.rgb, frame1.depth, camera);


    /***********************************************
   *g2o优化求解部分
   ************************************************/
    //初始化线性求解器
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering(false);
    //初始化块求解器
    SlamBlockSolver* blockSolver = new SlamBlockSolver( linearSolver );
    //选择优化方法
    g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg( blockSolver );
    //最总要的SparseOptimizer
    g2o::SparseOptimizer globalOptimizer;
    globalOptimizer.setAlgorithm(solver);

    //不输出调试信息
    globalOptimizer.setVerbose(false);

    //将初始帧加入到graph中
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId(start_index);
    v->setFixed( true );
    v->setEstimate( Eigen::Isometry3d::Identity());
    globalOptimizer.addVertex(v);

    keyframes.push_back(frame1);

    //是否检查回环
    bool check_loop_closure = pd.getData("check_loop_closure") == string("yes");

    for ( int i = start_index+1; i <= end_index; i++)
    {
        cout<<"Reading "<<i<<" file"<<endl;
        frame2.depth = cv::imread(depth_files[i], -1);
        frame2.rgb = cv::imread(rgb_files[i]);
        frame2.frameID = i;
        //将新帧与关键帧的最后一帧匹配比较
        CHECK_RESULT result = checkKeyframes( keyframes.back(), frame2, globalOptimizer, false, camera);

        switch (result)
        {
        case NOT_MATCHED:
            //没有匹配上，直接跳过
            cout<<RED"Not enough inliers."<<endl;
            break;
        case TOO_FAR_AWAY:
            //太远了
            cout<<RED"Too far away, may be an error"<<endl;
            break;
        case TOO_CLOSE:
            //太近了，没必要作为关键帧
            cout<<YELLOW"Too closed, not a key frame"<<endl;
            break;
        case KEYFRAME:
            cout<<GREEN"This is a new keyframe"<<endl;
            //只对关键帧进行操作！！！
            //检测回环
            if (check_loop_closure)
            {
                checkNearbyLoops(keyframes, frame2, globalOptimizer, camera);
                checkRandomLoops(keyframes, frame2, globalOptimizer, camera);
            }
            keyframes.push_back(frame2);
            break;
        default:
            break;
        }

     }


        //优化
        cout<<RESET"Optimizing pose graph, vertise: "<<globalOptimizer.vertices().size()<<endl;
        globalOptimizer.save("./data/result_before.g2o");
        globalOptimizer.initializeOptimization();
        globalOptimizer.optimize(100); //优化步数
        globalOptimizer.save("./data/result_after.g2o");
        cout<<"Optimization done"<<endl;

        //拼接点云地图
        cout<<"saving the point map ..."<<endl;
        PointCloud::Ptr tmp (new PointCloud());
        PointCloud::Ptr output (new PointCloud());

        //体素滤波器
        pcl::VoxelGrid<PointT> voxel;    //pointT在slambase.h中定义了
        static double gridesize = atof(pd.getData("voxel_grid").c_str()); //体素大小
        voxel.setLeafSize(gridesize, gridesize, gridesize);

        //ｚ方向滤波器，把远点删掉
        pcl::PassThrough<PointT> pass;
        pass.setFilterFieldName("z");
        pass.setFilterLimits(0.0, 4.0);   //小于0.0m，大于4.0m的统统不要

        //可视化
        pcl::visualization::CloudViewer viewer("viewer of frames");
        string visualize_pointcloud = pd.getData("visualize_pointcloud");
        bool visualize = visualize_pointcloud == string("yes");

        for (size_t i = 0; i < keyframes.size(); i++)
        {
            //从g2o中取一帧,还是忘记了智能指针．．．．．．．
            g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex(keyframes[i].frameID));
            //优化后的6自由度位姿
            Eigen::Isometry3d T_21 = vertex->estimate();
            PointCloud::Ptr newCloud = image2PointCloud(keyframes[i].rgb, keyframes[i].depth, camera);

            //滤波部分
            voxel.setInputCloud(newCloud);
            voxel.filter( *tmp );
            pass.setInputCloud(tmp);
            pass.filter( *newCloud );
            //拼接
//            joinPointCloud() 这个函数里没有定义滤波部分，暂时不用
            //将新点云转换到第一帧参考坐标系下
            pcl::transformPointCloud( *newCloud, *output, T_21.inverse().matrix());
            *cloud_raw += *output;

            //可视化
            if (visualize)
                viewer.showCloud( cloud_raw );

            tmp ->clear();
            output->clear();

        }

        //再来一次体素滤波
        voxel.setInputCloud( cloud_raw );
        voxel.filter( *tmp );

        //去除离群点
        double meank= atof( pd.getData("meank").c_str());
        double stddevmulthresh = atof(pd.getData("stddevmulthresh").c_str());
        PointCloud::Ptr after_filter ( new PointCloud());
        pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
        statistical_filter.setMeanK(meank);                     //设置进行统计时，考虑查询点相邻点的数量
        statistical_filter.setStddevMulThresh(stddevmulthresh);   //是否为离群点的阈值
        statistical_filter.setInputCloud(tmp);
        statistical_filter.filter(*after_filter);

        //点云保存
        pcl::io::savePCDFile("./data/result.pcd", *after_filter);

        cout<<GREEN"Final map is saved."<<endl;
        globalOptimizer.clear();

    return 0;
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
  * Description: 　检测两帧之间的关系，内点不足，运动过大，是关键帧．
  *Calls:
  *Called By:
  *Input:
  *Output:
  *Others:             当is_loops为false时候，为正常的里程计状态，添加节点和边的信息
  *                         到图中；当is_loops为true时，为回环检测状态，只需添加边的信息．
  *                         g2o::SparseOptimizer&　一定要按照引用出传递，复制构造函数删除了
**********************************************************************************/
CHECK_RESULT checkKeyframes( FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opt, bool is_loops, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    static ParameterReader pd;
    static int min_inliers = atoi(pd.getData("min_inliers").c_str());
    static double max_norm = atof( pd.getData("max_norm").c_str());
    static double max_norm_loop = atof( pd.getData("max_norm_loop").c_str());
    static double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str());
    //添加鲁棒核函数
    static g2o::RobustKernel* robustKernel = g2o::RobustKernelFactory::instance()->construct("Cauchy");

    static bool neidian;
    static RESULT_OF_PNP result;
    neidian = estimateMotion2( f1, f2, camera, result);
    if (neidian == false)                   //在estimateMotion２中定义的，少于4对配对点，不pnp，否则出错.
    {
        cout<<RED"Too few match points, can't solve PNP."<<endl;
        return NOT_MATCHED;
    }
    if ( result.inliers < min_inliers)  //内点过少
        return NOT_MATCHED;
    double norm = normofTransform(result.rvec, result.tvec);
    cout<<WHITE"Relative montion is "<<norm<<endl;
    if ( is_loops == false)   //是否处于回环检测模式
    {
        if ( norm > max_norm)
            return TOO_FAR_AWAY;
    }
    else
    {
        if (norm > max_norm_loop)
            return TOO_FAR_AWAY;
        //可以另外设置回环检测时最大的距离
    }

    if ( norm <= keyframe_threshold)
        return TOO_CLOSE;  //满足条件的都是关键帧咯

    if (is_loops == false)  //非回环检测模式要添加节点
    {
        g2o::VertexSE3* v = new g2o::VertexSE3();
        v->setId(f2.frameID);
        v->setEstimate( Eigen::Isometry3d::Identity());
        opt.addVertex(v);
    }

    //添加边
    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
    edge->vertices()[0] = opt.vertex(f1.frameID);
    edge->vertices()[1] = opt.vertex( f2.frameID);
    Eigen::Isometry3d measurement = cvMat2Eigen(result.rvec, result.tvec);
    edge->setMeasurement( measurement);  //这里为什么要加一个逆？
    edge->setRobustKernel(robustKernel);
    //信息矩阵
    Eigen::Matrix<double,6,6> information = Eigen::Matrix<double,6,6>::Identity();
    //pose是6-D的，说以协方差矩阵是6x6的．
    //假设位置和角度的的估计相互独立．
    information(0,0) = information(1,1) = information(2,2) = 100;
    information(3,3) = information(4,4) = information(5,5) = 100;
    edge->setInformation(information);

    opt.addEdge(edge);

    return KEYFRAME;
}

/*********************************************************************************
  * Description:    近距离回环检测，检测关键帧序列后几帧出现的回环，匹配成功条件一条边
  *Calls:
  *Called By:
  *Input:
  *Output:
  *Others:
**********************************************************************************/
void checkNearbyLoops( vector<FRAME>& frames, FRAME& frame, g2o::SparseOptimizer& opt, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    static ParameterReader pd;
    static  int nearby_loops = atoi( pd.getData("nearby_loops").c_str());

    cout<<MAGENTA"Checking nearby loops..."<<endl;
    //把frame和frames最后nearby_loops检测一遍
    if (frames.size() <= nearby_loops)
    {
        //关键帧不足，逐个比较
        for ( size_t i=0; i < frames.size(); i++)
        checkKeyframes(frame, frames[i], opt, true, camera);
    }
    else
    {
        for ( size_t i = frames.size() - nearby_loops; i<frames.size(); i++)
            checkKeyframes(frame, frames[i], opt, true, camera);
    }
}

/*********************************************************************************
  * Description:    随机回环检测，随机在关键帧序列中抽n个帧，与当前帧进行匹配，匹配成功，添加一条边
  *Calls:
  *Called By:
  *Input:
  *Output:
  *Others:
**********************************************************************************/
void checkRandomLoops( vector<FRAME>& frames, FRAME& frame, g2o::SparseOptimizer& opt, CAMERA_INTRINSIC_PARAMETERS& camera)
{
    static ParameterReader pd;
    static int random_loops = atoi(pd.getData("random_loops").c_str());
    //初始化随机数发生器，srand设置相同值，每次随机结果一样
    srand((unsigned)time(NULL));

    cout<<MAGENTA"Checking random loops..."<<endl;
    if (frames.size() <= random_loops)
    {
        //关键帧不足，逐个比较
        for ( size_t i=0; i < frames.size(); i++)
        checkKeyframes(frame, frames[i],opt, true, camera);
    }
    else
    {
        for (int i = 0; i<random_loops; i++)
        {
            int index = rand()%frames.size();
            checkKeyframes(frame, frames[index], opt, true, camera);
        }
    }
}


