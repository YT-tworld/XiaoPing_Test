#include <opencv2/opencv.hpp>
#include <iostream>
#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>

int main(int argc, char *argv[])
	{
    setlocale(LC_ALL,"");//避免日志中中文乱码
    ros::init(argc,argv,"node_image_pub"); //这个名称注册到master上
    ///1-1.读取图像
    cv::Mat image1=cv::imread("/home/yt/XiaoPingTest_ws/src/XiaoPingTest1/image_test/1.png");
    cv::Mat image2=cv::imread("/home/yt/XiaoPingTest_ws/src/XiaoPingTest1/image_test/2.png");
    ///1-2.发送图像
    ros::NodeHandle nh;// 创建节点句柄
    ros::Publisher img_pub = nh.advertise<sensor_msgs::Image>("/cube_image", 10);// 创建图像发布对象
    // 转换OpenCV图像为ROS消息并发布
    sensor_msgs::ImagePtr img_msg1 = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image1).toImageMsg();
    int count=1;
    ros::Rate rate(1);   //发布频率
    while(ros::ok())
    {
        ROS_INFO("开始发送图像第:%d张图",count); 
        img_pub.publish(img_msg1);
        count++;
        rate.sleep();
    }
    
    // ROS_INFO("开始显示图像");    
    // cv::imshow("input",image);//窗口名称，图像
    // cv::waitKey(0);//停多少秒，0就一直卡住
    // cv::destroyAllWindows();//关闭所有opcv窗口

	 return 0;
  }
