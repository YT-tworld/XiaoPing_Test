#include "ros/ros.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <vector>
#include <string>

class CubeDetector {
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    std::vector<cv::Point2f> corner_points_; // 四角点坐标
    int detected_number_;                    // 识别到的数字
public:    
    std::vector<cv::Mat> num_templates_;     // 数字蒙版模板（mask_1到mask_6）

public:
    // 1-0.对象初始化函数
    CubeDetector(ros::NodeHandle& nh, const std::string& template_dir = "./") 
        : nh_(nh), it_(nh), detected_number_(-1) {
        // 加载数字蒙版模板（mask_1.png到mask_6.png）
        for (int i = 1; i <= 24; i++) {
            std::string path = template_dir+ std::to_string(i) + ".png";
            cv::Mat temp = cv::imread(path, cv::IMREAD_GRAYSCALE);
            if (temp.empty()) {
                ROS_WARN("未找到蒙版模板: %s", path.c_str());
            } else {
                num_templates_.push_back(temp);
                ROS_INFO("加载蒙版模板: %s", path.c_str());
            }
        }

        // 订阅图像话题
        image_sub_ = it_.subscribe("/cube_image", 1, &CubeDetector::imageCallback, this);
        ROS_INFO("节点启动，开始订阅图像话题...");
    }

    // 2-0.图像回调函数：主处理流程
    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        try {
            // 1. 转换ROS图像为OpenCV格式（过）
            cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
            if (image.empty()) return;

            // 2. 颜色蒙版：提取方块所在区域
            cv::Mat color_mask = createColorMask(image);//得到一个黑白图，白色为目标区域
            cv::Mat cube_roi;
            image.copyTo(cube_roi, color_mask); // 只保留方块区域（把白色区域换回原来的颜色）
            //检测提取到的区域图像彩图
            cv::imshow("area_color",cube_roi);//窗口名称，图像
            cv::waitKey(1000);//停多少毫秒，0就一直卡住
            cv::destroyAllWindows();//关闭所有opcv窗口

            // 3. 预处理：二值化+轮廓检测
            cv::Mat gray_roi, binary_figure,binary_cube;
            cv::cvtColor(cube_roi, gray_roi, cv::COLOR_BGR2GRAY);//彩图转灰度图
            cv::threshold(gray_roi, binary_figure, 110, 255, cv::THRESH_BINARY); // 二值化，大于100为255白色，其他为黑
            cv::threshold(gray_roi, binary_cube, 1,   255, cv::THRESH_BINARY); // 二值化，大于100为255白色，其他为黑

            //检测原数字二值化图像
            cv::imshow("figure_binary",binary_figure);//窗口名称，图像
            cv::waitKey(1000);//停多少毫秒，0就一直卡住
            cv::destroyAllWindows();//关闭所有opcv窗口
            //检测原方块二值化图像
            cv::imshow("cube_binary",binary_cube);//窗口名称，图像
            cv::waitKey(1000);//停多少毫秒，0就一直卡住
            cv::destroyAllWindows();//关闭所有opcv窗口
            

            // 4. 模板匹配：识别数字
            int templet_num=matchNumberTemplate(binary_figure);
            detected_number_=-1;
            while(templet_num>0)
            {
                if(detected_number_==-1)
                    detected_number_++;
                templet_num-=4;
                detected_number_++;
            }
           
            // 5. 提取四角点
            if (detected_number_ != -1) {
                corner_points_ = extractCorners(binary_cube);
                // 输出结果到终端
                printResult(detected_number_, corner_points_);
            } else {
                corner_points_.clear();
            }

            // 6. 可视化
            visualizeResult(image, corner_points_, detected_number_);
            cv::waitKey(1);


        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("图像转换失败: %s", e.what());
        }
    }

    // 2-1.生成颜色蒙版：过滤方块区域（根据实际颜色调整HSV），最终得到白色文字，红色方块，其他全黑
    cv::Mat createColorMask(const cv::Mat& image) {
        // 步骤1：先将RGB颜色格式的image转为HSV颜色格式的hsv
        cv::Mat hsv, red_mask;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
        
        // 步骤2：【先提取红色的地方(方块)】提取红色方块的大致区域（红色蒙版）
        // 红色的HSV范围（只需要大致框住方块即可，不用太精确）
        cv::Scalar red_lower(0, 100, 50);   // 低阈值（放宽一点，确保方块都被包含）
        cv::Scalar red_upper(10, 255, 255); // 高阈值
        cv::inRange(hsv, red_lower, red_upper, red_mask);
        
        // 步骤3：【再提取白色区域(文字)（白色的HSV范围）
        cv::Mat white_mask;
        // 白色的HSV特征：高亮度、低饱和度（接近白色）
        cv::Scalar white_lower(0, 0, 200);   // S=0（灰度/白色），V≥200（高亮度）
        cv::Scalar white_upper(180, 30, 255); // H任意（白色无色调），S≤30（低饱和度）
        cv::inRange(hsv, white_lower, white_upper, white_mask);
        
        // 步骤4：合并蒙版——只保留“红色方块内的白色文字”和“红色方块本身”
        // （用逻辑或：红色区域 或 白色文字区域，都保留）
        cv::Mat final_mask;
        cv::bitwise_or(red_mask, white_mask, final_mask);
        
        // 步骤4：形态学操作去除噪点（让区域更完整）
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(final_mask, final_mask, cv::MORPH_CLOSE, kernel);
    
    return final_mask;
    }

    // 2-2.模板匹配：识别数字（1-6）
    // 整合版：包含裁剪正方形→缩放→模板匹配三步流程
    int matchNumberTemplate(const cv::Mat& binary_figure) {
        if (num_templates_.empty()) {
            ROS_WARN("模板库为空，无法识别");
            return -1;
        }
        // --------------------------
        // 第一步：裁剪为正方形（不剪到数字）
        // --------------------------
        cv::Mat square_cropped;
        std::vector<std::vector<cv::Point>> contours;
        // 查找数字轮廓（白色区域），把所有连续的白点作为一个轮廓存到contours
        cv::findContours(binary_figure, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        if (contours.empty()) {
            ROS_WARN("未找到数字轮廓，无法裁剪");
            return -1;
        }

        // 找到最大轮廓（假设是数字）
        int max_idx = 0;
        double max_area = cv::contourArea(contours[0]);
        for (size_t i = 1; i < contours.size(); ++i) {
            double area = cv::contourArea(contours[i]);
            if (area > max_area) {
                max_area = area;
                max_idx = i;
            }
        }

        // 数字的包围框
        cv::Rect num_rect = cv::boundingRect(contours[max_idx]);
        // 正方形边长=包围框宽高的最大值（确保装下数字）
        int side_len = std::max(num_rect.width, num_rect.height);
        // 正方形中心与数字中心对齐
        cv::Point2f num_center(
            num_rect.x + num_rect.width / 2.0,
            num_rect.y + num_rect.height / 2.0
        );
        int square_x = static_cast<int>(num_center.x - side_len / 2.0);
        int square_y = static_cast<int>(num_center.y - side_len / 2.0);
        cv::Rect square_rect(square_x, square_y, side_len, side_len);

        // 处理超出原图像的区域（用黑色填充）
        square_cropped = cv::Mat::zeros(side_len, side_len, binary_figure.type());
        cv::Rect overlap_rect = square_rect & cv::Rect(0, 0, binary_figure.cols, binary_figure.rows);
        cv::Rect roi_in_square(
            overlap_rect.x - square_rect.x,
            overlap_rect.y - square_rect.y,
            overlap_rect.width,
            overlap_rect.height
        );
        binary_figure(overlap_rect).copyTo(square_cropped(roi_in_square));
        ROS_INFO("第一步：裁剪正方形完成（尺寸: %dx%d）", side_len, side_len);
        //检测裁剪后的数字二值化图像
            cv::imshow("figure_image_cut",square_cropped);//窗口名称，图像
            cv::waitKey(1000);//停多少毫秒，0就一直卡住
            cv::destroyAllWindows();//关闭所有opcv窗口


        // --------------------------
        // 第二步：缩放至与模板相同尺寸
        // --------------------------
        cv::Mat resized_roi;
        cv::Size template_size = num_templates_[0].size(); // 用第一个模板的尺寸作为目标
        // 缩放图像（保持比例，确保与模板大小一致）
        cv::resize(square_cropped, resized_roi, template_size, 0, 0, cv::INTER_AREA);
        ROS_INFO("第二步：缩放到模板尺寸（%dx%d）", template_size.width, template_size.height);

        //检测裁缩放后的数字二值化图像
            cv::imshow("figure_image_zoom",resized_roi);//窗口名称，图像
            cv::waitKey(1000);//停多少毫秒，0就一直卡住
            cv::destroyAllWindows();//关闭所有opcv窗口

        // --------------------------
        // 第三步：模板匹配识别数字
        // --------------------------
        double max_score = 0.1; // 匹配阈值（0-1）（可根据实际调整）
        int best_idx = -1;

        for (size_t i = 0; i < num_templates_.size(); ++i) {
            cv::Mat result;
            // 计算相似度（归一化相关系数，范围-1~1）
            cv::matchTemplate(resized_roi, num_templates_[i], result, cv::TM_CCOEFF_NORMED);
            double score;
            cv::minMaxLoc(result, nullptr, &score); // 取最高相似度

            // 更新最佳匹配
            if (score > max_score) {
                max_score = score;
                best_idx = i + 1; // 模板索引0对应数字1
            }
        }
        ROS_INFO("第三步：模板匹配完成（最高相似度: %.2f，识别结果: %d号模版）", max_score, best_idx);
        return best_idx;
    }

    // 2-3.提取四角点（基于轮廓检测和多边形逼近）
    std::vector<cv::Point2f> extractCorners(const cv::Mat& binary_figure) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary_figure, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& cnt : contours) {
            double area = cv::contourArea(cnt);
            if (area < 300) continue; // 过滤小轮廓

            // 多边形逼近（找四边形）
            double peri = cv::arcLength(cnt, true);
            std::vector<cv::Point> approx;
            cv::approxPolyDP(cnt, approx, 0.02 * peri, true);

            if (approx.size() == 4) { // 四角形
                std::vector<cv::Point2f> corners;
                for (const auto& p : approx) corners.emplace_back(p);
                return sortCorners(corners); // 排序后返回
            }
        }
        return {};
    }

    // 2-4.排序四角点（顺时针：左上→右上→右下→左下）
    std::vector<cv::Point2f> sortCorners(const std::vector<cv::Point2f>& points) {
        if (points.size() != 4) return {};

        // 按x+y排序（左上最小，右下最大）
        std::vector<cv::Point2f> sorted = points;
        std::sort(sorted.begin(), sorted.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return (a.x + a.y) < (b.x + b.y);
        });

        // 区分右上和左下
        if (sorted[1].x > sorted[2].x) std::swap(sorted[1], sorted[2]);
        return sorted;
    }

    // 2-5.终端打印结果
    void printResult(int number, const std::vector<cv::Point2f>& corners) {
        if (number == -1 || corners.empty()) {
            ROS_INFO("未识别到有效数字或角点");
            return;
        }

        ROS_INFO("\n识别结果：");
        ROS_INFO("数字：%d", number);
        ROS_INFO("四角点坐标：");
        for (size_t i = 0; i < corners.size(); ++i) {
            ROS_INFO("角点%d：(%.1f, %.1f)", i+1, corners[i].x, corners[i].y);
        }
    }

    // 2-6.可视化结果
    void visualizeResult(cv::Mat& image, const std::vector<cv::Point2f>& corners, int number) {
        // 绘制四角点和边框
        if (!corners.empty()) {
            for (size_t i = 0; i < corners.size(); ++i) {
                cv::circle(image, corners[i], 5, cv::Scalar(0, 255, 0), -1);
                cv::line(image, corners[i], corners[(i+1)%4], cv::Scalar(255, 0, 0), 2);
            }
        }

        // 显示数字
        std::string text = (number != -1) ? "figure: " + std::to_string(number) : "未识别";
        cv::putText(image, text, cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);

        cv::imshow("识别结果", image);
        cv::waitKey(3000);//停多少毫秒，0就一直卡住
        cv::destroyAllWindows();//关闭所有opcv窗口

    }
};

int main(int argc, char**argv) {
    setlocale(LC_ALL,"");//避免日志中中文乱码
    ros::init(argc, argv, "cube_number_detector");
    ros::NodeHandle nh;

    // 初始化检测器（蒙版模板路径根据实际修改）
    CubeDetector detector(nh, "/home/yt/XiaoPingTest_ws/src/XiaoPingTest1/temple/");
    
    ros::spin();
    cv::destroyAllWindows();
    return 0;
}