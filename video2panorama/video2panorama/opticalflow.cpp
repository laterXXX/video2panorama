//#include <iostream>  
//#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>  
//#include <opencv2/highgui/highgui.hpp> 
//#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
//#include <opencv2/ml/ml.hpp>
//#include <fstream>
//
//using namespace cv;
//using namespace std;
//
//
//void duan_OpticalFlow3(Mat &frame, Mat & result);
//bool addNewPoints3();
//bool acceptTrackedPoint3(int i);
//
//
//Mat curgray;    // 当前图片
//Mat pregray;    // 预测图片
//vector<Point2f> point[2];   // point0为特征点的原来位置，point1为特征点的新位置
//vector<Point2f> initPoint;  // 初始化跟踪点的位置
//vector<Point2f> features;   // 检测的特征
//int maxCount = 500;         // 检测的最大特征数
//double qLevel = 0.01;   // 特征检测的等级
//double minDist = 10.0;  // 两特征点之间的最小距离
//vector<uchar> status;   // 跟踪特征的状态，特征的流发现为1，否则为0
//vector<float> err;
//ofstream file_optical_flow;
//
//
//
//int main()
//{
//
//	Mat matSrc;
//	Mat matRst;
//
//	string optical_flow = "d:/imgsource/panorama/906optical_flow.txt";
//	file_optical_flow.open(optical_flow);
//
//
//	VideoCapture cap;
//	cap.open("C:/Users/Administrator/Videos/SingleVideoStitching/videosource/s1.mp4");
//	//int totalFrameNumber = cap.get(CV_CAP_PROP_FRAME_COUNT);
//
//	// perform the tracking process
//	printf("Start the tracking process, press ESC to quit.\n");
//	while (cap.isOpened()) {
//		// get frame from the video
//		cap >> matSrc;
//		if (!matSrc.empty())
//		{
//			duan_OpticalFlow3(matSrc, matRst);
//
//		}
//		else
//		{
//			cout << "Error : Get picture is empty!" << endl;
//		}
//		if (waitKey(1) == 27) break;
//	}
//
//
//	waitKey(0);
//	file_optical_flow.close();
//	return 0;
//
//}
//
//
//
//void duan_OpticalFlow3(Mat &frame, Mat & result)
//{
//	cvtColor(frame, curgray, CV_BGR2GRAY);
//	frame.copyTo(result);
//
//	if (addNewPoints3())
//	{
//		goodFeaturesToTrack(curgray, features, maxCount, qLevel, minDist);
//		point[0].insert(point[0].end(), features.begin(), features.end());
//		initPoint.insert(initPoint.end(), features.begin(), features.end());
//	}
//
//
//	if (pregray.empty())
//	{
//		curgray.copyTo(pregray);
//	}
//
//	calcOpticalFlowPyrLK(pregray, curgray, point[0], point[1], status, err);
//
//
//	int k = 0;
//	for (size_t i = 0; i<point[1].size(); i++)
//	{
//		if (acceptTrackedPoint3(i))
//		{
//			initPoint[k] = initPoint[i];
//			point[1][k++] = point[1][i];
//		}
//	}
//
//	float shift_x_mean = 0;
//	vector<float> shift_x;
//	for (int i = 0; i < point[0].size(); i++) {
//		float shift_x_ = point[1][i].x - point[0][i].x;
//		shift_x_mean += shift_x_;
//		shift_x.push_back(shift_x_);
//		//file_shift_x_y << shift_x_ << endl;;
//	}
//	
//	shift_x_mean = shift_x_mean / point[0].size();
//
//	file_optical_flow << shift_x_mean << endl;
//
//
//	point[1].resize(k);
//	initPoint.resize(k);
//
//	for (size_t i = 0; i<point[1].size(); i++)
//	{
//		line(result, initPoint[i], point[1][i], Scalar(0, 0, 255));
//		circle(result, point[1][i], 3, Scalar(0, 255, 0), -1);
//	}
//
//
//	swap(point[1], point[0]);
//	swap(pregray, curgray);
//
//
//	imshow("Optical Flow Demo", result);
//	//waitKey(50);
//}
//
//
//bool addNewPoints3()
//{
//	return point[0].size() <= 10;
//}
//
//
//bool acceptTrackedPoint3(int i)
//{
//	return status[i] && ((abs(point[0][i].x - point[1][i].x) + abs(point[0][i].y - point[1][i].y)) > 2);//检测是否为移动点
//}