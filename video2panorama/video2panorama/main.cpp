//#include <opencv2/xfeatures2d/nonfree.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/opencv.hpp>
//#include <iostream>
//
//using namespace std;
//using namespace cv;
//using namespace xfeatures2d;
//
//vector<Point2f> point[2];
//vector<Point2f> features;
//vector<float> err;
//vector<uchar> status;
//int start_frame_num = 60;
//Mat cur_gray;
//Mat pre_gray;
//bool addNewPoints()
//{
//	return point[0].size() <= 10;
//}
//
//bool acceptTrackedPoint(int i)
//{
//	return status[i] && ((abs(point[0][i].x - point[1][i].x) + abs(point[0][i].y - point[1][i].y)) > 2);//检测是否为移动点
//}
//
//int stitching(Mat &pre_frame,Mat &cur_frame) {
//	cvtColor(cur_frame, cur_gray, CV_BGR2GRAY);
//	if (addNewPoints())
//	{
//		//goodFeaturesToTrack(pregray, features, maxCount, qLevel, minDist);
//		Ptr<ORB> orb = ORB::create(5000);
//		vector<KeyPoint> kps;
//		orb->detect(pre_frame, kps);
//		for (auto kp : kps)
//			features.push_back(kp.pt);
//		point[0].insert(point[0].end(), features.begin(), features.end());
//	}
//
//	calcOpticalFlowPyrLK(pre_gray, cur_gray, point[0], point[1], status, err);
//	int k = 0;
//	for (size_t i = 0; i<point[1].size(); i++)
//	{
//		if (acceptTrackedPoint(i))
//		{
//			//initPoint[k] = initPoint[i];
//			point[1][k++] = point[1][i];
//		}
//	}
//
//	float shift_x_mean = 0;
//	float shift_x_ = 0;
//	vector<float> shift_x;
//	for (int i = 0; i < point[0].size(); i++) {
//		shift_x_ = point[1][i].x - point[0][i].x;
//		shift_x_mean += shift_x_;
//		shift_x.push_back(shift_x_);
//		//file_shift_x_y << shift_x_ << endl;;
//	}
//
//	shift_x_mean = shift_x_mean / shift_x.size();
//
//	cout << "shift_x_mean" << shift_x_mean << endl;
//
//	point[1].resize(k);
//	swap(point[1], point[0]);
//	swap(pre_gray, cur_gray);
//	return 0;
//}
//
//
//
//int main() {
//	VideoCapture cap;
//	int open_video_result = cap.open("C:/Users/Administrator/Videos/SingleVideoStitching/videosource/s1.mp4");
//	if (!open_video_result)
//	{
//		return 0;
//	}
//
//	cap.set(CAP_PROP_POS_FRAMES, start_frame_num);
//	Mat start_frame;
//	cap >> start_frame;
//	
//	cvtColor(start_frame,pre_gray,CV_BGR2GRAY);
//
//	Mat frame;
//	
//	while (true) {
//		cap >> frame;
//		//cvtColor(frame, cur_gray,CV_BGR2GRAY);
//		if (!frame.empty())
//		{
//			stitching(start_frame,frame);
//			start_frame = frame;
//			/*swap(start_frame, frame);
//			swap(pre_gray, cur_gray);*/
//		}
//		else
//		{
//			cout << "frame_empty" << endl;
//			break;
//		}
//	}
//
//	system("pause");
//	return 0;
//}
