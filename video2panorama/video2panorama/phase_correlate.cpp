//#include <iostream>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
//#include <opencv2/core/core.hpp> 
//#include <opencv2/opencv.hpp>
//#include <fstream>
//
//using namespace cv;
//using namespace std;
//using namespace xfeatures2d;
//
//int time_watermark1 = 120;
//int date_watermark1 = 962;
//
//int preproccess(Mat src, Mat &no_watermark) {
//	no_watermark = src(Rect(0, time_watermark1, src.cols, src.rows - (src.rows - date_watermark1) - time_watermark1));
//	return 0;
//}
//
//int preproccess_from_point_WH(Mat src, Mat &no_watermark,Point left_top,int width,int height) {
//	//no_watermark = src(Rect(0, time_watermark, src.cols, src.rows - (src.rows - date_watermark) - time_watermark));
//	no_watermark = src(Rect(left_top.x,left_top.y,width,height));
//	return 0;
//}
//
//int get_frames_by_video(string video_path) {
//
//
//
//	return 0;
//}
//
//double phase_correlate(Mat left,Mat right) {
//	//double shift = 0.0;
//	Mat dst1, dst2;
//	if (left.empty()) { return -1.0; }
//	if (right.empty()) { return -2.0;}
//	
//	cvtColor(left, left, CV_BGR2GRAY);     //转换为灰度图像
//	left.convertTo(dst1, CV_64FC1);       //转换为32位浮点型
//	cvtColor(right, right, CV_BGR2GRAY);
//	right.convertTo(dst2, CV_64FC1);
//
//	Point2d phase_shift;
//	phase_shift = phaseCorrelate(dst1, dst2);
//	//cout << endl << "warp :" << endl << "\tX shift : " << phase_shift.x << "\tY shift : " << phase_shift.y << endl;
//
//	return phase_shift.x;
//}
//
//Mat orb_findhomographyFromframes1(Mat img1, Mat img2, bool watermark, string match_method = "BruteForce-Hamming") {
//	Mat homo;
//	vector<KeyPoint> keypoints1, keypoints2;
//	Mat des1, des2;
//	vector<DMatch> matches;
//	vector<DMatch> bf_goodmatches;
//	vector<DMatch> flann_goodmatches;
//	vector<Point2f> img_point2f_1, img_point2f_2;
//	vector<Point2f> point2f_1, point2f_2;
//	Ptr<ORB> orb = ORB::create(5000);
//
//	orb->detect(img1, keypoints1);
//	orb->detect(img2, keypoints2);
//
//	if (watermark) {
//	
//		for (int index = 0; index < keypoints1.size(); index++) {
//			Point2f point = keypoints1[index].pt;
//			if (time_watermark1 < point.y && point.y < date_watermark1)
//			{
//				point2f_1.push_back(point);
//			}
//		}
//
//		for (int index = 0; index < keypoints2.size(); index++) {
//			Point2f point = keypoints2[index].pt;
//			if (time_watermark1 < point.y && point.y < date_watermark1)
//			{
//				point2f_2.push_back(point);
//			}
//		}
//		KeyPoint::convert(point2f_1, keypoints1);
//		KeyPoint::convert(point2f_2, keypoints2);
//	}
//	orb->compute(img1, keypoints1, des1);
//	orb->compute(img2, keypoints2, des2);
//
//	if (match_method == "BruteForce-Hamming")
//	{
//		Ptr<DescriptorMatcher> des_matcher = DescriptorMatcher::create("BruteForce-Hamming");
//		des_matcher->match(des1, des2, matches);
//
//		double min_dist = 0, max_dist = 0;
//		for (int i = 0; i < des1.rows; i++)
//		{
//			if (min_dist < matches[i].distance)
//			{
//				min_dist = matches[i].distance;
//			}
//			if (max_dist > matches[i].distance)
//			{
//				max_dist = matches[i].distance;
//			}
//		}
//
//		for (int i = 0; i < des1.rows; i++)
//		{
//			if (matches[i].distance < max(min_dist * 2, 30.0))
//			{
//				bf_goodmatches.push_back(matches[i]);
//			}
//		}
//
//		for (int i = 0; i < bf_goodmatches.size(); i++)
//		{
//			img_point2f_1.push_back(keypoints1[bf_goodmatches[i].queryIdx].pt);
//			img_point2f_2.push_back(keypoints2[bf_goodmatches[i].trainIdx].pt);
//		}
//
//		
//		homo = findHomography(img_point2f_1, img_point2f_2, CV_RANSAC);
//
//	}
//
//	return homo;
//}
//int main() {
//
//	//Mat src1, src2;
//	//src1 = imread("D:/imgsource/frame/76.jpg");
//	
//	//src2 = imread("D:/imgsource/frame/77.jpg");
//	//Mat dst1, dst2;
//	//Mat homo = orb_findhomographyFromframes1(src1, src2,false);
//	//cout << homo << endl;
//
//	//dst1 = imread("D:/imgsource/frames/76.jpg");
//	//dst2 = imread("D:/imgsource/frames/77.jpg");
//	//Mat homo1 = orb_findhomographyFromframes1(dst1, dst2, true);
//	//cout << homo1 << endl;
//	//dst1 = imread("D:/imgsource/frames/76.jpg");
//	//dst2 = imread("D:/imgsource/frames/77.jpg");
//	//Mat homo2 = orb_findhomographyFromframes1(dst1, dst2, false);
//	//cout << homo2 << endl;
//
//	//cvtColor(src1, src1, CV_BGR2GRAY);     //转换为灰度图像
//	//src1.convertTo(dst1, CV_64FC1);       //转换为32位浮点型
//	//cvtColor(src2, src2, CV_BGR2GRAY);
//	//src2.convertTo(dst2, CV_64FC1);
//
//	//Point2d phase_shift;
//	//phase_shift = phaseCorrelate(dst1, dst2);
//	//cout << endl << "warp :" << endl << "\tX shift : " << phase_shift.x << "\tY shift : " << phase_shift.y << endl;
//
//	//waitKey(0);
//
//	string video_path = "C:/Users/Administrator/Videos/SingleVideoStitching/videosource/s1.mp4";
//	string homo_shift_path = "D:/imgsource/panorama/905homo_shift.txt";
//	string phase_shift_path = "D:/imgsource/panorama/911phase_shift.txt";
//	ofstream file;
//	ofstream file_phase;
//	file.open(homo_shift_path);
//	file_phase.open(phase_shift_path);
//	assert(file.is_open());
//	assert(file_phase.is_open());
//
//	VideoCapture cap;
//	cap.open(video_path);
//	if (!cap.isOpened()) {
//		cout << "video_open_failed"<<endl;
//		return -1;
//	}
//	//cap.set(CAP_PROP_POS_FRAMES,60);
//	Mat preMat,curMat,no_watermark_pre, no_watermark_cur,homo;
//	double homo_shift,phase_correlate_shift;
//	cap >> preMat;
//	if (preMat.empty()) {
//		cout << "preMat_empty" << endl;
//		return -1;
//	}
//
//	Point left_top;
//	int width = preMat.cols / 4;
//	int height = preMat.rows / 4;
//	//left_top.x = preMat.cols / 2 - (width / 2);
//	//left_top.y = preMat.rows / 2 - (height / 2);
//	left_top.x = 50;
//	left_top.y = preMat.rows - height;
//
//
//
//	while (cap.isOpened()) {
//		cap >> curMat;
//		if (!curMat.empty()) {
//
//		/*	preproccess(preMat, no_watermark_pre);
//			preproccess(curMat, no_watermark_cur);*/
//
//			preproccess_from_point_WH(preMat, no_watermark_pre, left_top, width, height);
//			preproccess_from_point_WH(curMat, no_watermark_cur, left_top, width, height);
//
//
//		/*	homo = orb_findhomographyFromframes1(no_watermark_pre, no_watermark_cur, false);
//			homo_shift = homo.at<double>(0, 2);
//			file << homo_shift << endl;
//			cout << "homo_shift" << homo_shift <<endl;*/
//
//			phase_correlate_shift = phase_correlate(no_watermark_pre, no_watermark_cur);
//			file_phase << phase_correlate_shift << endl;
//			cout << "phase_correlate_shift" << phase_correlate_shift <<endl;
//			swap(preMat, curMat);
//		}
//		else {
//			cout << "curMat_empty" << endl;
//			return -1;
//		}
//		//waitKey(20);
//	}
//	file.close();
//	file_phase.close();
//
//	
//	system("pause");
//	return 0;
//}