//#include <iostream>  
//#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>  
//#include <opencv2/highgui/highgui.hpp> 
//#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
//#include <opencv2/ml/ml.hpp>
//#include "opencv2/video/tracking.hpp"
//#include <fstream>
//#include <opencv2/xfeatures2d/nonfree.hpp>
//#include <time.h>
//using namespace cv;
//using namespace std;
//using namespace xfeatures2d;
//
//
//Mat curgray;    // 当前图片
//Mat pregray;    // 预测图片
//Mat curgray_opticalflowlk_features;    // 当前图片
//Mat pregray_opticalflowlk_features;    // 预测图片
//vector<Point2f> point[2];   // point0为特征点的原来位置，point1为特征点的新位置
//vector<Point2f> lk_orb_point[2];   // point0为特征点的原来位置，point1为特征点的新位置
//vector<Point2f> initPoint;  // 初始化跟踪点的位置
//vector<Point2f> features;   // 检测的特征
//vector<Point2f> lk_orb_features;   // 检测的特征
//int maxCount = 1500;         // 检测的最大特征数
//double qLevel = 0.01;   // 特征检测的等级
//double minDist = 10.0;  // 两特征点之间的最小距离
//vector<uchar> status;   // 跟踪特征的状态，特征的流发现为1，否则为0
//vector<uchar> status_opticalflowlk_features;   // 跟踪特征的状态，特征的流发现为1，否则为0
//vector<float> err;
//ofstream ofile;
//int cols_panorama = 1920;
//int frame_num = 50;
//Mat panorama;
//int digit_shfit = 0;
//Mat shift_mat;
//int time_watermark = 120;
//int date_watermark = 962;
//vector<int> vector_shift;
//KalmanFilter KF;
//const int stateNum = 3;
//const int measureNum = 1;
//Mat matSrc;
//Mat preMat;
//Mat matRst;
//int little_shift = 0;
//
//bool addNewPoints()
//{
//	return point[0].size() <= 10;
//}
//bool addNewPoints(vector<Point2f> point)
//{
//	return point.size() <= 10;
//}
//bool acceptTrackedPoint(int i)
//{
//	return status[i] && ((abs(point[0][i].x - point[1][i].x) + abs(point[0][i].y - point[1][i].y)) > 2);//检测是否为移动点
//}
//bool acceptTrackedPoint(int i, vector<uchar> status, vector<Point2f> point[2])
//{
//	return status[i] && ((abs(point[0][i].x - point[1][i].x) + abs(point[0][i].y - point[1][i].y)) > 2);//检测是否为移动点
//}
//
//
//int preproccess(Mat src, Mat &no_watermark) {
//	no_watermark = src(Rect(0, time_watermark, src.cols, src.rows - (src.rows - date_watermark) - time_watermark));
//	return 0;
//}
//
//
//int preproccess_from_point_WH(Mat src, Mat &no_watermark,Point left_top,int width,int height) {
//	//no_watermark = src(Rect(0, time_watermark, src.cols, src.rows - (src.rows - date_watermark) - time_watermark));
//	no_watermark = src(Rect(left_top.x,left_top.y,width,height));
//	return 0;
//}
//
//Mat orb_findhomographyFromframes(Mat img1, Mat img2, bool watermark, string match_method = "BruteForce-Hamming") {
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
//		//KeyPoint::convert(keypoints1, point2f_1);
//		//KeyPoint::convert(keypoints2, point2f_2);
//
//		//	for (int index = 0; index < img_point2f_1.size(); index++) {
//		//		//Point2f point = keypoints1[index].pt;
//		//		if (120 < img_point2f_1[index].y < 962)
//		//		{
//		//			point2f_1.push_back(img_point2f_1[index]);
//		//		}
//		//	}
//
//		//	for (int index = 0; index < img_point2f_2.size(); index++) {
//		//		//Point2f point = keypoints2[index].pt;
//		//		if (120 < img_point2f_2[index].y < 962)
//		//		{
//		//			point2f_2.push_back(img_point2f_2[index]);
//		//		}
//		//	}
//
//		for (int index = 0; index < keypoints1.size(); index++) {
//			Point2f point = keypoints1[index].pt;
//			if (time_watermark < point.y && point.y < date_watermark)
//			{
//				point2f_1.push_back(point);
//			}
//		}
//
//		for (int index = 0; index < keypoints2.size(); index++) {
//			Point2f point = keypoints2[index].pt;
//			if (time_watermark < point.y && point.y < date_watermark)
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
//		//if (watermark) {
//		//	for (int index = 0; index < img_point2f_1.size(); index++) {
//		//		//Point2f point = keypoints1[index].pt;
//		//		if (120 < img_point2f_1[index].y < 962)
//		//		{
//		//			point2f_1.push_back(img_point2f_1[index]);
//		//		}
//		//	}
//
//		//	for (int index = 0; index < img_point2f_2.size(); index++) {
//		//		//Point2f point = keypoints2[index].pt;
//		//		if (120 < img_point2f_2[index].y < 962)
//		//		{
//		//			point2f_2.push_back(img_point2f_2[index]);
//		//		}
//		//	}
//		//	homo = findHomography(point2f_1, point2f_2, CV_RANSAC);
//		//}
//		homo = findHomography(img_point2f_1, img_point2f_2, CV_RANSAC);
//
//	}
//
//	return homo;
//}
//
//
//int fusion(Mat preMat, Mat matSrc, Mat left, Mat right, int stich_region, int integer_shift_) {
//	int weight = 1;
//	int preMat_start = preMat.cols - stich_region * 2;
//	int matSrc_start = matSrc.cols - stich_region * 2 - integer_shift_;
//	int shift_mat_start = shift_mat.cols - integer_shift_ - stich_region - stich_region;
//	for (int i = 0; i < preMat.rows; i++) {
//		uchar* p_preMat = preMat.ptr<uchar>(i);
//		uchar* p_matSrc = matSrc.ptr<uchar>(i);
//		uchar* p_left = left.ptr<uchar>(i);
//		uchar* p_right = right.ptr<uchar>(i);
//		uchar* p_shift_mat = shift_mat.ptr<uchar>(i);
//		for (int j = 0; j < stich_region * 2; j++) {
//			weight = (stich_region * 2 - j) / (stich_region * 2);
//			p_shift_mat[(shift_mat_start + j) * 3] = p_preMat[(preMat_start + j) * 3] * weight + p_matSrc[(matSrc_start + j) * 3] * (1 - weight);
//			p_shift_mat[(shift_mat_start + j) * 3 + 1] = p_preMat[(preMat_start + j) * 3 + 1] * weight + p_matSrc[(matSrc_start + j) * 3 + 1] * (1 - weight);
//			p_shift_mat[(shift_mat_start + j) * 3 + 2] = p_preMat[(preMat_start + j) * 3 + 2] * weight + p_matSrc[(matSrc_start + j) * 3 + 2] * (1 - weight);
//		}
//	}
//	//imshow("",shift_mat);
//	//waitKey(1);
//	return 0;
//}
//
//int kalman_optimize(float measure) {
//
//	KF.transitionMatrix = (Mat_<float>(3, 3) << 1, 0, 1, 1, 0, 0, 0, 0, 1);  //转移矩阵A  
//																			 //setIdentity(KF.measurementMatrix);                                 //测量矩阵H  
//	KF.measurementMatrix = (Mat_<float>(1, 3) << 1, -1, 0);
//	cout << KF.measurementMatrix << endl;
//
//	setIdentity(KF.processNoiseCov, Scalar::all(1e-5));                //系统噪声方差矩阵Q  
//	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));            //测量噪声方差矩阵R  
//	setIdentity(KF.errorCovPost, Scalar::all(1));					   //后验错误估计协方差矩阵P
//	Mat measurement = Mat::zeros(measureNum, 1, CV_32F);
//
//	//初始状态值
//	KF.statePost = (Mat_<float>(3, 1) << 0, 0, 1);
//
//	//预测
//	Mat prediction = KF.predict();
//	//计算测量值
//	measurement.at<float>(0) = measure;
//
//	//更新
//	KF.correct(measurement);
//
//	return (int)(KF.statePost.at<float>(0) - KF.statePost.at<float>(1));
//}
//
//int stitching(Mat preMat, Mat matSrc, double shift, string img_path) {
//	if (cols_panorama == 1920)
//	{
//		shift_mat = preMat.clone();
//	}
//
//	/*四舍五入*/
//	int integer_shift_ = int(abs(shift));
//	//int integer_shift = int(abs(shift_x_mean));
//
//	integer_shift_ = abs(abs(shift) - integer_shift_) >= 0.5 ? (integer_shift_ + 1) : integer_shift_;
//	
//	
//	//little_shift += integer_shift_;
//	//if (integer_shift_ < 10) {
//	//	return 0;
//	//}
//	//
//
//	vector_shift.push_back(integer_shift_);
//	Mat r;
//	/*if (integer_shift_ > 50) {
//		double orb_shift = abs(orb_findhomographyFromframes(preMat, matSrc, true).at<double>(0, 2));
//		integer_shift_ = int(orb_shift);
//		integer_shift_ = abs(orb_shift - integer_shift_) >= 0.5 ? (integer_shift_ + 1) : integer_shift_;
//		vector_shift.pop_back();
//		vector_shift.push_back(integer_shift_);
//	}*/
//
//	if (integer_shift_ > 50) {
//		vector_shift.pop_back();
//		integer_shift_ = vector_shift[vector_shift.size() - 1];
//		vector_shift.push_back(integer_shift_);
//	}
//
//	//use kalman_algorithm
//	float kalman_result = (float)integer_shift_;
//	integer_shift_ = kalman_optimize(kalman_result);
//
//	cout << "integer" << integer_shift_ << endl;
//	cols_panorama += integer_shift_;
//	float camera_01_shift = 130;
//
//	if (integer_shift_ > 130)
//	{
//		cout << "shift err" << endl;
//		return -1;
//	}
//
//
//	int stich_region = int((camera_01_shift - integer_shift_) / 2);
//	if (stich_region < 0) {
//		cout << "stich_region err <0" << endl;
//	}
//	cout << "stich_region" << stich_region << endl;
//
//	if (integer_shift_ != 0)
//	{
//		int right_start_index = matSrc.cols - integer_shift_ - stich_region;
//		if (stich_region >= shift_mat.cols || right_start_index >= matSrc.cols) {
//			cout << "stich_region >= shift_mat.cols || right_start_index >= matSrc.cols" << endl;
//			return -1;
//		}
//		Mat left = shift_mat.colRange(0, shift_mat.cols - stich_region);
//		Mat right = matSrc.colRange(right_start_index, matSrc.cols);
//		hconcat(left, right, shift_mat);
//		//ofile << integer_shift_ << endl;
//		//fusion(preMat, matSrc, left, right, stich_region, integer_shift_);
//		//imwrite(img_path, shift_mat);
//	}
//	
//	return 0;
//}
//
//
//float compute_shift(vector<Point2f> point[2]) {
//
//	float shift_x_mean = 0;
//	float shift_x_mean_ = 0;
//	vector<float> shift_x;
//	int pint0_size = point[0].size();
//	for (int i = 0; i < pint0_size; i++) {
//		float shift_x_ = point[1][i].x - point[0][i].x;
//		shift_x_mean += shift_x_;
//		shift_x.push_back(shift_x_);
//	}
//
//	for (int i = 0; i < shift_x.size(); i++) {
//		if (shift_x[i] > 0 || shift_x[i] <-50)
//		{
//			shift_x.erase(shift_x.begin() + i);
//		}
//		else {
//			shift_x_mean_ += shift_x[i];
//		}
//	}
//	shift_x_mean_ = shift_x_mean_ / shift_x.size();
//	//shift_x_mean = shift_x_mean / point[0].size();
//	//Mat homo = findHomography(point[0], point[1], CV_RANSAC);
//	//cout << "shift_x_mean_" << shift_x_mean_ << endl;
//	//cout << "shift_x_mean" << shift_x_mean << endl;
//
//	/*四舍五入*/
//	int integer_shift_ = int(abs(shift_x_mean_));
//	//int integer_shift = int(abs(shift_x_mean));
//
//	integer_shift_ = abs(abs(shift_x_mean_) - integer_shift_) >= 0.5 ? (integer_shift_ + 1) : integer_shift_;
//	//integer_shift = abs(abs(shift_x_mean) - integer_shift) >= 0.5 ? (integer_shift + 1) : integer_shift;
//
//	cout << "pint0_size		" << pint0_size << endl;
//	cout << "integer_shift_		" << integer_shift_ << endl;
//	cout << "shift_x.size()		" << shift_x.size() << endl;
//	//cout << "integer_shift" << integer_shift << endl;
//
//	//Mat h = findHomography(point[0], point[1], CV_RANSAC);
//	//cout << "h" << homo.at<float>(0, 2) << endl;
//	//cout << "h" << h.inv() << endl;
//	cout << endl;
//	cout << endl;
//	cout << endl;
//	return integer_shift_;
//}
//
//int optical_flow_lk(Mat preMat, Mat matSrc, Mat matRst, string img_path) {
//
//	cvtColor(matSrc, curgray, CV_BGR2GRAY);
//	if (cols_panorama == 1920)
//	{
//		shift_mat = preMat.clone();
//	}
//	if (addNewPoints(lk_orb_point[0]))
//	{
//		Ptr<ORB> orb = ORB::create(5000);
//		vector<KeyPoint> kps_orb;
//		orb->detect(preMat, kps_orb);
//
//		//Ptr<SIFT> sift = SIFT::create();
//		//vector<KeyPoint> kps_sift;
//		//sift->detect(preMat,kps_sift);
//
//		if (lk_orb_features.size() > 0) {
//			lk_orb_features.clear();
//		}
//		if (lk_orb_point[0].size() > 0)
//		{
//			lk_orb_point[0].clear();
//		}
//		if (lk_orb_point[1].size() > 0)
//		{
//			lk_orb_point[1].clear();
//		}
//		for (auto kp : kps_orb)
//			lk_orb_features.push_back(kp.pt);
//
//		lk_orb_point[0].insert(lk_orb_point[0].end(), lk_orb_features.begin(), lk_orb_features.end());
//	}
//	/*if (addNewPoints(point[0]))
//	{
//	goodFeaturesToTrack(pregray, features, maxCount, qLevel, minDist);
//	point[0].insert(point[0].end(), features.begin(), features.end());
//	}*/
//	calcOpticalFlowPyrLK(pregray, curgray, lk_orb_point[0], lk_orb_point[1], status_opticalflowlk_features, err);
//	//calcOpticalFlowPyrLK(pregray, curgray, point[0], point[1], status, err);
//
//	int k = 0;
//	for (size_t i = 0; i<lk_orb_point[1].size(); i++)
//	{
//		if (acceptTrackedPoint(i, status_opticalflowlk_features, lk_orb_point))
//		{
//			lk_orb_point[1][k++] = lk_orb_point[1][i];
//		}
//	}
//
//	//int kk = 0;
//	//for (size_t i = 0; i<point[1].size(); i++)
//	//{
//	//	if (acceptTrackedPoint(i, status, point))
//	//	{
//	//		point[1][kk++] = point[1][i];
//	//	}
//	//}
//
//	float shift_orb = compute_shift(lk_orb_point);
//	//cout << "shift_orb" << shift_orb << endl;
//
//	//float shift_goodfeature = compute_shift(point);
//	//cout << "shift_goodfeature" << shift_goodfeature << endl;
//
//	digit_shfit = int(shift_orb);
//
//	cols_panorama += digit_shfit;
//	float camera_01_shift = 130;
//	if (digit_shfit > 130)
//	{
//		cout << "shift err" << endl;
//	}
//	int stich_region = int((camera_01_shift - digit_shfit) / 2);
//	if (stich_region < 0) {
//		cout << "stich_region err <0" << endl;
//	}
//	if (digit_shfit != 0)
//	{
//		int right_start_index = matSrc.cols - digit_shfit - stich_region;
//		if (stich_region >= shift_mat.cols || right_start_index >= matSrc.cols) {
//			cout << "stich_region >= shift_mat.cols || right_start_index >= matSrc.cols" << endl;
//			return -1;
//		}
//		Mat left = shift_mat.colRange(0, shift_mat.cols - stich_region);
//		Mat right = matSrc.colRange(right_start_index, matSrc.cols);
//		hconcat(left, right, shift_mat);
//	}
//
//	//imwrite(img_path,shift_mat);
//
//	lk_orb_point[1].resize(k);
//	//point[1].resize(kk);
//
//	swap(lk_orb_point[1], lk_orb_point[0]);
//	//swap(point[1], point[0]);
//	swap(pregray, curgray);
//
//	waitKey(10);
//	return 0;
//}
//double phase_correlate(Mat left, Mat right) {
//	//double shift = 0.0;
//	Mat dst1, dst2;
//	if (left.empty()) { return -1.0; }
//	if (right.empty()) { return -2.0; }
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
//int main()
//{
//	clock_t start = clock();
//	KF.init(stateNum, measureNum, 0);
//
//	string file_path = "d:/imgsource/panorama/911resize.txt";
//	/*ofile.open(file_path);
//	assert(ofile.is_open());*/
//	frame_num = 60;
//
//	VideoCapture cap;
//	cap.open("C:/Users/Administrator/Videos/SingleVideoStitching/videosource/s1.mp4");
//
//	// perform the tracking process
//	printf("Start the tracking process, press ESC to quit.\n");
//	int index = 0;
//
//	//set start_frame
//	//cap.set(CAP_PROP_POS_FRAMES, frame_num);
//
//	cap >> preMat;
//
//	float shift_x_mean_orb = 0;
//	cvtColor(preMat, pregray, CV_RGB2GRAY);
//	Mat result, no_watermark_pre, no_watermark_cur, homo;;
//	cvtColor(preMat, pregray_opticalflowlk_features, CV_RGB2GRAY);
//	//string imgPath = "d:/imgsource/panorama/824_fusion.jpg";
//	
//	double homo_shift, phase_correlate_shift;
//	Point left_top;
//	int width = preMat.cols / 4;
//	int height = preMat.rows / 4;
//	//left_top.x = preMat.cols / 2 - (width / 2);
//	//left_top.y = preMat.rows / 2 - (height / 2);
//	left_top.x = 50;
//	left_top.y = preMat.rows - height;
//
//	int scale = 4;
//	//int if_resize = 1;
//	preproccess(preMat, no_watermark_pre);
//	resize(no_watermark_pre, no_watermark_pre, Size(no_watermark_pre.cols / scale, no_watermark_pre.rows / scale));
//	while (cap.isOpened()) {
//		std::cout << frame_num << "---" << frame_num + 1 << endl;
//		frame_num++;
//		// get frame from the video
//		cap >> matSrc;
//
//		if (!matSrc.empty())
//		{
//			//optical_flow_lk(preMat, matSrc, matRst, "d:/imgsource/panorama/orb_pan_823.jpg");
//			//preMat = matSrc;
//		
//			preproccess(matSrc, no_watermark_cur);
//			resize(no_watermark_cur, no_watermark_cur, Size(no_watermark_cur.cols / scale, no_watermark_cur.rows / scale));
//			
//			/*preproccess_from_point_WH(preMat, no_watermark_pre, left_top, width, height);
//			preproccess_from_point_WH(matSrc, no_watermark_cur, left_top, width, height);*/
//
//			phase_correlate_shift = phase_correlate(no_watermark_pre, no_watermark_cur);
//			//result = orb_findhomographyFromframes(preMat, matSrc, true);
//			//cout << "orb_h		" << result.at<double>(0, 2) << endl;
//			//stitching(preMat, matSrc, result.at<double>(0, 2), "d:/imgsource/panorama/831_fusion.jpg");
//			phase_correlate_shift = abs(phase_correlate_shift * scale);
//
//			if (phase_correlate_shift > 10) {
//				stitching(preMat, matSrc, phase_correlate_shift, "d:/imgsource/panorama/906_fusion_optimize_scale.jpg");
//				cout << endl;
//				swap(no_watermark_pre, no_watermark_cur);
//			}
//
//		}
//		else
//		{
//			std::cout << "Error : Get picture is empty!" << endl;
//			break;
//		}
//
//	}
//
//	if (!shift_mat.empty()) {
//		imwrite("d:/imgsource/panorama/912_resize_scale4_optimize.jpg", shift_mat);
//	}
//	clock_t end = clock();
//	cout << "time_cost" << end - start << endl;
//	//if (!vector_shift.empty()) {
//	//	for (vector<int>::iterator it = vector_shift.begin(); it != vector_shift.end();++it) {
//	//		ofile << *it << endl;
//	//	}
//	//}
//	//ofile.close();
//	//cv::waitKey(0);
//	std::system("pause");
//	return 0;
//
//}
//
//Mat opticalflowlk_goodfeatures(Mat img1, Mat img2, float &shift_x_mean, bool watermark) {
//	shift_x_mean = 0;
//	cvtColor(img2, curgray, CV_BGR2GRAY);
//
//	if (addNewPoints(point[0]))
//	{
//		goodFeaturesToTrack(pregray, features, maxCount, qLevel, minDist);
//		point[0].insert(point[0].end(), features.begin(), features.end());
//		initPoint.insert(initPoint.end(), features.begin(), features.end());
//	}
//	calcOpticalFlowPyrLK(pregray, curgray, point[0], point[1], status, err);
//	int k = 0;
//	for (size_t i = 0; i<point[1].size(); i++)
//	{
//		if (acceptTrackedPoint(i, status, point))
//		{
//			initPoint[k] = initPoint[i];
//			point[1][k++] = point[1][i];
//		}
//	}
//	//float shift_x_mean = 0;
//	float shift_x_mean_handled = 0;
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
//	//cout << "shift_x_mean" << shift_x_mean << endl;
//	//cout << "shift_x.size()" << shift_x.size() << endl;
//	for (int index = 0; index < shift_x.size(); index++) {
//		if (shift_x[index] > 300 || shift_x[index] < -300)
//		{
//			shift_x.erase(shift_x.begin() + index);
//		}
//		else
//		{
//			shift_x_mean_handled += shift_x[index];
//		}
//	}
//	shift_x_mean_handled = shift_x_mean_handled / shift_x.size();
//	//cout << "shift_x_mean_handled" << shift_x_mean_handled << endl;
//	Mat homo = findHomography(point[0], point[1], CV_RANSAC);
//
//	point[1].resize(k);
//	initPoint.resize(k);
//
//	cv::swap(point[1], point[0]);
//	//cv::swap(pregray, curgray);
//
//	return homo;
//}
//
//void duan_OpticalFlow(Mat &preMat, Mat &frame, Mat & result)
//{
//	//ofstream file_shift_x_y;
//	//string path_name = "d:/imgsource/shift/" + to_string(frame_num) + "-" + to_string(frame_num+1)+".txt";
//	//file_shift_x_y.open(path_name);
//
//	/*if (cols_panorama == 1920)
//	{
//	shift_mat = preMat.clone();
//	}*/
//	cvtColor(frame, curgray, CV_BGR2GRAY);
//	//frame.copyTo(result);
//
//	if (addNewPoints())
//	{
//		//goodFeaturesToTrack(pregray, features, maxCount, qLevel, minDist);
//		Ptr<ORB> orb = ORB::create(5000);
//		vector<KeyPoint> kps;
//		orb->detect(preMat, kps);
//		for (auto kp : kps)
//			features.push_back(kp.pt);
//
//		point[0].insert(point[0].end(), features.begin(), features.end());
//		//initPoint.insert(initPoint.end(), features.begin(), features.end());
//	}
//
//
//	/*if (pregray.empty())
//	{
//	curgray.copyTo(pregray);
//	}*/
//
//	calcOpticalFlowPyrLK(pregray, curgray, point[0], point[1], status, err);
//
//
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
//	cout << "shift_x_mean" << shift_x_mean << endl;
//
//	Mat homo = findHomography(point[0], point[1], CV_RANSAC);
//
//	//cout << homo << endl;
//
//	digit_shfit = abs(int(shift_x_mean));
//
//	Mat imageWarp_shift_x_mean;
//	Mat imageWarp_homo;
//
//	cols_panorama += digit_shfit;
//	float camera_01_shift = 130;
//	int stich_region = int((camera_01_shift - digit_shfit) / 2);
//
//	//if (digit_shfit != 0)
//	//{
//	//	Mat left = shift_mat.colRange(0, shift_mat.cols - stich_region);
//	//	Mat right = frame.colRange(frame.cols - digit_shfit - stich_region, frame.cols);
//	//	hconcat(left, right, shift_mat);
//	//}
//
//
//	//imwrite("d:/imgsource/panorama/pan.jpg", shift_mat);
//	//imshow("panorama", shift_mat);
//	//warpPerspective(frame, imageWarp, homo.inv(), Size(1920,1080));
//
//	//imshow("warp", imageWarp);
//
//	point[1].resize(k);
//	//initPoint.resize(k);
//
//	//for (size_t i = 0; i<point[1].size(); i++)
//	//{
//	//	line(result, initPoint[i], point[1][i], Scalar(0, 0, 255));
//	//	circle(result, point[1][i], 3, Scalar(0, 255, 0), -1);
//	//}
//
//	swap(point[1], point[0]);
//	swap(pregray, curgray);
//
//	//imshow("Optical Flow Demo", result);
//	waitKey(10);
//	//file_shift_x_y.close();
//
//}
//
