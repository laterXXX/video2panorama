#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;
static inline Point calcPoint(Point2f center, double R, double angle)
{
	return center + Point2f((float)cos(angle), (float)-sin(angle))*(float)R;
}

static void help()
{
	printf("\nExample of c calls to OpenCV's Kalman filter.\n"
		"   Tracking of rotating point.\n"
		"   Rotation speed is constant.\n"
		"   Both state and measurements vectors are 1D (a point angle),\n"
		"   Measurement is the real point angle + gaussian noise.\n"
		"   The real and the estimated points are connected with yellow line segment,\n"
		"   the real and the measured points are connected with red line segment.\n"
		"   (if Kalman filter works correctly,\n"
		"    the yellow segment should be shorter than the red one).\n"
		"\n"
		"   Pressing any key (except ESC) will reset the tracking with a different speed.\n"
		"   Pressing ESC will stop the program.\n"
	);
}

int main_3(int, char**)
{

	/*
    CV_PROP_RW Mat statePre;           //!< predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
    CV_PROP_RW Mat statePost;          //!< corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
    CV_PROP_RW Mat transitionMatrix;   //!< state transition matrix (A)
    CV_PROP_RW Mat controlMatrix;      //!< control matrix (B) (not used if there is no control)
    CV_PROP_RW Mat measurementMatrix;  //!< measurement matrix (H)
    CV_PROP_RW Mat processNoiseCov;    //!< process noise covariance matrix (Q)
    CV_PROP_RW Mat measurementNoiseCov;//!< measurement noise covariance matrix (R)
    CV_PROP_RW Mat errorCovPre;        //!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)
	CV_PROP_RW Mat gain;               //!< Kalman gain matrix (K(k)): K(k)=P'(k)*Ht*inv(H*P'(k)*Ht+R)
	CV_PROP_RW Mat errorCovPost;       //!< posteriori error estimate covariance matrix (P(k)): P(k)=(I-K(k)*H)*P'(k)
	*/

	help();
	Mat img(500, 500, CV_8UC3);
	KalmanFilter KF(2, 1, 0);
	Mat state(2, 1, CV_32F); /* (phi, delta_phi) */ //x'(k)
	Mat processNoise(2, 1, CV_32F);	//Q
	Mat measurement = Mat::zeros(1, 1, CV_32F);	//Z
	char code = (char)-1;

	for (;;)
	{
		randn(state, Scalar::all(0), Scalar::all(0.1));
		cout << state << endl;
		KF.transitionMatrix = (Mat_<float>(2, 2) << 1, 1, 0, 1);	//那第一个应该是位置，第二个是速度

		setIdentity(KF.measurementMatrix);	//H
		setIdentity(KF.processNoiseCov, Scalar::all(1e-5));//Q
		setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));//R
		setIdentity(KF.errorCovPost, Scalar::all(1));	//(P(k)): P(k)=(I-K(k)*H)*P'(k)

		randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));	//!< corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))

		for (;;)
		{
			Point2f center(img.cols*0.5f, img.rows*0.5f);
			float R = img.cols / 3.f;
			double stateAngle = state.at<float>(0);
			Point statePt = calcPoint(center, R, stateAngle);

			Mat prediction = KF.predict();
			double predictAngle = prediction.at<float>(0);	//(x'(k)): x(k)=A*x(k-1)+B*u(k)
			Point predictPt = calcPoint(center, R, predictAngle);

			randn(measurement, Scalar::all(0), Scalar::all(KF.measurementNoiseCov.at<float>(0)));

			// generate measurement
			measurement += KF.measurementMatrix*state;

			double measAngle = measurement.at<float>(0);
			Point measPt = calcPoint(center, R, measAngle);

			// plot points
#define drawCross( center, color, d )                                        \
                line( img, Point( center.x - d, center.y - d ),                          \
                             Point( center.x + d, center.y + d ), color, 1, LINE_AA, 0); \
                line( img, Point( center.x + d, center.y - d ),                          \
                             Point( center.x - d, center.y + d ), color, 1, LINE_AA, 0 )

			img = Scalar::all(0);
			drawCross(statePt, Scalar(255, 255, 255), 3);
			drawCross(measPt, Scalar(0, 0, 255), 3);
			drawCross(predictPt, Scalar(0, 255, 0), 3);
			line(img, statePt, measPt, Scalar(0, 0, 255), 3, LINE_AA, 0);
			line(img, statePt, predictPt, Scalar(0, 255, 255), 3, LINE_AA, 0);

			if (theRNG().uniform(0, 4) != 0)
				KF.correct(measurement);

			randn(processNoise, Scalar(0), Scalar::all(sqrt(KF.processNoiseCov.at<float>(0, 0))));
			state = KF.transitionMatrix*state + processNoise;

			imshow("Kalman", img);
			code = (char)waitKey(100);

			if (code > 0)
				break;
		}
		if (code == 27 || code == 'q' || code == 'Q')
			break;
	}

	return 0;
}
