// Skeleton: http://fahmifahim.com/2011/05/16/kinect-and-opencv/
// Tracking colored objects: http://www.aishack.in/2010/07/tracking-colored-objects-in-opencv/
// Tracking ball: http://projectproto.blogspot.com/2012/04/android-opencv-object-tracking.html
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <libfreenect.h>
#include <pthread.h>

#define CV_NO_BACKWARD_COMPATIBILITY

#include <cv.h>
#include <highgui.h>

#define FREENECTOPENCV_WINDOW_D "Depthimage"
#define FREENECTOPENCV_WINDOW_N "Normalimage"
#define FREENECTOPENCV_RGB_DEPTH 3
#define FREENECTOPENCV_DEPTH_DEPTH 1
#define FREENECTOPENCV_RGB_WIDTH 640
#define FREENECTOPENCV_RGB_HEIGHT 480
#define FREENECTOPENCV_DEPTH_WIDTH 640
#define FREENECTOPENCV_DEPTH_HEIGHT 480

IplImage* depthimg = 0;
IplImage* rgbimg = 0;
IplImage* tempimg = 0;
IplImage* red_img = 0;
IplImage* orange_img = 0;
IplImage* canny_temp = 0;
pthread_mutex_t mutex_depth = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex_rgb = PTHREAD_MUTEX_INITIALIZER;
pthread_t cv_thread;

// callback for depthimage, called by libfreenect
void depth_cb(freenect_device *dev, void *depth, uint32_t timestamp)

{
        cv::Mat depth8;
        cv::Mat mydepth = cv::Mat( FREENECTOPENCV_DEPTH_WIDTH,FREENECTOPENCV_DEPTH_HEIGHT, CV_16UC1, depth);

        mydepth.convertTo(depth8, CV_8UC1, 1.0/4.0);
        pthread_mutex_lock( &mutex_depth );
        memcpy(depthimg->imageData, depth8.data, 640*480);
        // unlock mutex
        pthread_mutex_unlock( &mutex_depth );

}

// callback for rgbimage, called by libfreenect
void rgb_cb(freenect_device *dev, void *rgb, uint32_t timestamp)
{
        // lock mutex for opencv rgb image
        pthread_mutex_lock( &mutex_rgb );
        memcpy(rgbimg->imageData, rgb, FREENECT_VIDEO_RGB_SIZE);
        // unlock mutex
        pthread_mutex_unlock( &mutex_rgb );
}

// Looks for red cups
IplImage* getRedImage(IplImage* img)
{
	// Convert the image into an HSV image
	IplImage* imgHSV = cvCreateImage(cvGetSize(img), 8, 3);
	cvCvtColor(img, imgHSV, CV_BGR2HSV);
	IplImage* imgThreshed = cvCreateImage(cvGetSize(img), 8, 1);
	IplImage* redHigh = cvCreateImage(cvGetSize(img), 8, 1);
	IplImage* redLow = cvCreateImage(cvGetSize(img), 8, 1);
	cvInRangeS(imgHSV, cvScalar(175, 250, 75), cvScalar(255, 255, 255), redHigh);
	cvInRangeS(imgHSV, cvScalar(0, 250, 25), cvScalar(1, 255, 255), redLow);
	cvOr(redHigh, redLow, imgThreshed);
	cvReleaseImage(&imgHSV);
	cvReleaseImage(&redHigh);
	cvReleaseImage(&redLow);
	cvSmooth(imgThreshed, imgThreshed, CV_MEDIAN, 7);
	return imgThreshed;
}

// Looks for orange ball
IplImage* getOrangeImage(IplImage* img)
{
	// Convert the image into an HSV image
	IplImage* imgHSV = cvCreateImage(cvGetSize(img), 8, 3);
	cvCvtColor(img, imgHSV, CV_BGR2HSV);
	IplImage* imgThreshed = cvCreateImage(cvGetSize(img), 8, 1);
	cvInRangeS(imgHSV, cvScalar(9, 180, 75), cvScalar(19, 300, 300), imgThreshed);
	//cvInRangeS(imgHSV, cvScalar(0, 0, 0), cvScalar(90, 255, 255), imgThreshed);
	cvReleaseImage(&imgHSV);
	cvSmooth(imgThreshed, imgThreshed, CV_MEDIAN, 7);
	//cvSmooth(imgThreshed, imgThreshed, CV_GAUSSIAN, 9, 9);
	return imgThreshed;
}


/*
 * thread for displaying the opencv content
 */
void *cv_threadfunc (void *ptr) {
        cvNamedWindow( FREENECTOPENCV_WINDOW_D, CV_WINDOW_AUTOSIZE );
        cvNamedWindow( FREENECTOPENCV_WINDOW_N, CV_WINDOW_AUTOSIZE );
	cvNamedWindow( "Cup Contours", CV_WINDOW_AUTOSIZE );
	cvNamedWindow( "CUP CAM", CV_WINDOW_AUTOSIZE );
	cvNamedWindow( "BALL CAM", CV_WINDOW_AUTOSIZE );
	cvNamedWindow( "Ball Outlines", CV_WINDOW_AUTOSIZE );
        depthimg = cvCreateImage(cvSize(FREENECTOPENCV_DEPTH_WIDTH, FREENECTOPENCV_DEPTH_HEIGHT), IPL_DEPTH_8U, FREENECTOPENCV_DEPTH_DEPTH);
        rgbimg = cvCreateImage(cvSize(FREENECTOPENCV_RGB_WIDTH, FREENECTOPENCV_RGB_HEIGHT), IPL_DEPTH_8U, FREENECTOPENCV_RGB_DEPTH);
        tempimg = cvCreateImage(cvSize(FREENECTOPENCV_RGB_WIDTH, FREENECTOPENCV_RGB_HEIGHT), IPL_DEPTH_8U, FREENECTOPENCV_RGB_DEPTH);
	red_img = cvCreateImage(cvSize(FREENECTOPENCV_RGB_WIDTH, FREENECTOPENCV_RGB_HEIGHT), IPL_DEPTH_8U, 1);
	orange_img = cvCreateImage(cvSize(FREENECTOPENCV_RGB_WIDTH, FREENECTOPENCV_RGB_HEIGHT), IPL_DEPTH_8U, 1);
	canny_temp = cvCreateImage(cvSize(FREENECTOPENCV_DEPTH_WIDTH, FREENECTOPENCV_DEPTH_HEIGHT), IPL_DEPTH_8U, FREENECTOPENCV_DEPTH_DEPTH);

        // use image polling
        while (1) {
                //lock mutex for depth image
                pthread_mutex_lock( &mutex_depth );
                // show image to window
                cvCanny(depthimg, canny_temp, 50.0, 200.0, 3);
		cvCvtColor(depthimg,tempimg,CV_GRAY2BGR);
                //cvCvtColor(tempimg,tempimg,CV_HSV2BGR);

                cvShowImage(FREENECTOPENCV_WINDOW_D,tempimg);
                //unlock mutex for depth image
                pthread_mutex_unlock( &mutex_depth );

                //lock mutex for rgb image
                pthread_mutex_lock( &mutex_rgb );
                // show image to window
                cvCvtColor(rgbimg,tempimg,CV_BGR2RGB);
                cvShowImage(FREENECTOPENCV_WINDOW_N, tempimg);
		red_img = getRedImage(tempimg);
		orange_img = getOrangeImage(tempimg);
		cvShowImage("CUP CAM", red_img);
		cvShowImage("BALL CAM", orange_img);
		

		// Canny filter
		cvCanny(red_img, red_img, 50.0, 200.0, 3);
		//cvShowImage("Canny Image", red_img);
		cv::Mat cups(red_img);
		//cv::smooth(cups, cups);
		std::vector<std::vector <cv::Point> > cupContours;
		std::vector<cv::Vec4i> cupHierarchy;
		cv::findContours(cups, cupContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point());
		//cv::boxFilter(cups, cups, -1, cv::Size(3, 3));
		//cv::imshow("Canny Image", cups);

		cv::Mat drawing = cv::Mat::zeros(cups.size(), CV_8UC3 );
	        double biggestSize = -1;
		int biggestContour = -1;	
		for(unsigned int i = 0; i < cupContours.size(); i++ )
		{
			std::vector <cv::Point> cont = cupContours[i];
			double s = cv::contourArea(cont);
			if (s > biggestSize) {
				biggestSize = s;
				biggestContour = i;
			}
		}
		cv::Rect cupRect;
		if (cupContours.size() > 0)
		{
			cupRect = cv::boundingRect(cv::Mat(cupContours[biggestContour]));
		}
		cv::Scalar color = cv::Scalar(100,100,100);
		drawContours( drawing, cupContours, biggestContour, color, 2, 8, cupHierarchy, 0, cv::Point());

		// Look for circles (ball)
		cv::Mat balls(orange_img);
		std::vector<std::vector <cv::Point> > ballContours;
		std::vector<cv::Vec4i> ballHierarchy;
		cv::findContours(balls, ballContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point());
	        biggestSize = -1;
		biggestContour = -1;	
		for(unsigned int i = 0; i < ballContours.size(); i++ )
		{
			std::vector <cv::Point> cont = ballContours[i];
			double s = cv::contourArea(cont);
			if (s > biggestSize) {
				biggestSize = s;
				biggestContour = i;
			}
		}
		cv::Rect ballRect;
		if (ballContours.size() > 0)
		{
			ballRect = cv::boundingRect(cv::Mat(ballContours[biggestContour]));
		}
		drawContours( drawing, ballContours, biggestContour, color, 2, 8, ballHierarchy, 0, cv::Point());

		cv::imshow("Ball Outline", drawing);

                //unlock mutex
                pthread_mutex_unlock( &mutex_rgb );

                // wait for quit key
                if( cvWaitKey( 15 )==27 )
			break;

        }
        pthread_exit(NULL);
		return NULL;

}

int main(int argc, char **argv)
{
        freenect_context *f_ctx;
        freenect_device *f_dev;

        int res = 0;
        int die = 0;
        printf("Kinect camera test\n");

        if (freenect_init(&f_ctx, NULL) < 0) {
                        printf("freenect_init() failed\n");
                        return 1;
                }

                if (freenect_open_device(f_ctx, &f_dev, 0) < 0) {
                        printf("Could not open device\n");
                        return 1;
                }

        freenect_set_depth_callback(f_dev, depth_cb);
        freenect_set_video_callback(f_dev, rgb_cb);
        freenect_set_video_format(f_dev, FREENECT_VIDEO_RGB);
	freenect_set_tilt_degs(f_dev, 0);
        // create opencv display thread
        res = pthread_create(&cv_thread, NULL, cv_threadfunc, (void*) depthimg);
        if (res) {
                printf("pthread_create failed\n");
                return 1;
        }

        printf("init done\n");

        freenect_start_depth(f_dev);
        freenect_start_video(f_dev);

        while(!die && freenect_process_events(f_ctx) >= 0 );

}
