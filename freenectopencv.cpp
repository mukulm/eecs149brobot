// Skeleton: http://fahmifahim.com/2011/05/16/kinect-and-opencv/
// Tracking colored objects: http://www.aishack.in/2010/07/tracking-colored-objects-in-opencv/
// Tracking ball: http://projectproto.blogspot.com/2012/04/android-opencv-object-tracking.html
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>

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
IplImage* imgHSV = 0;
IplImage* thresholded = 0;
uint8_t cupdist = 0;
pthread_mutex_t mutex_depth = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex_rgb = PTHREAD_MUTEX_INITIALIZER;
pthread_t cv_thread;
CvScalar hsv_min = cvScalar(150, 84, 130, 0);
CvScalar hsv_max = cvScalar(358, 256, 255, 0);

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
	cvInRangeS(imgHSV, cvScalar(170, 225, 35), cvScalar(255, 255, 255), redHigh);
	cvInRangeS(imgHSV, cvScalar(0, 225, 25), cvScalar(6, 255, 255), redLow);
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
	cvInRangeS(imgHSV, hsv_min, hsv_max, imgThreshed);
	cvReleaseImage(&imgHSV);
	//cvSmooth(imgThreshed, imgThreshed, CV_MEDIAN, 7);
	cvSmooth(imgThreshed, imgThreshed, CV_GAUSSIAN, 9, 9);
	return imgThreshed;
}


/*
 * thread for displaying the opencv content
 */
void *cv_threadfunc (void *ptr) {
    cvNamedWindow( FREENECTOPENCV_WINDOW_D, CV_WINDOW_AUTOSIZE );
    cvNamedWindow( FREENECTOPENCV_WINDOW_N, CV_WINDOW_AUTOSIZE );
	cvNamedWindow( "BALL CAM", CV_WINDOW_AUTOSIZE );
	cvNamedWindow( "HSV", CV_WINDOW_AUTOSIZE );
	cvNamedWindow( "AFTER FILTER", CV_WINDOW_AUTOSIZE );
	cvNamedWindow( "Ball Contours", CV_WINDOW_AUTOSIZE );
    depthimg = cvCreateImage(cvSize(FREENECTOPENCV_DEPTH_WIDTH, FREENECTOPENCV_DEPTH_HEIGHT), IPL_DEPTH_8U, FREENECTOPENCV_DEPTH_DEPTH);
    rgbimg = cvCreateImage(cvSize(FREENECTOPENCV_RGB_WIDTH, FREENECTOPENCV_RGB_HEIGHT), IPL_DEPTH_8U, FREENECTOPENCV_RGB_DEPTH);
    tempimg = cvCreateImage(cvSize(FREENECTOPENCV_RGB_WIDTH, FREENECTOPENCV_RGB_HEIGHT), IPL_DEPTH_8U, FREENECTOPENCV_RGB_DEPTH);
	imgHSV = cvCreateImage(cvSize(FREENECTOPENCV_RGB_WIDTH, FREENECTOPENCV_RGB_HEIGHT), 8, 3);
	thresholded = cvCreateImage(cvSize(FREENECTOPENCV_RGB_WIDTH, FREENECTOPENCV_RGB_HEIGHT), IPL_DEPTH_8U, 1);
	int frameCount = 0;
    // use image polling
    while (1) {
		frameCount++;
        //lock mutex for depth image
        pthread_mutex_lock( &mutex_depth );
        // show image to window
        cvCanny(depthimg, canny_temp, 50.0, 200.0, 3);
		cvCvtColor(depthimg,tempimg,CV_GRAY2BGR);
        cvShowImage(FREENECTOPENCV_WINDOW_D,depthimg);
        //unlock mutex for depth image
        pthread_mutex_unlock( &mutex_depth );

        //lock mutex for rgb image
        pthread_mutex_lock( &mutex_rgb );
        // show image to window
        cvCvtColor(rgbimg,tempimg,CV_BGR2RGB);

		//TRACK BALL
		//convert colorspace to HSV
		cvCvtColor(tempimg, imgHSV, CV_BGR2HSV);
		//Filter out colors exceeding range.
		cvInRangeS(imgHSV, hsv_min, hsv_max, thresholded);
		//Memory for hough circles
		CvMemStorage* storage = cvCreateMemStorage(0);
		//hough detector works better with some smoothing of the image
		cvSmooth(thresholded, thresholded, CV_GAUSSIAN, 9, 9);
		CvSeq* circles = cvHoughCircles(thresholded, storage, CV_HOUGH_GRADIENT, 2, thresholded->height/4, 100, 50, 10 , 400);
		for (int i = 0; i < circles->total; i++){
			float* p = (float*)cvGetSeqElem(circles, i);
			printf("Ball! x=%f y=%f r=%f\n\r", p[0],p[1],p[2]);
			cvCircle(tempimg, cvPoint(cvRound(p[0]),cvRound(p[1])), 3, CV_RGB(0,255,0), -1, 8, 0);
			cvCircle(tempimg, cvPoint(cvRound(p[0]),cvRound(p[1])),cvRound(p[2]), CV_RGB(255,0,0), 3, 8, 0 );
		}
		cvShowImage("BALL CAM", tempimg);
		cvShowImage("HSV", imgHSV);
		cvShowImage("AFTER FILTER", thresholded);	
		cvReleaseMemStorage(&storage);

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
