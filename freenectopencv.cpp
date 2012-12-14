/************************************************************
 * Mukul Murthy, Jordan Smith, Tim Brown, and Kunal Agarwal *
 * UC Berkeley EECS 149 - Embedded Systems                  *
 * Fall 2012                                                *
 * Instructors: Edward Lee, Sanjit Seshia                   *
 * Mentor: Zach Wasson                                      *
 *							    *
 * BROBOT VISION SYSTEM					    * 
 * **********************************************************/

// Sources we borrowed code from:
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

// Constants for image size
#define FREENECTOPENCV_WINDOW_D "Depthimage"
#define FREENECTOPENCV_WINDOW_N "Normalimage"
#define FREENECTOPENCV_RGB_DEPTH 3
#define FREENECTOPENCV_DEPTH_DEPTH 1
#define FREENECTOPENCV_RGB_WIDTH 640
#define FREENECTOPENCV_RGB_HEIGHT 480
#define FREENECTOPENCV_DEPTH_WIDTH 640
#define FREENECTOPENCV_DEPTH_HEIGHT 480

// Constants we use in our algorithms and models
#define PI 3.14159265
#define ALPHA .7
#define DEPTH_MIN_DIST 23.0
#define DEPTH_MAX_DIST 73.0
#define X_OFFSET_TO_DEG 8.7
#define FRAMES_TO_CONVERGENCE 30
#define DISTANCE_THRESHOLD 1.0
#define OFFSET_THRESHOLD 0.3
#define X3_COEFF .0000664733958
#define X2_COEFF .03033943
#define X1_COEFF 4.844920
#define X0_COEFF 240.279

// Image arrays
IplImage* depthimg = 0;
IplImage* rgbimg = 0;
IplImage* tempimg = 0;
IplImage* red_img = 0;
IplImage* orange_img = 0;
IplImage* canny_temp = 0;

// Raw distance to a cup, in 8-bit depth units
uint8_t cupdist = 0;
// Center of cup
uint16_t centX = 0;
uint16_t centY = 0;
// Distance to cup, in inches
double inches = 0.0;
// Distance to cup in the previous frame. Used for filtering.
double prevInches = 0.0;
// Displacement from the center line, in inches
double offsetInches = 0.0;
// Displacement from center line in previous frame. For filtering.
double prevOffsetInches = 0.0;
// The last distance and displacement before a drastic change.
double stableInches = 0.0;
double stableOffsetInches = 0.0;
// Number of frames since a cup was removed.
int keyFrameCount = 0;
// Number of consecutive frames we think we've seen a hit. 
unsigned short hitCount = 0;
// Mutexes for depth and RGB images, to make sure each arry is one frame.
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
        pthread_mutex_unlock( &mutex_depth );

}

// callback for rgbimage, called by libfreenect
void rgb_cb(freenect_device *dev, void *rgb, uint32_t timestamp)
{
        pthread_mutex_lock( &mutex_rgb );
        memcpy(rgbimg->imageData, rgb, FREENECT_VIDEO_RGB_SIZE);
        pthread_mutex_unlock( &mutex_rgb );
}

// Takes in an BGR image and returns a smoothed binary image of red objects.
IplImage* getRedImage(IplImage* img)
{
	IplImage* imgHSV = cvCreateImage(cvGetSize(img), 8, 3);
	cvCvtColor(img, imgHSV, CV_BGR2HSV);
	IplImage* imgThreshed = cvCreateImage(cvGetSize(img), 8, 1);
	IplImage* redHigh = cvCreateImage(cvGetSize(img), 8, 1);
	IplImage* redLow = cvCreateImage(cvGetSize(img), 8, 1);
	cvInRangeS(imgHSV, cvScalar(165, 130, 45), cvScalar(255, 255, 255), redHigh);
	cvInRangeS(imgHSV, cvScalar(0, 120, 45), cvScalar(6, 255, 255), redLow);
	cvOr(redHigh, redLow, imgThreshed);
	cvReleaseImage(&imgHSV);
	cvReleaseImage(&redHigh);
	cvReleaseImage(&redLow);
	cvSmooth(imgThreshed, imgThreshed, CV_MEDIAN, 7);
	return imgThreshed;
}

// Takes in an BGR image and returns a smoothed binary image of red objects.
IplImage* getOrangeImage(IplImage* img)
{
	IplImage* imgHSV = cvCreateImage(cvGetSize(img), 8, 3);
	cvCvtColor(img, imgHSV, CV_BGR2HSV);
	IplImage* imgThreshed = cvCreateImage(cvGetSize(img), 8, 1);
	cvInRangeS(imgHSV, cvScalar(10, 170, 50), cvScalar(17, 255, 255), imgThreshed);
	cvReleaseImage(&imgHSV);
	cvSmooth(imgThreshed, imgThreshed, CV_MEDIAN, 7);
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
	cvNamedWindow( "Ball Contours", CV_WINDOW_AUTOSIZE );
        depthimg = cvCreateImage(cvSize(FREENECTOPENCV_DEPTH_WIDTH, FREENECTOPENCV_DEPTH_HEIGHT), IPL_DEPTH_8U, FREENECTOPENCV_DEPTH_DEPTH);
        rgbimg = cvCreateImage(cvSize(FREENECTOPENCV_RGB_WIDTH, FREENECTOPENCV_RGB_HEIGHT), IPL_DEPTH_8U, FREENECTOPENCV_RGB_DEPTH);
        tempimg = cvCreateImage(cvSize(FREENECTOPENCV_RGB_WIDTH, FREENECTOPENCV_RGB_HEIGHT), IPL_DEPTH_8U, FREENECTOPENCV_RGB_DEPTH);
	red_img = cvCreateImage(cvSize(FREENECTOPENCV_RGB_WIDTH, FREENECTOPENCV_RGB_HEIGHT), IPL_DEPTH_8U, 1);
	orange_img = cvCreateImage(cvSize(FREENECTOPENCV_RGB_WIDTH, FREENECTOPENCV_RGB_HEIGHT), IPL_DEPTH_8U, 1);
	canny_temp = cvCreateImage(cvSize(FREENECTOPENCV_DEPTH_WIDTH, FREENECTOPENCV_DEPTH_HEIGHT), IPL_DEPTH_8U, FREENECTOPENCV_DEPTH_DEPTH);
	cupdist = 0;
	keyFrameCount = 0;
        // use image polling
        while (1) {
		keyFrameCount++;

		// Display the depth and color images
                pthread_mutex_lock( &mutex_depth );
                cvCanny(depthimg, canny_temp, 50.0, 200.0, 3);
		cvCvtColor(depthimg,tempimg,CV_GRAY2BGR);
                cvShowImage(FREENECTOPENCV_WINDOW_D,depthimg);
                pthread_mutex_unlock( &mutex_depth );
                pthread_mutex_lock( &mutex_rgb );
                cvCvtColor(rgbimg,tempimg,CV_BGR2RGB);
                cvShowImage(FREENECTOPENCV_WINDOW_N, tempimg);
                pthread_mutex_unlock( &mutex_rgb );

		//Threshold the red and orange images
		red_img = getRedImage(tempimg);
		orange_img = getOrangeImage(tempimg);
		cvShowImage("CUP CAM", red_img);
		cvShowImage("BALL CAM", orange_img);
		

		// Canny filter
		cvCanny(red_img, red_img, 50.0, 200.0, 3);
		cv::Mat cups(red_img);
		// This will hold the array of cup contours.
		std::vector<std::vector <cv::Point> > cupContours;
		std::vector<cv::Vec4i> cupHierarchy;
		cv::findContours(cups, cupContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point());
		cv::Scalar color = cv::Scalar(100,100,100);
		cv::Mat drawing = cv::Mat::zeros(cups.size(), CV_8UC3 );

		// Calculate the biggest red object
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
		// If one contour was found, handle cup distance
		if (cupContours.size() > 0)
		{
			cupRect = cv::boundingRect(cv::Mat(cupContours[biggestContour]));
			centX = cupRect.x + cupRect.width/2;
			centY = cupRect.y + cupRect.height/2;
			cupdist = depthimg->imageData[centX + centY*FREENECTOPENCV_RGB_WIDTH];
			inches = X3_COEFF*cupdist*cupdist*cupdist - X2_COEFF*cupdist*cupdist + X1_COEFF*cupdist - X0_COEFF;
			offsetInches = inches * sin(((centX - FREENECTOPENCV_RGB_WIDTH/2.0) / X_OFFSET_TO_DEG) * PI / 180.0);
			if ((inches > DEPTH_MIN_DIST) && (inches < DEPTH_MAX_DIST)) {
				inches = ALPHA * prevInches + (1-ALPHA) * inches;
				offsetInches =  ALPHA * prevOffsetInches + (1-ALPHA) * offsetInches;
				// If distance or offset is different, cup might have been removed.
				if ((keyFrameCount > FRAMES_TO_CONVERGENCE) 
				    && ((abs(inches - stableInches) > DISTANCE_THRESHOLD)
					|| (abs(offsetInches - stableOffsetInches > OFFSET_THRESHOLD)))) {
			       		hitCount += 1;
				}
				if (hitCount < 2) {
					stableInches = inches;
					stableOffsetInches = offsetInches;
				}
				if (hitCount > 3) {
					printf("Cup at distance %f, offset %f was hit!\n", prevInches, prevOffsetInches);
					keyFrameCount = 0;
					hitCount = 0;
				}
				else {
					printf("Cup dist %f offset %f\n", inches, offsetInches);
				}
				prevInches = inches;
				prevOffsetInches = offsetInches;
			}
		}
		drawContours(drawing, cupContours, biggestContour, color, 2, 8, cupHierarchy, 0, cv::Point());
		cv::imshow("Cup Contours", drawing);

		// Look for circles (ball)
		cv::Mat balls(orange_img);
		std::vector<std::vector <cv::Point> > ballContours;
		std::vector<cv::Vec4i> ballHierarchy;
		cv::findContours(balls, ballContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point());
		drawing = cv::Mat::zeros(cups.size(), CV_8UC3 );
		// Find the biggest orange object
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
		cv::imshow("Ball Contours", drawing);


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
	// Tilt the physical Kinect device to 0 degrees. 
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
