#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <bits/stdc++.h>
#include "opencv2/video/video.hpp"
#include <time.h>

using namespace cv;
using namespace std;

int main()
{
  int stsize = 6;  //state consists of x,y,v_x,v_y,width and height of box
  int msize = 4;   //measurements x,y,width and height of box
  KalmanFilter kf(6,4,0,CV_32F);
  Mat state(6,1,CV_32F);
  Mat meas(4,1,CV_32F);
  setIdentity(kf.transitionMatrix);
  kf.measurementMatrix = Mat::zeros(4, 6, CV_32F);
  kf.measurementMatrix.at<float>(0) = 1.0f;
  kf.measurementMatrix.at<float>(7) = 1.0f;
  kf.measurementMatrix.at<float>(16) = 1.0f;
  kf.measurementMatrix.at<float>(23) = 1.0f;
  
  kf.processNoiseCov.at<float>(0) = 1e-2;
  kf.processNoiseCov.at<float>(7) = 1e-2;
  kf.processNoiseCov.at<float>(14) = 5.0f;
  kf.processNoiseCov.at<float>(21) = 5.0f;
  kf.processNoiseCov.at<float>(28) = 1e-2;
  kf.processNoiseCov.at<float>(35) = 1e-2;

  setIdentity(kf.measurementNoiseCov,Scalar(1e-1));
  
  int count,j;
  double tic = 0;
  int found = 0;
  VideoCapture cap("frontcam.mp4");
    if(!cap.isOpened())
        return -1;
    
    Mat original;
    namedWindow("Original",1);
    //namedWindow("Mask",1);
    //namedWindow("Filtered",1);
    while(1)
    {
      double toc = tic;
      tic = (double) cv::getTickCount();
      double dT = (tic - toc) / cv::getTickFrequency(); //gets the interval
	
      Mat frame;
      cap>>frame;
      if(frame.empty()) break;
      original = frame.clone();

      if(found)
        {
            // >>>> Matrix A
            kf.transitionMatrix.at<float>(2) = dT;
            kf.transitionMatrix.at<float>(9) = dT;
            // <<<< Matrix A
 
            state = kf.predict();
           
            Rect predRect; //predicts the box which bounds the ball
            predRect.width = state.at<float>(4);
            predRect.height = state.at<float>(5);
            predRect.x = state.at<float>(0) - predRect.width / 2;
            predRect.y = state.at<float>(1) - predRect.height / 2;
 
            Point center;
            center.x = state.at<float>(0);
            center.y = state.at<float>(1);
            circle(original, center, 2, CV_RGB(255,0,0), -1);
 
            rectangle(original, predRect, CV_RGB(255,0,0), 2);
        }
      
      Mat org_hsv(original.rows,original.cols,CV_8UC3,Scalar(0,0,0));   
      Mat red_low(original.rows,original.cols,CV_8UC3,Scalar(0,0,0));	  
      Mat red_high(original.rows,original.cols,CV_8UC3,Scalar(0,0,0));/*detection*/   Mat red(original.rows,original.cols,CV_8UC3,Scalar(0,0,0));
      Mat filtered(original.rows,original.cols,CV_8UC3,Scalar(0,0,0));
      
      cvtColor(frame,org_hsv,CV_BGR2HSV);
      inRange(org_hsv,Scalar(0,10,10),Scalar(20,255,255),red_low);
      inRange(org_hsv,Scalar(160,10,10),Scalar(179,255,255),red_high);
      addWeighted(red_low,1.0,red_high,1.0,0.0,red);
      
      bitwise_and(original,original,filtered,red);
   
      int i;
  vector<vector<Point> > contours;
  Mat contour_output = red.clone();
  findContours(contour_output,contours,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);
  Scalar color(0,255,0);
  
  vector<vector<Point> > contours_poly( contours.size() );
  vector<Rect> boundrec( contours.size() );
  vector<Rect> balls;
  
    for(i = 0; i < contours.size(); i++ )
  {
    approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
    boundrec[i] = boundingRect( Mat(contours_poly[i]) );
  }
    int count = 0;
    for(i=0; i<contours.size(); i++)
      {
	if((boundrec[i].width * boundrec[i].height) > 2000) //finds large boxes
	  balls.push_back(boundrec[i]);
      }
    for(i=0; i<balls.size(); i++)
      {
	rectangle(original, balls[i].tl(), balls[i].br(), color, 2, 8, 0);
      }
    if(balls.size())
        {
            meas.at<float>(0) = balls[0].x + balls[0].width / 2;
            meas.at<float>(1) = balls[0].y + balls[0].height / 2;
            meas.at<float>(2) = (float)balls[0].width;
            meas.at<float>(3) = (float)balls[0].height;
 
            if (!found) // First detection!
            {
                // >>>> Initialization
                kf.errorCovPre.at<float>(0) = 1;
                kf.errorCovPre.at<float>(7) = 1;
                kf.errorCovPre.at<float>(14) = 1;
                kf.errorCovPre.at<float>(21) = 1;
                kf.errorCovPre.at<float>(28) = 1;
                kf.errorCovPre.at<float>(35) = 1; 
 
                state.at<float>(0) = meas.at<float>(0);
                state.at<float>(1) = meas.at<float>(1);
                state.at<float>(2) = 0;
                state.at<float>(3) = 0;
                state.at<float>(4) = meas.at<float>(2);
                state.at<float>(5) = meas.at<float>(3);
                // <<<< Initialization
 
                kf.statePost = state;
                
                found = 1;
            }
            else
                kf.correct(meas); // Kalman Correction
        }
      
      imshow("Original",original);
      //imshow("Mask",red);
      //imshow("Filtered",filtered);
      waitKey(10);
    }
    return 0;
}
