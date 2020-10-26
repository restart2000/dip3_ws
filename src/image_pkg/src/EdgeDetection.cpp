#include <cmath>
#include <cstring>
#include <complex>
#include <vector>
#include <cv.h>
#include <highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Float32.h"
#include<geometry_msgs/Twist.h>
#include "sensor_msgs/Image.h"
#define LINEAR_X 0
#define sqr(x) ((x)*(x))

const double e = 2.718281828f;
const double pi = 3.141592654f;
const float ZeroF = 0.000001f;


using namespace cv;
using namespace std;

//dip2_ws Gaussian filter
double **GetGaussianKernal(int size,double sigma)
{
	int i,j;
	double **kernal=new double *[size];
	for(i=0;i<size;i++)
	{
		kernal[i]=new double[size];
	}
	
	int center_i,center_j;
	center_i=center_j=size/2; //定义原点的位置
	double sum;
	sum=0;
	
	for(i=0;i<size;i++)
	{
		for(j=0;j<size;j++)
		{
			kernal[i][j]=exp(  (-1)*(((i-center_i)*(i-center_i)+(j-center_j)*(j-center_j)) / (2*sigma*sigma)));
			sum+=kernal[i][j];
		}
	}
	
	for(i=0;i<size;i++)
	{
		for(j=0;j<size;j++)
		{
			kernal[i][j]/=sum;  //归一化求权值
		}
	}
	return kernal;
}


void GaussianFilter(Mat *src,double **kernal,int size)
{
	Mat temp=(*src).clone();
	for(int i=0;i<(*src).rows;i++)
	{
		for(int j=0;j<(*src).cols;j++)
		{
			if(i>(size/2)-1 && j>(size/2)-1 && i<(*src).rows-(size/2) && j<(*src).cols-(size/2))
			//i > (size / 2) - 1 && j > (size / 2) - 1 忽略边缘
			//i < (*src).rows - (size / 2) && j < (*src).cols - (size / 2) 忽略下边缘
			{
				double sum=0;
				for(int k=0;k<size;k++)
				{
					for(int l=0;l<size;l++)
					{
						sum=sum + (*src).ptr<uchar>(i-k+(size/2))[j-l+(size/2)] * kernal[k][l];
					}
				}
				temp.ptr<uchar>(i)[j]=sum;
						//image.ptr<type>(i)[j],OpenCV中访问像素的方法，指针遍历
						//也可以image.at<type>(i, j)[channel]
			}
		}
	}
	*src=temp.clone();
}


Mat Gaussian(Mat input,Mat output,int size,double sigma)
{
	std::vector<cv::Mat> channels;
	cv::split(input,channels);
	
	double **kernal=GetGaussianKernal(size,sigma);
	for(int i=0;i<3;i++)
	{
		GaussianFilter(&channels[i],kernal,size);
	}

	cv::merge(channels,output);
	return output;
}



////////////////////边缘检测//////////////////

// Gradient Calculation
Mat Gradient(Mat img, Mat mskX, Mat mskY) {
	Mat ret(img.rows, img.cols, CV_32FC(2), Scalar::all(0));
	if (mskX.rows != mskY.rows || mskX.cols != mskY.cols) {
		printf("\n2 Mask's sizes DO NOT match!\n");
		return ret;
	}
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			float tmpX = 0.0f, tmpY = 0.0f;
			for (int s = i - mskX.rows / 2; s - mskX.rows < i - mskX.rows / 2; s++) {
				if (s < 0 || s > img.rows) continue;
				for (int t = j - mskX.cols / 2; t - mskX.cols < j - mskX.cols / 2; t++) {
					if (t < 0 || t > img.cols) continue;
					tmpX += img.at<uchar>(s, t) * mskX.at<float>(s - i + mskX.rows / 2, t - j + mskX.cols / 2);
					tmpY += img.at<uchar>(s, t) * mskY.at<float>(s - i + mskY.rows / 2, t - j + mskY.cols / 2);
				}
			}
			ret.at<Vec2f>(i, j)[0] = abs(tmpX) + abs(tmpY);
			ret.at<Vec2f>(i, j)[1] = abs(tmpX) < ZeroF ? (tmpY < 0 ? -pi / 2 : pi / 2 ) : atan(tmpY / tmpX);
		}
	return ret;
}

// Non-maximum suppression
void NMS(Mat input, Mat &output, int size) {
	output = input.clone();
	if (size < 2)
		return;
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++) {
			float maxP = 0.0f;
			float &phi = input.at<Vec2f>(i, j)[1];
			float &mag = input.at<Vec2f>(i, j)[0];
			if (abs(phi) < pi / 8) { // Horizontal
				for (int s = i - size / 2; s - size < i - size / 2; s++) {
					if (s < 0 || s >= input.rows) continue;
					maxP = maxP < input.at<Vec2f>(s, j)[0] ? input.at<Vec2f>(s, j)[0] : maxP;
				}
			} else if (phi > 3 * pi / 8 || phi < -3 * pi / 8) { // Vertical
				for (int t = j - size / 2; t - size < j - size / 2; t++) {
					if (t < 0 || t >= input.cols) continue;
					maxP = maxP < input.at<Vec2f>(i, t)[0] ? input.at<Vec2f>(i, t)[0] : maxP;
				}
			} else if (phi > 0) { //+45 degree
				for (int k = -size / 2; k - (-size / 2) < size; k++) {
					if (i + k < 0 || i + k >= input.rows || j + k < 0 || j + k >= input.cols) continue;
					maxP = maxP < input.at<Vec2f>(i + k, j + k)[0] ? input.at<Vec2f>(i + k, j + k)[0] : maxP;
				}
			} else {	// -45 degree
				for (int k = -size / 2; k - (-size / 2) < size; k++) {
					if (i + k < 0 || i + k >= input.rows || j - k < 0 || j - k >= input.cols) continue;
					maxP = maxP < input.at<Vec2f>(i + k, j - k)[0] ? input.at<Vec2f>(i + k, j - k)[0] : maxP;
				}
			}
			float &outij = output.at<Vec2f>(i, j)[0];
			outij = abs(outij - maxP) < ZeroF ? outij  : 0;
		}
	vector<Mat> chs;
	split(output, chs);
	output = chs[0].clone();
	return ;
}

// Delay Thresholding
void delayThresholding(Mat input, Mat & output, int tH, float k, int size) {
	input.copyTo(output);
	int tL = tH / k;
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++) {
			if (input.at<float>(i, j) < tL)
				output.at<float>(i, j) = 0;
			else if (input.at<float>(i, j) < tH) {
				bool find = false;
				for (int s = i - size / 2; s - i + size / 2 < size && !find; s++)
					for (int t = j - size / 2; t - j + size / 2 < size && !find; t++) {
						if (s < 0 || s >= input.rows || t < 0 || t >= input.cols) continue;
						if (input.at<float>(s, t) > tH)
							find = true;
					}
				if (!find)
					output.at<float>(i, j) = 0;
				else
					output.at<float>(i, j) = 255;
			} else
				output.at<float>(i, j) = 255;
		}
}

// cannyEdgeDetection
void cannyEdgeDetection(Mat input, Mat & output) {
	Mat frame_guassian;
	Mat gray;
	Mat frIn=input.clone();

	int size=7; //gaussian size
	int element_size=7;
	Mat element(element_size,element_size,CV_8U,Scalar(1));
	double sigma=1;
	frame_guassian=Gaussian(frIn,frame_guassian,size,sigma);
	cvtColor(frame_guassian, gray, COLOR_BGR2GRAY);

	gray.copyTo(output);

	Mat grd;
	Mat sobelX = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	Mat sobelY = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	// Get gradient (with Sobel operator)
	grd = Gradient(gray, sobelX, sobelY);
	// Non-maximum suppression
	NMS(grd, grd, 3);
	// Delay Thresholding
	delayThresholding(grd, grd,80, 2, 5);
	//Normalize to [0, 255]
	normalize(grd, grd, 0, 255, NORM_MINMAX);
	grd.convertTo(output, CV_8U);
}
////////////////////////////////////////////////



/////////////////////霍夫线变换//////////////////
void Hough_Line(Mat input ,Mat &output){
	output = Mat(input.rows, input.cols,CV_8U);
	cvtColor(input, output, COLOR_GRAY2RGB);
	int r_max=cvRound(sqrt(input.rows*input.rows+input.cols*input.cols));//rounded Diagonal length 
	//cout<<"r_max"<<r_max<<endl;
	int theta_max=  180;
	int threshold = 160; 

	vector<int> r_line;
	vector<int> theta_line;
	typedef vector<int> T;
	T zeros(r_max,0);
	vector<T> accumulator(theta_max,zeros);
	vector<T> flag(theta_max,zeros);
	//定义一个二维累加器，行为r，列为theta
	for(int i=0;i<input.rows;i++)
	{
		for(int j=0;j<input.cols;j++)
		{
			if(input.at<uchar>(i,j)==255)
			{
					for(int theta=0;theta<theta_max;theta++)
				{
					int temp_r=cvRound(i*cos(pi*theta/180)+j*sin(pi*theta/180));
					if(temp_r<r_max && temp_r>0)
					{
						accumulator[theta][temp_r]++;
						if(flag[theta][temp_r]==0 && accumulator[theta][temp_r]>threshold)
						{
							r_line.push_back(temp_r);
							theta_line.push_back(theta);
							flag[theta][temp_r]=1;
						}
					}
				}
			}	
		}
	}

	//draw the line
	Point pt1,pt2;
	for(int i=0;i<r_line.size();i++)
	{
		int r_ho=r_line[i], theta_ho=theta_line[i];
		double a=cos(pi*theta_ho/180),b=sin(pi*theta_ho/180);
		int x0=r_ho*a;
		int y0=r_ho*b;

		pt1.y = cvRound(x0 + 600*(-b));   //line函数的xy定义与图像的xy定义相反             
		pt1.x = cvRound(y0 + 600*(a));                 
		pt2.y = cvRound(x0 - 600*(-b));
		pt2.x = cvRound(y0 - 600*(a));
		line(output, pt1, pt2, Scalar(0,0,255), 1, CV_AA);
	}
}
////////////////////////////////////////////////

//////////////////////霍夫圆变换//////////////////
// void Hough_Circle(Mat input ,Mat &output){
// 	output = Mat(input.rows, input.cols, CV_8U, Scalar(0));
	
// }

// Circle Detection via Hough Transform
// void Hough_Circle(Mat input, Mat & output) {
// 	img=input.clone();
// 	output = Mat(input.rows, input.cols, CV_8U, Scalar(0));
	
// 	Ptr<CvMat> dx, dy;  //Ptr是智能指针模板，将CvMat对象封装成指针
// 	dx = cvCreateMat( img.rows, img.cols, CV_16SC1 );//16位单通道图像，用来存储二值边缘图像的x方向的一阶导数
// 	dy = cvCreateMat( img.rows, img.cols, CV_16SC1 );//y方向的
// 	cvSobel( img, dx, 1, 0, 3 );//计算x方向的一阶导数
// 	cvSobel( img, dy, 0, 1, 3 );//计算y方向的一阶导数

	
// }

#if 0
void Hough_Circle(Mat input)
{
	Mat output=input.clone();
        cvtColor(input,output,COLOR_GRAY2BGR,3);
	imshow("output",output);
	Mat Sobel_gradient = Mat::zeros(input.rows,input.cols,CV_16U);
	//Mat angle = Mat::zeros(input.rows,input.cols,CV_16U);
	Mat accumulator = Mat::zeros(input.rows,input.cols,CV_16U);
	for(int i=1;i<input.rows-1;i++)
    	{
        	for(int j=1;j<input.cols-1;j++)
       		{
            		if(input.at<uchar>(i,j)==255)
			{
				float gx=0;
				float gy=0;
				float g=0;
				gx = input.at<uchar>(i-1,j+1) + 2*input.at<uchar>(i,j+1) + input.at<uchar>(i+1,j+1) - (input.at<uchar>(i-1,j-1) + 2*input.at<uchar>(i,j-1) + input.at<uchar>(i+1,j-1));
				gy = input.at<uchar>(i-1,j-1) + 2*input.at<uchar>(i-1,j) +input.at<uchar>(i-1,j+1) -(input.at<uchar>(i+1,j-1) + 2*input.at<uchar>(i+1,j) + input.at<uchar>(i+1,j+1));
				g = sqrt(gx * gx + gy * gy);
				Sobel_gradient.at<uchar>(i,j) = g;
				if(fabs(gx) < 1e-5)
				{
					gx=1e-5;
				}
				for(int x=1;x<input.rows-1;x++)
				{
					for(int y=1;y<input.cols-1;y++)
					{
						if( x != i)
						{
							if( fabs( gy/gx  - (y-j)/(x-i) ) < 1e-5  )
							{
								accumulator.at<uchar>(x,y)=accumulator.at<uchar>(x,y)+1;
							}
						}
					} 
				}	
			}
                }
        }
	
	int r_min=50;
	int r_max=500;
	int r_length;
	r_length=r_max-r_min;
	int center_count=0;
	Mat center = Mat::zeros(input.rows,input.cols,CV_16U);
	for(int x=1;x<input.rows-1;x++)
    	{
        	for(int y=1;y<input.cols-1;y++)
       		{
			if( accumulator.at<uchar>(x,y) > 160 && accumulator.at<uchar>(x,y) > accumulator.at<uchar>(x-1,y) && accumulator.at<uchar>(x,y) > accumulator.at<uchar>(x+1,y) && 		 				    accumulator.at<uchar>(x,y) > accumulator.at<uchar>(x,y-1) && accumulator.at<uchar>(x,y) > accumulator.at<uchar>(x,y+1) )
			{
				center.at<uchar>(x,y) = 1;
				center_count+=1;
			}
		}
	}
	Mat r = Mat::zeros(center_count,r_length+1,CV_16U);
	for(int x=1;x<input.rows-1;x++)
    	{
        	for(int y=1;y<input.cols-1;y++)
       		{
			if( center.at<uchar>(x,y) != 0 )
			{
				int count=0;
				for(int i=1;i<input.rows-1;i++)
				{
					for(int j=1;j<input.cols-1;j++)
					{
						int rho=0;
						if( Sobel_gradient.at<uchar>(i,j) != 0 )
						{
							rho=(int)sqrt((x-i)*(x-i)+(y-j)*(y-j));
							r.at<uchar>(count,rho)+=1;
						}
					}
				}
				count+=1;
			}
		}
	}
	int x0[center_count]={0};
	int y0[center_count]={0};
	for(int x=1;x<input.rows-1;x++)  //将序号跟坐标一一对应
    	{
        	for(int y=1;y<input.cols-1;y++)
       		{
			if( center.at<uchar>(x,y) != 0 )
			{
				for(int i=0;i<center_count;i++)
				{
					x0[i]=x;
					y0[i]=y;
				}
			}
		}
	}
	for(int i=0;i<center_count;i++)
	{
		for(int j=0;j<r_length+1;j++)
		{
			if( r.at<uchar>(i,j) > 40 )
			{
				circle(output,Point(x0[i],y0[i]),j,Scalar(0,0,255), 3, 8, 0);
			}
		}
	}
}
#endif

////////////////////////////////////////////////

int main(int argc,char **argv)
{
	VideoCapture capture;
	capture.open(0);

	ROS_WARN("*****START");
	ros::init(argc,argv,"trafficLaneTrack");
	ros::NodeHandle n;

	//ros::Rate loop_rate(10);
	ros::Publisher pub = n.advertise<geometry_msgs::Twist>("/smoother_cmd_vel",5);

	if(!capture.isOpened())
	{
		printf("摄像头没有正常打开，重新插拔工控机上的摄像头\n");
		return 0;
	}
	waitKey(10);
	Mat frame;//canny
	Mat frame_HoughLine;//houghline
	Mat frame_HoughCircle;//houghcircle

	int nFrames = 0;//当前帧数
	int frameWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);//图片宽度
	int frameHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);//图片高度

	while(ros::ok())
	{
		// capture.read(frame);
		// if(frame.empty())
		// {
		//     break;
		// }

	frame=imread("./lena.jpg");  //如果使用该段代码，请注意将相机capture.read关闭
	if (frame.empty())
	{
		printf("Can not load image\n");
		return 0;
	}
	imshow("lena",frame);

	frame_HoughLine=imread("./line.jpg");
	if (frame_HoughLine.empty())
	{
		printf("Can not load image\n");
		return 0;
	}
	imshow("Line",frame_HoughLine);

	frame_HoughCircle=imread("./test2.png");
	if (frame_HoughCircle.empty())
	{
		printf("Can not load image\n");
		return 0;
	}
	imshow("Circle",frame_HoughCircle);

	Mat output_canny = frame.clone();
	cannyEdgeDetection(frame, output_canny);
	imshow("output_canny", output_canny);

	Mat output_houghline;
	cannyEdgeDetection(frame_HoughLine,output_houghline);
	Mat output2;
	Hough_Line(output_houghline,output2);
	imshow("output_houghline",output2);

	Mat output_houghcircle;
	cannyEdgeDetection(frame_HoughCircle,output_houghcircle);
	Mat output3;
	vector<Vec3f>circles;
	HoughCircles(output_houghcircle,circles,CV_HOUGH_GRADIENT, 1, output_houghcircle.rows/20, 50, 46, 0, 0);
	for (size_t i = 0; i < circles.size(); i++){
		//提取出圆心坐标
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		//提取出圆半径
		int radius = cvRound(circles[i][2]);
		//圆心
		circle( frame_HoughCircle, center, 3, Scalar(0,0,0), -1, 8, 0 );
		//圆
		circle( frame_HoughCircle, center, radius, Scalar(0,0,255), 3, 8, 0 );
	}
	imshow("frame_HoughCircle", frame_HoughCircle);
	//Hough_Circle(output_houghcircle);
	//imshow("output_houghcircle",output3);

	ros::spinOnce();
	waitKey(5);
	}
	return 0;
}

