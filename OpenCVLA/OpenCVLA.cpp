#include "stdafx.h"
//#include "ExposureFusion.h"
#include <opencv2\opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ximgproc/edge_filter.hpp"
//#include <opencv2/saliency.hpp>
#include <fstream>
#include <stdint.h>
#include <vector>
#include <cstdint>
#include <math.h>
//#include <opencv2/saliency/saliencySpecializedClasses.hpp>
//#include "opencv2/xphoto.hpp"
//#include <opencv2/xphoto/white_balance.hpp>
//#include "opencv2/ximgproc.hpp"
#include <iostream>
#include <sstream>
using namespace std;
using namespace cv;

//using namespace saliency;

vector<Mat> bulid_Gaussian_Pyr(Mat& input, vector<Mat> Img_pyr, int level);
vector<Mat> bulid_Laplacian_Pyr(vector<Mat> Img_Gaussian_pyr, vector<Mat> Img_Laplacian_pyr, int level);
vector<Mat> blend_Laplacian_Pyr(vector<Mat> Img1_lp, vector<Mat> Img2_lp, vector<Mat> mask_gau, vector<Mat> blend_lp);
Mat blend(Mat& result_higest_level, vector<Mat> blend_lp);

vector<Mat> GaussianPyramid(Mat input, int level){
	vector<Mat> _GaussianPyramid;
	Mat currentImage = input;
	for (int i = 0; i < level; i++){
		Mat downSampleImage;
		pyrDown(currentImage, downSampleImage);
		_GaussianPyramid.push_back(downSampleImage);
		currentImage = downSampleImage;
	}
	return _GaussianPyramid;
}


vector<Mat_<float>> NewLapPyramid(Mat input, int level){
	vector<Mat_<float>> _NewLapPyramid;
	Mat currentImage = input;
	for (int i = 0; i < level; i++){
		Mat upSampleImage;
		pyrUp(currentImage, upSampleImage);
		_NewLapPyramid.push_back(upSampleImage);
		currentImage = upSampleImage;
	}
	return _NewLapPyramid;;
}

vector<Mat_<float>> LaplacianPyramid(vector<Mat_<float>> gaussPyrImg){
	int levels = gaussPyrImg.size();
	vector<Mat_<float>> _LaplacianPyramid;
	_LaplacianPyramid.push_back(gaussPyrImg[levels - 1]);
	for (int i = levels - 2; i >= 0; i--){
		Mat upsampleImage;
		pyrUp(gaussPyrImg[i + 1], upsampleImage, gaussPyrImg[i].size());
		Mat currentImage = gaussPyrImg[i] - upsampleImage;
		_LaplacianPyramid.push_back(currentImage);
	}
	return _LaplacianPyramid;


}

vector<Mat_<float>> blendPyramid(vector<Mat_<float>> PyrA, vector<Mat_<float>> PyrB, vector<Mat_<float>> Pyrmask){
	int level = Pyrmask.size();
	vector<Mat_<float>> blendedPyramid;
	//_LaplacianPyramid.push_back(gaussPyrImg[levels - 1]);
	for (int i = level; i < level; i++){
		Mat blendedImage;
		blendedImage = PyrA[i].mul(1.0 - Pyrmask[level - 1 - i]) + PyrB[i].mul(1.0 - Pyrmask[level - 1 - i]);
		blendedPyramid.push_back(blendedImage);
	}
	return blendedPyramid;

}

Mat collapsePyramid(vector<Mat_<float>> blendedPyramid){
	int level = blendedPyramid.size();
	Mat currentImage = blendedPyramid[0];
	for (int i = 0; i < level; i++){
		pyrUp(currentImage, currentImage, blendedPyramid[i].size());
		currentImage += blendedPyramid[i];
	}
	Mat blendedImage;
	convertScaleAbs(currentImage, blendedImage, 255.0);
	return blendedImage;
}

vector<Mat> bulid_Gaussian_Pyr(Mat& input, vector<Mat> Img_pyr, int level)
{
	/*
	Parameter Description:
	Parameter 1: The input Mat type to be calculated Gaussian pyramid image
	Parameter 2: The output Gaussian pyramid (stored in vector<Mat> type, you can use .at() to get the content of a certain layer)
	Parameter 3: The number of levels of the Gaussian pyramid (Special attention should be paid here: the real number of levels is level + 1!)
	*/

	Img_pyr.push_back(input);
	Mat dst;
	for (int i = 0; i < level; i++)
	{
		pyrDown(input, dst, Size(input.cols / 2, input.rows / 2));
		Img_pyr.push_back(dst);
		input = dst;
	}
	return Img_pyr;
}


//--------------------------------- The construction of Function 2 Laplace Pyramid------ -------------------------------------------------- -//
vector<Mat> bulid_Laplacian_Pyr(vector<Mat> Img_Gaussian_pyr, vector<Mat> Img_Laplacian_pyr, int level)
{
	/*
	Parameter Description:
	Parameter 1: The input Gaussian pyramid vector<Mat> type, each element represents each layer
	Parameter 2: Laplacian pyramid to be solved
	Parameter 3: The number of levels of the Laplace pyramid level
	*/

	vector<Mat> Img_Gaussian_pyr_temp;
	Img_Gaussian_pyr_temp.assign(Img_Gaussian_pyr.begin(), Img_Gaussian_pyr.end());   //Because the vector object cannot use = copy, here use assign to copy

	Mat for_sub, for_up;
	for (int i = 0; i < level; i++)
	{
		Mat for_up = Img_Gaussian_pyr_temp.back();       //Get a reference to the current highest image of the Gaussian pyramid
		Img_Gaussian_pyr_temp.pop_back();                //Delete the last element

		for_sub = Img_Gaussian_pyr_temp.back();          //Get the subtracted image

		Mat up2;
		pyrUp(for_up, up2, Size(for_up.cols * 2, for_up.rows * 2));    //Upsample

		Mat lp;
		/*
		cout<<"size1"<<for_sub.size();
		cout<<"c size2"<<up2.size();
		*/
		lp = for_sub - up2;
		Img_Laplacian_pyr.push_back(lp);
	}
	reverse(Img_Laplacian_pyr.begin(), Img_Laplacian_pyr.end());       //Do a reversal (0->maximum size pyramid layer)
	return Img_Laplacian_pyr;
}

//------------------------------------Function 3 Construction of the Hybrid Laplacian Pyramid-- ---------------------------------------//
vector<Mat> blend_Laplacian_Pyr(vector<Mat> Img1_lp, vector<Mat> Img2_lp, vector<Mat> mask_gau, vector<Mat> blend_lp)
{
	/*Parameter Description:
	Parameter 1: Laplacian pyramid vector<Mat> type of image 1 to be fused (level layer)
	Parameter 2: Laplacian pyramid vector<Mat> type of image 2 to be fused
	Parameter 3: Gaussian pyramid of mask (level+1)
	Parameter 4: The mixed Laplacian pyramid vector<Mat> type to be returned
	*/

	int level = Img1_lp.size();

	//cout<<"level"<<level; Confirm the number of level 

	for (int i = 0; i < level; i++)                                        //Note that 0 means the largest picture, indicating that lp is merged from the bottom
	{
		Mat A = (Img1_lp.at(i)).mul(mask_gau.at(i));                      //According to mask (as weight) 

		Mat antiMask = Scalar(1.0, 1.0, 1.0) - mask_gau[i];
		Mat B = Img2_lp[i].mul(antiMask);
		Mat blendedLevel = A + B;                                         //The corresponding layer of the Laplacian pyramid of the image to be fused is fused according to the mask
		blend_lp.push_back(blendedLevel);                                 //Save to blend_lp as the i-th layer
	}
	return blend_lp;
}

//--------------Function 4 image fusion---------------------//
Mat blend(Mat& result_higest_level, vector<Mat> blend_lp)
{
	/*Parameter Description:
	Parameter 1: The starting point of image mixing Mat (that is, the result of the weighted summation of the two highest layers of Gaussian pyramids with fused images according to mask
	Parameter 2: The mixed Laplacian pyramid vector<Mat> type obtained by Function 3
	*/

	int level = blend_lp.size();
	Mat for_up, temp_add;
	for (int i = 0; i < level; i++)
	{
		pyrUp(result_higest_level, for_up, Size(result_higest_level.cols * 2, result_higest_level.rows * 2));
		temp_add = blend_lp.back() + for_up;
		blend_lp.pop_back();              //Because here is to delete the last element directly, before calling this function, as the subsequent code also needs blend_lp, you need to save the copy first
		result_higest_level = temp_add;
	}
	return temp_add;
}

Mat PerfectReflectionAlgorithm(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	int HistRGB[767] = { 0 };
	int MaxVal = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			MaxVal = max(MaxVal, (int)src.at<Vec3b>(i, j)[0]);
			MaxVal = max(MaxVal, (int)src.at<Vec3b>(i, j)[1]);
			MaxVal = max(MaxVal, (int)src.at<Vec3b>(i, j)[2]);
			int sum = src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2];
			HistRGB[sum]++;
			//cout << "HistRGB: " << HistRGB[i] << endl;
		}
	}
	int Threshold = 0;
	int sum = 0;
	for (int i = 766; i >= 0; i--) {
		sum += HistRGB[i];
		//cout << "Sum: " << sum << endl;
		if (sum > row* col * 0.1) {
			Threshold = i;
			//cout << "Treshold: " << Threshold << endl;
			break;
		}
	}
	int AvgB = 0;
	int AvgG = 0;
	int AvgR = 0;
	int cnt = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int sumP = src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2];
			if (sumP > Threshold) {
				AvgB += src.at<Vec3b>(i, j)[0];
				AvgG += src.at<Vec3b>(i, j)[1];
				AvgR += src.at<Vec3b>(i, j)[2];
				cnt++;
			}
		}
	}
	//Get Average Value of RGB from input
	AvgB /= cnt;
	AvgG /= cnt;
	AvgR /= cnt;
	//cout << "B val: " << AvgB << endl;
	//cout << "G val: " << AvgG << endl;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Blue = src.at<Vec3b>(i, j)[0] * MaxVal / AvgB;
			int Green = src.at<Vec3b>(i, j)[1] * MaxVal / AvgG;
			int Red = src.at<Vec3b>(i, j)[2] * MaxVal / AvgR;
			if (Red > 255) {
				Red = 255;
			}
			else if (Red < 0) {
				Red = 0;
			}
			if (Green > 255) {
				Green = 255;
			}
			else if (Green < 0) {
				Green = 0;
			}
			if (Blue > 255) {
				Blue = 255;
			}
			else if (Blue < 0) {
				Blue = 0;
			}
			dst.at<Vec3b>(i, j)[0] = Blue;
			dst.at<Vec3b>(i, j)[1] = Green;
			dst.at<Vec3b>(i, j)[2] = Red;
		}
	}
	return dst;
}


//Normalize
double normalizedWeightWB(double wc1, double ws1, double we1, double wl1, double wc2, double ws2, double we2, double wl2){

	double normalized1 = (wl1 + wc1 + ws1 + we1) / (wl1 + wc1 + ws1 + we1 + wl2 + wc2 + ws2 + we2);
	//cout << normalized1 << endl;
	//normalized1 *= 255;
	return normalized1;
}

double normalizedWeightClahe(double wc1, double ws1, double we1, double wl1, double wc2, double ws2, double we2, double wl2){


	double normalized2 = (wl2 + wc2 + ws2 + we2) / (wl1 + wc1 + ws1 + we1 + wl2 + wc2 + ws2 + we2);
	//normalized2 *= 255;
	return normalized2;
}



void convolve(Mat src, float kernelX[], Mat& dst, int sizeKernel){

	//calculate padding needed to keep src image the same
	double pad = (sizeKernel - 1.0) / 2.0;

	//add padding
	Mat paddedImg = Mat(src.rows + 2 * pad, src.cols + 2 * pad, src.type(), double(0.0));
	copyMakeBorder(src, paddedImg, pad, pad, pad, pad, BORDER_REFLECT);

	//create temp matrices for convolution
	Mat convolvedImg = Mat(paddedImg.rows, paddedImg.cols, paddedImg.type(), double(0.0));
	Mat convolvedImgX = Mat(paddedImg.rows, paddedImg.cols, paddedImg.type(), double(0.0));

	double sum;

	//convolve with horizontal kernel over the rows of the images
	for (int r = 0; r<paddedImg.rows; r++){

		for (int c = 0; c<paddedImg.cols - 2 * pad; c++){
			sum = 0.0;
			for (int k = 0; k<sizeKernel; k++){
				sum += paddedImg.at<uchar>(r, c + k) * kernelX[k];
			}
			//store convolution value into temp matrix
			convolvedImgX.at<uchar>(r, c + pad) = sum;
		}
	}

	//convolve on previous result columns, using vertical kernel
	for (int c = pad; c<paddedImg.cols - pad; c++){
		for (int r = 0; r<paddedImg.rows - 2 * pad; r++){
			sum = 0.0;
			for (int k = 0; k<sizeKernel; k++){
				sum += convolvedImgX.at<uchar>(r + k, c) * kernelX[k];
			}
			convolvedImg.at<uchar>(r + pad, c) = sum;
		}
	}

	//remove added padding
	convolvedImg(Range(0 + pad, convolvedImg.rows - pad), Range(0 + pad, convolvedImg.cols - pad)).copyTo(dst);
}

void applyGaussianFilter(Mat src, Mat& dst){

	//create kernel coefficients (coeff used in build-in fct pyrDown of opencv)
	float kernelX[5] = { 1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0 };
	int sizeKernel = 5;

	//convolve img src with kernel
	convolve(src, kernelX, dst, sizeKernel);

	/*
	double data[25] = {1.0,4.0,6.0,4.0,1.0,4.0,16.0,24.0,16.0,4.0,6.0,24.0,36.0,24.0,6.0,4.0,16.0,24.0,16.0,4.0,1.0,4.0,6.0,4.0,1.0};
	double coeff = 1.0/256.0;

	//apply coeff to Mat
	Mat kernel = Mat(5, 5, CV_64F, data);
	kernel *= coeff;

	//convolve img scr with kernel
	filter2D(src, dst, -1, kernel, Point(-1,-1), 0, BORDER_DEFAULT);
	*/
}



void SimplestCB(Mat& in, Mat& out, float percent) {
	assert(in.channels() == 3);
	assert(percent > 0 && percent < 100);

	float half_percent = percent / 200.0f;

	vector<Mat> tmpsplit; split(in, tmpsplit);
	for (int i = 0; i<3; i++) {
		//find the low and high precentile values (based on the input percentile)
		Mat flat; tmpsplit[i].reshape(1, 1).copyTo(flat);
		cv::sort(flat, flat, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
		int lowval = flat.at<uchar>(cvFloor(((float)flat.cols) * half_percent));
		int highval = flat.at<uchar>(cvCeil(((float)flat.cols) * (1.0 - half_percent)));
		//cout << lowval << " " << highval << endl;

		//saturate below the low percentile and above the high percentile
		tmpsplit[i].setTo(lowval, tmpsplit[i] < lowval);
		tmpsplit[i].setTo(highval, tmpsplit[i] > highval);

		//scale the channel
		normalize(tmpsplit[i], tmpsplit[i], 0, 255, NORM_MINMAX);
	}
	merge(tmpsplit, out);
}

void gammaCorrection(const Mat &src, Mat &dst, const float gamma)
{
	float invGamma = 1 / gamma;

	Mat table(1, 256, CV_8U);
	uchar *p = table.ptr();
	for (int i = 0; i < 256; ++i) {
		p[i] = (uchar)(pow(i / 255.0, invGamma) * 255);
	}

	LUT(src, table, dst);
}

//Expos
double exponentialCalculation(double val, float sigma, double dst) {
	double delta = ((val / 255.0) - 0.5);
	double twoSigmaSq = (2 * sigma * sigma);
	return dst = exp(-delta * delta / twoSigmaSq);

}


string intToStr(int i, string path){
	string bla = "0";
	stringstream ss;
	ss << i;
	string ret = "";
	ss >> ret;
	string name = bla.substr(0, bla.size() - ret.size());
	name = path + name + ret + ".jpg";
	return name;
}


int main(int argc, char** argv)
{

	Mat img, grayImg, edges, result, result2, W1, W1s, W2, W2s, LaplaceOutput, FusiOutput1;

	/*
		read images from "Dataset" folder
	*/
	for (int u = 12; u <= 50; u++){
		img = imread("./Dataset/" + to_string(u) + ".jpg", CV_LOAD_IMAGE_COLOR);
		cout << u << endl;

		cv::resize(img, img, cv::Size(720, 480));

		String folder0 = "./Dataset/Input21/"; //Output Folder
		String file0 = folder0 + to_string(u) + ".jpg";
		imwrite(file0, img);

		Mat clone = img.clone();

		int height = clone.rows;
		int width = clone.cols;

		//White Balance
		result = PerfectReflectionAlgorithm(clone);
		//cv::resize(result, result, cv::Size(720, 480));
		//imshow("White Balance", result);

		//CLAHE 
		Mat clone2 = result.clone();
		cv::Mat lab_image, subImage;
		cv::cvtColor(result, lab_image, CV_BGR2Lab);
		//imshow("CLAHE Lab", lab_image);
		Mat labInput = lab_image.clone();

		// Extract the L channel
		vector<cv::Mat> lab_planes(3);
		cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

		// apply the CLAHE algorithm to the L channel
		Ptr<CLAHE> clahe = createCLAHE();
		clahe->setClipLimit(4);
		Mat dst, dst2, laplace_done;
		clahe->apply(lab_planes[0], dst);
		// Merge the the color planes back into an Lab image
		dst.copyTo(lab_planes[0]);
		cv::merge(lab_planes, lab_image);

		// convert back to RGB
		cv::Mat image_clahe;

		Mat image_claheLab = lab_image.clone();
		//imshow("CLAHE Lab Done", image_claheLab);
		cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);
		//cv::resize(image_clahe, image_clahe, cv::Size(720, 480));
		//imshow("CLAHE", image_clahe);

		//Mat gabung;
		//hconcat(result, image_clahe, gabung);
		//imshow("Gabung", gabung);

		Mat clone3 = image_clahe.clone();

		//Histogram CLAHE
		/// Establish the number of bins
		int histSize = 256;

		vector<Mat> bgr_planes;
		split(result, bgr_planes);
		/// Set the ranges ( for B,G,R) )
		float range[] = { 0, 256 };
		const float* histRange = { range };

		bool uniform = true; bool accumulate = false;

		Mat b_hist, g_hist, r_hist;

		/// Compute the histograms:
		calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

		// Draw the histograms for B, G and R
		int hist_w = 512; int hist_h = 400;
		int bin_w = cvRound((double)hist_w / histSize);

		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

		/// Normalize the result to [ 0, histImage.rows ]
		normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

		/// Draw for each channel
		for (int i = 1; i < histSize; i++)
		{
			line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
				Scalar(255, 0, 0), 2, 8, 0);
			line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
				Scalar(0, 255, 0), 2, 8, 0);
			line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
				Scalar(0, 0, 255), 2, 8, 0);
		}

		/// Display
		//imshow("calcHist Input", histImage);


		//Clahe Laplace
		int kernel_size = 3, scale = 1, delta = 0, ddepth = CV_32F;
		Mat gambar_clahe, gambar_gray, ddd;
		cvtColor(image_clahe, gambar_gray, COLOR_BGR2GRAY); // Convert the image to grayscale
		//imshow("CLAHElaplaceGray", gambar_gray);
		GaussianBlur(gambar_gray, gambar_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);
		//imshow("CLAHElaplaceGrayBlur", gambar_gray);
		//cvtColor(gambar_clahe, gambar_gray, COLOR_BGR2GRAY); // Convert the image to grayscale
		Mat abs_dst;
		//cvtColor(gambar_clahe, gambar_gray, COLOR_BGR2GRAY);
		//normalize(image_claheLab, image_claheLab, 0, 1, NORM_MINMAX, CV_32F);
		//image_claheLab.convertTo(image_claheLab, CV_32FC3, 1.f / 255);
		//cvtColor(image_claheLab, image_claheLab, COLOR_BGR2GRAY); // Convert the image to grayscale
		Laplacian(gambar_gray, ddd, ddepth, 3, 1, 0);
		normalize(ddd, ddd, 0, 1, NORM_MINMAX, CV_32F, Mat());
		//convertScaleAbs(ddd, ddd,128,128);
		//cv::threshold(ddd, ddd, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
		//cv::resize(ddd, ddd, cv::Size(720, 480));
		//imshow("Laplacian Contrast Clahe (WC2)", ddd);


		//Clahe Local Contrast
		Mat localgray, grayBlur, deezLocal, divideLocalBlur, img32F, img32FGrey, unsharpMas32F;
		double threshold = 5;
		//cvtColor(clone2, img32F, COLOR_BGR2Lab);
		//cv::cvtColor(img32F, img32F, CV_Lab2BGR);
		//cvtColor(clone2, img32F, COLOR_BGR2GRAY);

		clone2.convertTo(img32F, CV_32F);
		//imshow("im32f", img32F);
		cvtColor(img32F, img32F, COLOR_BGR2GRAY); // Convert the image to grayscale
		//imshow("localgray", localgray);
		cv::GaussianBlur(img32F, grayBlur, Size(5, 5), 0, 0, BORDER_REPLICATE);
		//subtract(img32F, grayBlur, divideLocal);
		//absdiff(img32F, grayBlur, divideLocal);
		absdiff(img32F, grayBlur, deezLocal);
		Sobel(deezLocal, deezLocal, CV_32F, 1, 0);
		double minValWB, maxValWB;
		minMaxLoc(deezLocal, &minValWB, &maxValWB); //find minimum and maximum intensities
		deezLocal.convertTo(deezLocal, CV_8U, 255.0 / (maxValWB - minValWB), -minValWB * 255.0 / (maxValWB - minValWB));
		img32F.convertTo(img32F, CV_8U);
		grayBlur.convertTo(grayBlur, CV_8U);

		//imshow("LocalGrayInput", img32F);
		//imshow("LocalGray", localgray);
		//imshow("LocalGrayBlur", grayBlur);
		//imshow("WB LocalContrast(WL1)", deezLocal);

		Mat localgray0, localgrayShow0, grayBlur0, divideLocal0, divideLocalBlur0, img32F0, imgBlur32F0, unsharpMas32F0, deezLocal0;
		//double threshold = 5;
		//Clahe Local Contrast

		//cvtColor(clone3, img32F0, COLOR_BGR2Lab);
		clone3.convertTo(img32F0, CV_32F);
		cvtColor(img32F0, localgray0, COLOR_BGR2GRAY); // Convert the image to grayscale
		//imshow("localgray0", localgray0);
		cv::GaussianBlur(localgray0, grayBlur0, Size(5, 5), 0, 0, BORDER_REPLICATE);

		//subtract(img32F, grayBlur, divideLocal);
		absdiff(localgray0, grayBlur0, deezLocal0);
		Sobel(deezLocal0, deezLocal0, CV_32F, 1, 0);
		double minVal, maxVal;
		minMaxLoc(deezLocal0, &minVal, &maxVal); //find minimum and maximum intensities
		deezLocal0.convertTo(deezLocal0, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
		localgray0.convertTo(localgrayShow0, CV_8U);
		grayBlur0.convertTo(grayBlur0, CV_8U);
		//imshow("BlurGray0", localgrayShow0);
		//imshow("ClaheLocalContrast", grayBlur0);
		//imshow("Clahe Local Contrast (WL2)", deezLocal0);

		
		vector<Mat> Gau_Pyr, lp_Pyr, Gau_Pyr2, lp_Pyr2;
		Mat resultPyr, image_clahePyr;
		vector<Mat> maskGaussianPyramid;

		resize(result, resultPyr, Size(720, 480));
		resize(image_clahe, image_clahePyr, Size(720, 480));

		//cv::cvtColor(resultPyr, resultPyr, COLOR_BGR2GRAY);
		//cv::cvtColor(image_clahePyr, image_clahePyr, COLOR_BGR2GRAY);

		resultPyr.convertTo(resultPyr, CV_32F);                  //Convert to CV_32F, which is used to match the type of mask (in addition, CV_32F has high precision and is beneficial to calculation)
		image_clahePyr.convertTo(image_clahePyr, CV_32F);

		//imshow("COBAAAAA", image_clahePyr);

		Gau_Pyr = bulid_Gaussian_Pyr(resultPyr, Gau_Pyr, 4);       //Calculate the Gaussian pyramid of two pictures
		Gau_Pyr2 = bulid_Gaussian_Pyr(image_clahePyr, Gau_Pyr2, 4);

		//imshow("0th layer of the L pyramid", Gau_Pyr.at(0));
		//imshow("1th layer of the L pyramid", Gau_Pyr.at(1));
		//imshow("2th layer of the L pyramid", Gau_Pyr.at(2));
		//imshow("3th layer of the L pyramid", Gau_Pyr.at(3));

		

		Mat newThis0 = Gau_Pyr2[0];
		Mat newThis1 = Gau_Pyr2[1];
		Mat newThis2 = Gau_Pyr2[2];
		Mat newThis3 = Gau_Pyr2[3];
		//imshow("0th layer of the G pyramid", newThis0 / 255);
		//imshow("1st layer of the G pyramid", newThis1 / 255);
		//imshow("2nd layer of the G pyramid", newThis2 / 255);
		//imshow("3rd layer of the G pyramid", newThis3 / 255);


		

		lp_Pyr = bulid_Laplacian_Pyr(Gau_Pyr, lp_Pyr, 4);     //Calculate the Laplacian pyramid of the two images
		lp_Pyr2 = bulid_Laplacian_Pyr(Gau_Pyr2, lp_Pyr2, 4);
		/*
		imshow("0th layer of the L pyramid", lp_Pyr.at(0));
		imshow("1st layer of the L pyramid", lp_Pyr.at(1));
		imshow("2nd layer of the L pyramid", lp_Pyr.at(2));
		imshow("3rd layer of the L pyramid", lp_Pyr.at(3));
		*/

		/*
		imshow("0th layer of the L pyramid", lp_Pyr2.at(0));
		imshow("1st layer of the L pyramid", lp_Pyr2.at(1));
		imshow("2nd layer of the L pyramid", lp_Pyr2.at(2));
		imshow("3rd layer of the L pyramid", lp_Pyr2.at(3));
		*/

		// Create 1280x480 mat for window
		//cv::Mat win_mat(cv::Size(1280, 480), CV_8UC3);

		// Copy small images into big mat
		//lp_Pyr.at(0).copyTo(win_mat(cv::Rect(0, 0, 640, 480)));
		//lp_Pyr.at(1).copyTo(win_mat(cv::Rect(640, 0, 640, 480)));

		// Display big mat
		//cv::imshow("Images", win_mat);


		Mat mask = Mat::zeros(height, width, CV_32FC1);           //Construction mask mask, CV_32FC1 type, the same size as Img1
		//mask(Range::all(), Range(0, mask.cols * 0.5)) = 0;      //All lines of the mask, then the left half is 1, and the right half is 0 (meaning half and half fusion)

		cvtColor(mask, mask, CV_GRAY2BGR);                        //Because the mask is single channel at this time, Img is 3channel, so cvtColor is also needed

		//cout<<"The type of mask now"<<mask.type()<<endl;
		//cout<<"The current type of lp"<<lp_Pyr.at(0).type()<<endl;

		vector<Mat> mask_Pyr, blend_lp, Fusi_blend_lp;
		Mat result_higest_level;                                  //The starting point of image fusion
		mask_Pyr = bulid_Gaussian_Pyr(mask, mask_Pyr, 4);         //The Gaussian pyramid of the mask is also level+1

		//Follow the top layer of the Gaussian pyramid of Img1 and Img2 according to the mask
		result_higest_level = Gau_Pyr.back().mul(mask_Pyr.back()) + ((Gau_Pyr2.back()).mul(Scalar(1.0, 1.0, 1.0) - mask_Pyr.back()));

		/*
		imshow("mask the 0th layer of the Gaussian pyramid", mask_Pyr.at(0));
		imshow("mask the first layer of the Gaussian pyramid", mask_Pyr.at(1));
		imshow("mask the second layer of the Gaussian pyramid", mask_Pyr.at(2));
		imshow("mask the third layer of the Gaussian pyramid", mask_Pyr.at(3));
		imshow("mask the third layer of the Gaussian pyramid", mask_Pyr.at(4));
		*/

		blend_lp = blend_Laplacian_Pyr(lp_Pyr, lp_Pyr2, mask_Pyr, blend_lp);

		/*
		imshow("The 0th layer of the fusion Laplace pyramid", blend_lp.at(0));
		imshow("Fuse the first layer of the Laplace pyramid", blend_lp.at(1));
		imshow("The second layer of the fusion Laplace pyramid", blend_lp.at(2));
		imshow("The third layer of the fusion Laplace pyramid", blend_lp.at(3));
		*/
		//imshow("The starting point of image fusion", result_higest_level);



		Mat output;
		output = blend(result_higest_level, blend_lp);
		output.convertTo(output, CV_8U);

		//addWeighted(image_clahe, 0.5, result, 0.5, 0.0, output);
		//cv::cvtColor(output, output, CV_GRAY2BGR);
		//imshow("Laplace Pyr", output);
		

		/*
		vector<Mat> Fusi_Gau_Pyr, Fusi_lp_Pyr, Fusi_Gau_Pyr2, Fusi_lp_Pyr2;

		Mat output2;

		vector<Mat> Fusi_maskGaussianPyramid;

		W1.convertTo(W1, CV_32F);                  //Convert to CV_32F, which is used to match the type of mask (in addition, CV_32F has high precision and is beneficial to calculation)
		W2.convertTo(W2, CV_32F);

		//resize(output2, output2, Size(720, 480));
		//resize(, image_clahePyr, Size(720, 480));

		Fusi_Gau_Pyr = bulid_Gaussian_Pyr(W1, Fusi_Gau_Pyr, 4);       //Calculate the Gaussian pyramid of two pictures
		Fusi_Gau_Pyr2 = bulid_Gaussian_Pyr(W2, Fusi_Gau_Pyr2, 4);

		//Fusi_lp_Pyr = bulid_Laplacian_Pyr(Fusi_Gau_Pyr, Fusi_lp_Pyr, 4);     //Calculate the Laplacian pyramid of the two images
		//Fusi_lp_Pyr2 = bulid_Laplacian_Pyr(Fusi_Gau_Pyr2, Fusi_lp_Pyr2, 4);


		Mat Fusi_result_higest_level;
		//Follow the top layer of the Gaussian pyramid of Img1 and Img2 according to the mask
		Fusi_result_higest_level = Fusi_Gau_Pyr.back().mul(mask_Pyr.back()) + ((Fusi_Gau_Pyr2.back()).mul(Scalar(1.0, 1.0, 1.0) - mask_Pyr.back()));


		//Fusi_blend_lp = blend_Laplacian_Pyr(Fusi_lp_Pyr, Fusi_lp_Pyr2, mask_Pyr, Fusi_blend_lp);


		Mat Fusi_output;
		//Fusi_output = blend(Fusi_result_higest_level, Fusi_blend_lp);
		//Fusi_output.convertTo(Fusi_output, CV_8UC3);
		Fusi_result_higest_level.convertTo(Fusi_result_higest_level, CV_8UC3);
		//imshow("Fusi_Fusion", Fusi_result_higest_level);
		//imwrite("Fusioutput1.jpg",Fusi_output); // A JPG FILE IS BEING SAVED
		//pyrDown(clone2, clone2, Size(clone2.cols * 2, clone2.rows * 2));
		//imshow("Pyrdown",clone2);
		*/


		// EXPOS CLAHE
		//float newSigma = 0.2;
		Mat clone4 = image_clahe.clone();
		clone4.convertTo(clone4, CV_8U);
		Mat clone5 = clone4.clone();
		cv::cvtColor(clone5, clone5, COLOR_BGR2GRAY);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				double b = clone4.at<Vec3b>(i, j)[0];
				double g = clone4.at<Vec3b>(i, j)[1];
				double r = clone4.at<Vec3b>(i, j)[2];
				double e = exponentialCalculation(r, 0.25, r)* exponentialCalculation(g, 0.25, g) * exponentialCalculation(b, 0.25, b);
				e *= 255;
				//cout << e << endl;
				clone5.at<uchar>(i, j) = e;

			}
		}
		//cout << "B val: " << AvgB << endl;
		//cout << "G val: " << AvgG << endl;
		//imshow("EXPOS CLAHE(WE2)", clone5);

		//EXPOS WB
		Mat clone6 = result.clone();
		clone6.convertTo(clone6, CV_8U);
		Mat clone7 = clone6.clone();
		cv::cvtColor(clone7, clone7, COLOR_BGR2GRAY);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				double b0 = clone6.at<Vec3b>(i, j)[0];
				double g0 = clone6.at<Vec3b>(i, j)[1];
				double r0 = clone6.at<Vec3b>(i, j)[2];
				double e0 = exponentialCalculation(r0, 0.25, r0)* exponentialCalculation(g0, 0.25, g0) * exponentialCalculation(b0, 0.25, b0);
				e0 *= 255;
				//cout << e << endl;
				clone7.at<uchar>(i, j) = e0;
			}
		}
		//imshow("EXPOS WB(WE1)", clone7);

		//Mat gammaImg;
		//gammaCorrection(result, gammaImg, 2.2);
		//imshow("Gmma", gammaImg);

		/*Mat m = ddd;
		int radius = 3;
		Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(radius * 2 + 1, radius * 2 + 1));
		double sum = cv::sum(cv::sum(kernel))[0]; // use cv::sum twice to get sum of all elements
		kernel.convertTo(kernel, CV_64F, 1 / sum);
		cv::filter2D(m, m, CV_64F, kernel);
		m.convertTo(m, CV_8U);
		imshow("WBcontrast", m);
		*/

		// Reduce noise by blurring with a Gaussian filter ( kernel size = 3 )
		//cv::GaussianBlur(img, dst2, Size(3, 3), 0, 0, BORDER_DEFAULT);
		//cv::cvtColor(dst2, dst2, COLOR_BGR2GRAY); // Convert the image to grayscale
		//cv::Laplacian(dst, dst2, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
		// converting back to CV_8U
		//cv::convertScaleAbs(dst2, laplace_done);
		//imshow("CLAHE_laplace", laplace_done);


		//

		// CLAHE >>> Saliency Detection
		Mat blur0;
		cv::GaussianBlur(image_clahe, blur0, Size(3, 3), 0, 0, BORDER_DEFAULT);//Color Blur
		cv::cvtColor(blur0, blur0, CV_BGR2Lab);
		//imshow("Blur0", blur0);

	cv:Scalar tempVal0 = mean(blur0);

		double Lmean0 = tempVal0.val[0];
		double amean0 = tempVal0.val[1];
		double bmean0 = tempVal0.val[2];

		Mat salientMap0 = Mat::zeros(height, width, CV_32F);                        //Define the saliency map, because it is a double type, so the initialization type here is CV_64F

		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				//Saliency calculation, corresponding to sm = (l-lm).^2 + (a-am).^2 + (b-bm).^2 in the matlab code;
				salientMap0.at<float>(i, j) = ((float)blur0.at<Vec3b>(i, j)[0] - Lmean0)*((float)blur0.at<Vec3b>(i, j)[0] - Lmean0)
					+ ((float)blur0.at<Vec3b>(i, j)[1] - amean0)*((float)blur0.at<Vec3b>(i, j)[1] - amean0)
					+ ((float)blur0.at<Vec3b>(i, j)[2] - bmean0)*((float)blur0.at<Vec3b>(i, j)[2] - bmean0);

			}
		}

		normalize(salientMap0, salientMap0, 0, 1, NORM_MINMAX, CV_32F, Mat());
		//cv::cvtColor(salientMap0, salientMap0, COLOR_Lab2BGR);
		//cv::cvtColor(salientMap0, salientMap0, COLOR_BGR2GRAY);
		//imshow("CLAHE Saliency (WS2)", salientMap0);


		//white balance >>> Laplace
		cv::Mat wb_done;
		cv::cvtColor(result, result, COLOR_BGR2GRAY); // Convert the image to grayscale
		//imshow("Wb Gray", result);
		cv::GaussianBlur(result, result, Size(3, 3), 0, 0, BORDER_DEFAULT);
		//imshow("Wb GrayBlur", result);
		cv::Laplacian(result, result2, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
		normalize(result2, wb_done, 0, 1, NORM_MINMAX, CV_32F, Mat());
		//cv::convertScaleAbs(result2, wb_done);
		//cv::resize(wb_done, wb_done, cv::Size(720, 480));
		//cv::cvtColor(wb_done, wb_done, COLOR_BGR2GRAY);
		//imshow("WB Laplacian(WC1)", wb_done);

		// White Balance >>> Saliency Detection
		Mat blur, lm, am, bm, L, a, b, sm, p1, p2, p3;
		cv::GaussianBlur(clone2, blur, Size(3, 3), 0, 0, BORDER_DEFAULT);
		cv::cvtColor(blur, blur, CV_BGR2Lab);
		Scalar tempVal = mean(blur);

		double Lmean = tempVal.val[0];
		double amean = tempVal.val[1];
		double bmean = tempVal.val[2];

		Mat salientMap = Mat::zeros(height, width, CV_32F);                        //Define the saliency map, because it is a double type, so the initialization type here is CV_64F
		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++){
				//Saliency calculation, corresponding to sm = (l-lm).^2 + (a-am).^2 + (b-bm).^2 in the matlab code;
				salientMap.at<float>(i, j) = ((float)blur.at<Vec3b>(i, j)[0] - Lmean)*((float)blur.at<Vec3b>(i, j)[0] - Lmean)
					+ ((float)blur.at<Vec3b>(i, j)[1] - amean)*((float)blur.at<Vec3b>(i, j)[1] - amean)
					+ ((float)blur.at<Vec3b>(i, j)[2] - bmean)*((float)blur.at<Vec3b>(i, j)[2] - bmean);

			}
		}

		normalize(salientMap, salientMap, 0, 1, NORM_MINMAX, CV_32F, Mat());
		//cv::cvtColor(salientMap, salientMap, COLOR_BGR2GRAY);
		//imshow("WB Saliency(WS1)", salientMap);



		/*
			//NORMALIZED WEIGHT function
		*/
		
		Mat WC1 = deezLocal; //WC1
		Mat WL1 = wb_done; //WL1
		Mat WS1 = salientMap; //WS1
		Mat WE1 = clone5; //WE1
		Mat WS1NEW = clone5; //WE1
		Mat WL1NEW = clone5;

		Mat WC2 = deezLocal0; //WC2
		Mat WL2 = ddd; //WL2
		Mat WS2 = salientMap0; //WS2
		Mat WE2 = clone7; //WE2
		Mat WS2NEW = clone7;
		Mat WL2NEW = clone7;

		//double wc1, double ws1, double we1, double wl1
		Mat WBNormalized = clone7.clone();
		Mat ClaheNormalized = clone5.clone();

		Mat WBNormalizedAnother = WBNormalized.clone();
		Mat ClaheNormalizedAnother = ClaheNormalized.clone();

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				//WS1NEW.at<uchar>(i, j) = WS1.at<double>(i, j);

				WL1NEW.at<uchar>(i, j) = WL1.at<float>(i, j) * 255;
				WL2NEW.at<uchar>(i, j) = WL2.at<float>(i, j) * 255;
				WS1NEW.at<uchar>(i, j) = WS1.at<float>(i, j) * 255;
				WS2NEW.at<uchar>(i, j) = WS2.at<float>(i, j) * 255;
				double WBNValue = normalizedWeightWB(WC1.at<uchar>(i, j), WS1NEW.at<uchar>(i, j), WE1.at<uchar>(i, j), WL1NEW.at<uchar>(i, j), WC2.at<uchar>(i, j), WS2NEW.at<uchar>(i, j), WE2.at<uchar>(i, j), WL2NEW.at<uchar>(i, j));
				double ClaheNValue = normalizedWeightClahe(WC1.at<uchar>(i, j), WS1NEW.at<uchar>(i, j), WE1.at<uchar>(i, j), WL1NEW.at<uchar>(i, j), WC2.at<uchar>(i, j), WS2NEW.at<uchar>(i, j), WE2.at<uchar>(i, j), WL2NEW.at<uchar>(i, j));
				WBNormalized.at<uchar>(i, j) = WBNValue;
				ClaheNormalized.at<uchar>(i, j) = ClaheNValue;
				WBNormalizedAnother.at<uchar>(i, j) = WBNValue;
				ClaheNormalizedAnother.at<uchar>(i, j) = ClaheNValue;

			}
		}

		Mat ClaheClone = ClaheNormalized.clone();
		Mat WBClone = WBNormalized.clone();

		//img.convertTo(ClaheClone, CV_32F);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				ClaheClone.at<uchar>(i, j) = ClaheClone.at<uchar>(i, j) / 255;
				WBClone.at<uchar>(i, j) = WBClone.at<uchar>(i, j) / 255;
			}
		}
		img.convertTo(ClaheClone, CV_32F);
		//imshow("ClaheNormalized(W2)", ClaheClone);
		//imshow("WBNormalized(W1)", WBClone);
		//imshow("ClaheNormalized(W2)", ClaheNormalized);
		//imshow("WBNormalized(W1)", WBNormalized);

		
		Mat Clahepyr = image_clahe.clone();
		Mat WBPyr = result.clone();
		Mat WBclone = WBNormalized.clone();
		Mat CLAHEclone = ClaheNormalized.clone();

		/////////////////
		vector<Mat> Gau_PyrWB, lp_PyrWB, Gau_PyrCLAHE, lp_PyrCLAHE;
		vector<Mat> newMaskGaussianPyramid;

		CLAHEclone.convertTo(CLAHEclone, CV_32F);                  //Convert to CV_32F, which is used to match the type of mask (in addition, CV_32F has high precision and is beneficial to calculation)
		WBclone.convertTo(WBclone, CV_32F);

		Gau_PyrWB = bulid_Gaussian_Pyr(CLAHEclone, Gau_PyrWB, 4);       //Calculate the Gaussian pyramid of two pictures
		Gau_PyrCLAHE = bulid_Gaussian_Pyr(WBclone, Gau_PyrCLAHE, 4);

		lp_PyrWB = bulid_Laplacian_Pyr(Gau_PyrWB, lp_PyrWB, 4);     //Calculate the Laplacian pyramid of the two images
		lp_PyrCLAHE = bulid_Laplacian_Pyr(Gau_PyrCLAHE, lp_PyrCLAHE, 4);


		Mat maskGausian = Mat::zeros(height, width, CV_32FC1);           //Construction mask mask, CV_32FC1 type, the same size as Img1
		//maskGausian(Range::all(), Range(0, maskGausian.cols)) = 0.5;      //All lines of the mask, then the left half is 1, and the right half is 0 (meaning half and half fusion)

		//cvtColor(maskGausian, maskGausian, CV_GRAY2BGR);                        //Because the mask is single channel at this time, Img is 3channel, so cvtColor is also needed

		vector<Mat> mask_PyrGausian, blend_Gausian, Fusi_blend_Gausian;

		Mat result_higest_levelGausian;                                  //The starting point of image fusion

		mask_PyrGausian = bulid_Gaussian_Pyr(maskGausian, mask_PyrGausian, 4);         //The Gaussian pyramid of the mask is also level+1

		result_higest_levelGausian = Gau_PyrWB.back().mul(mask_PyrGausian.back()) + ((Gau_PyrCLAHE.back()).mul(Scalar(1.0, 1.0, 1.0) - mask_PyrGausian.back()));

		blend_Gausian = blend_Laplacian_Pyr(lp_PyrWB, lp_PyrCLAHE, mask_PyrGausian, blend_Gausian);

		Mat GausianFusion;
		GausianFusion = blend(result_higest_levelGausian, blend_Gausian);
		GausianFusion.convertTo(GausianFusion, CV_8U);


		//addWeighted(WBNormalized, 0.5, ClaheNormalized, 0.5, 0.0, GausianFusion);
		cvtColor(GausianFusion, GausianFusion, COLOR_GRAY2BGR);
		//imshow("Weight Pyr", GausianFusion);

		Mat Final;
		addWeighted(GausianFusion, 0.3, output, 0.8, 0.0, Final);
		//imshow("Final Output", Final);
		String folder = "./Dataset/Output21/";
		String file = folder + to_string(u) + ".jpg";
		imwrite(file, Final);
		waitKey(1);

	}
	return 0;

}