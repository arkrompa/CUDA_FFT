
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdlib>
#include <cuComplex.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <cmath>
#include <windows.h>
#include <conio.h>
#include<string>

using namespace cv;
using namespace std;
#define TILE_SIZE 16
#define TILE_MASK 3


__global__ void mulComplexWithTimer(cufftComplex *d_rgbSpectre, cufftComplex *d_kernelSpectre, cufftComplex *od_Spectre, const int paddedImageSize, unsigned long long *time)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned long long start = clock();

	if (i < paddedImageSize)
	{
		cufftComplex rgbSpectre = d_rgbSpectre[i];
		cufftComplex kernelSpectre = d_kernelSpectre[i];

		od_Spectre[i] = { rgbSpectre.x * kernelSpectre.x - rgbSpectre.y * kernelSpectre.y, rgbSpectre.x * kernelSpectre.y + rgbSpectre.y * kernelSpectre.x };

	}
	unsigned long long stop = clock();
	time[i] = stop - start;

}

__global__ void mulComplex(cufftComplex *d_rgbSpectre, cufftComplex *d_kernelSpectre, const int paddedImageSize)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < paddedImageSize)
	{
		cufftComplex rgbSpectre = d_rgbSpectre[i];
		cufftComplex kernelSpectre = d_kernelSpectre[i];
		d_rgbSpectre[i] = { rgbSpectre.x * kernelSpectre.x - rgbSpectre.y * kernelSpectre.y, rgbSpectre.x * kernelSpectre.y + rgbSpectre.y * kernelSpectre.x };
	}

}

__global__ void sharedConvolution(char *inputImageKernel, char *outputImagekernel, int imageWidth, int imageHeight, char *mask)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ char shInputBGR[(TILE_SIZE + 2)][(TILE_SIZE + 2)];
	__shared__ char shMask[TILE_MASK*TILE_MASK];

	if (threadIdx.x <3 && threadIdx.y < 3)
		shMask[threadIdx.x*TILE_MASK + threadIdx.y] = mask[threadIdx.x*TILE_MASK + threadIdx.y];


	bool x_lmargin = (threadIdx.x == 0);
	bool x_rmargin = (threadIdx.x == TILE_SIZE - 1);
	bool y_umargin = (threadIdx.y == 0);
	bool y_bmargin = (threadIdx.y == TILE_SIZE - 1);


	if (x_lmargin)
		shInputBGR[threadIdx.x][threadIdx.y + 1] = 0;
	else if (x_rmargin)
		shInputBGR[threadIdx.x + 2][threadIdx.y + 1] = 0;
	if (y_umargin) {
		shInputBGR[threadIdx.x + 1][threadIdx.y] = 0;
		if (x_lmargin)
			shInputBGR[threadIdx.x][threadIdx.y] = 0;
		else if (x_rmargin)
			shInputBGR[threadIdx.x + 2][threadIdx.y] = 0;
	}
	else if (y_bmargin) {
		shInputBGR[threadIdx.x + 1][threadIdx.y + 2] = 0;
		if (x_rmargin)
			shInputBGR[threadIdx.x + 2][threadIdx.y + 2] = 0;
		else if (x_lmargin)
			shInputBGR[threadIdx.x][threadIdx.y + 2] = 0;
	}

	shInputBGR[threadIdx.x + 1][threadIdx.y + 1] = inputImageKernel[row*imageWidth + col];
	if (x_lmargin && (col > 0))
		shInputBGR[threadIdx.x][threadIdx.y + 1] = inputImageKernel[row*imageWidth + (col - 1)];
	else if (x_rmargin && (col < imageWidth - 1))
		shInputBGR[threadIdx.x + 2][threadIdx.y + 1] = inputImageKernel[row*imageWidth + (col + 1)];
	if (y_umargin && (row > 0)) {
		shInputBGR[threadIdx.x + 1][threadIdx.y] = inputImageKernel[(row - 1)*imageWidth + col];
		if (x_lmargin)
			shInputBGR[threadIdx.x][threadIdx.y] = inputImageKernel[(row - 1)*imageWidth + (col - 1)];
		else if (x_rmargin)
			shInputBGR[threadIdx.x + 2][threadIdx.y] = inputImageKernel[(row - 1)*imageWidth + (col + 1)];
	}
	else if (y_bmargin && (row < imageHeight - 1)) {
		shInputBGR[threadIdx.x + 1][threadIdx.y + 2] = inputImageKernel[(row + 1)*imageWidth + col];
		if (x_rmargin)
			shInputBGR[threadIdx.x + 2][threadIdx.y + 2] = inputImageKernel[(row + 1)*imageWidth + (col + 1)];
		else if (x_lmargin)
			shInputBGR[threadIdx.x][threadIdx.y + 2] = inputImageKernel[(row + 1)*imageWidth + (col - 1)];
	}

	__syncthreads();


	float result = 0;

	for (int i = 0; i < TILE_MASK; i++)
		for (int j = 0; j<TILE_MASK; j++)
			result += shInputBGR[threadIdx.x + i][threadIdx.y + j] * shMask[i*TILE_MASK + j];

	outputImagekernel[row*imageWidth + col] = (char)(result);



}
void zeroPad(float *inputData, int size)
{
	for (int i = 0; i < size; i++)
		inputData[i] = 0;
}

void initKernel(float *inputVector, int width)
{
	inputVector[0] = 0;
	inputVector[1] = -1;
	inputVector[2] = 0;

	inputVector[width] = -1;
	inputVector[width + 1] = 4;
	inputVector[width + 2] = -1;

	inputVector[2 * width] = 0;
	inputVector[2 * width + 1] = -1;
	inputVector[2 * width + 2] = 0;

}
void initMask(char *inputVector)
{
	inputVector[0] = 0;
	inputVector[1] = -1;
	inputVector[2] = 0;

	inputVector[3] = -1;
	inputVector[4] = 4;
	inputVector[5] = -1;

	inputVector[6] = 0;
	inputVector[7] = -1;
	inputVector[8] = 0;

}

double kernelTimer(unsigned long long *h_time, int threadsPerBlock, int blockNum)
{
	unsigned long long bufMax = 0;
	for (int i = 0; i < threadsPerBlock*blockNum; i++)
	{
		bufMax += h_time[i];
	}
	double czas = bufMax;
	czas = czas / (threadsPerBlock*blockNum);
	cout << "Average time inside mulKernel per color: " << czas << "  clock ticks\n";

	return czas;
}
typedef struct processingParams
{
	int rows;
	int cols;
	int paddedImageWidth;
	int paddedImageHeight;
	int paddedImageSize;
	int hermitianMatrixSize;
	int memComplex_size_paddedImage;
	int memReal_size_paddedImage;

	int threadsPerBlock;
	int blockNum;


};
typedef struct timersSet
{
	double bgrfromImageTime;
	double bgrtoImageTime;
	double paddingTime;
	double preparingMemoryTime;
	double planMakingTime;
	double kernelFFTtime;
	double preparingParamsTime;
	double fileReadingTime;
	cudaEvent_t startTime;
	cudaEvent_t stopTime;
	float GPUtimeMemToGPU;
	float GPUtimeMemToCPU;
	float GPUtimeR2C;
	float GPUtimeMul;
	float GPUtimeC2R;
	float GPUtimeConv;

};


void copyBGRtoImage(Mat *image, float* bgrValues, int k, processingParams processParams)
{
	for (int i = 1; i < processParams.paddedImageHeight - 1; i++)
	{
		for (int j = 1; j < processParams.paddedImageWidth - 1; j++)
		{
			image->at<Vec3b>(i - 1, j - 1)[k] = abs(bgrValues[i*processParams.paddedImageWidth + j] / (processParams.paddedImageSize));
		}
	}

}
void copyBGRfromImage(float* bgrValues, Mat *image, int k, processingParams processParams)
{
	for (int i = 0; i < processParams.rows; i++)
	{
		for (int j = 0; j < processParams.cols; j++)
		{
			bgrValues[i*processParams.paddedImageWidth + j] = image->at<cv::Vec3b>(i, j)[k];
		}
	}

}


void copyBGRtoImageConv(Mat *image, char *bgrValues, int k, processingParams processParams)
{
	for (int i = 0; i < processParams.paddedImageHeight; i++)
	{
		for (int j = 0; j < processParams.paddedImageWidth; j++)
		{
			image->at<Vec3b>(i, j)[k] = abs(((int)bgrValues[i*processParams.paddedImageWidth + j]));
		}
	}

}

void copyBGRfromImageConv(char *bgrValues, Mat *image, int k, processingParams processParams)
{
	for (int i = 0; i < processParams.rows; i++)
	{
		for (int j = 0; j < processParams.cols; j++)
		{
			bgrValues[i*processParams.paddedImageWidth + j] = (unsigned char)image->at<cv::Vec3b>(i, j)[k];
		}
	}

}




void processData(float* bgrValues, cufftReal* d_bgrValues, cufftComplex *d_bgrValuesComplex, cufftComplex *d_kernelDataComplex, processingParams processParams, cufftHandle *plan, cufftHandle *planInv, timersSet *timers)
{
	cout << "Copying memory to GPU";
	cudaEventRecord(timers->startTime, 0);
	cudaMemcpy(d_bgrValues, bgrValues, processParams.memReal_size_paddedImage, cudaMemcpyHostToDevice);
	cudaEventRecord(timers->stopTime, 0);
	cudaEventSynchronize(timers->startTime);
	cudaEventSynchronize(timers->stopTime);
	cudaEventElapsedTime(&timers->GPUtimeMemToGPU, timers->startTime, timers->stopTime);
	cout << "\t\t" << timers->GPUtimeMemToGPU << " ms\n";

	cout << "Executing R2C fft";
	cudaEventRecord(timers->startTime, 0);
	cufftExecR2C(*plan, d_bgrValues, d_bgrValuesComplex);
	cudaDeviceSynchronize();
	cudaEventRecord(timers->stopTime, 0);
	cudaEventSynchronize(timers->startTime);
	cudaEventSynchronize(timers->stopTime);
	cudaEventElapsedTime(&timers->GPUtimeR2C, timers->startTime, timers->stopTime);
	cout << "\t\t" << timers->GPUtimeR2C << " ms\n";

	cout << "Mul of complex values on GPU";
	cudaEventRecord(timers->startTime, 0);
	mulComplex << <processParams.blockNum, processParams.threadsPerBlock >> >(d_bgrValuesComplex, d_kernelDataComplex, processParams.hermitianMatrixSize);
	cudaDeviceSynchronize();
	cudaEventRecord(timers->stopTime, 0);
	cudaEventSynchronize(timers->startTime);
	cudaEventSynchronize(timers->stopTime);
	cudaEventElapsedTime(&timers->GPUtimeMul, timers->startTime, timers->stopTime);
	cout << "\t" << timers->GPUtimeMul << " ms\n";

	cout << "Executing C2R fft";
	cudaEventRecord(timers->startTime, 0);
	cufftExecC2R(*planInv, d_bgrValuesComplex, d_bgrValues);
	cudaDeviceSynchronize();
	cudaEventRecord(timers->stopTime, 0);
	cudaEventSynchronize(timers->startTime);
	cudaEventSynchronize(timers->stopTime);
	cudaEventElapsedTime(&timers->GPUtimeC2R, timers->startTime, timers->stopTime);
	cout << "\t\t" << timers->GPUtimeC2R << " ms\n";

	cout << "Copying memory to CPU";
	cudaEventRecord(timers->startTime, 0);
	cudaMemcpy(bgrValues, d_bgrValues, processParams.memReal_size_paddedImage, cudaMemcpyDeviceToHost);
	cudaEventRecord(timers->stopTime, 0);
	cudaEventSynchronize(timers->startTime);
	cudaEventSynchronize(timers->stopTime);
	cudaEventElapsedTime(&timers->GPUtimeMemToCPU, timers->startTime, timers->stopTime);
	cout << "\t\t" << timers->GPUtimeMemToCPU << " ms\n";

}

void processDataConv(char *d_mask, char *d_inputBGR, char *d_outputBGR, char *inputBGR, char *outputBGR, timersSet *timers, processingParams processParamsConv, int maskDim, int maskSize)
{
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid((int)ceil((float)processParamsConv.paddedImageWidth / (float)TILE_SIZE),
		(int)ceil((float)processParamsConv.paddedImageHeight / (float)TILE_SIZE));

	cout << "Copying memory to GPU";
	cudaEventRecord(timers->startTime, 0);
	cudaMemcpy(d_inputBGR, inputBGR, processParamsConv.memReal_size_paddedImage, cudaMemcpyHostToDevice);
	cudaEventRecord(timers->stopTime, 0);
	cudaEventSynchronize(timers->startTime);
	cudaEventSynchronize(timers->stopTime);
	cudaEventRecord(timers->startTime, 0);
	cudaEventElapsedTime(&timers->GPUtimeMemToGPU, timers->startTime, timers->stopTime);
	cout << "\t\t" << timers->GPUtimeMemToGPU << " ms\n";

	cout << "Applying convolution on GPU";
	cudaEventRecord(timers->startTime, 0);
	sharedConvolution << <dimGrid, dimBlock >> >(d_inputBGR, d_outputBGR, processParamsConv.paddedImageWidth, processParamsConv.paddedImageHeight, d_mask);
	cudaDeviceSynchronize();
	cudaEventRecord(timers->stopTime, 0);
	cudaEventSynchronize(timers->startTime);
	cudaEventSynchronize(timers->stopTime);
	cudaEventElapsedTime(&timers->GPUtimeConv, timers->startTime, timers->stopTime);
	cout << "\t" << timers->GPUtimeConv << " ms\n";

	cout << "Copying memory to CPU";
	cudaEventRecord(timers->startTime, 0);
	cudaMemcpy(outputBGR, d_outputBGR, processParamsConv.memReal_size_paddedImage, cudaMemcpyDeviceToHost);
	cudaEventRecord(timers->stopTime, 0);
	cudaEventSynchronize(timers->startTime);
	cudaEventSynchronize(timers->stopTime);
	cudaEventElapsedTime(&timers->GPUtimeMemToCPU, timers->startTime, timers->stopTime);
	cout << "\t\t" << timers->GPUtimeMemToCPU << " ms\n";


}

void processKernel(float *kernelData, cufftReal *d_kernelData, cufftComplex *d_kernelDataComplex, processingParams processParams, cufftHandle *plan)
{
	cudaMemcpy(d_kernelData, kernelData, processParams.memReal_size_paddedImage, cudaMemcpyHostToDevice);
	cufftExecR2C(*plan, d_kernelData, d_kernelDataComplex);
	cudaMemcpy(d_kernelData, kernelData, processParams.memReal_size_paddedImage, cudaMemcpyHostToDevice);

}
double totalTimeFFT(timersSet *timers)
{
	double Time = 0;
	Time += timers->bgrfromImageTime;
	Time += timers->bgrtoImageTime;
	Time += timers->paddingTime;
	Time += (double)timers->GPUtimeMemToGPU;
	Time += (double)timers->GPUtimeR2C;
	Time += (double)timers->GPUtimeC2R;
	Time += (double)timers->GPUtimeMemToCPU;

	return Time;
}
double totalTimeConv(timersSet *timers)
{
	double Time = 0;
	Time += timers->bgrfromImageTime;
	Time += timers->bgrtoImageTime;
	Time += (double)timers->GPUtimeMemToGPU;
	Time += (double)timers->GPUtimeConv;
	Time += (double)timers->GPUtimeMemToCPU;

	return Time;
}

int main()
{
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

	timersSet *timers = new timersSet();
	unsigned long long start, stop;
	cudaEventCreate(&timers->startTime);
	cudaEventCreate(&timers->stopTime);
	string filename;
	char command = '0';

	do {
		SetConsoleTextAttribute(hConsole, 7);
		cout << "Wybierz obraz do filtracji: \n";
		cout << "1.Moon\n";
		cout << "2.Earth\n";
		cout << "3.Lego\n";
		cout << "4.Lena\n";
		cout << "5.Hubble8k\n";
		cout << "6.Exit\n";
		command = getche();

		switch (command)
		{
		case '1':
			filename = "Moon.bmp";
			break;
		case '2':
			filename = "Earth.bmp";
			break;
		case '3':
			filename = "Lego.bmp";
			break;
		case '4':
			filename = "Lena.bmp";
			break;
		case '5':
			filename = "Hubble8k.bmp";
			break;
		case '6':
			exit(0);
			break;
		default:
			filename = "failed";
			break;
		}

		system("cls");
		SetConsoleTextAttribute(hConsole, 8);
		start = clock();
		Mat image;
		cout << "Reading file";
		image = imread(filename, IMREAD_COLOR); // Read the file

		if (!image.data) // Check for invalid input
		{
			cout << "Could not open or find the image\n" << std::endl;
			system("PAUSE");
			return -1;
		}

		stop = clock();
		timers->fileReadingTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t\t\t" << timers->fileReadingTime << " ms\n";

		cout << "Preparing params";
		start = clock();
		processingParams processParams;
		processParams.cols = image.cols;
		processParams.rows = image.rows;
		processParams.paddedImageWidth = processParams.cols + 2;
		processParams.paddedImageHeight = processParams.rows + 2;
		processParams.paddedImageSize = processParams.paddedImageWidth*processParams.paddedImageHeight;
		processParams.hermitianMatrixSize = processParams.paddedImageHeight*((int)(processParams.paddedImageWidth / 2) + 1);
		processParams.memComplex_size_paddedImage = sizeof(cufftComplex)*processParams.hermitianMatrixSize;
		processParams.memReal_size_paddedImage = sizeof(float)*processParams.paddedImageSize;
		processParams.threadsPerBlock = 1024;
		processParams.blockNum = processParams.hermitianMatrixSize / processParams.threadsPerBlock + 1;

		cufftComplex *d_bgrValuesComplex;
		cufftReal *d_bgrValues;
		cufftReal *d_kernelData;
		cufftComplex *d_kernelDataComplex;

		cudaError_t cudaStatus;
		cufftHandle plan;
		cufftHandle planInv;
		stop = clock();
		timers->preparingParamsTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t\t" << timers->preparingParamsTime << " ms\n";

		cout << "First time GPU run and malloc";
		cudaStatus = cudaMalloc((void **)&d_kernelData, processParams.memReal_size_paddedImage);
		if (cudaStatus != cudaSuccess) {
			cout << "Cannot alloc memory on GPU";
		}
		cudaStatus = cudaMalloc((void **)&d_kernelDataComplex, processParams.memComplex_size_paddedImage);
		if (cudaStatus != cudaSuccess) {
			cout << "Cannot alloc memory on GPU";
		}
		stop = clock();
		timers->preparingMemoryTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t" << timers->preparingMemoryTime << " ms\n";

		cout << "Making cufftPlan2D";
		start = clock();
		cufftPlan2d(&plan, processParams.paddedImageHeight, processParams.paddedImageWidth, CUFFT_R2C);
		cufftPlan2d(&planInv, processParams.paddedImageHeight, processParams.paddedImageWidth, CUFFT_C2R);
		stop = clock();
		timers->planMakingTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t\t" << timers->planMakingTime << " ms\n";


		cout << "KERNEL fft and freeing memory";
		start = clock();
		float *kernelData = new float[processParams.paddedImageSize];
		zeroPad(kernelData, processParams.paddedImageSize);
		initKernel(kernelData, processParams.paddedImageWidth);
		processKernel(kernelData, d_kernelData, d_kernelDataComplex, processParams, &plan);
		cudaFree(d_kernelData);
		free(kernelData);
		stop = clock();
		timers->kernelFFTtime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t" << timers->kernelFFTtime << " ms\n";

		cout << "Preparing memory for IMAGE fft";
		start = clock();
		cudaStatus = cudaMalloc((void **)&d_bgrValues, processParams.memReal_size_paddedImage);
		if (cudaStatus != cudaSuccess) {
			cout << "Cannot alloc memory on GPU";
		}
		cudaStatus = cudaMalloc((void **)&d_bgrValuesComplex, processParams.memComplex_size_paddedImage);
		if (cudaStatus != cudaSuccess) {
			cout << "Cannot alloc memory on GPU";
		}
		float *bgrValue = new float[processParams.paddedImageSize];
		stop = clock();
		timers->preparingMemoryTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t" << timers->preparingMemoryTime << " ms\n";


		cout << "\n\n---FFT PROCESSING BLUE COLOR---\n";
		cout << "Padding IMAGE with zeros";
		start = clock();
		zeroPad(bgrValue, processParams.paddedImageSize);
		stop = clock();
		timers->paddingTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t" << timers->paddingTime << " ms\n";

		cout << "Copying BGR from IMAGE to host";
		start = clock();
		copyBGRfromImage(bgrValue, &image, 0, processParams);
		stop = clock();
		timers->bgrfromImageTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t" << timers->bgrfromImageTime << " ms\n";

		processData(bgrValue, d_bgrValues, d_bgrValuesComplex, d_kernelDataComplex, processParams, &plan, &planInv, timers);
		cout << "Copying filtered BGR to IMAGE";
		start = clock();
		copyBGRtoImage(&image, bgrValue, 0, processParams);
		stop = clock();
		timers->bgrtoImageTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t" << timers->bgrtoImageTime << " ms\n";
		cout << "Total time";
		cout << "\t" << totalTimeFFT(timers) << " ms\n";


		cout << "\n\n---FFT PROCESSING GREEN COLOR---\n";
		cout << "Padding IMAGE with zeros";
		start = clock();
		zeroPad(bgrValue, processParams.paddedImageSize);
		stop = clock();
		timers->paddingTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t" << timers->paddingTime << " ms\n";

		cout << "Copying BGR from IMAGE to host";
		start = clock();
		copyBGRfromImage(bgrValue, &image, 1, processParams);
		stop = clock();
		timers->bgrfromImageTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t" << timers->bgrfromImageTime << " ms\n";

		processData(bgrValue, d_bgrValues, d_bgrValuesComplex, d_kernelDataComplex, processParams, &plan, &planInv, timers);
		cout << "Copying filtered BGR to IMAGE";
		start = clock();
		copyBGRtoImage(&image, bgrValue, 1, processParams);
		stop = clock();
		timers->bgrtoImageTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t" << timers->bgrtoImageTime << " ms\n";
		cout << "Total time";
		cout << "\t" << totalTimeFFT(timers) << " ms\n";

		cout << "\n\n---FFT PROCESSING RED COLOR---\n";
		cout << "Padding IMAGE with zeros";
		start = clock();
		zeroPad(bgrValue, processParams.paddedImageSize);
		stop = clock();
		timers->paddingTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t" << timers->paddingTime << " ms\n";

		cout << "Copying BGR from IMAGE to host";
		start = clock();
		copyBGRfromImage(bgrValue, &image, 2, processParams);
		stop = clock();
		timers->bgrfromImageTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t" << timers->bgrfromImageTime << " ms\n";

		processData(bgrValue, d_bgrValues, d_bgrValuesComplex, d_kernelDataComplex, processParams, &plan, &planInv, timers);
		cout << "Copying filtered BGR to IMAGE";
		start = clock();
		copyBGRtoImage(&image, bgrValue, 2, processParams);
		stop = clock();
		timers->bgrtoImageTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t" << timers->bgrtoImageTime << " ms\n";
		cout << "Total time";
		cout << "\t" << totalTimeFFT(timers) << " ms\n";

		imwrite("resultFFT.bmp", image);

		cufftDestroy(plan);
		cufftDestroy(planInv);
		cudaFree(d_bgrValues);
		cudaFree(d_bgrValuesComplex);
		cudaFree(d_kernelDataComplex);
		free(bgrValue);

		Mat imageConv;
		imageConv = imread(filename, IMREAD_COLOR);

		processingParams processParamsConv;
		processParamsConv.cols = image.cols;
		processParamsConv.rows = image.rows;
		processParamsConv.paddedImageWidth = processParamsConv.cols;
		processParamsConv.paddedImageHeight = processParamsConv.rows;
		processParamsConv.paddedImageSize = processParamsConv.paddedImageWidth*processParamsConv.paddedImageHeight;
		processParamsConv.memReal_size_paddedImage = sizeof(char)*processParamsConv.paddedImageSize;

		int maskDim = 3;
		int maskSize = sizeof(char)*maskDim*maskDim;

		char *inputBGR = new char[processParamsConv.paddedImageSize];
		char *outputBGR = new char[processParamsConv.paddedImageSize];
		char *mask = new char[maskDim*maskDim];


		char *d_mask;
		char *d_inputBGR;
		char *d_outputBGR;

		cudaStatus = cudaMalloc((void **)&d_mask, maskSize);
		if (cudaStatus != cudaSuccess) {
			cout << "Cannot alloc memory on GPU";
		}
		cudaStatus = cudaMalloc((void **)&d_inputBGR, processParamsConv.memReal_size_paddedImage);
		if (cudaStatus != cudaSuccess) {
			cout << "Cannot alloc memory on GPU";
		}
		cudaStatus = cudaMalloc((void **)&d_outputBGR, processParamsConv.memReal_size_paddedImage);
		if (cudaStatus != cudaSuccess) {
			cout << "Cannot alloc memory on GPU";
		}

		initMask(mask);
		cudaMemcpy(d_mask, mask, maskSize, cudaMemcpyHostToDevice);

		SetConsoleTextAttribute(hConsole, 2);
		cout << "\n\n---CONV PROCESSING BLUE COLOR---\n";
		cout << "Copying BGR from IMAGE to host";
		start = clock();
		copyBGRfromImageConv(inputBGR, &imageConv, 0, processParamsConv);
		stop = clock();
		timers->bgrfromImageTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t" << timers->bgrfromImageTime << " ms\n";

		processDataConv(d_mask, d_inputBGR, d_outputBGR, inputBGR, outputBGR, timers, processParamsConv, maskDim, maskSize);

		cout << "Copying filtered BGR to IMAGE";
		start = clock();
		copyBGRtoImageConv(&imageConv, outputBGR, 0, processParamsConv);
		stop = clock();
		timers->bgrtoImageTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t" << timers->bgrtoImageTime << " ms\n";

		cout << "Total time";
		cout << "\t" << totalTimeConv(timers) << " ms\n";


		cout << "\n\n---CONV PROCESSING GREEN COLOR---\n";
		cout << "Copying BGR from IMAGE to host";
		start = clock();
		copyBGRfromImageConv(inputBGR, &imageConv, 1, processParamsConv);
		stop = clock();
		timers->bgrfromImageTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t" << timers->bgrfromImageTime << " ms\n";

		processDataConv(d_mask, d_inputBGR, d_outputBGR, inputBGR, outputBGR, timers, processParamsConv, maskDim, maskSize);

		cout << "Copying filtered BGR to IMAGE";
		start = clock();
		copyBGRtoImageConv(&imageConv, outputBGR, 1, processParamsConv);
		stop = clock();
		timers->bgrtoImageTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t" << timers->bgrtoImageTime << " ms\n";

		cout << "Total time";
		cout << "\t" << totalTimeConv(timers) << " ms\n";

		cout << "\n\n---CONV PROCESSING RED COLOR---\n";
		cout << "Copying BGR from IMAGE to host";
		start = clock();
		copyBGRfromImageConv(inputBGR, &imageConv, 2, processParamsConv);
		stop = clock();
		timers->bgrfromImageTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t" << timers->bgrfromImageTime << " ms\n";

		processDataConv(d_mask, d_inputBGR, d_outputBGR, inputBGR, outputBGR, timers, processParamsConv, maskDim, maskSize);

		cout << "Copying filtered BGR to IMAGE";
		start = clock();
		copyBGRtoImageConv(&imageConv, outputBGR, 2, processParamsConv);
		stop = clock();
		timers->bgrtoImageTime = double(stop - start) / ((double)CLOCKS_PER_SEC / 1000);
		cout << "\t" << timers->bgrtoImageTime << " ms\n";

		cout << "Total time";
		cout << "\t" << totalTimeConv(timers) << " ms\n";

		imwrite("resultConv.bmp", imageConv);

		ShellExecute(NULL, "open", filename.c_str(), NULL, NULL, SW_SHOW);
		Sleep(10);
		ShellExecute(NULL, "open", "resultFFT.bmp", NULL, NULL, SW_SHOW);
		Sleep(10);
		ShellExecute(NULL, "open", "resultConv.bmp", NULL, NULL, SW_SHOW);
		Sleep(10);

		cudaFree(d_outputBGR);
		cudaFree(d_inputBGR);
		cudaFree(d_mask);
		free(outputBGR);
		free(inputBGR);
		free(mask);

		system("PAUSE");
		system("cls");
	} while (TRUE);

	return 0;
}

