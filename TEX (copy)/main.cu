#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <helper_gl.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>
#include <GL/freeglut.h>
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GL/glx.h>

#define uint unsigned int
#define uchar unsigned char
using namespace std;
using namespace cv;
#define block_size_x 2
#define block_size_y 128

#define rows 992   //948*1500
#define cols 1420
#define disp_max 160

int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__global__ void stereo_kernel(uint (*a)[cols],uint (*b)[cols],uchar (*disp)[cols])
{
	const uint x=(blockIdx.x*blockDim.x)+threadIdx.x;
	const uint y=(blockIdx.y*blockDim.y)+threadIdx.y;
	if(y<disp_max)
	{
		disp[x][y]=0;
	}
	else
	{
		disp[x][y]=0;
		uint cost=abs((float)(a[x][y]-b[x][y]));
		uint cost_now;
		for(int d=1;d<disp_max+1;d++)
		{
			cost_now=abs((float)(a[x][y]-b[x][y-d]));
			if(cost>cost_now)
			{
				disp[x][y]=d;
				cost=cost_now;
			}
		}
		//disp[x][y]*=15;
	}
}

int main()
{
	dim3 threads(block_size_x,block_size_y);
	dim3 blocks(iDivUp(rows,block_size_x),iDivUp(cols,block_size_y));

	uint (*cpu_p1)[cols];
	uint (*cpu_p2)[cols];
	uint (*gpu_p1)[cols];
	uint (*gpu_p2)[cols];
	uchar (*gpu_p3)[cols];

	//uint (*cpu_p1_1)[cols];
	//uint (*cpu_p2_1)[cols];
	//uint (*gpu_p1_1)[cols];
	//uint (*gpu_p2_1)[cols];
	//uchar (*gpu_p3_1)[cols];

	Mat im1,im2,im3;
	im3.create(rows,cols,CV_8UC1);
	im1=imread("im0.png");
	im2=imread("im1.png");

	cout<<1<<endl;
	cudaHostAlloc( (void**)&cpu_p1,rows*cols*sizeof(uint),cudaHostAllocDefault);
	cudaHostAlloc( (void**)&cpu_p2,rows*cols*sizeof(uint),cudaHostAllocDefault);
	//cudaHostAlloc((void**)&cpu_p1_1,rows*cols*sizeof(uint),cudaHostAllocDefault);
	//cudaHostAlloc((void**)&cpu_p2_1,rows*cols*sizeof(uint),cudaHostAllocDefault);



	for(int x=0;x<rows;x++)
	{
		for(int y=0;y<cols;y++)
		{
			cpu_p1[x][y]=im1.at<Vec3b>(x,y)[0]+(im1.at<Vec3b>(x,y)[1]<<8)+(im1.at<Vec3b>(x,y)[2]<<16);
			cpu_p2[x][y]=im2.at<Vec3b>(x,y)[0]+(im2.at<Vec3b>(x,y)[1]<<8)+(im2.at<Vec3b>(x,y)[2]<<16);
			//cpu_p1_1[x][y]=im1.at<Vec3b>(x+rows,y)[0]+(im1.at<Vec3b>(x+rows,y)[1]<<8)+(im1.at<Vec3b>(x+rows,y)[2]<<16);
			//cpu_p2_1[x][y]=im2.at<Vec3b>(x+rows,y)[0]+(im2.at<Vec3b>(x+rows,y)[1]<<8)+(im2.at<Vec3b>(x+rows,y)[2]<<16);
		}
	}



	cudaMalloc((void **)&gpu_p1,rows*cols*sizeof(uint));
	cudaMalloc((void **)&gpu_p2,rows*cols*sizeof(uint));
	cudaMalloc((void **)&gpu_p3,rows*cols*sizeof(uchar));

	//cudaMalloc((void **)&gpu_p1_1,rows*cols*sizeof(uint));
	//cudaMalloc((void **)&gpu_p2_1,rows*cols*sizeof(uint));
	//cudaMalloc((void **)&gpu_p3_1,rows*cols*sizeof(uchar));

	cudaMemcpyAsync(gpu_p1,cpu_p1,rows*cols*sizeof(uint),cudaMemcpyHostToDevice);
	cudaMemcpyAsync(gpu_p2,cpu_p2,rows*cols*sizeof(uint),cudaMemcpyHostToDevice);
	//cudaMemcpyAsync(gpu_p1_1,cpu_p1_1,rows*cols*sizeof(uint),cudaMemcpyHostToDevice);
	//cudaMemcpyAsync(gpu_p2_1,cpu_p2_1,rows*cols*sizeof(uint),cudaMemcpyHostToDevice);

  //cudaMemcpy(gpu_p1,cpu_p1,rows*cols*sizeof(uint),cudaMemcpyHostToDevice);
	//cudaMemcpy(gpu_p2,cpu_p2,rows*cols*sizeof(uint),cudaMemcpyHostToDevice);
	//cudaMemcpy(gpu_p1_1,cpu_p1_1,rows*cols*sizeof(uint),cudaMemcpyHostToDevice);
	//cudaMemcpy(gpu_p2_1,cpu_p2_1,rows*cols*sizeof(uint),cudaMemcpyHostToDevice);


	stereo_kernel<<<blocks,threads>>>(gpu_p1,gpu_p2,gpu_p3);
	//stereo_kernel<<<blocks,threads>>>(gpu_p1_1,gpu_p2_1,gpu_p3_1);

	cudaMemcpy(im3.data,gpu_p3,rows*cols*sizeof(uchar),cudaMemcpyDeviceToHost);
	//cudaMemcpy(im3.data+(rows*cols),gpu_p3_1,rows*cols*sizeof(uchar),cudaMemcpyDeviceToHost);

	imshow("   ",im3);
	waitKey(0);

	return 0;
}
