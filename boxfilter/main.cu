#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define uint unsigned int
#define uchar unsigned char

#define rows 288
#define cols 384

uchar *gpu_data;
uchar *out;

dim3 threads(16,16);
dim3 blocks(18,24);

__global__ void box_kernel_x(uchar *input,uchar *output,int r)
{
	uint x=(blockIdx.x*blockDim.x)+threadIdx.x;
	uint y=(blockIdx.y*blockDim.y)+threadIdx.y;
	uint offset=y+x*blockDim.y*gridDim.y;
	if((x>=r)&&(x<rows-1-r)&&(y>=r)&&(y<cols-1-r))
	{
		int sum=0;
		for(int i=x-r;i<x+r+1;i++)
		{
			sum+=input[i*cols+y];
		}
		output[offset]=sum/(r<<1+1);
	}
	else
		output[offset]=0;
}

__global__ void box_kernel_y(uchar *input,uchar *output,int r)
{
	uint x=(blockIdx.x*blockDim.x)+threadIdx.x;
	uint y=(blockIdx.y*blockDim.y)+threadIdx.y;
	uint offset=y+x*blockDim.y*gridDim.y;
	if((x>=r)&&(x<rows-1-r)&&(y>=r)&&(y<cols-1-r))
	{
		int sum=0;
		for(int j=y-r;j<y+r+1;j++)
		{
			sum+=input[x*cols+j];
		}
		output[offset]=sum/(r<<1+1);
	}
	else
		output[offset]=0;
}

void boxfilter(uchar *input,uchar *output,int r,dim3 block,dim3 thread)
{
	uchar *temp1;
	cudaMalloc((void **)&temp1,sizeof(uchar)*rows*cols);
	box_kernel_x<<<block,thread>>>(input,temp1,r);
	box_kernel_y<<<block,thread>>>(temp1,output,r);
}

int main()
{
	//不使用纹理内存
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	Mat img=imread("im2.ppm",0);
	imshow("原图像",img);
	cudaMalloc((void **)&gpu_data,sizeof(uchar)*rows*cols);
	cudaMalloc((void **)&out,sizeof(uchar)*rows*cols);
	cudaMemcpy(gpu_data,img.data,sizeof(uchar)*rows*cols,cudaMemcpyHostToDevice);
	cudaEventRecord(start,0);

	boxfilter(gpu_data,out,7,blocks,threads);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time,start,stop);
	cudaMemcpy(img.data,out,sizeof(uchar)*rows*cols,cudaMemcpyDeviceToHost);
	imshow("不使用纹理内存盒式滤波后的图像",img);
	cout<<"不使用纹理内存所用时间:"<<time<<"ms"<<endl;
	cudaFree(gpu_data);
	cudaFree(out);

	waitKey(0);
	return 0;
}

