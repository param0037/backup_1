#pragma once

#include "../../basic/basic.h"
#include "../../classes/matrix.h"
#include "../../classes/device_in_place_cls.h"
#include "../../cv/device/cv_basic_proc.cuh"
#include "cv/cv_classes/cv_classes.h"


namespace de
{
	namespace vis
	{
		_DECX_API_ 
		void _GaussianBlur2D(de::vis::dev_Img& src, de::vis::dev_Img& dst, const de::Point2D radius, const de::Point2D_d sigma);


		_DECX_API_ 
		void _GaussianBlur2D_border_C(de::vis::dev_Img& src, de::vis::dev_Img& dst, const de::Point2D radius, const de::Point2D_d sigma, const float border);


		_DECX_API_ 
		void _GaussianBlur2D_border_mirror(de::vis::dev_Img& src, de::vis::dev_Img& dst, const de::Point2D radius, const de::Point2D_d sigma);


		_DECX_API_ 
		void _GaussianBlur3D(de::vis::dev_Img& src, de::vis::dev_Img& dst, const de::Point2D radius, const de::Point2D_d sigma);


		_DECX_API_ 
		void _GaussianBlur3D_border_C(de::vis::dev_Img& src, de::vis::dev_Img& dst, const de::Point2D radius, const de::Point2D_d sigma, const float border);


		_DECX_API_ 
		void _GaussianBlur3D_border_mirror(de::vis::dev_Img& src, de::vis::dev_Img& dst, const de::Point2D radius, const de::Point2D_d sigma);


		_DECX_API_ 
		void _NLM_2D(de::vis::dev_Img& src, de::vis::dev_Img& dst, const de::Point2D search_dim, const de::Point2D neigbor_dim, const float h, int step);


		_DECX_API_ 
		void _NLM_3D(de::vis::dev_Img& src, de::vis::dev_Img& dst, const de::Point2D search_dim, const de::Point2D neigbor_dim, const float h, int step);
	}
}



void de::vis::_GaussianBlur2D(de::vis::dev_Img& src, de::vis::dev_Img& dst, const de::Point2D radius, const de::Point2D_d sigma)
{
	if (cuP.is_init != true) {
		printf("CUDA should be initialized first\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	if (src.channel != 1) {
		printf("The input Image array is not a 1D array\n");
		system("pause");
		exit(EXIT_FAILURE);
	}
	__dev_Img* sub_ptr_src = dynamic_cast<__dev_Img*>(&src);
	__dev_Img* sub_ptr_dst = dynamic_cast<__dev_Img*>(&dst);

	const uint Wsrc = src.width;
	const uint Hsrc = src.height;

	// allocate a one dimensional kernel vector
	uchar* dev_src = sub_ptr_src->dev_Mat,
		* dev_dst = sub_ptr_dst->dev_Mat,
		* dev_mid;
	float* kernel_x, * kernel_y;

	int2 ker_lenXY;
	ker_lenXY.x = (radius.x << 1) + 1;
	ker_lenXY.y = (radius.y << 1) + 1;

	// ~.x : width, ~.y : height
	dim3 dstDim(Wsrc - (radius.x << 1), Hsrc - (radius.y << 1));

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&kernel_x), (ker_lenXY.x + ker_lenXY.y) * sizeof(float)));
	kernel_y = kernel_x + ker_lenXY.x;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_mid), dstDim.x * Hsrc * sizeof(uchar)));

	cudaStream_t S_K;
	
	checkCudaErrors(cudaStreamCreateWithFlags(&S_K, cudaStreamNonBlocking));

	// generating the Gaussian kernel vector
	Gaussian_vec_Gen << <1, ker_lenXY.x, 0, S_K >> > (kernel_x,
		sigma.x,
		radius.x,
		ker_lenXY.x);

	Gaussian_vec_Gen << <1, ker_lenXY.y, 0, S_K >> > (kernel_y,
		sigma.y,
		radius.y,
		ker_lenXY.y);

	// configure the launch parameters
	dim3 grid_1(_cu_ceil(Hsrc, Gauss_threads_x), _cu_ceil(dstDim.x, Gauss_threads_y));
	dim3 grid_2(_cu_ceil(dstDim.y, Gauss_threads_x), _cu_ceil(dstDim.x, Gauss_threads_y));
	dim3 threads(Gauss_threads_x, Gauss_threads_y);

	checkCudaErrors(cudaDeviceSynchronize());

	// launch the kernel
	Gaussian_blur_hor << <grid_1, threads, 0, S_K >> > (dev_src,
		kernel_x,
		dev_mid,
		ker_lenXY.x,
		radius.x,
		dstDim,
		dim3(Wsrc, Hsrc));

	Gaussian_blur_ver << <grid_2, threads, 0, S_K >> > (dev_mid,
		kernel_y,
		dev_dst,
		ker_lenXY.y,
		radius.y,
		dstDim,
		dim3(Wsrc, Hsrc));

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(kernel_x));
	checkCudaErrors(cudaFree(dev_mid));
	checkCudaErrors(cudaStreamDestroy(S_K));
}



void de::vis::_GaussianBlur2D_border_C(de::vis::dev_Img& src, de::vis::dev_Img& dst, const de::Point2D radius, const de::Point2D_d sigma, const float border)
{
	if (cuP.is_init != true) {
		printf("CUDA should be initialized first\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	if (src.channel != 1) {
		printf("The input Image array is not a 1D array\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	__dev_Img* sub_ptr_src = dynamic_cast<__dev_Img*>(&src);
	__dev_Img* sub_ptr_dst = dynamic_cast<__dev_Img*>(&dst);

	const uint Wsrc = sub_ptr_src->width;
	const uint Hsrc = sub_ptr_src->height;

	// allocate a one dimensional kernel vector
	uchar* dev_src = sub_ptr_src->dev_Mat,
		* dev_dst = sub_ptr_dst->dev_Mat,
		* dev_mid;
	float* kernel_x, *kernel_y;

	int2 ker_lenXY;
	ker_lenXY.x = (radius.x << 1) + 1;
	ker_lenXY.y = (radius.y << 1) + 1;

	// ~.x : width, ~.y : height
	dim3 dstDim(Wsrc, Hsrc);

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&kernel_x), (ker_lenXY.x + ker_lenXY.y) * sizeof(float)));
	kernel_y = kernel_x + ker_lenXY.x;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_mid), dstDim.x * Hsrc * sizeof(uchar)));

	cudaStream_t S_K;
	checkCudaErrors(cudaStreamCreateWithFlags(&S_K, cudaStreamNonBlocking));

	// generating the Gaussian kernel vector
	Gaussian_vec_Gen << <1, ker_lenXY.x, 0, S_K >> > (kernel_x,
		sigma.x,
		radius.x,
		ker_lenXY.x);

	Gaussian_vec_Gen << <1, ker_lenXY.y, 0, S_K >> > (kernel_y,
		sigma.y,
		radius.y,
		ker_lenXY.y);

	// configure the launch parameters
	dim3 grid(_cu_ceil(dstDim.y, Gauss_threads_x), _cu_ceil(dstDim.x, Gauss_threads_y));
	dim3 threads(Gauss_threads_x, Gauss_threads_y);

	// launch the kernel
	Gaussian_blur_hor_border << <grid, threads, 0, S_K >> > (dev_src,
		kernel_x,
		dev_mid,
		ker_lenXY.x,
		dstDim,
		radius.x,
		border);

	Gaussian_blur_ver_border << <grid, threads, 0, S_K >> > (dev_mid,
		kernel_y,
		dev_dst,
		ker_lenXY.y,
		dstDim,
		radius.y,
		border);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(kernel_x));
	checkCudaErrors(cudaFree(dev_mid));
	checkCudaErrors(cudaStreamDestroy(S_K));
}



void de::vis::_GaussianBlur2D_border_mirror(de::vis::dev_Img& src, de::vis::dev_Img& dst, const de::Point2D radius, const de::Point2D_d sigma)
{
	if (cuP.is_init != true) {
		printf("CUDA should be initialized first\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	if (src.channel != 1) {
		printf("The input Image array is not a 1D array\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	__dev_Img* sub_ptr_src = dynamic_cast<__dev_Img*>(&src);
	__dev_Img* sub_ptr_dst = dynamic_cast<__dev_Img*>(&dst);

	const uint Wsrc = sub_ptr_src->width;
	const uint Hsrc = sub_ptr_src->height;

	// allocate a one dimensional kernel vector
	uchar* dev_src = sub_ptr_src->dev_Mat,
		* dev_dst = sub_ptr_dst->dev_Mat,
		* dev_mid;
	float* kernel_x, * kernel_y;

	int2 ker_lenXY;
	ker_lenXY.x = (radius.x << 1) + 1;
	ker_lenXY.y = (radius.y << 1) + 1;

	// ~.x : width, ~.y : height
	dim3 dstDim(Wsrc, Hsrc);

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&kernel_x), (ker_lenXY.x + ker_lenXY.y) * sizeof(float)));
	kernel_y = kernel_x + ker_lenXY.x;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_mid), dstDim.x * Hsrc * sizeof(uchar)));

	cudaStream_t S_K;
	checkCudaErrors(cudaStreamCreateWithFlags(&S_K, cudaStreamNonBlocking));

	// generating the Gaussian kernel vector
	Gaussian_vec_Gen << <1, ker_lenXY.x, 0, S_K >> > (kernel_x,
		sigma.x,
		radius.x,
		ker_lenXY.x);

	Gaussian_vec_Gen << <1, ker_lenXY.y, 0, S_K >> > (kernel_y,
		sigma.y,
		radius.y,
		ker_lenXY.y);

	// configure the launch parameters
	dim3 grid(_cu_ceil(dstDim.y, Gauss_threads_x), _cu_ceil(dstDim.x, Gauss_threads_y));
	dim3 threads(Gauss_threads_x, Gauss_threads_y);

	// launch the kernel
	Gaussian_blur_hor_mirror << <grid, threads, 0, S_K >> > (dev_src,
		kernel_x,
		dev_mid,
		ker_lenXY.x,
		dstDim,
		radius.x);

	Gaussian_blur_ver_mirror << <grid, threads, 0, S_K >> > (dev_mid,
		kernel_y,
		dev_dst,
		ker_lenXY.y,
		dstDim,
		radius.y);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(kernel_x));
	checkCudaErrors(cudaFree(dev_mid));
	checkCudaErrors(cudaStreamDestroy(S_K));
}



void de::vis::_GaussianBlur3D(de::vis::dev_Img& src, de::vis::dev_Img& dst, const de::Point2D radius, const de::Point2D_d sigma)
{
	if (cuP.is_init != true) {
		printf("CUDA should be initialized first\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	if (src.channel != 4) {
		printf("The input Image array is not a 4D array, a BGR colored image is usually in 4-bytes aligned\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	__dev_Img* sub_ptr_src = dynamic_cast<__dev_Img*>(&src);
	__dev_Img* sub_ptr_dst = dynamic_cast<__dev_Img*>(&dst);

	const uint Wsrc = sub_ptr_src->width;
	const uint Hsrc = sub_ptr_src->height;

	// ~.x : width; ~.y : height
	const dim3 srcDim(Wsrc, Hsrc);
	// ~.x : width; ~.y : height
	const dim3 dstDim(Wsrc - (radius.x << 1), Hsrc - (radius.y << 1));

	// allocate a one dimensional kernel vector
	uchar4* dev_src,
		* dev_dst,
		* dev_mid;
	float* kernel_x, * kernel_y;

	int2 ker_lenXY;
	ker_lenXY.x = (radius.x << 1) + 1;
	ker_lenXY.y = (radius.y << 1) + 1;

	dev_src = (uchar4*)(sub_ptr_src->dev_Mat);
	dev_dst = (uchar4*)(sub_ptr_dst->dev_Mat);

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_mid), dstDim.x * srcDim.y * sizeof(uchar4)));

	//checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_mid_1), dstDim.x * srcDim.y * __SPACE__));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&kernel_x), (ker_lenXY.x + ker_lenXY.y) * sizeof(float)));
	kernel_y = kernel_x + ker_lenXY.x;

	cudaStream_t S_K_0;		// always used in executing the kernel function, e.g.
	checkCudaErrors(cudaStreamCreate(&S_K_0));

	// generating the Gaussian kernel vector
	Gaussian_vec_Gen << <1, ker_lenXY.x, 0, S_K_0 >> > (kernel_x,
		sigma.x,
		radius.x,
		ker_lenXY.x);

	Gaussian_vec_Gen << <1, ker_lenXY.y, 0, S_K_0 >> > (kernel_y,
		sigma.y,
		radius.y,
		ker_lenXY.y);

	// configure the launch parameters
	dim3 grid_1(_cu_ceil(srcDim.y, Gauss_threads_x), _cu_ceil(dstDim.x, Gauss_threads_y));
	dim3 grid_2(_cu_ceil(dstDim.y, Gauss_threads_x), _cu_ceil(dstDim.x, Gauss_threads_y));
	dim3 threads(Gauss_threads_x, Gauss_threads_y);

	checkCudaErrors(cudaDeviceSynchronize());

	// launch the kernel
	Gaussian_blur_hor3D << < grid_1, threads, 0, S_K_0 >> > (dev_src,
		kernel_x,
		dev_mid,
		ker_lenXY.x,
		radius.x,
		dstDim,
		dim3(Wsrc, Hsrc),
		radius.x);

	Gaussian_blur_ver3D << <grid_2, threads, 0, S_K_0 >> > (dev_mid,
		kernel_y,
		dev_dst,
		ker_lenXY.y,
		radius.y,
		dstDim,
		dim3(Wsrc, Hsrc));

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaStreamDestroy(S_K_0));
}



void de::vis::_GaussianBlur3D_border_C(de::vis::dev_Img& src, de::vis::dev_Img& dst, const de::Point2D radius, const de::Point2D_d sigma, const float border)
{
	if (cuP.is_init != true) {
		printf("CUDA should be initialized first\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	if (src.channel != 4) {
		printf("The input Image array is not a 4D array, a BGR colored image is usually in 4-bytes aligned\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	__dev_Img* sub_ptr_src = dynamic_cast<__dev_Img*>(&src);
	__dev_Img* sub_ptr_dst = dynamic_cast<__dev_Img*>(&dst);

	const uint Wsrc = sub_ptr_src->width;
	const uint Hsrc = sub_ptr_src->height;

	// ~.x : width; ~.y : height
	const dim3 srcDim(Wsrc, Hsrc);
	// ~.x : width; ~.y : height
	const dim3 dstDim = srcDim;

	// allocate a one dimensional kernel vector
	uchar4* dev_src,
		* dev_dst,
		* dev_mid;
	float* kernel_x, * kernel_y;

	int2 ker_lenXY;
	ker_lenXY.x = (radius.x << 1) + 1;
	ker_lenXY.y = (radius.y << 1) + 1;

	dev_src = (uchar4*)(sub_ptr_src->dev_Mat);
	dev_dst = (uchar4*)(sub_ptr_dst->dev_Mat);

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_mid), dstDim.x * srcDim.y * sizeof(uchar4)));

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&kernel_x), (ker_lenXY.x + ker_lenXY.y) * sizeof(float)));
	kernel_y = kernel_x + ker_lenXY.x;

	cudaStream_t S_K_0;		// always used in executing the kernel function, e.g. cu_filling and cu_Conv
	checkCudaErrors(cudaStreamCreate(&S_K_0));
	
	// generating the Gaussian kernel vector
	Gaussian_vec_Gen << <1, ker_lenXY.x, 0, S_K_0 >> > (kernel_x,
		sigma.x,
		radius.x,
		ker_lenXY.x);

	Gaussian_vec_Gen << <1, ker_lenXY.y, 0, S_K_0 >> > (kernel_y,
		sigma.y,
		radius.y,
		ker_lenXY.y);

	// configure the launch parameters
	dim3 grid(_cu_ceil(dstDim.y, Gauss_threads_x), _cu_ceil(dstDim.x, Gauss_threads_y));
	dim3 threads(Gauss_threads_x, Gauss_threads_y);

	checkCudaErrors(cudaDeviceSynchronize());

	Gaussian_blur_hor_border3D << < grid, threads, 0, S_K_0 >> > (dev_src,
		kernel_x,
		dev_mid,
		ker_lenXY.x,
		srcDim,
		radius.y,
		border);

	Gaussian_blur_ver_border3D << <grid, threads, 0, S_K_0 >> > (dev_mid,
		kernel_y,
		dev_dst,
		ker_lenXY.y,
		srcDim,
		radius.y,
		border);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaStreamDestroy(S_K_0));
}



void de::vis::_GaussianBlur3D_border_mirror(de::vis::dev_Img& src, de::vis::dev_Img& dst, const de::Point2D radius, const de::Point2D_d sigma)
{
	if (cuP.is_init != true) {
		printf("CUDA should be initialized first\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	if (src.channel != 4) {
		printf("The input Image array is not a 4D array, a BGR colored image is usually in 4-bytes aligned\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	__dev_Img* sub_ptr_src = dynamic_cast<__dev_Img*>(&src);
	__dev_Img* sub_ptr_dst = dynamic_cast<__dev_Img*>(&dst);

	const uint Wsrc = sub_ptr_src->width;
	const uint Hsrc = sub_ptr_src->height;

	// ~.x : width; ~.y : height
	const dim3 srcDim(Wsrc, Hsrc);
	// ~.x : width; ~.y : height
	const dim3 dstDim = srcDim;

	// allocate a one dimensional kernel vector
	uchar4* dev_src,
		* dev_dst,
		* dev_mid;
	float* kernel_x, * kernel_y;

	int2 ker_lenXY;
	ker_lenXY.x = (radius.x << 1) + 1;
	ker_lenXY.y = (radius.y << 1) + 1;

	dev_src = (uchar4*)(sub_ptr_src->dev_Mat);
	dev_dst = (uchar4*)(sub_ptr_dst->dev_Mat);

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_mid), dstDim.x * srcDim.y * sizeof(uchar4)));

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&kernel_x), (ker_lenXY.x + ker_lenXY.y) * sizeof(float)));
	kernel_y = kernel_x + ker_lenXY.x;

	cudaStream_t S_K_0;		// always used in executing the kernel function, e.g. cu_filling and cu_Conv
	checkCudaErrors(cudaStreamCreate(&S_K_0));

	// generating the Gaussian kernel vector
	Gaussian_vec_Gen << <1, ker_lenXY.x, 0, S_K_0 >> > (kernel_x,
		sigma.x,
		radius.x,
		ker_lenXY.x);

	Gaussian_vec_Gen << <1, ker_lenXY.y, 0, S_K_0 >> > (kernel_y,
		sigma.y,
		radius.y,
		ker_lenXY.y);

	// configure the launch parameters
	dim3 grid(_cu_ceil(dstDim.y, Gauss_threads_x), _cu_ceil(dstDim.x, Gauss_threads_y));
	dim3 threads(Gauss_threads_x, Gauss_threads_y);

	checkCudaErrors(cudaDeviceSynchronize());

	Gaussian_blur_hor_mirror3D << < grid, threads, 0, S_K_0 >> > (dev_src,
		kernel_x,
		dev_mid,
		ker_lenXY.x,
		srcDim,
		radius.x);

	Gaussian_blur_ver_mirror3D << <grid, threads, 0, S_K_0 >> > (dev_mid,
		kernel_y,
		dev_dst,
		ker_lenXY.y,
		srcDim,
		radius.y);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaStreamDestroy(S_K_0));
}



void de::vis::_NLM_2D(de::vis::dev_Img& src, de::vis::dev_Img& dst, const de::Point2D search_radius, const de::Point2D neigbor_radius, const float h, int step)
{
	if (cuP.is_init != true) {
		printf("CUDA should be initialized first\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	if (src.channel != 1) {
		printf("The input Image array is not a 1D array\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	__dev_Img* sub_ptr_src = dynamic_cast<__dev_Img*>(&src);
	__dev_Img* sub_ptr_dst = dynamic_cast<__dev_Img*>(&dst);

	int2 srcDims;
	srcDims.x = src.width;
	srcDims.y = src.height;

	int2 search_dims;
	search_dims.x = (search_radius.x << 1) + 1;
	search_dims.y = (search_radius.y << 1) + 1;

	// the search area must bigger than neigbor area
	if (search_radius.x < neigbor_radius.x || search_radius.y < neigbor_radius.y) {
		printf("the search area must bigger than neigbor area\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	int Ne_area = ((neigbor_radius.x << 1) + 1) * ((neigbor_radius.y << 1) + 1);
	int2 Ne_radius;
	Ne_radius.x = neigbor_radius.x;
	Ne_radius.y = neigbor_radius.y;

	int2 with_Ne;
	with_Ne.x = (neigbor_radius.x << 1) + srcDims.x;
	with_Ne.y = (neigbor_radius.y << 1) + srcDims.y;

	int2 diff_active_size;
	diff_active_size.x = (search_radius.x << 1) + srcDims.x;
	diff_active_size.x = (search_radius.y << 1) + srcDims.y;

	//this is the dims for the working area, mirrored borders, both are OUT_OF_CONV
	int2 workspace;
	workspace.x = (search_radius.x << 1) + with_Ne.x;
	workspace.y = (search_radius.y << 1) + with_Ne.y;

	// first allocate the requier arrays
	uchar* padded,
		* dev_src,		// in place array
		* dev_centre_Ne;		/* also works as the subed image in each loop, its dims should be equal to
						with_Ne, in each loop, the program will apply src - var_sub, however, it
						will be done on single workspace by __global__ cu_ImgDiff_Sq()*/
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&padded), workspace.x * workspace.y * sizeof(uchar)));
	dev_src = sub_ptr_dst->dev_Mat;
	uchar* dev_dst = sub_ptr_dst->dev_Mat;

	// malloc the device space of Ne_constant subed image
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_centre_Ne), with_Ne.x * with_Ne.y * sizeof(uchar)));

	float2* loop_accu;	/*accumulate the result in the loop
						~.x : the result heaven't been normalized
						~.y : the accumulating weights----the normalize scale
						*/
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&loop_accu), srcDims.x * srcDims.y * sizeof(float2)));

	cudaStream_t S_K, S_C;
	checkCudaErrors(cudaStreamCreate(&S_K));
	checkCudaErrors(cudaStreamCreate(&S_C));

	// difference image on host, and is a mapped memory, dims -> with_Ne
	float* d_diff;	// device pointer

	size_t diff_bytes = with_Ne.x * with_Ne.y * sizeof(float);
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_diff), diff_bytes));

	// configure the launch parameters
	// srcDims based params
	dim3 thread_src(NLM_Thr_blx, NLM_Thr_bly);
	dim3 grid_src(_cu_ceil(srcDims.y, NLM_Thr_blx), _cu_ceil(srcDims.x, NLM_Thr_bly));

	// workspace dims based params
	dim3 thread_WS = thread_src;
	dim3 grid_WS(_cu_ceil(workspace.y, NLM_Thr_blx), _cu_ceil(workspace.x, NLM_Thr_bly));

	// Ne_expanded dims based params
	dim3 thread_Ne = thread_src;
	dim3 grid_Ne(_cu_ceil(with_Ne.y, NLM_Thr_blx), _cu_ceil(with_Ne.x, NLM_Thr_bly));

	checkCudaErrors(cudaDeviceSynchronize());

	// make border for workspace
	int2 pre_offset;
	pre_offset.x = search_radius.x + neigbor_radius.x;
	pre_offset.y = search_radius.y + neigbor_radius.y;
	// srcDims -> worksapce
	cu_border_mirror << < grid_WS, thread_WS, 0, S_K >> > (dev_src,
		padded,
		srcDims,
		workspace,
		pre_offset);

	// srcDims -> with_Ne, create the constant centre Ne subed image
	pre_offset.x = neigbor_radius.x;
	pre_offset.y = neigbor_radius.y;
	cu_border_mirror << <grid_Ne, thread_Ne, 0, S_C >> > (dev_src,
		dev_centre_Ne,
		srcDims,
		with_Ne,
		pre_offset);

	int2 __proc;
	__proc.x = search_dims.x / step;
	__proc.y = search_dims.y / step;

	float h_2 = h * h;

	checkCudaErrors(cudaDeviceSynchronize());

	/*the constant subed matrix is actually the original image src, so the parameter subed of cu_ImgDiff_Sq
	* is dev_src*/
	// let's start the loop
	for (int i_sr = 0; i_sr < __proc.y; i_sr += step)
	{
		for (int j_sr = 0; j_sr < __proc.x; j_sr += step)
		{
			int2 _shf;
			_shf.x = i_sr;
			_shf.y = j_sr;
			/* requires a subed image(src) and the workspace, the subed image will be subtracted by the segment of
			 workspacesrc, the difference image dst is a mapped memory d_diff, sub is always dev_src*/
			cu_ImgDiff_Sq << <grid_Ne, thread_Ne, 0, S_K >> > (padded,
				d_diff,
				dev_centre_Ne,
				_shf,
				with_Ne,
				workspace.x);

			cu_NLM_calc << <grid_src, thread_src, 0, S_K >> > (d_diff,
				padded,
				loop_accu,
				_shf,
				Ne_radius,
				Ne_area,
				srcDims,
				with_Ne.x,
				workspace.x,
				h_2);
		}
	}
	checkCudaErrors(cudaDeviceSynchronize());

	cu_NLM_final << <grid_src, thread_src, 0, S_K >> > (loop_accu, dev_dst, srcDims);

	checkCudaErrors(cudaFree(padded));
	checkCudaErrors(cudaFree(d_diff));
	checkCudaErrors(cudaStreamSynchronize(S_K));

	checkCudaErrors(cudaFree(loop_accu));
	checkCudaErrors(cudaFree(dev_centre_Ne));
	checkCudaErrors(cudaStreamDestroy(S_K));

	checkCudaErrors(cudaStreamSynchronize(S_C));
	checkCudaErrors(cudaStreamDestroy(S_C));
}



void de::vis::_NLM_3D(de::vis::dev_Img& src, de::vis::dev_Img& dst, const de::Point2D search_radius, const de::Point2D neigbor_radius, const float h, int step)
{
	if (cuP.is_init != true) {
		printf("CUDA should be initialized first\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	if (src.channel != 4) {
		printf("The input Image array is not a 4D array, a BGR colored image is usually in 4-bytes aligned\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	int2 srcDims;
	srcDims.x = src.width;
	srcDims.y = src.height;

	int2 search_dims;
	search_dims.x = (search_radius.x << 1) + 1;
	search_dims.y = (search_radius.y << 1) + 1;

	// the search area must bigger than neigbor area
	if (search_radius.x < neigbor_radius.x || search_radius.y < neigbor_radius.y) {
		printf("the search area must bigger than neigbor area\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	__dev_Img* sub_ptr_src = dynamic_cast<__dev_Img*>(&src);
	__dev_Img* sub_ptr_dst = dynamic_cast<__dev_Img*>(&dst);

	int Ne_area = ((neigbor_radius.x << 1) + 1) * ((neigbor_radius.y << 1) + 1);
	int2 Ne_radius;
	Ne_radius.x = neigbor_radius.x;
	Ne_radius.y = neigbor_radius.y;

	int2 with_Ne;
	with_Ne.x = (neigbor_radius.x << 1) + srcDims.x;
	with_Ne.y = (neigbor_radius.y << 1) + srcDims.y;

	int2 diff_active_size;
	diff_active_size.x = (search_radius.x << 1) + srcDims.x;
	diff_active_size.x = (search_radius.y << 1) + srcDims.y;

	//this is the dims for the working area, mirrored borders, both are OUT_OF_CONV
	int2 workspace;
	workspace.x = (search_radius.x << 1) + with_Ne.x;
	workspace.y = (search_radius.y << 1) + with_Ne.y;

	// first allocate the requier arrays
	uchar* padded,
		* dev_src,		// in place array
		* dev_centre_Ne;		/* also works as the subed image in each loop, its dims should be equal to
						with_Ne, in each loop, the program will apply src - var_sub, however, it
						will be done on single workspace by __global__ cu_ImgDiff_Sq()*/
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&padded), workspace.x * workspace.y * sizeof(uchar4)));
	dev_src = sub_ptr_src->dev_Mat;
	uchar* dev_dst = sub_ptr_dst->dev_Mat;

	// malloc the device space of Ne_constant subed image
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_centre_Ne), with_Ne.x * with_Ne.y * sizeof(uchar4)));

	float6* loop_accu;	/*accumulate the result in the loop
						~.x : the result heaven't been normalized
						~.y : the accumulating weights----the normalize scale
						*/
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&loop_accu), srcDims.x * srcDims.y * sizeof(float6)));

	cudaStream_t S_K, S_C;
	checkCudaErrors(cudaStreamCreate(&S_K));
	checkCudaErrors(cudaStreamCreate(&S_C));

	// difference image on host, and is a mapped memory, dims -> with_Ne
	float4* d_diff;	// device pointer

	size_t diff_bytes = with_Ne.x * with_Ne.y * sizeof(float4);
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_diff), diff_bytes));

	// configure the launch parameters
	// srcDims based params
	dim3 thread_src(NLM_Thr_blx, NLM_Thr_bly);
	dim3 grid_src(_cu_ceil(srcDims.y, NLM_Thr_blx), _cu_ceil(srcDims.x, NLM_Thr_bly));

	// workspace dims based params
	dim3 thread_WS = thread_src;
	dim3 grid_WS(_cu_ceil(workspace.y, NLM_Thr_blx), _cu_ceil(workspace.x, NLM_Thr_bly));

	// Ne_expanded dims based params
	dim3 thread_Ne = thread_src;
	dim3 grid_Ne(_cu_ceil(with_Ne.y, NLM_Thr_blx), _cu_ceil(with_Ne.x, NLM_Thr_bly));

	checkCudaErrors(cudaDeviceSynchronize());

	// make border for workspace
	int2 pre_offset;
	pre_offset.x = search_radius.x + neigbor_radius.x;
	pre_offset.y = search_radius.y + neigbor_radius.y;
	// srcDims -> worksapce
	cu_border_mirror_3D << < grid_WS, thread_WS, 0, S_K >> > (dev_src,
		padded,
		srcDims,
		workspace,
		pre_offset);

	// srcDims -> with_Ne, create the constant centre Ne subed image
	pre_offset.x = neigbor_radius.x;
	pre_offset.y = neigbor_radius.y;
	cu_border_mirror_3D << <grid_Ne, thread_Ne, 0, S_C >> > (dev_src,
		dev_centre_Ne,
		srcDims,
		with_Ne,
		pre_offset);

	int2 __proc;
	__proc.x = search_dims.x / step;
	__proc.y = search_dims.y / step;

	float h_2 = h * h;

	checkCudaErrors(cudaDeviceSynchronize());

	cudaMemset(loop_accu, 0, srcDims.x * srcDims.y * sizeof(float6));

	/*the constant subed matrix is actually the original image src, so the parameter subed of cu_ImgDiff_Sq
	* is dev_src*/
	// let's start the loop
	for (int i_sr = 0; i_sr < __proc.y; i_sr += step)
	{
		for (int j_sr = 0; j_sr < __proc.x; j_sr += step)
		{
			int2 _shf;
			_shf.x = i_sr;
			_shf.y = j_sr;
			/* requires a subed image(src) and the workspace, the subed image will be subtracted by the segment of
			 workspacesrc, the difference image dst is a mapped memory d_diff, sub is always dev_src*/
			cu_ImgDiff_Sq3D << <grid_Ne, thread_Ne, 0, S_K >> > (padded,
				d_diff,
				dev_centre_Ne,
				_shf,
				with_Ne,
				workspace.x);

			cu_NLM_calc_3D << <grid_src, thread_src, 0, S_K >> > (d_diff,
				padded,
				loop_accu,
				_shf,
				Ne_radius,
				Ne_area,
				srcDims,
				with_Ne.x,
				workspace.x,
				h_2);
		}
	}
	checkCudaErrors(cudaDeviceSynchronize());

	cu_NLM_final_3D << <grid_src, thread_src, 0, S_K >> > (loop_accu, dev_dst, srcDims);

	checkCudaErrors(cudaFree(padded));
	checkCudaErrors(cudaFree(d_diff));
	checkCudaErrors(cudaStreamSynchronize(S_K));

	checkCudaErrors(cudaFree(loop_accu));
	checkCudaErrors(cudaFree(dev_centre_Ne));
	checkCudaErrors(cudaStreamDestroy(S_K));

	checkCudaErrors(cudaStreamSynchronize(S_C));
	checkCudaErrors(cudaStreamDestroy(S_C));
}