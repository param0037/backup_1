#pragma once
#include "../../basic/basic.h"
#include "../../classes/matrix.h"
#include "../../cv/device/cv_basic_proc.cuh"
#include "cv/cv_classes/cv_classes.h"


namespace de
{
	namespace vis
	{
		_DECX_API_ de::DH GaussianBlur2D(de::vis::Img& src, de::vis::Img& dst, const de::Point2D radius, const de::Point2D_d sigma);


		_DECX_API_ de::DH GaussianBlur2D_border_C(de::vis::Img& src, de::vis::Img& dst, const de::Point2D radius, const de::Point2D_d sigma, const float border);


		_DECX_API_ de::DH GaussianBlur2D_border_mirror(de::vis::Img& src, de::vis::Img& dst, const de::Point2D radius, const de::Point2D_d sigma);


		_DECX_API_ de::DH GaussianBlur3D(de::vis::Img& src, de::vis::Img& dst, const de::Point2D radius, const de::Point2D_d sigma);


		_DECX_API_ de::DH GaussianBlur3D_border_C(de::vis::Img& src, de::vis::Img& dst, const de::Point2D radius, const de::Point2D_d sigma, const float border);


		_DECX_API_ de::DH GaussianBlur3D_border_mirror(de::vis::Img& src, de::vis::Img& dst, const de::Point2D radius, const de::Point2D_d sigma);


		_DECX_API_ de::DH NLM_2D(de::vis::Img& src, de::vis::Img& dst, const de::Point2D search_dim, const de::Point2D neigbor_dim, const float h, int step);


		_DECX_API_ de::DH NLM_3D(de::vis::Img& src, de::vis::Img& dst, const de::Point2D search_dim, const de::Point2D neigbor_dim, const float h, int step);
	}
}



#if 0
/*apply the Gaussian blur on a matrix, which means only one channel to be proccessed
* 高斯核是分离的， 一维的， 因此将其放入常量存储器中
*/
de::DH de::vis::GaussianBlur2D(de::vis::Img &src, de::vis::Img &dst, const de::Point2D radius, const de::Point2D_d sigma)
{
	de::DH handle;

	if (cuP.is_init != true) {
		handle.error_string = "CUDA should be initialize first";
		handle.error_type = DECX_FAIL_not_init;
		return handle;
	}

	if (src.channel != 1) {
		handle.error_string = "Source matrix must be a vis::Img class with DE_UC4";
		handle.error_type = DECX_FAIL_ChannelError;
		return handle;
	}

	const uint Wsrc = src.width;
	const uint Hsrc = src.height;
	
	// allocate a one dimensional kernel vector
	uchar *dev_src,
		*dev_mid;
	float* kernel_x, * kernel_y;

	int2 ker_lenXY;
	ker_lenXY.x = (radius.x << 1) + 1;
	ker_lenXY.y = (radius.y << 1) + 1;

	// ~.x : width, ~.y : height
	dim3 dstDim(Wsrc - (radius.x << 1), Hsrc - (radius.y << 1));

	checkCudaErrors(cudaMalloc(&kernel_x, (ker_lenXY.x + ker_lenXY.y) * sizeof(float)));
	kernel_y = kernel_x + ker_lenXY.x;

	checkCudaErrors(cudaMalloc(&dev_src, src.total_bytes));
	checkCudaErrors(cudaMalloc(&dev_mid, dstDim.x * Hsrc * sizeof(uchar)));

	cudaStream_t S_C,		// used in transfering the datas between host and device 
		S_K;
	checkCudaErrors(cudaStreamCreateWithFlags(&S_C, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&S_K, cudaStreamNonBlocking));
	
	// start copying the src.Mat from host to device 
	checkCudaErrors(cudaMemcpyAsync(dev_src, src.Mat, src.total_bytes, cudaMemcpyHostToDevice, S_C));

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
		dev_src,
		ker_lenXY.y,
		radius.y,
		dstDim,
		dim3(Wsrc, Hsrc));

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpyAsync(dst.Mat, dev_src, dst.total_bytes, cudaMemcpyDeviceToHost, S_C));

	checkCudaErrors(cudaFree(kernel_x));
	checkCudaErrors(cudaFree(dev_mid));
	checkCudaErrors(cudaStreamDestroy(S_K));

	checkCudaErrors(cudaStreamSynchronize(S_C));
	checkCudaErrors(cudaFree(dev_src));
	checkCudaErrors(cudaStreamDestroy(S_C));

	handle.error_string = "No error";
	handle.error_type = DECX_SUCCESS;
	return handle;
}



de::DH de::vis::GaussianBlur2D_border_C(de::vis::Img& src, de::vis::Img& dst, const de::Point2D radius, const de::Point2D_d sigma, const float border)
{
	de::DH handle;

	if (cuP.is_init != true) {
		handle.error_string = "CUDA should be initialize first";
		handle.error_type = DECX_FAIL_not_init;
		return handle;
	}

	if (src.channel != 1) {
		handle.error_string = "Source matrix must be a vis::Img class with DE_UC4";
		handle.error_type = DECX_FAIL_ChannelError;
		return handle;
	}

	const uint Wsrc = src.width;
	const uint Hsrc = src.height;

	// allocate a one dimensional kernel vector
	uchar *dev_src,
		*dev_dst,
		*dev_mid;
	float* kernel_x, * kernel_y;

	int2 ker_lenXY;
	ker_lenXY.x = (radius.x << 1) + 1;
	ker_lenXY.y = (radius.y << 1) + 1;

	// ~.x : width, ~.y : height
	dim3 dstDim(Wsrc, Hsrc);

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&kernel_x), (ker_lenXY.x + ker_lenXY.y) * sizeof(float)));
	kernel_y = kernel_x + ker_lenXY.x;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_src), src.total_bytes));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_mid), dstDim.x * Hsrc * sizeof(uchar)));

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_dst), dstDim.x * dstDim.y * sizeof(uchar)));

	cudaStream_t S_C,		// used in transfering the datas between host and device 
		S_K;
	checkCudaErrors(cudaStreamCreateWithFlags(&S_C, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&S_K, cudaStreamNonBlocking));
	
	// start copying the src.Mat from host to device 
	checkCudaErrors(cudaMemcpyAsync(dev_src, src.Mat, src.total_bytes, cudaMemcpyHostToDevice, S_C));

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

	checkCudaErrors(cudaDeviceSynchronize());

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
	checkCudaErrors(cudaMemcpyAsync(dst.Mat, dev_dst, dst.total_bytes, cudaMemcpyDeviceToHost, S_C));

	checkCudaErrors(cudaFree(dev_src));
	checkCudaErrors(cudaFree(kernel_x));
	checkCudaErrors(cudaFree(dev_mid));
	checkCudaErrors(cudaStreamDestroy(S_K));

	checkCudaErrors(cudaStreamSynchronize(S_C));
	checkCudaErrors(cudaFree(dev_dst));
	checkCudaErrors(cudaStreamDestroy(S_C));

	handle.error_string = "No error";
	handle.error_type = DECX_SUCCESS;
	return handle;
}



de::DH de::vis::GaussianBlur2D_border_mirror(de::vis::Img& src, de::vis::Img&dst, const de::Point2D radius, const de::Point2D_d sigma)
{
	de::DH handle;

	if (cuP.is_init != true) {
		handle.error_string = "CUDA should be initialize first";
		handle.error_type = DECX_FAIL_not_init;
		return handle;
	}

	if (src.channel != 1) {
		handle.error_string = "Source matrix must be a vis::Img class with DE_UC4";
		handle.error_type = DECX_FAIL_ChannelError;
		return handle;
	}

	const uint Wsrc = src.width;
	const uint Hsrc = src.height;

	// allocate a one dimensional kernel vector
	uchar* dev_src,
		* dev_dst,
		* dev_mid;
	float* kernel_x, * kernel_y;

	int2 ker_lenXY;
	ker_lenXY.x = (radius.x << 1) + 1;
	ker_lenXY.y = (radius.y << 1) + 1;

	// ~.x : width, ~.y : height
	dim3 dstDim(Wsrc, Hsrc);

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&kernel_x), (ker_lenXY.x + ker_lenXY.y) * sizeof(float)));
	kernel_y = kernel_x + ker_lenXY.x;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_src), src.total_bytes));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_mid), dstDim.x * Hsrc * sizeof(uchar)));

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_dst), dstDim.x * dstDim.y * sizeof(uchar)));

	cudaStream_t S_C,		// used in transfering the datas between host and device 
		S_K,
		S_C_1;				// used in launching the CUDA kernel functions
	checkCudaErrors(cudaStreamCreateWithFlags(&S_C, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&S_K, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&S_C_1, cudaStreamNonBlocking));

	// start copying the src.Mat from host to device 
	checkCudaErrors(cudaMemcpyAsync(dev_src, src.Mat, src.total_bytes, cudaMemcpyHostToDevice, S_C));

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

	checkCudaErrors(cudaDeviceSynchronize());

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

	checkCudaErrors(cudaStreamSynchronize(S_K));
	checkCudaErrors(cudaMemcpyAsync(dst.Mat, dev_dst, dst.total_bytes, cudaMemcpyDeviceToHost, S_C));

	checkCudaErrors(cudaFree(dev_src));
	checkCudaErrors(cudaFree(kernel_x));
	checkCudaErrors(cudaFree(dev_mid));
	checkCudaErrors(cudaStreamDestroy(S_K));

	checkCudaErrors(cudaStreamSynchronize(S_C));
	checkCudaErrors(cudaFree(dev_dst));
	checkCudaErrors(cudaStreamDestroy(S_C));

	handle.error_string = "No error";
	handle.error_type = DECX_SUCCESS;
	return handle;
}
#endif


struct arr_info
{
	void** arrP;
	size_t malloc_size;
	arr_info();
	arr_info(void** _arrP, size_t _size) { 
		this->arrP = _arrP; 
		this->malloc_size = _size; 
	}
};




#if 0
de::DH de::vis::GaussianBlur3D(de::vis::Img& src, de::vis::Img& dst, const de::Point2D radius, const de::Point2D_d sigma)
{
	de::DH handle;

	if (cuP.is_init != true) {
		handle.error_string = "CUDA should be initialize first";
		handle.error_type = DECX_FAIL_not_init;
		return handle;
	}

	if (src.channel != 4) {
		handle.error_string = "Source matrix must be a vis::Img class with DE_UC4";
		handle.error_type = DECX_FAIL_ChannelError;
		return handle;
	}
	//const uint ker_len = (radius << 1) + 1;
	const uint depth = src.channel;
	const uint Wsrc = src.width;
	const uint Hsrc = src.height;

	// ~.x : width; ~.y : height
	const dim3 srcDim(Wsrc, Hsrc, depth);
	// ~.x : width; ~.y : height
	const dim3 dstDim(Wsrc - (radius.x << 1), Hsrc - (radius.y << 1), depth);

	// allocate a one dimensional kernel vector
	uchar4* dev_src,
		* dev_dst,
		* dev_mid;
	float* kernel_x, * kernel_y;

	int2 ker_lenXY;
	ker_lenXY.x = (radius.x << 1) + 1;
	ker_lenXY.y = (radius.y << 1) + 1;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_src), src.ImgPlane * sizeof(uchar4)));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_dst), dst.ImgPlane * sizeof(uchar4)));

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_mid), dstDim.x * srcDim.y * sizeof(uchar4)));
	
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&kernel_x), (ker_lenXY.x + ker_lenXY.y) * sizeof(float)));
	kernel_y = kernel_x + ker_lenXY.x;

	cudaStream_t S_K_0,		// always used in executing the kernel function, e.g. cu_filling and cu_Conv
		S_C_0;		// always used in copying datas from host to deice, e.g. src and ker

	checkCudaErrors(cudaStreamCreate(&S_K_0));
	checkCudaErrors(cudaStreamCreate(&S_C_0));

	checkCudaErrors(cudaMemcpyAsync(dev_src, src.Mat, src.ImgPlane * sizeof(uchar4), cudaMemcpyHostToDevice, S_C_0));

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

	checkCudaErrors(cudaMemcpyAsync(dst.Mat, dev_dst, dst.ImgPlane * sizeof(uchar4), cudaMemcpyDeviceToHost, S_K_0));

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaStreamDestroy(S_K_0));
	checkCudaErrors(cudaStreamDestroy(S_C_0));

	handle.error_string = "No error";
	handle.error_type = DECX_SUCCESS;
	return handle;
}



//#define __malloc_align
de::DH de::vis::GaussianBlur3D_border_C(de::vis::Img& src, de::vis::Img& dst, const de::Point2D radius, const de::Point2D_d sigma, const float border)
{
	de::DH handle;

	if (cuP.is_init != true) {
		handle.error_string = "CUDA should be initialize first";
		handle.error_type = DECX_FAIL_not_init;
		return handle;
	}

	if (src.channel != 4) {
		handle.error_string = "Source matrix must be a vis::Img class with DE_UC4";
		handle.error_type = DECX_FAIL_ChannelError;
		return handle;
	}

	//const uint ker_len = (radius << 1) + 1;
	const uint depth = src.channel;
	const uint Wsrc = src.width;
	const uint Hsrc = src.height;

	// ~.x : width; ~.y : height
	const dim3 srcDim(Wsrc, Hsrc, depth);
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

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_src), src.ImgPlane * sizeof(uchar4)));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_dst), dst.ImgPlane * sizeof(uchar4)));

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_mid), dstDim.x * srcDim.y * sizeof(uchar4)));

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&kernel_x), (ker_lenXY.x + ker_lenXY.y) * sizeof(float)));
	kernel_y = kernel_x + ker_lenXY.x;

	cudaStream_t S_K_0,		// always used in executing the kernel function, e.g. cu_filling and cu_Conv
		S_C_0;		// always used in copying datas from host to deice, e.g. src and ker

	checkCudaErrors(cudaStreamCreate(&S_K_0));
	checkCudaErrors(cudaStreamCreate(&S_C_0));

	checkCudaErrors(cudaMemcpyAsync(dev_src, src.Mat, src.ImgPlane * sizeof(uchar4), cudaMemcpyHostToDevice, S_C_0));

	// generating the Gaussian kernel vector
	//double _dex_mapping = 3 * (double)sigma / (double)radius;
	//uint iter = _GetHighest(ker_len);

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

	checkCudaErrors(cudaMemcpyAsync(dst.Mat, dev_dst, dst.ImgPlane * sizeof(uchar4), cudaMemcpyDeviceToHost, S_K_0));

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaStreamDestroy(S_K_0));
	checkCudaErrors(cudaStreamDestroy(S_C_0));

	handle.error_string = "No error";
	handle.error_type = DECX_SUCCESS;
	return handle;
}



de::DH de::vis::GaussianBlur3D_border_mirror(de::vis::Img& src, de::vis::Img& dst, const de::Point2D radius, const de::Point2D_d sigma)
{
	de::DH handle;

	if (cuP.is_init != true) {
		handle.error_string = "CUDA should be initialize first";
		handle.error_type = DECX_FAIL_not_init;
		return handle;
	}

	if (src.channel != 4) {
		handle.error_string = "Source matrix must be a vis::Img class with DE_UC4";
		handle.error_type = DECX_FAIL_ChannelError;
		return handle;
	}

	//const uint ker_len = (radius << 1) + 1;
	const uint depth = src.channel;
	const uint Wsrc = src.width;
	const uint Hsrc = src.height;

	// ~.x : width; ~.y : height
	const dim3 srcDim(Wsrc, Hsrc, depth);
	// ~.x : width; ~.y : height
	const dim3 dstDim = srcDim;

	// allocate a one dimensional kernel vector
	uchar4* dev_src,
		* dev_dst,
		* dev_mid;
	float* kernel_x, *kernel_y;

	int2 ker_lenXY;
	ker_lenXY.x = (radius.x << 1) + 1;
	ker_lenXY.y = (radius.y << 1) + 1;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_src), src.ImgPlane * sizeof(uchar4)));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_dst), dst.ImgPlane * sizeof(uchar4)));

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_mid), dstDim.x * srcDim.y * sizeof(uchar4)));

	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&kernel_x), (ker_lenXY.x + ker_lenXY.y) * sizeof(float)));
	kernel_y = kernel_x + ker_lenXY.x;

	cudaStream_t S_K_0,		// always used in executing the kernel function, e.g. cu_filling and cu_Conv
		S_C_0;		// always used in copying datas from host to deice, e.g. src and ker

	checkCudaErrors(cudaStreamCreate(&S_K_0));
	checkCudaErrors(cudaStreamCreate(&S_C_0));

	checkCudaErrors(cudaMemcpyAsync(dev_src, src.Mat, src.ImgPlane * sizeof(uchar4), cudaMemcpyHostToDevice, S_C_0));

	// generating the Gaussian kernel vector
	//double _dex_mapping = 3 * (double)sigma / (double)radius;
	//uint iter = _GetHighest(ker_len);

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
	
	checkCudaErrors(cudaMemcpyAsync(dst.Mat, dev_dst, dst.ImgPlane * sizeof(uchar4), cudaMemcpyDeviceToHost, S_K_0));

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaStreamDestroy(S_K_0));
	checkCudaErrors(cudaStreamDestroy(S_C_0));

	handle.error_string = "No error";
	handle.error_type = DECX_SUCCESS;
	return handle;
}
#endif


// search_radius.x : how many rows, ~.y : how many cols, so is neigbor_radius
// dst matrix has the same size with src
de::DH de::vis::NLM_2D(de::vis::Img& src, de::vis::Img& dst, const de::Point2D search_radius, const de::Point2D neigbor_radius, const float h, int step)
{
	de::DH handle;

	if (cuP.is_init != true) {
		handle.error_string = "CUDA should be initialize first";
		handle.error_type = DECX_FAIL_not_init;
		return handle;
	}

	if (src.channel != 1) {
		handle.error_string = "Source matrix must be a vis::Img class with DE_UC1";
		handle.error_type = DECX_FAIL_ChannelError;
		return handle;
	}

	int2 srcDims;
	srcDims.x = src.width;
	srcDims.y = src.height;

	int2 search_dims;
	search_dims.x = (search_radius.x << 1) + 1;
	search_dims.y = (search_radius.y << 1) + 1;

	// the search area must bigger than neigbor area
	if (search_radius.x < neigbor_radius.x || search_radius.y < neigbor_radius.y){
		handle.error_string = "Search area must be larger than the neibour area";
		handle.error_type = DECX_FAIL_CVNLM_BadArea;
		return handle;
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
		*dev_src,		// in place array
		*dev_centre_Ne;		/* also works as the subed image in each loop, its dims should be equal to 
						with_Ne, in each loop, the program will apply src - var_sub, however, it
						will be done on single workspace by __global__ cu_ImgDiff_Sq()*/
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&padded), workspace.x * workspace.y * sizeof(uchar)));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_src), src.total_bytes));
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
	float *d_diff;	// device pointer

	size_t diff_bytes = with_Ne.x * with_Ne.y * sizeof(float);
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_diff), diff_bytes));

	// copy src
	checkCudaErrors(cudaMemcpyAsync(dev_src, src.Mat, src.total_bytes, cudaMemcpyHostToDevice, S_C));

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
	cu_border_mirror << < grid_WS, thread_WS, 0, S_K >> > 
		(dev_src, padded, srcDims, workspace, pre_offset);

	// srcDims -> with_Ne, create the constant centre Ne subed image
	pre_offset.x = neigbor_radius.x;
	pre_offset.y = neigbor_radius.y;
	cu_border_mirror << <grid_Ne, thread_Ne, 0, S_C >> > 
		(dev_src, dev_centre_Ne, srcDims, with_Ne, pre_offset);

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
			cu_ImgDiff_Sq << <grid_Ne, thread_Ne, 0, S_K >> > 
				(padded, d_diff, dev_centre_Ne, _shf, with_Ne, workspace.x);

			cu_NLM_calc << <grid_src, thread_src, 0, S_K >> > 
				(d_diff, padded, loop_accu, _shf, Ne_radius, Ne_area, srcDims, with_Ne.x, workspace.x, h_2);
		}
	}
	checkCudaErrors(cudaDeviceSynchronize());

	cu_NLM_final << <grid_src, thread_src, 0, S_K >> > (loop_accu, dev_src, srcDims);

	checkCudaErrors(cudaFree(padded));
	checkCudaErrors(cudaFree(d_diff));
	checkCudaErrors(cudaStreamSynchronize(S_K));

	checkCudaErrors(cudaMemcpyAsync(dst.Mat, dev_src, src.total_bytes, cudaMemcpyDeviceToHost, S_C));

	checkCudaErrors(cudaFree(loop_accu));
	checkCudaErrors(cudaFree(dev_centre_Ne));
	checkCudaErrors(cudaStreamDestroy(S_K));

	checkCudaErrors(cudaStreamSynchronize(S_C));
	checkCudaErrors(cudaFree(dev_src));
	checkCudaErrors(cudaStreamDestroy(S_C));

	handle.error_string = "No error";
	handle.error_type = DECX_SUCCESS;
	return handle;
}




de::DH de::vis::NLM_3D(de::vis::Img& src, de::vis::Img& dst, const de::Point2D search_radius, const de::Point2D neigbor_radius, const float h, int step)
{
	de::DH handle;

	if (cuP.is_init != true) {
		handle.error_string = "CUDA should be initialize first";
		handle.error_type = DECX_FAIL_not_init;
		return handle;
	}

	if (src.channel != 4) {
		handle.error_string = "Source matrix must be a vis::Img class with DE_UC4";
		handle.error_type = DECX_FAIL_ChannelError;
		return handle;
	}

	int2 srcDims;
	srcDims.x = src.width;
	srcDims.y = src.height;

	int2 search_dims;
	search_dims.x = (search_radius.x << 1) + 1;
	search_dims.y = (search_radius.y << 1) + 1;

	// the search area must bigger than neigbor area
	if (search_radius.x < neigbor_radius.x || search_radius.y < neigbor_radius.y) {
		handle.error_string = "Search area must be larger than the neibour area";
		handle.error_type = DECX_FAIL_CVNLM_BadArea;
		return handle;
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
		*dev_src,		// in place array
		*dev_centre_Ne;		/* also works as the subed image in each loop, its dims should be equal to 
						with_Ne, in each loop, the program will apply src - var_sub, however, it
						will be done on single workspace by __global__ cu_ImgDiff_Sq()*/
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&padded), workspace.x * workspace.y * sizeof(uchar4)));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_src), src.total_bytes));
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
	float4 *d_diff;	// device pointer

	size_t diff_bytes = with_Ne.x * with_Ne.y * sizeof(float4);
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_diff), diff_bytes));

	// copy src
	checkCudaErrors(cudaMemcpyAsync(dev_src, src.Mat, src.total_bytes, cudaMemcpyHostToDevice, S_C));

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

	cu_NLM_final_3D << <grid_src, thread_src, 0, S_K >> > (loop_accu, dev_src, srcDims);

	checkCudaErrors(cudaFree(padded));
	checkCudaErrors(cudaFree(d_diff));
	checkCudaErrors(cudaStreamSynchronize(S_K));

	checkCudaErrors(cudaMemcpyAsync(dst.Mat, dev_src, src.total_bytes, cudaMemcpyDeviceToHost, S_C));

	checkCudaErrors(cudaFree(loop_accu));
	checkCudaErrors(cudaFree(dev_centre_Ne));
	checkCudaErrors(cudaStreamDestroy(S_K));

	checkCudaErrors(cudaStreamSynchronize(S_C));
	checkCudaErrors(cudaFree(dev_src));
	checkCudaErrors(cudaStreamDestroy(S_C));

	handle.error_string = "No error";
	handle.error_type = DECX_SUCCESS;
	return handle;
}