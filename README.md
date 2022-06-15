# backup_1
Config files for my GitHub profile.
# 宗旨
## 本项目本来是我学习CUDA和cpu并发编程的记录。如果您是正在做毕设的大学生，不想配置主流复杂的BLAS库，并且在C++的层面编程，不妨试一下我写的BLAS库，希望可以帮到您。

# 警告
## 目前只适配了Windows端

# 做了以下更新：
## 1. 重写了方阵矩阵乘的CUDA内核，速度大幅度提升。
## 2. 重写了二维滑窗卷积CUDA内核，优化了速度。
## 3. 增加了基于im2col方法的卷积，适用于卷积神经网络。
## 4. 增加了三个内存池，优化了内存管理。
## 5. 增加了简易的线程池，基于c++11的future和function实现，目前的缺点：在低频（2.2GHz）下cpu负荷只有80%。
## 6. 增加了cpu端的矩阵乘算子。
## 7. 重写了fft1D和fft2D的CUDA内核，向量化访存，并尽可能地利用共享内存提速。
## 8. 增加了Tensor，TensorArray类，并给每个类都增加了其GPU版本，避免了在主机端分配内存并拷贝。
## 9. 重写了NLM（非局部均值滤波）内核，直接读取uchar，并应用共享内存，速度提升

# 若有留言请通过1722336300@qq.com联系我。


# purpose
## Originally, I create this project to store some of my notes when I was learning CUDA and CPU concurrency programming. If you are a collage student who is busy with your graduation projects, and you are somehow not allowed to use or tired to config the mainstream BLAS library, maybe it's a good way to have a try on my library. I hope I can help you.

# WARNINGS
## Currently the source codes only work on Windows.

# updates
## 1. I rewrote the CUDA kernel used in matrix multiplication. The speed is boost dramatically.
## 2. I rewrote the CUDA kernel used in 2D convolution. The speed is boost dramatically.
## 3. A new method to improve performance in 2D convolution called "im2col" is implemented, which is suitable for convolution neuro networks.
## 4. Three memory pools is added, which optimizes the memory management.
## 5. I also create a thread pool based on the new features in c++11. But it has some drabacks, like, the occupancy only reaches to 80% of CPU when the frequency is relatively low (2.2GHz).
## 6. New GEMM operators on CPU is added.
## 7. I rewrote CUDA kernels used in fft1D and fft2D by vectorizing memory access, and using shared memory when it is possible.
## 8. I increase Tensor and TensorArray classes. At the meantime, I created the corresponding classes operates on GPU to avoid some unnecessary allocation on host and memory copying between host and device.
## 9. I rewrote NLM (non-local means) kernels, which can read data in uchar directly, and makes full use of shared memory.

# Do not hesitate to contact me via 1722336300@qq.com if you have any comment.
