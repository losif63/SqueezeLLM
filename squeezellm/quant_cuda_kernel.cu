#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>

// half-tensor
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDATensorMethods.cuh>

// atomicAdd for double-precision floating-point numbers on hardware with
// compute capability < 6.0 from:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(
    double* address,
    double val
) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(
      address_as_ull,
      assumed,
      __double_as_longlong(val + __longlong_as_double(assumed))
    );

  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

const int BLOCKWIDTH  = 128;
const int BLOCKHEIGHT3 =  12;
const int BLOCKHEIGHT4 =  16;

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__device__ inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}

/**********************************************************************/
// FP32

__global__ void VecQuant3MatMulKernelNUQPerChannel(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
);

__global__ void VecQuant4MatMulKernelNUQPerChannel(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
);

__global__ void VecQuant3MatMulKernelNUQPerChannelBatched(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
);

__global__ void VecQuant4MatMulKernelNUQPerChannelBatched(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
);

template <typename scalar_t>
__global__ void SPMV_ATOMIC(
    const       int* __restrict__ rows,
    const       int* __restrict__ cols,
    const  scalar_t* __restrict__ mat,
    const  scalar_t* __restrict__ vec,
	         scalar_t* __restrict__ mul,
    const  int num_rows
);

template <typename scalar_t>
__global__ void SPMV_ATOMIC_BATCHED(
    const       int* __restrict__ rows,
    const       int* __restrict__ cols,
    const  scalar_t* __restrict__ mat,
    const  scalar_t* __restrict__ vec,
	         scalar_t* __restrict__ mul,
    const  int num_rows,
    int batch,
    int vec_height
);

__global__ void DenseMatVecKernel(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int topX,
    int full_width
);

__global__ void DenseMatVecKernelBatched(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int topX,
    int full_width,
    int batch,
    int vec_height,
    int matwidth
);

/**********************************************************************/
// FP16

__global__ void VecQuant3MatMulKernelNUQPerChannel_fp16(
  const  float* __restrict__ vec,
  const    int* __restrict__ mat,
         float* __restrict__ mul,
  const  float* __restrict__ lookup_table,
  int height,
  int width
);

__global__ void VecQuant4MatMulKernelNUQPerChannel_fp16(
  const  float* __restrict__ vec,
  const    int* __restrict__ mat,
         float* __restrict__ mul,
  const  float* __restrict__ lookup_table,
  int height,
  int width
);

__global__ void VecQuant3MatMulKernelNUQPerChannelBatched_fp16(
  const  float* __restrict__ vec,
  const    int* __restrict__ mat,
         float* __restrict__ mul,
  const  float* __restrict__ lookup_table,
  int height,
  int width,
  int batch,
  int vec_height
);

__global__ void VecQuant4MatMulKernelNUQPerChannelBatched_fp16(
  const  float* __restrict__ vec,
  const    int* __restrict__ mat,
         float* __restrict__ mul,
  const  float* __restrict__ lookup_table,
  int height,
  int width,
  int batch,
  int vec_height
);

template <typename scalar_t>
__global__ void SPMV_ATOMIC_fp16(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows
);

template <typename scalar_t>
__global__ void SPMV_ATOMIC_BATCHED_fp16(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows,
  int batch,
  int vec_height
);

__global__ void DenseMatVecKernel_fp16(
  const  float* __restrict__ vec,
  const  float* __restrict__ full_rows,
  const  int* __restrict__ full_row_indices,
         float* __restrict__ mul,
  int topX,
  int full_width
);

__global__ void DenseMatVecKernelBatched_fp16(
  const  float* __restrict__ vec,
  const  float* __restrict__ full_rows,
  const  int* __restrict__ full_row_indices,
         float* __restrict__ mul,
  int topX,
  int full_width,
  int batch,
  int vec_height,
  int matwidth
);

/**********************************************************************/
// FP8

// __global__ void VecQuant3MatMulKernelNUQPerChannel_fp8(
//   const  float* __restrict__ vec,
//   const    int* __restrict__ mat,
//          float* __restrict__ mul,
//   const  float* __restrict__ lookup_table,
//   int height,
//   int width
// );

// __global__ void VecQuant4MatMulKernelNUQPerChannel_fp8(
//   const  float* __restrict__ vec,
//   const    int* __restrict__ mat,
//          float* __restrict__ mul,
//   const  float* __restrict__ lookup_table,
//   int height,
//   int width
// );

// __global__ void VecQuant3MatMulKernelNUQPerChannelBatched_fp8(
//   const  float* __restrict__ vec,
//   const    int* __restrict__ mat,
//          float* __restrict__ mul,
//   const  float* __restrict__ lookup_table,
//   int height,
//   int width,
//   int batch,
//   int vec_height
// );

// __global__ void VecQuant4MatMulKernelNUQPerChannelBatched_fp8(
//   const  float* __restrict__ vec,
//   const    int* __restrict__ mat,
//          float* __restrict__ mul,
//   const  float* __restrict__ lookup_table,
//   int height,
//   int width,
//   int batch,
//   int vec_height
// );

// template <typename scalar_t>
// __global__ void SPMV_ATOMIC_fp8(
//   const       int* __restrict__ rows,
//   const       int* __restrict__ cols,
//   const  scalar_t* __restrict__ mat,
//   const  scalar_t* __restrict__ vec,
//          scalar_t* __restrict__ mul,
//   const  int num_rows
// );

// template <typename scalar_t>
// __global__ void SPMV_ATOMIC_BATCHED_fp8(
//   const       int* __restrict__ rows,
//   const       int* __restrict__ cols,
//   const  scalar_t* __restrict__ mat,
//   const  scalar_t* __restrict__ vec,
//          scalar_t* __restrict__ mul,
//   const  int num_rows,
//   int batch,
//   int vec_height
// );

// __global__ void DenseMatVecKernel_fp8(
//   const  float* __restrict__ vec,
//   const  float* __restrict__ full_rows,
//   const  int* __restrict__ full_row_indices,
//          float* __restrict__ mul,
//   int topX,
//   int full_width
// );

// __global__ void DenseMatVecKernelBatched_fp8(
//   const  float* __restrict__ vec,
//   const  float* __restrict__ full_rows,
//   const  int* __restrict__ full_row_indices,
//          float* __restrict__ mul,
//   int topX,
//   int full_width,
//   int batch,
//   int vec_height,
//   int matwidth
// );

/**********************************************************************/
// BFP16

__global__ void VecQuant3MatMulKernelNUQPerChannel_bfp16(
  const  float* __restrict__ vec,
  const    int* __restrict__ mat,
         float* __restrict__ mul,
  const  float* __restrict__ lookup_table,
  int height,
  int width
);

__global__ void VecQuant4MatMulKernelNUQPerChannel_bfp16(
  const  float* __restrict__ vec,
  const    int* __restrict__ mat,
         float* __restrict__ mul,
  const  float* __restrict__ lookup_table,
  int height,
  int width
);

__global__ void VecQuant3MatMulKernelNUQPerChannelBatched_bfp16(
  const  float* __restrict__ vec,
  const    int* __restrict__ mat,
         float* __restrict__ mul,
  const  float* __restrict__ lookup_table,
  int height,
  int width,
  int batch,
  int vec_height
);

__global__ void VecQuant4MatMulKernelNUQPerChannelBatched_bfp16(
  const  float* __restrict__ vec,
  const    int* __restrict__ mat,
         float* __restrict__ mul,
  const  float* __restrict__ lookup_table,
  int height,
  int width,
  int batch,
  int vec_height
);

template <typename scalar_t>
__global__ void SPMV_ATOMIC_bfp16(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows
);

template <typename scalar_t>
__global__ void SPMV_ATOMIC_BATCHED_bfp16(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows,
  int batch,
  int vec_height
);

__global__ void DenseMatVecKernel_bfp16(
  const  float* __restrict__ vec,
  const  float* __restrict__ full_rows,
  const  int* __restrict__ full_row_indices,
         float* __restrict__ mul,
  int topX,
  int full_width
);

__global__ void DenseMatVecKernelBatched_bfp16(
  const  float* __restrict__ vec,
  const  float* __restrict__ full_rows,
  const  int* __restrict__ full_row_indices,
         float* __restrict__ mul,
  int topX,
  int full_width,
  int batch,
  int vec_height,
  int matwidth
);

/**********************************************************************/
// CUSTOM

__global__ void VecQuant3MatMulKernelNUQPerChannel_custom(
  const  float* __restrict__ vec,
  const    int* __restrict__ mat,
         float* __restrict__ mul,
  const  float* __restrict__ lookup_table,
  int height,
  int width
);

__global__ void VecQuant4MatMulKernelNUQPerChannel_custom(
  const  float* __restrict__ vec,
  const    int* __restrict__ mat,
         float* __restrict__ mul,
  const  float* __restrict__ lookup_table,
  int height,
  int width
);

__global__ void VecQuant3MatMulKernelNUQPerChannelBatched_custom(
  const  float* __restrict__ vec,
  const    int* __restrict__ mat,
         float* __restrict__ mul,
  const  float* __restrict__ lookup_table,
  int height,
  int width,
  int batch,
  int vec_height
);

__global__ void VecQuant4MatMulKernelNUQPerChannelBatched_custom(
  const  float* __restrict__ vec,
  const    int* __restrict__ mat,
         float* __restrict__ mul,
  const  float* __restrict__ lookup_table,
  int height,
  int width,
  int batch,
  int vec_height
);

template <typename scalar_t>
__global__ void SPMV_ATOMIC_custom(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows
);

template <typename scalar_t>
__global__ void SPMV_ATOMIC_BATCHED_custom(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows,
  int batch,
  int vec_height
);

__global__ void DenseMatVecKernel_custom(
  const  float* __restrict__ vec,
  const  float* __restrict__ full_rows,
  const  int* __restrict__ full_row_indices,
         float* __restrict__ mul,
  int topX,
  int full_width
);

__global__ void DenseMatVecKernelBatched_custom(
  const  float* __restrict__ vec,
  const  float* __restrict__ full_rows,
  const  int* __restrict__ full_row_indices,
         float* __restrict__ mul,
  int topX,
  int full_width,
  int batch,
  int vec_height,
  int matwidth
);

/**********************************************************************/
// FP32

void vecquant3matmul_nuq_perchannel_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannel<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width
  );
}

// 4-bit matvec kernel (LUT-based)
void vecquant4matmul_nuq_perchannel_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannel<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width
  );
}

// 3-bit batched matvec kernel (LUT-based)
void vecquant3matmul_nuq_perchannel_batched_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelBatched<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, batch, vec_height
  );
}

// 4-bit batched matvec kernel (LUT-based)
void vecquant4matmul_nuq_perchannel_batched_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelBatched<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, batch, vec_height
  );
}

//NUQ + Sparse
void vecquant3matmul_spmv_nuq_perchannel_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  VecQuant3MatMulKernelNUQPerChannel<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3
  );
}

//NUQ + Sparse
void vecquant4matmul_spmv_nuq_perchannel_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  VecQuant4MatMulKernelNUQPerChannel<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4
  );
}


//NUQ + Sparse
void vecquant3matmul_spmv_nuq_perchannel_batched_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelBatched<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );

}

//NUQ + Sparse
void vecquant4matmul_spmv_nuq_perchannel_batched_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelBatched<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );
}


//NUQ + hybrid sparse kernel
void vecquant3matmul_spmv_hybrid_nuq_perchannel_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannel<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  //dense matvec kernel here!
  DenseMatVecKernel<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width
  );

}


//NUQ + hybrid sparse kernel
void vecquant4matmul_spmv_hybrid_nuq_perchannel_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannel<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  //dense matvec kernel here!
  DenseMatVecKernel<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width
  );

}

//NUQ + hybrid sparse kernel
void vecquant3matmul_spmv_hybrid_nuq_perchannel_batched_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelBatched<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  int matwidth = mul.size(1);

  //dense matvec kernel here!
  DenseMatVecKernelBatched<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width,
    batch,
    vec_height,
    matwidth
  );

}


//NUQ + hybrid sparse kernel
void vecquant4matmul_spmv_hybrid_nuq_perchannel_batched_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelBatched<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  int matwidth = mul.size(1);

  //dense matvec kernel here!
  DenseMatVecKernelBatched<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width,
    batch,
    vec_height,
    matwidth
  );

}


__global__ void VecQuant3MatMulKernelNUQPerChannel(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
) {

  // BLOCKHEIGHT3 = 12 = BLOCKWIDTH / 32 * 3
  int row = BLOCKHEIGHT3 * blockIdx.x;
  // BLOCKWIDTH = 128
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT3) * BLOCKWIDTH + threadIdx.x];

  //Modified dequant block
  __shared__ float deq2[8][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = (((row / 3) * 32) + off) * 8;
  // Current row number is (row / 3) * 32

  for (int val = 0; val < 8; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }
  // There are BLOCKWIDTH (128) columns in deq2
  // Each column are the centroid values needed for each element in the BLOCKWIDTH elements of blockvec

  int i = width * row + col;
  int k = 0;

  float res = 0;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  __syncthreads();

  // The following calculation procedure always processes three mat[i] values at once
  // Therefore, we can always be sure that the first mat[i] always is synchronized with the 3-bit granularity!!
  // 32 bits * 3 = 96 bits --> basically, calculating 32 elements at a time!!
  while (k < BLOCKWIDTH) {
    tmp1 = as_unsigned(mat[i]);

    res += deq2[(tmp1 >>  0) & 0x7][k + 0] * blockvec[k + 0];
    res += deq2[(tmp1 >>  3) & 0x7][k + 1] * blockvec[k + 1];
    res += deq2[(tmp1 >>  6) & 0x7][k + 2] * blockvec[k + 2];
    res += deq2[(tmp1 >>  9) & 0x7][k + 3] * blockvec[k + 3];
    res += deq2[(tmp1 >>  12) & 0x7][k + 4] * blockvec[k + 4];
    res += deq2[(tmp1 >>  15) & 0x7][k + 5] * blockvec[k + 5];
    res += deq2[(tmp1 >>  18) & 0x7][k + 6] * blockvec[k + 6];
    res += deq2[(tmp1 >>  21) & 0x7][k + 7] * blockvec[k + 7];
    res += deq2[(tmp1 >>  24) & 0x7][k + 8] * blockvec[k + 8];
    res += deq2[(tmp1 >>  27) & 0x7][k + 9] * blockvec[k + 9];

    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
    tmp2 >>= 1;
    res += deq2[(tmp >>  0) & 0x7][k + 10] * blockvec[k + 10];
    k += 11;
    res += deq2[(tmp2 >>  0) & 0x7][k + 0] * blockvec[k + 0];
    res += deq2[(tmp2 >>  3) & 0x7][k + 1] * blockvec[k + 1];
    res += deq2[(tmp2 >>  6) & 0x7][k + 2] * blockvec[k + 2];
    res += deq2[(tmp2 >>  9) & 0x7][k + 3] * blockvec[k + 3];
    res += deq2[(tmp2 >>  12) & 0x7][k + 4] * blockvec[k + 4];
    res += deq2[(tmp2 >>  15) & 0x7][k + 5] * blockvec[k + 5];
    res += deq2[(tmp2 >>  18) & 0x7][k + 6] * blockvec[k + 6];
    res += deq2[(tmp2 >>  21) & 0x7][k + 7] * blockvec[k + 7];
    res += deq2[(tmp2 >>  24) & 0x7][k + 8] * blockvec[k + 8];
    res += deq2[(tmp2 >>  27) & 0x7][k + 9] * blockvec[k + 9];

    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
    tmp1 >>= 2;
    res += deq2[(tmp >>  0) & 0x7][k + 10] * blockvec[k + 10];
    k += 11;
    res += deq2[(tmp1 >>  0) & 0x7][k + 0] * blockvec[k + 0];
    res += deq2[(tmp1 >>  3) & 0x7][k + 1] * blockvec[k + 1];
    res += deq2[(tmp1 >>  6) & 0x7][k + 2] * blockvec[k + 2];
    res += deq2[(tmp1 >>  9) & 0x7][k + 3] * blockvec[k + 3];
    res += deq2[(tmp1 >>  12) & 0x7][k + 4] * blockvec[k + 4];
    res += deq2[(tmp1 >>  15) & 0x7][k + 5] * blockvec[k + 5];
    res += deq2[(tmp1 >>  18) & 0x7][k + 6] * blockvec[k + 6];
    res += deq2[(tmp1 >>  21) & 0x7][k + 7] * blockvec[k + 7];
    res += deq2[(tmp1 >>  24) & 0x7][k + 8] * blockvec[k + 8];
    res += deq2[(tmp1 >>  27) & 0x7][k + 9] * blockvec[k + 9];
    i += width;
    k += 10;
  }

  atomicAdd(&mul[col], res);
}

//4-bit per-channel
__global__ void VecQuant4MatMulKernelNUQPerChannel(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
) {

  // BLOCKHEIGHT4 = 16 = BLOCKWIDTH / 32 * 4
  int row = BLOCKHEIGHT4 * blockIdx.x;
  int col =  BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT4) * BLOCKWIDTH + threadIdx.x];

  //Modified dequant block
  __shared__ float deq2[16][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = (((row / 4) * 32) + off) * 16;
  for (int val = 0; val < 16; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

  __syncthreads();

  float res = 0;
  int i = width * row + col;
  int k = 0;

  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp = as_unsigned(mat[i]);

    res += deq2[(tmp >>  0) & 0xf][k + 0] * blockvec[k + 0];
    res += deq2[(tmp >>  4) & 0xf][k + 1] * blockvec[k + 1];
    res += deq2[(tmp >>  8) & 0xf][k + 2] * blockvec[k + 2];
    res += deq2[(tmp >>  12) & 0xf][k + 3] * blockvec[k + 3];
    res += deq2[(tmp >>  16) & 0xf][k + 4] * blockvec[k + 4];
    res += deq2[(tmp >>  20) & 0xf][k + 5] * blockvec[k + 5];
    res += deq2[(tmp >>  24) & 0xf][k + 6] * blockvec[k + 6];
    res += deq2[(tmp >>  28) & 0xf][k + 7] * blockvec[k + 7];

    i += width;
    k += 8;
  }

  atomicAdd(&mul[col], res);
}


//batched version (3-bit)
__global__ void VecQuant3MatMulKernelNUQPerChannelBatched(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
) {

  int row = BLOCKHEIGHT3 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];

  __shared__ float deq2[8][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = (((row / 3) * 32) + off) * 8;
  // Current row number is (row / 3) * 32
  for (int val = 0; val < 8; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

  int i;
  float res;
  int k;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  for (int b = 0; b < batch; ++b){
    //initialize vars
    i = width * row + col;
    res = 0;
    k = 0;

    __syncthreads();
    blockvec[threadIdx.x] = vec[b * vec_height + (row / BLOCKHEIGHT3) * BLOCKWIDTH + threadIdx.x];
    __syncthreads();

    while (k < BLOCKWIDTH) {
      tmp1 = as_unsigned(mat[i]);

      res += deq2[(tmp1 >>  0) & 0x7][k + 0] * blockvec[k + 0];
      res += deq2[(tmp1 >>  3) & 0x7][k + 1] * blockvec[k + 1];
      res += deq2[(tmp1 >>  6) & 0x7][k + 2] * blockvec[k + 2];
      res += deq2[(tmp1 >>  9) & 0x7][k + 3] * blockvec[k + 3];
      res += deq2[(tmp1 >>  12) & 0x7][k + 4] * blockvec[k + 4];
      res += deq2[(tmp1 >>  15) & 0x7][k + 5] * blockvec[k + 5];
      res += deq2[(tmp1 >>  18) & 0x7][k + 6] * blockvec[k + 6];
      res += deq2[(tmp1 >>  21) & 0x7][k + 7] * blockvec[k + 7];
      res += deq2[(tmp1 >>  24) & 0x7][k + 8] * blockvec[k + 8];
      res += deq2[(tmp1 >>  27) & 0x7][k + 9] * blockvec[k + 9];

      i += width;
      tmp2 = as_unsigned(mat[i]);
      tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
      tmp2 >>= 1;
      res += deq2[(tmp >>  0) & 0x7][k + 10] * blockvec[k + 10];
      k += 11;
      res += deq2[(tmp2 >>  0) & 0x7][k + 0] * blockvec[k + 0];
      res += deq2[(tmp2 >>  3) & 0x7][k + 1] * blockvec[k + 1];
      res += deq2[(tmp2 >>  6) & 0x7][k + 2] * blockvec[k + 2];
      res += deq2[(tmp2 >>  9) & 0x7][k + 3] * blockvec[k + 3];
      res += deq2[(tmp2 >>  12) & 0x7][k + 4] * blockvec[k + 4];
      res += deq2[(tmp2 >>  15) & 0x7][k + 5] * blockvec[k + 5];
      res += deq2[(tmp2 >>  18) & 0x7][k + 6] * blockvec[k + 6];
      res += deq2[(tmp2 >>  21) & 0x7][k + 7] * blockvec[k + 7];
      res += deq2[(tmp2 >>  24) & 0x7][k + 8] * blockvec[k + 8];
      res += deq2[(tmp2 >>  27) & 0x7][k + 9] * blockvec[k + 9];

      i += width;
      tmp1 = as_unsigned(mat[i]);
      tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
      tmp1 >>= 2;
      res += deq2[(tmp >>  0) & 0x7][k + 10] * blockvec[k + 10];
      k += 11;
      res += deq2[(tmp1 >>  0) & 0x7][k + 0] * blockvec[k + 0];
      res += deq2[(tmp1 >>  3) & 0x7][k + 1] * blockvec[k + 1];
      res += deq2[(tmp1 >>  6) & 0x7][k + 2] * blockvec[k + 2];
      res += deq2[(tmp1 >>  9) & 0x7][k + 3] * blockvec[k + 3];
      res += deq2[(tmp1 >>  12) & 0x7][k + 4] * blockvec[k + 4];
      res += deq2[(tmp1 >>  15) & 0x7][k + 5] * blockvec[k + 5];
      res += deq2[(tmp1 >>  18) & 0x7][k + 6] * blockvec[k + 6];
      res += deq2[(tmp1 >>  21) & 0x7][k + 7] * blockvec[k + 7];
      res += deq2[(tmp1 >>  24) & 0x7][k + 8] * blockvec[k + 8];
      res += deq2[(tmp1 >>  27) & 0x7][k + 9] * blockvec[k + 9];
      i += width;
      k += 10;
    }

    atomicAdd(&mul[b * width + col], res);
  }
}

//batched version (4-bit)
__global__ void VecQuant4MatMulKernelNUQPerChannelBatched(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
) {

  int row = BLOCKHEIGHT4 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  __shared__ float blockvec[BLOCKWIDTH];

  //Modified dequant block
  __shared__ float deq2[16][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = (((row / 4) * 32) + off) * 16;
  for (int val = 0; val < 16; val += 1) {
    int lut_index = row_offset + (val & 0xf);
    deq2[val][off] = lookup_table[lut_index];
  }

  int i;
  float res;
  int k;
  unsigned int tmp;

  for (int b = 0; b < batch; ++b){
    i = width * row + col;
    res = 0;
    k = 0;

    __syncthreads();
    blockvec[threadIdx.x] = vec[b * vec_height + (row / BLOCKHEIGHT4) * BLOCKWIDTH + threadIdx.x];
    __syncthreads();

    while (k < BLOCKWIDTH) {
      tmp = as_unsigned(mat[i]);

      res += deq2[(tmp >>  0) & 0xf][k + 0] * blockvec[k + 0];
      res += deq2[(tmp >>  4) & 0xf][k + 1] * blockvec[k + 1];
      res += deq2[(tmp >>  8) & 0xf][k + 2] * blockvec[k + 2];
      res += deq2[(tmp >>  12) & 0xf][k + 3] * blockvec[k + 3];
      res += deq2[(tmp >>  16) & 0xf][k + 4] * blockvec[k + 4];
      res += deq2[(tmp >>  20) & 0xf][k + 5] * blockvec[k + 5];
      res += deq2[(tmp >>  24) & 0xf][k + 6] * blockvec[k + 6];
      res += deq2[(tmp >>  28) & 0xf][k + 7] * blockvec[k + 7];

      i += width;
      k += 8;
    }

    atomicAdd(&mul[b * width + col], res);
  }
}

template <typename scalar_t>
__global__ void SPMV_ATOMIC(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0;
        int start_elem = rows[row];
        int end_elem = rows[row+1];
        for (int i = start_elem; i < end_elem; i++) {
            dot += mat[i] * vec[cols[i]];
        }
        atomicAdd(&mul[row], dot);
    }
}

template <typename scalar_t>
__global__ void SPMV_ATOMIC_BATCHED(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows,
  int batch,
  int vec_height
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        int start_elem = rows[row];
        int end_elem = rows[row+1];
        for (int b = 0; b < batch; ++b){
            float dot = 0;
            for (int i = start_elem; i < end_elem; i++) {
                dot += mat[i] * vec[b * vec_height + cols[i]];
                // dot += mat[i] * vec[cols[i] * batch + b];
                // dot += mat[i] * vec[cols[i]];
            }
            atomicAdd(&mul[b * num_rows + row], dot);
            // atomicAdd(&mul[row * batch + b], dot);
            // atomicAdd(&mul[row], dot);
        }
    }
}

// Dense kernel for only a subset of rows
__global__ void DenseMatVecKernel(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int height,
    int width
) {

  int row = BLOCKWIDTH * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[row + threadIdx.x];

  __syncthreads();

  int i = width * row + col;
  int k = 0;
  float res = 0;

  if (threadIdx.x < width) {
    while (k < BLOCKWIDTH) {
      res += full_rows[i] * blockvec[k];
      k += 1;
      i += width;
    }

    int col_idx = full_row_indices[col];
    atomicAdd(&mul[col_idx], res);
  }
}


// Dense kernel for only a subset of rows
__global__ void DenseMatVecKernelBatched(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int height,
    int width,
    int batch,
    int vec_height,
    int matwidth
) {

  int row = BLOCKWIDTH * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];

  for (int b = 0; b < batch; ++b){
    int i = width * row + col;
    int k = 0;
    float res = 0;

    __syncthreads();
    blockvec[threadIdx.x] = vec[b * vec_height + row + threadIdx.x];
    __syncthreads();

    if (threadIdx.x < width) {
      while (k < BLOCKWIDTH) {
        res += full_rows[i] * blockvec[k];
        k += 1;
        i += width;
      }

      int col_idx = full_row_indices[col];
      atomicAdd(&mul[b * matwidth + col_idx], res);
    }
  }
}

/**********************************************************************/
// FP16

void vecquant3matmul_nuq_perchannel_fp16_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannel_fp16<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width
  );
}

// 4-bit matvec kernel (LUT-based)
void vecquant4matmul_nuq_perchannel_fp16_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannel_fp16<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width
  );
}

// 3-bit batched matvec kernel (LUT-based)
void vecquant3matmul_nuq_perchannel_batched_fp16_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelBatched_fp16<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, batch, vec_height
  );
}

// 4-bit batched matvec kernel (LUT-based)
void vecquant4matmul_nuq_perchannel_batched_fp16_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelBatched_fp16<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, batch, vec_height
  );
}

//NUQ + Sparse
void vecquant3matmul_spmv_nuq_perchannel_fp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC_fp16<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  VecQuant3MatMulKernelNUQPerChannel_fp16<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3
  );
}

//NUQ + Sparse
void vecquant4matmul_spmv_nuq_perchannel_fp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC_fp16<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  VecQuant4MatMulKernelNUQPerChannel_fp16<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4
  );
}


//NUQ + Sparse
void vecquant3matmul_spmv_nuq_perchannel_batched_fp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelBatched_fp16<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED_fp16<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );

}

//NUQ + Sparse
void vecquant4matmul_spmv_nuq_perchannel_batched_fp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelBatched_fp16<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED_fp16<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );
}


//NUQ + hybrid sparse kernel
void vecquant3matmul_spmv_hybrid_nuq_perchannel_fp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannel_fp16<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC_fp16<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  //dense matvec kernel here!
  DenseMatVecKernel_fp16<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width
  );

}


//NUQ + hybrid sparse kernel
void vecquant4matmul_spmv_hybrid_nuq_perchannel_fp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannel_fp16<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC_fp16<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  //dense matvec kernel here!
  DenseMatVecKernel_fp16<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width
  );

}

//NUQ + hybrid sparse kernel
void vecquant3matmul_spmv_hybrid_nuq_perchannel_batched_fp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelBatched_fp16<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED_fp16<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  int matwidth = mul.size(1);

  //dense matvec kernel here!
  DenseMatVecKernelBatched_fp16<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width,
    batch,
    vec_height,
    matwidth
  );

}


//NUQ + hybrid sparse kernel
void vecquant4matmul_spmv_hybrid_nuq_perchannel_batched_fp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelBatched_fp16<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED_fp16<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  int matwidth = mul.size(1);

  //dense matvec kernel here!
  DenseMatVecKernelBatched_fp16<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width,
    batch,
    vec_height,
    matwidth
  );

}

#define HMUL_FLOATS(a, b) (__half2float(__hmul(__float2half(a), __float2half(b))))

__global__ void VecQuant3MatMulKernelNUQPerChannel_fp16(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
) {

  // BLOCKHEIGHT3 = 12 = BLOCKWIDTH / 32 * 3
  int row = BLOCKHEIGHT3 * blockIdx.x;
  // BLOCKWIDTH = 128
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT3) * BLOCKWIDTH + threadIdx.x];

  //Modified dequant block
  __shared__ float deq2[8][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = (((row / 3) * 32) + off) * 8;
  // Current row number is (row / 3) * 32

  for (int val = 0; val < 8; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }
  // There are BLOCKWIDTH (128) columns in deq2
  // Each column are the centroid values needed for each element in the BLOCKWIDTH elements of blockvec

  int i = width * row + col;
  int k = 0;

  float res = 0;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  __syncthreads();

  // The following calculation procedure always processes three mat[i] values at once
  // Therefore, we can always be sure that the first mat[i] always is synchronized with the 3-bit granularity!!
  // 32 bits * 3 = 96 bits --> basically, calculating 32 elements at a time!!
  while (k < BLOCKWIDTH) {
    tmp1 = as_unsigned(mat[i]);

    res += HMUL_FLOATS(deq2[(tmp1 >>  0) & 0x7][k + 0], blockvec[k + 0]);
    res += HMUL_FLOATS(deq2[(tmp1 >>  3) & 0x7][k + 1], blockvec[k + 1]);
    res += HMUL_FLOATS(deq2[(tmp1 >>  6) & 0x7][k + 2], blockvec[k + 2]);
    res += HMUL_FLOATS(deq2[(tmp1 >>  9) & 0x7][k + 3], blockvec[k + 3]);
    res += HMUL_FLOATS(deq2[(tmp1 >>  12) & 0x7][k + 4], blockvec[k + 4]);
    res += HMUL_FLOATS(deq2[(tmp1 >>  15) & 0x7][k + 5], blockvec[k + 5]);
    res += HMUL_FLOATS(deq2[(tmp1 >>  18) & 0x7][k + 6], blockvec[k + 6]);
    res += HMUL_FLOATS(deq2[(tmp1 >>  21) & 0x7][k + 7], blockvec[k + 7]);
    res += HMUL_FLOATS(deq2[(tmp1 >>  24) & 0x7][k + 8], blockvec[k + 8]);
    res += HMUL_FLOATS(deq2[(tmp1 >>  27) & 0x7][k + 9], blockvec[k + 9]);

    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
    tmp2 >>= 1;
    res += HMUL_FLOATS(deq2[(tmp >>  0) & 0x7][k + 10], blockvec[k + 10]);
    k += 11;
    res += HMUL_FLOATS(deq2[(tmp2 >>  0) & 0x7][k + 0], blockvec[k + 0]);
    res += HMUL_FLOATS(deq2[(tmp2 >>  3) & 0x7][k + 1], blockvec[k + 1]);
    res += HMUL_FLOATS(deq2[(tmp2 >>  6) & 0x7][k + 2], blockvec[k + 2]);
    res += HMUL_FLOATS(deq2[(tmp2 >>  9) & 0x7][k + 3], blockvec[k + 3]);
    res += HMUL_FLOATS(deq2[(tmp2 >>  12) & 0x7][k + 4], blockvec[k + 4]);
    res += HMUL_FLOATS(deq2[(tmp2 >>  15) & 0x7][k + 5], blockvec[k + 5]);
    res += HMUL_FLOATS(deq2[(tmp2 >>  18) & 0x7][k + 6], blockvec[k + 6]);
    res += HMUL_FLOATS(deq2[(tmp2 >>  21) & 0x7][k + 7], blockvec[k + 7]);
    res += HMUL_FLOATS(deq2[(tmp2 >>  24) & 0x7][k + 8], blockvec[k + 8]);
    res += HMUL_FLOATS(deq2[(tmp2 >>  27) & 0x7][k + 9], blockvec[k + 9]);

    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
    tmp1 >>= 2;
    res += HMUL_FLOATS(deq2[(tmp >>  0) & 0x7][k + 10], blockvec[k + 10]);
    k += 11;
    res += HMUL_FLOATS(deq2[(tmp1 >>  0) & 0x7][k + 0], blockvec[k + 0]);
    res += HMUL_FLOATS(deq2[(tmp1 >>  3) & 0x7][k + 1], blockvec[k + 1]);
    res += HMUL_FLOATS(deq2[(tmp1 >>  6) & 0x7][k + 2], blockvec[k + 2]);
    res += HMUL_FLOATS(deq2[(tmp1 >>  9) & 0x7][k + 3], blockvec[k + 3]);
    res += HMUL_FLOATS(deq2[(tmp1 >>  12) & 0x7][k + 4], blockvec[k + 4]);
    res += HMUL_FLOATS(deq2[(tmp1 >>  15) & 0x7][k + 5], blockvec[k + 5]);
    res += HMUL_FLOATS(deq2[(tmp1 >>  18) & 0x7][k + 6], blockvec[k + 6]);
    res += HMUL_FLOATS(deq2[(tmp1 >>  21) & 0x7][k + 7], blockvec[k + 7]);
    res += HMUL_FLOATS(deq2[(tmp1 >>  24) & 0x7][k + 8], blockvec[k + 8]);
    res += HMUL_FLOATS(deq2[(tmp1 >>  27) & 0x7][k + 9], blockvec[k + 9]);
    i += width;
    k += 10;
  }

  atomicAdd(&mul[col], res);
}

//4-bit per-channel
__global__ void VecQuant4MatMulKernelNUQPerChannel_fp16(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
) {

  // BLOCKHEIGHT4 = 16 = BLOCKWIDTH / 32 * 4
  int row = BLOCKHEIGHT4 * blockIdx.x;
  int col =  BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT4) * BLOCKWIDTH + threadIdx.x];

  //Modified dequant block
  __shared__ float deq2[16][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = (((row / 4) * 32) + off) * 16;
  for (int val = 0; val < 16; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

  __syncthreads();

  float res = 0;
  int i = width * row + col;
  int k = 0;

  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp = as_unsigned(mat[i]);

    res += HMUL_FLOATS(deq2[(tmp >>  0) & 0xf][k + 0], blockvec[k + 0]);
    res += HMUL_FLOATS(deq2[(tmp >>  4) & 0xf][k + 1], blockvec[k + 1]);
    res += HMUL_FLOATS(deq2[(tmp >>  8) & 0xf][k + 2], blockvec[k + 2]);
    res += HMUL_FLOATS(deq2[(tmp >>  12) & 0xf][k + 3], blockvec[k + 3]);
    res += HMUL_FLOATS(deq2[(tmp >>  16) & 0xf][k + 4], blockvec[k + 4]);
    res += HMUL_FLOATS(deq2[(tmp >>  20) & 0xf][k + 5], blockvec[k + 5]);
    res += HMUL_FLOATS(deq2[(tmp >>  24) & 0xf][k + 6], blockvec[k + 6]);
    res += HMUL_FLOATS(deq2[(tmp >>  28) & 0xf][k + 7], blockvec[k + 7]);

    i += width;
    k += 8;
  }

  atomicAdd(&mul[col], res);
}


//batched version (3-bit)
__global__ void VecQuant3MatMulKernelNUQPerChannelBatched_fp16(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
) {

  int row = BLOCKHEIGHT3 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];

  __shared__ float deq2[8][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = (((row / 3) * 32) + off) * 8;
  // Current row number is (row / 3) * 32
  for (int val = 0; val < 8; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

  int i;
  float res;
  int k;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  for (int b = 0; b < batch; ++b){
    //initialize vars
    i = width * row + col;
    res = 0;
    k = 0;

    __syncthreads();
    blockvec[threadIdx.x] = vec[b * vec_height + (row / BLOCKHEIGHT3) * BLOCKWIDTH + threadIdx.x];
    __syncthreads();

    while (k < BLOCKWIDTH) {
      tmp1 = as_unsigned(mat[i]);

      res += HMUL_FLOATS(deq2[(tmp1 >>  0) & 0x7][k + 0], blockvec[k + 0]);
      res += HMUL_FLOATS(deq2[(tmp1 >>  3) & 0x7][k + 1], blockvec[k + 1]);
      res += HMUL_FLOATS(deq2[(tmp1 >>  6) & 0x7][k + 2], blockvec[k + 2]);
      res += HMUL_FLOATS(deq2[(tmp1 >>  9) & 0x7][k + 3], blockvec[k + 3]);
      res += HMUL_FLOATS(deq2[(tmp1 >>  12) & 0x7][k + 4], blockvec[k + 4]);
      res += HMUL_FLOATS(deq2[(tmp1 >>  15) & 0x7][k + 5], blockvec[k + 5]);
      res += HMUL_FLOATS(deq2[(tmp1 >>  18) & 0x7][k + 6], blockvec[k + 6]);
      res += HMUL_FLOATS(deq2[(tmp1 >>  21) & 0x7][k + 7], blockvec[k + 7]);
      res += HMUL_FLOATS(deq2[(tmp1 >>  24) & 0x7][k + 8], blockvec[k + 8]);
      res += HMUL_FLOATS(deq2[(tmp1 >>  27) & 0x7][k + 9], blockvec[k + 9]);

      i += width;
      tmp2 = as_unsigned(mat[i]);
      tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
      tmp2 >>= 1;
      res += HMUL_FLOATS(deq2[(tmp >>  0) & 0x7][k + 10], blockvec[k + 10]);
      k += 11;
      res += HMUL_FLOATS(deq2[(tmp2 >>  0) & 0x7][k + 0], blockvec[k + 0]);
      res += HMUL_FLOATS(deq2[(tmp2 >>  3) & 0x7][k + 1], blockvec[k + 1]);
      res += HMUL_FLOATS(deq2[(tmp2 >>  6) & 0x7][k + 2], blockvec[k + 2]);
      res += HMUL_FLOATS(deq2[(tmp2 >>  9) & 0x7][k + 3], blockvec[k + 3]);
      res += HMUL_FLOATS(deq2[(tmp2 >>  12) & 0x7][k + 4], blockvec[k + 4]);
      res += HMUL_FLOATS(deq2[(tmp2 >>  15) & 0x7][k + 5], blockvec[k + 5]);
      res += HMUL_FLOATS(deq2[(tmp2 >>  18) & 0x7][k + 6], blockvec[k + 6]);
      res += HMUL_FLOATS(deq2[(tmp2 >>  21) & 0x7][k + 7], blockvec[k + 7]);
      res += HMUL_FLOATS(deq2[(tmp2 >>  24) & 0x7][k + 8], blockvec[k + 8]);
      res += HMUL_FLOATS(deq2[(tmp2 >>  27) & 0x7][k + 9], blockvec[k + 9]);

      i += width;
      tmp1 = as_unsigned(mat[i]);
      tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
      tmp1 >>= 2;
      res += HMUL_FLOATS(deq2[(tmp >>  0) & 0x7][k + 10], blockvec[k + 10]);
      k += 11;
      res += HMUL_FLOATS(deq2[(tmp1 >>  0) & 0x7][k + 0], blockvec[k + 0]);
      res += HMUL_FLOATS(deq2[(tmp1 >>  3) & 0x7][k + 1], blockvec[k + 1]);
      res += HMUL_FLOATS(deq2[(tmp1 >>  6) & 0x7][k + 2], blockvec[k + 2]);
      res += HMUL_FLOATS(deq2[(tmp1 >>  9) & 0x7][k + 3], blockvec[k + 3]);
      res += HMUL_FLOATS(deq2[(tmp1 >>  12) & 0x7][k + 4], blockvec[k + 4]);
      res += HMUL_FLOATS(deq2[(tmp1 >>  15) & 0x7][k + 5], blockvec[k + 5]);
      res += HMUL_FLOATS(deq2[(tmp1 >>  18) & 0x7][k + 6], blockvec[k + 6]);
      res += HMUL_FLOATS(deq2[(tmp1 >>  21) & 0x7][k + 7], blockvec[k + 7]);
      res += HMUL_FLOATS(deq2[(tmp1 >>  24) & 0x7][k + 8], blockvec[k + 8]);
      res += HMUL_FLOATS(deq2[(tmp1 >>  27) & 0x7][k + 9], blockvec[k + 9]);
      i += width;
      k += 10;
    }

    atomicAdd(&mul[b * width + col], res);
  }
}

//batched version (4-bit)
__global__ void VecQuant4MatMulKernelNUQPerChannelBatched_fp16(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
) {

  int row = BLOCKHEIGHT4 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  __shared__ float blockvec[BLOCKWIDTH];

  //Modified dequant block
  __shared__ float deq2[16][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = (((row / 4) * 32) + off) * 16;
  for (int val = 0; val < 16; val += 1) {
    int lut_index = row_offset + (val & 0xf);
    deq2[val][off] = lookup_table[lut_index];
  }

  int i;
  float res;
  int k;
  unsigned int tmp;

  for (int b = 0; b < batch; ++b){
    i = width * row + col;
    res = 0;
    k = 0;

    __syncthreads();
    blockvec[threadIdx.x] = vec[b * vec_height + (row / BLOCKHEIGHT4) * BLOCKWIDTH + threadIdx.x];
    __syncthreads();

    while (k < BLOCKWIDTH) {
      tmp = as_unsigned(mat[i]);

      res += HMUL_FLOATS(deq2[(tmp >>  0) & 0xf][k + 0], blockvec[k + 0]);
      res += HMUL_FLOATS(deq2[(tmp >>  4) & 0xf][k + 1], blockvec[k + 1]);
      res += HMUL_FLOATS(deq2[(tmp >>  8) & 0xf][k + 2], blockvec[k + 2]);
      res += HMUL_FLOATS(deq2[(tmp >>  12) & 0xf][k + 3], blockvec[k + 3]);
      res += HMUL_FLOATS(deq2[(tmp >>  16) & 0xf][k + 4], blockvec[k + 4]);
      res += HMUL_FLOATS(deq2[(tmp >>  20) & 0xf][k + 5], blockvec[k + 5]);
      res += HMUL_FLOATS(deq2[(tmp >>  24) & 0xf][k + 6], blockvec[k + 6]);
      res += HMUL_FLOATS(deq2[(tmp >>  28) & 0xf][k + 7], blockvec[k + 7]);

      i += width;
      k += 8;
    }

    atomicAdd(&mul[b * width + col], res);
  }
}

template <typename scalar_t>
__global__ void SPMV_ATOMIC_fp16(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0;
        int start_elem = rows[row];
        int end_elem = rows[row+1];
        for (int i = start_elem; i < end_elem; i++) {
            dot += HMUL_FLOATS(mat[i], vec[cols[i]]);
        }
        atomicAdd(&mul[row], dot);
    }
}

template <typename scalar_t>
__global__ void SPMV_ATOMIC_BATCHED_fp16(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows,
  int batch,
  int vec_height
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        int start_elem = rows[row];
        int end_elem = rows[row+1];
        for (int b = 0; b < batch; ++b){
            float dot = 0;
            for (int i = start_elem; i < end_elem; i++) {
                dot +=HMUL_FLOATS(mat[i], vec[b * vec_height + cols[i]]);
                // dot += mat[i] * vec[cols[i] * batch + b];
                // dot += mat[i] * vec[cols[i]];
            }
            atomicAdd(&mul[b * num_rows + row], dot);
            // atomicAdd(&mul[row * batch + b], dot);
            // atomicAdd(&mul[row], dot);
        }
    }
}

// Dense kernel for only a subset of rows
__global__ void DenseMatVecKernel_fp16(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int height,
    int width
) {

  int row = BLOCKWIDTH * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[row + threadIdx.x];

  __syncthreads();

  int i = width * row + col;
  int k = 0;
  float res = 0;

  if (threadIdx.x < width) {
    while (k < BLOCKWIDTH) {
      res += HMUL_FLOATS(full_rows[i], blockvec[k]);
      k += 1;
      i += width;
    }

    int col_idx = full_row_indices[col];
    atomicAdd(&mul[col_idx], res);
  }
}


// Dense kernel for only a subset of rows
__global__ void DenseMatVecKernelBatched_fp16(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int height,
    int width,
    int batch,
    int vec_height,
    int matwidth
) {

  int row = BLOCKWIDTH * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];

  for (int b = 0; b < batch; ++b){
    int i = width * row + col;
    int k = 0;
    float res = 0;

    __syncthreads();
    blockvec[threadIdx.x] = vec[b * vec_height + row + threadIdx.x];
    __syncthreads();

    if (threadIdx.x < width) {
      while (k < BLOCKWIDTH) {
        res += HMUL_FLOATS(full_rows[i], blockvec[k]);
        k += 1;
        i += width;
      }

      int col_idx = full_row_indices[col];
      atomicAdd(&mul[b * matwidth + col_idx], res);
    }
  }
}

/**********************************************************************/
// BFP16

void vecquant3matmul_nuq_perchannel_bfp16_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannel_bfp16<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width
  );
}

// 4-bit matvec kernel (LUT-based)
void vecquant4matmul_nuq_perchannel_bfp16_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannel_bfp16<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width
  );
}

// 3-bit batched matvec kernel (LUT-based)
void vecquant3matmul_nuq_perchannel_batched_bfp16_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelBatched_bfp16<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, batch, vec_height
  );
}

// 4-bit batched matvec kernel (LUT-based)
void vecquant4matmul_nuq_perchannel_batched_bfp16_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelBatched_bfp16<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, batch, vec_height
  );
}

//NUQ + Sparse
void vecquant3matmul_spmv_nuq_perchannel_bfp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC_bfp16<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  VecQuant3MatMulKernelNUQPerChannel_bfp16<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3
  );
}

//NUQ + Sparse
void vecquant4matmul_spmv_nuq_perchannel_bfp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC_bfp16<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  VecQuant4MatMulKernelNUQPerChannel_bfp16<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4
  );
}


//NUQ + Sparse
void vecquant3matmul_spmv_nuq_perchannel_batched_bfp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelBatched_bfp16<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED_bfp16<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );

}

//NUQ + Sparse
void vecquant4matmul_spmv_nuq_perchannel_batched_bfp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelBatched_bfp16<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED_bfp16<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );
}


//NUQ + hybrid sparse kernel
void vecquant3matmul_spmv_hybrid_nuq_perchannel_bfp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannel_bfp16<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC_bfp16<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  //dense matvec kernel here!
  DenseMatVecKernel_bfp16<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width
  );

}


//NUQ + hybrid sparse kernel
void vecquant4matmul_spmv_hybrid_nuq_perchannel_bfp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannel_bfp16<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC_bfp16<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  //dense matvec kernel here!
  DenseMatVecKernel_bfp16<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width
  );

}

//NUQ + hybrid sparse kernel
void vecquant3matmul_spmv_hybrid_nuq_perchannel_batched_bfp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelBatched_bfp16<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED_bfp16<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  int matwidth = mul.size(1);

  //dense matvec kernel here!
  DenseMatVecKernelBatched_bfp16<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width,
    batch,
    vec_height,
    matwidth
  );

}


//NUQ + hybrid sparse kernel
void vecquant4matmul_spmv_hybrid_nuq_perchannel_batched_bfp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelBatched_bfp16<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED_bfp16<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  int matwidth = mul.size(1);

  //dense matvec kernel here!
  DenseMatVecKernelBatched_bfp16<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width,
    batch,
    vec_height,
    matwidth
  );

}

__device__ __nv_bfloat16 __bmul(__nv_bfloat16 a, __nv_bfloat16 b) {
  unsigned short* aAsInt = reinterpret_cast<unsigned short*>(&a);
  unsigned short* bAsInt = reinterpret_cast<unsigned short*>(&b);

  unsigned short sign_a = *aAsInt >> 15;
  unsigned short exponent_a = (*aAsInt >> 7) & 0xFF;
  unsigned short mantissa_a = *aAsInt & 0x7F;
  if(exponent_a > 0) {
    mantissa_a |= 0x80;
  }

  unsigned short sign_b = *bAsInt >> 15;
  unsigned short exponent_b = (*bAsInt >> 7) & 0xFF;
  unsigned short mantissa_b = *bAsInt & 0x7F;
  if(exponent_b > 0) {
    mantissa_b |= 0x80;
  }
  
  // Combine the components to form the bfloat16 representation
  unsigned short result = 0;
  result |= (sign_a ^ sign_b) << 15;
  // Calculate product of mantissas
  unsigned int product = (unsigned int)mantissa_a * (unsigned int)mantissa_b;

  if (exponent_a > 0 && exponent_b > 0) {
    // Handle exponent bias
    short exponent_result = exponent_a + exponent_b - 127;
    short carry = (product >> 15) & 0x1;
    exponent_result += carry;

    if (exponent_result <= 0) {
      // Number tooo small, record exponent as zero
      short shift = 8 + carry - exponent_result;
      result |= (unsigned short)((product >> shift) & 0x7F);  // Adjust the shift to fit into 7 bits
    }
    else {
      result |= (exponent_result & 0xFF) << 7;
      unsigned short shift = 7 + carry;
      result |= (unsigned short)((product >> shift) & 0x7F);  // Adjust the shift to fit into 7 bits
    }
  } else if (exponent_a > 0 || exponent_b > 0) {
    short exponent_result = exponent_a + exponent_b - 126;
    short even_smaller = __clz(product) - 17;
    if (exponent_result - even_smaller <= 0) {
      if (exponent_result > 0) {
        product = product << exponent_result;
        exponent_result = 0;
      }
      short shift = 8 - exponent_result;
      result |= (unsigned short)((product >> shift) & 0x7F);  // Adjust the shift to fit into 7 bits
    } else {
      exponent_result -= even_smaller;
      result |= (exponent_result & 0xFF) << 7;
      product = product << even_smaller;
      short shift = 7;
      result |= (unsigned short)((product >> shift) & 0x7F);  // Adjust the shift to fit into 7 bits
    }
  }
  
  return *reinterpret_cast<__nv_bfloat16*>(&result);
}

#define BMUL_FLOATS(a, b) (__bfloat162float(__bmul(__float2bfloat16(a), __float2bfloat16(b))))

__global__ void VecQuant3MatMulKernelNUQPerChannel_bfp16(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
) {

  // BLOCKHEIGHT3 = 12 = BLOCKWIDTH / 32 * 3
  int row = BLOCKHEIGHT3 * blockIdx.x;
  // BLOCKWIDTH = 128
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT3) * BLOCKWIDTH + threadIdx.x];

  //Modified dequant block
  __shared__ float deq2[8][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = (((row / 3) * 32) + off) * 8;
  // Current row number is (row / 3) * 32

  for (int val = 0; val < 8; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }
  // There are BLOCKWIDTH (128) columns in deq2
  // Each column are the centroid values needed for each element in the BLOCKWIDTH elements of blockvec

  int i = width * row + col;
  int k = 0;

  float res = 0;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  __syncthreads();

  // The following calculation procedure always processes three mat[i] values at once
  // Therefore, we can always be sure that the first mat[i] always is synchronized with the 3-bit granularity!!
  // 32 bits * 3 = 96 bits --> basically, calculating 32 elements at a time!!
  while (k < BLOCKWIDTH) {
    tmp1 = as_unsigned(mat[i]);

    res += BMUL_FLOATS(deq2[(tmp1 >>  0) & 0x7][k + 0], blockvec[k + 0]);
    res += BMUL_FLOATS(deq2[(tmp1 >>  3) & 0x7][k + 1], blockvec[k + 1]);
    res += BMUL_FLOATS(deq2[(tmp1 >>  6) & 0x7][k + 2], blockvec[k + 2]);
    res += BMUL_FLOATS(deq2[(tmp1 >>  9) & 0x7][k + 3], blockvec[k + 3]);
    res += BMUL_FLOATS(deq2[(tmp1 >>  12) & 0x7][k + 4], blockvec[k + 4]);
    res += BMUL_FLOATS(deq2[(tmp1 >>  15) & 0x7][k + 5], blockvec[k + 5]);
    res += BMUL_FLOATS(deq2[(tmp1 >>  18) & 0x7][k + 6], blockvec[k + 6]);
    res += BMUL_FLOATS(deq2[(tmp1 >>  21) & 0x7][k + 7], blockvec[k + 7]);
    res += BMUL_FLOATS(deq2[(tmp1 >>  24) & 0x7][k + 8], blockvec[k + 8]);
    res += BMUL_FLOATS(deq2[(tmp1 >>  27) & 0x7][k + 9], blockvec[k + 9]);

    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
    tmp2 >>= 1;
    res += BMUL_FLOATS(deq2[(tmp >>  0) & 0x7][k + 10], blockvec[k + 10]);
    k += 11;
    res += BMUL_FLOATS(deq2[(tmp2 >>  0) & 0x7][k + 0], blockvec[k + 0]);
    res += BMUL_FLOATS(deq2[(tmp2 >>  3) & 0x7][k + 1], blockvec[k + 1]);
    res += BMUL_FLOATS(deq2[(tmp2 >>  6) & 0x7][k + 2], blockvec[k + 2]);
    res += BMUL_FLOATS(deq2[(tmp2 >>  9) & 0x7][k + 3], blockvec[k + 3]);
    res += BMUL_FLOATS(deq2[(tmp2 >>  12) & 0x7][k + 4], blockvec[k + 4]);
    res += BMUL_FLOATS(deq2[(tmp2 >>  15) & 0x7][k + 5], blockvec[k + 5]);
    res += BMUL_FLOATS(deq2[(tmp2 >>  18) & 0x7][k + 6], blockvec[k + 6]);
    res += BMUL_FLOATS(deq2[(tmp2 >>  21) & 0x7][k + 7], blockvec[k + 7]);
    res += BMUL_FLOATS(deq2[(tmp2 >>  24) & 0x7][k + 8], blockvec[k + 8]);
    res += BMUL_FLOATS(deq2[(tmp2 >>  27) & 0x7][k + 9], blockvec[k + 9]);

    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
    tmp1 >>= 2;
    res += BMUL_FLOATS(deq2[(tmp >>  0) & 0x7][k + 10], blockvec[k + 10]);
    k += 11;
    res += BMUL_FLOATS(deq2[(tmp1 >>  0) & 0x7][k + 0], blockvec[k + 0]);
    res += BMUL_FLOATS(deq2[(tmp1 >>  3) & 0x7][k + 1], blockvec[k + 1]);
    res += BMUL_FLOATS(deq2[(tmp1 >>  6) & 0x7][k + 2], blockvec[k + 2]);
    res += BMUL_FLOATS(deq2[(tmp1 >>  9) & 0x7][k + 3], blockvec[k + 3]);
    res += BMUL_FLOATS(deq2[(tmp1 >>  12) & 0x7][k + 4], blockvec[k + 4]);
    res += BMUL_FLOATS(deq2[(tmp1 >>  15) & 0x7][k + 5], blockvec[k + 5]);
    res += BMUL_FLOATS(deq2[(tmp1 >>  18) & 0x7][k + 6], blockvec[k + 6]);
    res += BMUL_FLOATS(deq2[(tmp1 >>  21) & 0x7][k + 7], blockvec[k + 7]);
    res += BMUL_FLOATS(deq2[(tmp1 >>  24) & 0x7][k + 8], blockvec[k + 8]);
    res += BMUL_FLOATS(deq2[(tmp1 >>  27) & 0x7][k + 9], blockvec[k + 9]);
    i += width;
    k += 10;
  }

  atomicAdd(&mul[col], res);
}

//4-bit per-channel
__global__ void VecQuant4MatMulKernelNUQPerChannel_bfp16(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
) {

  // BLOCKHEIGHT4 = 16 = BLOCKWIDTH / 32 * 4
  int row = BLOCKHEIGHT4 * blockIdx.x;
  int col =  BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT4) * BLOCKWIDTH + threadIdx.x];

  //Modified dequant block
  __shared__ float deq2[16][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = (((row / 4) * 32) + off) * 16;
  for (int val = 0; val < 16; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

  __syncthreads();

  float res = 0;
  int i = width * row + col;
  int k = 0;

  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp = as_unsigned(mat[i]);

    res += BMUL_FLOATS(deq2[(tmp >>  0) & 0xf][k + 0], blockvec[k + 0]);
    res += BMUL_FLOATS(deq2[(tmp >>  4) & 0xf][k + 1], blockvec[k + 1]);
    res += BMUL_FLOATS(deq2[(tmp >>  8) & 0xf][k + 2], blockvec[k + 2]);
    res += BMUL_FLOATS(deq2[(tmp >>  12) & 0xf][k + 3], blockvec[k + 3]);
    res += BMUL_FLOATS(deq2[(tmp >>  16) & 0xf][k + 4], blockvec[k + 4]);
    res += BMUL_FLOATS(deq2[(tmp >>  20) & 0xf][k + 5], blockvec[k + 5]);
    res += BMUL_FLOATS(deq2[(tmp >>  24) & 0xf][k + 6], blockvec[k + 6]);
    res += BMUL_FLOATS(deq2[(tmp >>  28) & 0xf][k + 7], blockvec[k + 7]);

    i += width;
    k += 8;
  }

  atomicAdd(&mul[col], res);
}


//batched version (3-bit)
__global__ void VecQuant3MatMulKernelNUQPerChannelBatched_bfp16(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
) {

  int row = BLOCKHEIGHT3 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];

  __shared__ float deq2[8][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = (((row / 3) * 32) + off) * 8;
  // Current row number is (row / 3) * 32
  for (int val = 0; val < 8; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

  int i;
  float res;
  int k;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  for (int b = 0; b < batch; ++b){
    //initialize vars
    i = width * row + col;
    res = 0;
    k = 0;

    __syncthreads();
    blockvec[threadIdx.x] = vec[b * vec_height + (row / BLOCKHEIGHT3) * BLOCKWIDTH + threadIdx.x];
    __syncthreads();

    while (k < BLOCKWIDTH) {
      tmp1 = as_unsigned(mat[i]);

      res += BMUL_FLOATS(deq2[(tmp1 >>  0) & 0x7][k + 0], blockvec[k + 0]);
      res += BMUL_FLOATS(deq2[(tmp1 >>  3) & 0x7][k + 1], blockvec[k + 1]);
      res += BMUL_FLOATS(deq2[(tmp1 >>  6) & 0x7][k + 2], blockvec[k + 2]);
      res += BMUL_FLOATS(deq2[(tmp1 >>  9) & 0x7][k + 3], blockvec[k + 3]);
      res += BMUL_FLOATS(deq2[(tmp1 >>  12) & 0x7][k + 4], blockvec[k + 4]);
      res += BMUL_FLOATS(deq2[(tmp1 >>  15) & 0x7][k + 5], blockvec[k + 5]);
      res += BMUL_FLOATS(deq2[(tmp1 >>  18) & 0x7][k + 6], blockvec[k + 6]);
      res += BMUL_FLOATS(deq2[(tmp1 >>  21) & 0x7][k + 7], blockvec[k + 7]);
      res += BMUL_FLOATS(deq2[(tmp1 >>  24) & 0x7][k + 8], blockvec[k + 8]);
      res += BMUL_FLOATS(deq2[(tmp1 >>  27) & 0x7][k + 9], blockvec[k + 9]);

      i += width;
      tmp2 = as_unsigned(mat[i]);
      tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
      tmp2 >>= 1;
      res += BMUL_FLOATS(deq2[(tmp >>  0) & 0x7][k + 10], blockvec[k + 10]);
      k += 11;
      res += BMUL_FLOATS(deq2[(tmp2 >>  0) & 0x7][k + 0], blockvec[k + 0]);
      res += BMUL_FLOATS(deq2[(tmp2 >>  3) & 0x7][k + 1], blockvec[k + 1]);
      res += BMUL_FLOATS(deq2[(tmp2 >>  6) & 0x7][k + 2], blockvec[k + 2]);
      res += BMUL_FLOATS(deq2[(tmp2 >>  9) & 0x7][k + 3], blockvec[k + 3]);
      res += BMUL_FLOATS(deq2[(tmp2 >>  12) & 0x7][k + 4], blockvec[k + 4]);
      res += BMUL_FLOATS(deq2[(tmp2 >>  15) & 0x7][k + 5], blockvec[k + 5]);
      res += BMUL_FLOATS(deq2[(tmp2 >>  18) & 0x7][k + 6], blockvec[k + 6]);
      res += BMUL_FLOATS(deq2[(tmp2 >>  21) & 0x7][k + 7], blockvec[k + 7]);
      res += BMUL_FLOATS(deq2[(tmp2 >>  24) & 0x7][k + 8], blockvec[k + 8]);
      res += BMUL_FLOATS(deq2[(tmp2 >>  27) & 0x7][k + 9], blockvec[k + 9]);

      i += width;
      tmp1 = as_unsigned(mat[i]);
      tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
      tmp1 >>= 2;
      res += BMUL_FLOATS(deq2[(tmp >>  0) & 0x7][k + 10], blockvec[k + 10]);
      k += 11;
      res += BMUL_FLOATS(deq2[(tmp1 >>  0) & 0x7][k + 0], blockvec[k + 0]);
      res += BMUL_FLOATS(deq2[(tmp1 >>  3) & 0x7][k + 1], blockvec[k + 1]);
      res += BMUL_FLOATS(deq2[(tmp1 >>  6) & 0x7][k + 2], blockvec[k + 2]);
      res += BMUL_FLOATS(deq2[(tmp1 >>  9) & 0x7][k + 3], blockvec[k + 3]);
      res += BMUL_FLOATS(deq2[(tmp1 >>  12) & 0x7][k + 4], blockvec[k + 4]);
      res += BMUL_FLOATS(deq2[(tmp1 >>  15) & 0x7][k + 5], blockvec[k + 5]);
      res += BMUL_FLOATS(deq2[(tmp1 >>  18) & 0x7][k + 6], blockvec[k + 6]);
      res += BMUL_FLOATS(deq2[(tmp1 >>  21) & 0x7][k + 7], blockvec[k + 7]);
      res += BMUL_FLOATS(deq2[(tmp1 >>  24) & 0x7][k + 8], blockvec[k + 8]);
      res += BMUL_FLOATS(deq2[(tmp1 >>  27) & 0x7][k + 9], blockvec[k + 9]);
      i += width;
      k += 10;
    }

    atomicAdd(&mul[b * width + col], res);
  }
}

//batched version (4-bit)
__global__ void VecQuant4MatMulKernelNUQPerChannelBatched_bfp16(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
) {

  int row = BLOCKHEIGHT4 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  __shared__ float blockvec[BLOCKWIDTH];

  //Modified dequant block
  __shared__ float deq2[16][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = (((row / 4) * 32) + off) * 16;
  for (int val = 0; val < 16; val += 1) {
    int lut_index = row_offset + (val & 0xf);
    deq2[val][off] = lookup_table[lut_index];
  }

  int i;
  float res;
  int k;
  unsigned int tmp;

  for (int b = 0; b < batch; ++b){
    i = width * row + col;
    res = 0;
    k = 0;

    __syncthreads();
    blockvec[threadIdx.x] = vec[b * vec_height + (row / BLOCKHEIGHT4) * BLOCKWIDTH + threadIdx.x];
    __syncthreads();

    while (k < BLOCKWIDTH) {
      tmp = as_unsigned(mat[i]);

      res += BMUL_FLOATS(deq2[(tmp >>  0) & 0xf][k + 0], blockvec[k + 0]);
      res += BMUL_FLOATS(deq2[(tmp >>  4) & 0xf][k + 1], blockvec[k + 1]);
      res += BMUL_FLOATS(deq2[(tmp >>  8) & 0xf][k + 2], blockvec[k + 2]);
      res += BMUL_FLOATS(deq2[(tmp >>  12) & 0xf][k + 3], blockvec[k + 3]);
      res += BMUL_FLOATS(deq2[(tmp >>  16) & 0xf][k + 4], blockvec[k + 4]);
      res += BMUL_FLOATS(deq2[(tmp >>  20) & 0xf][k + 5], blockvec[k + 5]);
      res += BMUL_FLOATS(deq2[(tmp >>  24) & 0xf][k + 6], blockvec[k + 6]);
      res += BMUL_FLOATS(deq2[(tmp >>  28) & 0xf][k + 7], blockvec[k + 7]);

      i += width;
      k += 8;
    }

    atomicAdd(&mul[b * width + col], res);
  }
}

template <typename scalar_t>
__global__ void SPMV_ATOMIC_bfp16(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0;
        int start_elem = rows[row];
        int end_elem = rows[row+1];
        for (int i = start_elem; i < end_elem; i++) {
            dot += BMUL_FLOATS(mat[i], vec[cols[i]]);
        }
        atomicAdd(&mul[row], dot);
    }
}

template <typename scalar_t>
__global__ void SPMV_ATOMIC_BATCHED_bfp16(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows,
  int batch,
  int vec_height
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        int start_elem = rows[row];
        int end_elem = rows[row+1];
        for (int b = 0; b < batch; ++b){
            float dot = 0;
            for (int i = start_elem; i < end_elem; i++) {
                dot +=BMUL_FLOATS(mat[i], vec[b * vec_height + cols[i]]);
                // dot += mat[i] * vec[cols[i] * batch + b];
                // dot += mat[i] * vec[cols[i]];
            }
            atomicAdd(&mul[b * num_rows + row], dot);
            // atomicAdd(&mul[row * batch + b], dot);
            // atomicAdd(&mul[row], dot);
        }
    }
}

// Dense kernel for only a subset of rows
__global__ void DenseMatVecKernel_bfp16(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int height,
    int width
) {

  int row = BLOCKWIDTH * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[row + threadIdx.x];

  __syncthreads();

  int i = width * row + col;
  int k = 0;
  float res = 0;

  if (threadIdx.x < width) {
    while (k < BLOCKWIDTH) {
      res += BMUL_FLOATS(full_rows[i], blockvec[k]);
      k += 1;
      i += width;
    }

    int col_idx = full_row_indices[col];
    atomicAdd(&mul[col_idx], res);
  }
}


// Dense kernel for only a subset of rows
__global__ void DenseMatVecKernelBatched_bfp16(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int height,
    int width,
    int batch,
    int vec_height,
    int matwidth
) {

  int row = BLOCKWIDTH * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];

  for (int b = 0; b < batch; ++b){
    int i = width * row + col;
    int k = 0;
    float res = 0;

    __syncthreads();
    blockvec[threadIdx.x] = vec[b * vec_height + row + threadIdx.x];
    __syncthreads();

    if (threadIdx.x < width) {
      while (k < BLOCKWIDTH) {
        res += BMUL_FLOATS(full_rows[i], blockvec[k]);
        k += 1;
        i += width;
      }

      int col_idx = full_row_indices[col];
      atomicAdd(&mul[b * matwidth + col_idx], res);
    }
  }
}

/**********************************************************************/
// CUSTOM


void vecquant3matmul_nuq_perchannel_custom_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannel_custom<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width
  );
}

// 4-bit matvec kernel (LUT-based)
void vecquant4matmul_nuq_perchannel_custom_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannel_custom<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width
  );
}

// 3-bit batched matvec kernel (LUT-based)
void vecquant3matmul_nuq_perchannel_batched_custom_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelBatched_custom<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, batch, vec_height
  );
}

// 4-bit batched matvec kernel (LUT-based)
void vecquant4matmul_nuq_perchannel_batched_custom_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelBatched_custom<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, batch, vec_height
  );
}

//NUQ + Sparse
void vecquant3matmul_spmv_nuq_perchannel_custom_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC_custom<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  VecQuant3MatMulKernelNUQPerChannel_custom<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3
  );
}

//NUQ + Sparse
void vecquant4matmul_spmv_nuq_perchannel_custom_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC_custom<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  VecQuant4MatMulKernelNUQPerChannel_custom<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4
  );
}


//NUQ + Sparse
void vecquant3matmul_spmv_nuq_perchannel_batched_custom_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelBatched_custom<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED_custom<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );

}

//NUQ + Sparse
void vecquant4matmul_spmv_nuq_perchannel_batched_custom_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelBatched_custom<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED_custom<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );
}


//NUQ + hybrid sparse kernel
void vecquant3matmul_spmv_hybrid_nuq_perchannel_custom_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannel_custom<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC_custom<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  //dense matvec kernel here!
  DenseMatVecKernel_custom<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width
  );

}


//NUQ + hybrid sparse kernel
void vecquant4matmul_spmv_hybrid_nuq_perchannel_custom_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannel_custom<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic", ([&] {
      SPMV_ATOMIC_custom<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  //dense matvec kernel here!
  DenseMatVecKernel_custom<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width
  );

}

//NUQ + hybrid sparse kernel
void vecquant3matmul_spmv_hybrid_nuq_perchannel_batched_custom_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks3(
    (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads3(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelBatched_custom<<<blocks3, threads3>>>(
    vec.data_ptr<float>(),
    mat3.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height3, width3, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED_custom<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  int matwidth = mul.size(1);

  //dense matvec kernel here!
  DenseMatVecKernelBatched_custom<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width,
    batch,
    vec_height,
    matwidth
  );

}


//NUQ + hybrid sparse kernel
void vecquant4matmul_spmv_hybrid_nuq_perchannel_batched_custom_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor full_rows,
  torch::Tensor full_row_indices,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  dim3 blocks4(
    (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads4(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelBatched_custom<<<blocks4, threads4>>>(
    vec.data_ptr<float>(),
    mat4.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height4, width4, batch, vec_height
  );

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  AT_DISPATCH_FLOATING_TYPES(
    mat.type(), "spmv_atomic_batched", ([&] {
      SPMV_ATOMIC_BATCHED_custom<<<num_blocks, block_size>>>(
        rows.data<int>(),
        cols.data<int>(),
        mat.data<scalar_t>(),
        vec.data<scalar_t>(),
        mul.data<scalar_t>(),
        num_rows,
        batch,
        vec_height
      );
    })
  );

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  dim3 blocks_topX(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads_topX(BLOCKWIDTH);

  int matwidth = mul.size(1);

  //dense matvec kernel here!
  DenseMatVecKernelBatched_custom<<<blocks_topX, threads_topX>>>(
    vec.data_ptr<float>(),
    full_rows.data_ptr<float>(),
    full_row_indices.data_ptr<int>(),
    mul.data_ptr<float>(),
    height,
    width,
    batch,
    vec_height,
    matwidth
  );

}

// CONFIGURATION: 1BIT SIGN + C_EXPONENT BITS + C_MANTISSA BITS
// ASSUME RIGHT ALIGN
#define C_EXPONENT 5
#define C_MANTISSA 10

__device__ unsigned int __cmul(unsigned int a, unsigned int b) {
  unsigned int EXP_MASK = (1 << C_EXPONENT) - 1;
  unsigned int MAN_MASK = (1 << C_MANTISSA) - 1;
  int EXP_BIAS = (1 << (C_EXPONENT - 1)) - 1;
  
  unsigned int sign_a = a >> (C_EXPONENT + C_MANTISSA);
  unsigned int exponent_a = (a >> C_MANTISSA) & EXP_MASK;
  unsigned int mantissa_a = a & MAN_MASK;
  if(exponent_a > 0) {
    mantissa_a |= (1 << C_MANTISSA);
  }

  unsigned short sign_b = b >> (C_EXPONENT + C_MANTISSA);
  unsigned short exponent_b = (b >> C_MANTISSA) & EXP_MASK;
  unsigned short mantissa_b = b & MAN_MASK;
  if(exponent_b > 0) {
    mantissa_b |= (1 << C_MANTISSA);
  }

  // Combine the components to form the custom representation
  unsigned int result = 0;
  result |= (sign_a ^ sign_b) << (C_EXPONENT + C_MANTISSA);
  // Calculate product of mantissas
  unsigned int product = (unsigned int)mantissa_a * (unsigned int)mantissa_b;
  
  if (exponent_a > 0 && exponent_b > 0) {
    // Handle exponent bias
    int exponent_result = exponent_a + exponent_b - EXP_BIAS;
    short carry = (product >> ((2 * C_MANTISSA) + 1)) & 0x1;
    exponent_result += carry;

    if (exponent_result <= 0) {
      // Number tooo small, record exponent as zero
      int shift = C_MANTISSA + 1 + carry - exponent_result;
      result |= (unsigned short)((product >> shift) & MAN_MASK); 
    }
    else {
      result |= (exponent_result & EXP_MASK) << C_MANTISSA;
      int shift = C_MANTISSA + carry;
      result |= (unsigned short)((product >> shift) & MAN_MASK);
    }
  } else if (exponent_a > 0 || exponent_b > 0) {
    short exponent_result = exponent_a + exponent_b - (EXP_BIAS - 1);
    short even_smaller = __clz(product) - (32 - 2 * C_MANTISSA - 1);
    if (exponent_result - even_smaller <= 0) {
      if (exponent_result > 0) {
        product = product << exponent_result;
        exponent_result = 0;
      }
      short shift = C_MANTISSA + 1 - exponent_result;
      result |= (unsigned short)((product >> shift) & MAN_MASK);
    } else {
      exponent_result -= even_smaller;
      result |= (exponent_result & EXP_MASK) << C_MANTISSA;
      product = product << even_smaller;
      short shift = C_MANTISSA;
      result |= (unsigned short)((product >> shift) & MAN_MASK);
    }
  }
  return result;
}

__device__ unsigned int float2custom(float x) {
  unsigned int EXP_MASK = (1 << 8) - 1;
  unsigned int MAN_MASK = (1 << 23) - 1;
  // This is also equivalent to the largest possible exponent value
  unsigned int C_EXP_MASK = (1 << C_EXPONENT) - 1; 
  unsigned int C_MAN_MASK = (1 << C_MANTISSA) - 1;
  unsigned int C_EXP_BIAS = (1 << (C_EXPONENT - 1)) - 1;

  unsigned int a = __float_as_uint(x);
  unsigned int sign = a >> 31;
  unsigned int exponent = ((a >> 23) & EXP_MASK);
  unsigned int mantissa = a & MAN_MASK;

  unsigned int result = 0;
  result |= sign << (C_EXPONENT + C_MANTISSA);

  // When converting to a larger range, need to be careful when exponent = 0
  if (C_EXPONENT > 8 && exponent == 0) {
    if (mantissa == 0) {
      // if (threadIdx.x == 0)
      //   printf("Float -> Custom, %x, %x\n", a, result);
      return __uint_as_float(result);
    } 
    exponent = -126 + C_EXP_BIAS;
    while (exponent > 0 && (mantissa >> (C_MANTISSA - 1)) == 0) {
      exponent -= 1;
      mantissa = (mantissa << 1) & MAN_MASK;
    }
    if (exponent == 0) {
      if (C_MANTISSA <= 23) result |= mantissa >> (23 - C_MANTISSA);
      else result |= mantissa << (C_MANTISSA - 23);
    } else {
      exponent -= 1;
      mantissa = (mantissa << 1) & MAN_MASK;
      result |= exponent << C_MANTISSA;
      if (C_MANTISSA <= 23) result |= mantissa >> (23 - C_MANTISSA);
      else result |= mantissa << (C_MANTISSA - 23);
    }
  }
  else {
    exponent = exponent - 127 + C_EXP_BIAS;
    if (exponent < C_EXP_MASK && exponent > 0) {
      result |= exponent << C_MANTISSA;
      if (C_MANTISSA <= 23) result |= mantissa >> (23 - C_MANTISSA);
      else result |= mantissa << (C_MANTISSA - 23);
    } else if (exponent <= 0) {
      if (C_MANTISSA <= 23) result |= mantissa >> (23 - C_MANTISSA);
      else result |= mantissa << (C_MANTISSA - 23);
    } else {
      result |= C_EXP_MASK << C_MANTISSA;
    }
  }
  // if (threadIdx.x == 0)
  //   printf("Float -> Custom, %x, %x\n", a, result);
  return result;
}

__device__ float custom2float(unsigned int a) {
  unsigned int EXP_MASK = (1 << 8) - 1;
  unsigned int MAN_MASK = (1 << 23) - 1;
  // This is also equivalent to the largest possible exponent value
  unsigned int C_EXP_MASK = (1 << C_EXPONENT) - 1; 
  unsigned int C_MAN_MASK = (1 << C_MANTISSA) - 1;
  unsigned int C_EXP_BIAS = (1 << (C_EXPONENT - 1)) - 1;

  unsigned int sign = a >> (C_EXPONENT + C_MANTISSA);
  unsigned int exponent = ((a >> C_MANTISSA) & C_EXP_MASK);
  unsigned int mantissa = a & C_MAN_MASK;\
  unsigned int result = 0;
  result |= sign << 31;

  // When converting to a larger range, need to be careful when exponent = 0
  if (8 > C_EXPONENT && exponent == 0) {
    if (mantissa == 0) {
      // if (threadIdx.x == 0)
      //   printf("Custom -> Float, %x, %x\n", a, result);
      return __uint_as_float(result);
    }
    exponent = 1 - C_EXP_BIAS + 127;
    while (exponent > 0 && (mantissa >> (C_MANTISSA - 1)) == 0) {
      exponent -= 1;
      mantissa = (mantissa << 1) & C_MAN_MASK;
    }
    if (exponent == 0) {
      if (C_MANTISSA <= 23) result |= mantissa << (23 - C_MANTISSA);
      else result |= mantissa >> (C_MANTISSA - 23);
    } else {
      exponent -= 1;
      mantissa = (mantissa << 1) & C_MAN_MASK;
      result |= exponent << 23;
      if (C_MANTISSA <= 23) result |= mantissa << (23 - C_MANTISSA);
      else result |= mantissa >> (C_MANTISSA - 23);
    }
  }
  else {
    exponent = exponent - C_EXP_BIAS + 127;
    if (exponent < EXP_MASK && exponent > 0) {
      result |= exponent << 23;
      if (C_MANTISSA <= 23) result |= mantissa << (23 - C_MANTISSA);
      else result |= mantissa >> (C_MANTISSA - 23);
    } else if (exponent <= 0) {
      if (C_MANTISSA <= 23) result |= mantissa << (23 - C_MANTISSA);
      else result |= mantissa >> (C_MANTISSA - 23);
    } else {
      result |= EXP_MASK << 23;
    }
  }
  // if (threadIdx.x == 0)
  //   printf("Custom -> Float, %x, %x\n", a, result);
  return __uint_as_float(result);
}

#define CMUL_FLOATS(a, b) custom2float(__cmul(float2custom(a), float2custom(b)))

__global__ void VecQuant3MatMulKernelNUQPerChannel_custom(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
) {

  // BLOCKHEIGHT3 = 12 = BLOCKWIDTH / 32 * 3
  int row = BLOCKHEIGHT3 * blockIdx.x;
  // BLOCKWIDTH = 128
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT3) * BLOCKWIDTH + threadIdx.x];

  //Modified dequant block
  __shared__ float deq2[8][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = (((row / 3) * 32) + off) * 8;
  // Current row number is (row / 3) * 32

  for (int val = 0; val < 8; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }
  // There are BLOCKWIDTH (128) columns in deq2
  // Each column are the centroid values needed for each element in the BLOCKWIDTH elements of blockvec

  int i = width * row + col;
  int k = 0;

  float res = 0;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  __syncthreads();

  // The following calculation procedure always processes three mat[i] values at once
  // Therefore, we can always be sure that the first mat[i] always is synchronized with the 3-bit granularity!!
  // 32 bits * 3 = 96 bits --> basically, calculating 32 elements at a time!!
  while (k < BLOCKWIDTH) {
    tmp1 = as_unsigned(mat[i]);

    res += CMUL_FLOATS(deq2[(tmp1 >>  0) & 0x7][k + 0], blockvec[k + 0]);
    res += CMUL_FLOATS(deq2[(tmp1 >>  3) & 0x7][k + 1], blockvec[k + 1]);
    res += CMUL_FLOATS(deq2[(tmp1 >>  6) & 0x7][k + 2], blockvec[k + 2]);
    res += CMUL_FLOATS(deq2[(tmp1 >>  9) & 0x7][k + 3], blockvec[k + 3]);
    res += CMUL_FLOATS(deq2[(tmp1 >>  12) & 0x7][k + 4], blockvec[k + 4]);
    res += CMUL_FLOATS(deq2[(tmp1 >>  15) & 0x7][k + 5], blockvec[k + 5]);
    res += CMUL_FLOATS(deq2[(tmp1 >>  18) & 0x7][k + 6], blockvec[k + 6]);
    res += CMUL_FLOATS(deq2[(tmp1 >>  21) & 0x7][k + 7], blockvec[k + 7]);
    res += CMUL_FLOATS(deq2[(tmp1 >>  24) & 0x7][k + 8], blockvec[k + 8]);
    res += CMUL_FLOATS(deq2[(tmp1 >>  27) & 0x7][k + 9], blockvec[k + 9]);

    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
    tmp2 >>= 1;
    res += CMUL_FLOATS(deq2[(tmp >>  0) & 0x7][k + 10], blockvec[k + 10]);
    k += 11;
    res += CMUL_FLOATS(deq2[(tmp2 >>  0) & 0x7][k + 0], blockvec[k + 0]);
    res += CMUL_FLOATS(deq2[(tmp2 >>  3) & 0x7][k + 1], blockvec[k + 1]);
    res += CMUL_FLOATS(deq2[(tmp2 >>  6) & 0x7][k + 2], blockvec[k + 2]);
    res += CMUL_FLOATS(deq2[(tmp2 >>  9) & 0x7][k + 3], blockvec[k + 3]);
    res += CMUL_FLOATS(deq2[(tmp2 >>  12) & 0x7][k + 4], blockvec[k + 4]);
    res += CMUL_FLOATS(deq2[(tmp2 >>  15) & 0x7][k + 5], blockvec[k + 5]);
    res += CMUL_FLOATS(deq2[(tmp2 >>  18) & 0x7][k + 6], blockvec[k + 6]);
    res += CMUL_FLOATS(deq2[(tmp2 >>  21) & 0x7][k + 7], blockvec[k + 7]);
    res += CMUL_FLOATS(deq2[(tmp2 >>  24) & 0x7][k + 8], blockvec[k + 8]);
    res += CMUL_FLOATS(deq2[(tmp2 >>  27) & 0x7][k + 9], blockvec[k + 9]);

    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
    tmp1 >>= 2;
    res += CMUL_FLOATS(deq2[(tmp >>  0) & 0x7][k + 10], blockvec[k + 10]);
    k += 11;
    res += CMUL_FLOATS(deq2[(tmp1 >>  0) & 0x7][k + 0], blockvec[k + 0]);
    res += CMUL_FLOATS(deq2[(tmp1 >>  3) & 0x7][k + 1], blockvec[k + 1]);
    res += CMUL_FLOATS(deq2[(tmp1 >>  6) & 0x7][k + 2], blockvec[k + 2]);
    res += CMUL_FLOATS(deq2[(tmp1 >>  9) & 0x7][k + 3], blockvec[k + 3]);
    res += CMUL_FLOATS(deq2[(tmp1 >>  12) & 0x7][k + 4], blockvec[k + 4]);
    res += CMUL_FLOATS(deq2[(tmp1 >>  15) & 0x7][k + 5], blockvec[k + 5]);
    res += CMUL_FLOATS(deq2[(tmp1 >>  18) & 0x7][k + 6], blockvec[k + 6]);
    res += CMUL_FLOATS(deq2[(tmp1 >>  21) & 0x7][k + 7], blockvec[k + 7]);
    res += CMUL_FLOATS(deq2[(tmp1 >>  24) & 0x7][k + 8], blockvec[k + 8]);
    res += CMUL_FLOATS(deq2[(tmp1 >>  27) & 0x7][k + 9], blockvec[k + 9]);
    i += width;
    k += 10;
  }

  atomicAdd(&mul[col], res);
}

//4-bit per-channel
__global__ void VecQuant4MatMulKernelNUQPerChannel_custom(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
) {

  // BLOCKHEIGHT4 = 16 = BLOCKWIDTH / 32 * 4
  int row = BLOCKHEIGHT4 * blockIdx.x;
  int col =  BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT4) * BLOCKWIDTH + threadIdx.x];

  //Modified dequant block
  __shared__ float deq2[16][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = (((row / 4) * 32) + off) * 16;
  for (int val = 0; val < 16; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

  __syncthreads();

  float res = 0;
  int i = width * row + col;
  int k = 0;

  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp = as_unsigned(mat[i]);

    res += CMUL_FLOATS(deq2[(tmp >>  0) & 0xf][k + 0], blockvec[k + 0]);
    res += CMUL_FLOATS(deq2[(tmp >>  4) & 0xf][k + 1], blockvec[k + 1]);
    res += CMUL_FLOATS(deq2[(tmp >>  8) & 0xf][k + 2], blockvec[k + 2]);
    res += CMUL_FLOATS(deq2[(tmp >>  12) & 0xf][k + 3], blockvec[k + 3]);
    res += CMUL_FLOATS(deq2[(tmp >>  16) & 0xf][k + 4], blockvec[k + 4]);
    res += CMUL_FLOATS(deq2[(tmp >>  20) & 0xf][k + 5], blockvec[k + 5]);
    res += CMUL_FLOATS(deq2[(tmp >>  24) & 0xf][k + 6], blockvec[k + 6]);
    res += CMUL_FLOATS(deq2[(tmp >>  28) & 0xf][k + 7], blockvec[k + 7]);

    i += width;
    k += 8;
  }

  atomicAdd(&mul[col], res);
}


//batched version (3-bit)
__global__ void VecQuant3MatMulKernelNUQPerChannelBatched_custom(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
) {

  int row = BLOCKHEIGHT3 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];

  __shared__ float deq2[8][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = (((row / 3) * 32) + off) * 8;
  // Current row number is (row / 3) * 32
  for (int val = 0; val < 8; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

  int i;
  float res;
  int k;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  for (int b = 0; b < batch; ++b){
    //initialize vars
    i = width * row + col;
    res = 0;
    k = 0;

    __syncthreads();
    blockvec[threadIdx.x] = vec[b * vec_height + (row / BLOCKHEIGHT3) * BLOCKWIDTH + threadIdx.x];
    __syncthreads();

    while (k < BLOCKWIDTH) {
      tmp1 = as_unsigned(mat[i]);

      res += CMUL_FLOATS(deq2[(tmp1 >>  0) & 0x7][k + 0], blockvec[k + 0]);
      res += CMUL_FLOATS(deq2[(tmp1 >>  3) & 0x7][k + 1], blockvec[k + 1]);
      res += CMUL_FLOATS(deq2[(tmp1 >>  6) & 0x7][k + 2], blockvec[k + 2]);
      res += CMUL_FLOATS(deq2[(tmp1 >>  9) & 0x7][k + 3], blockvec[k + 3]);
      res += CMUL_FLOATS(deq2[(tmp1 >>  12) & 0x7][k + 4], blockvec[k + 4]);
      res += CMUL_FLOATS(deq2[(tmp1 >>  15) & 0x7][k + 5], blockvec[k + 5]);
      res += CMUL_FLOATS(deq2[(tmp1 >>  18) & 0x7][k + 6], blockvec[k + 6]);
      res += CMUL_FLOATS(deq2[(tmp1 >>  21) & 0x7][k + 7], blockvec[k + 7]);
      res += CMUL_FLOATS(deq2[(tmp1 >>  24) & 0x7][k + 8], blockvec[k + 8]);
      res += CMUL_FLOATS(deq2[(tmp1 >>  27) & 0x7][k + 9], blockvec[k + 9]);

      i += width;
      tmp2 = as_unsigned(mat[i]);
      tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
      tmp2 >>= 1;
      res += CMUL_FLOATS(deq2[(tmp >>  0) & 0x7][k + 10], blockvec[k + 10]);
      k += 11;
      res += CMUL_FLOATS(deq2[(tmp2 >>  0) & 0x7][k + 0], blockvec[k + 0]);
      res += CMUL_FLOATS(deq2[(tmp2 >>  3) & 0x7][k + 1], blockvec[k + 1]);
      res += CMUL_FLOATS(deq2[(tmp2 >>  6) & 0x7][k + 2], blockvec[k + 2]);
      res += CMUL_FLOATS(deq2[(tmp2 >>  9) & 0x7][k + 3], blockvec[k + 3]);
      res += CMUL_FLOATS(deq2[(tmp2 >>  12) & 0x7][k + 4], blockvec[k + 4]);
      res += CMUL_FLOATS(deq2[(tmp2 >>  15) & 0x7][k + 5], blockvec[k + 5]);
      res += CMUL_FLOATS(deq2[(tmp2 >>  18) & 0x7][k + 6], blockvec[k + 6]);
      res += CMUL_FLOATS(deq2[(tmp2 >>  21) & 0x7][k + 7], blockvec[k + 7]);
      res += CMUL_FLOATS(deq2[(tmp2 >>  24) & 0x7][k + 8], blockvec[k + 8]);
      res += CMUL_FLOATS(deq2[(tmp2 >>  27) & 0x7][k + 9], blockvec[k + 9]);

      i += width;
      tmp1 = as_unsigned(mat[i]);
      tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
      tmp1 >>= 2;
      res += CMUL_FLOATS(deq2[(tmp >>  0) & 0x7][k + 10], blockvec[k + 10]);
      k += 11;
      res += CMUL_FLOATS(deq2[(tmp1 >>  0) & 0x7][k + 0], blockvec[k + 0]);
      res += CMUL_FLOATS(deq2[(tmp1 >>  3) & 0x7][k + 1], blockvec[k + 1]);
      res += CMUL_FLOATS(deq2[(tmp1 >>  6) & 0x7][k + 2], blockvec[k + 2]);
      res += CMUL_FLOATS(deq2[(tmp1 >>  9) & 0x7][k + 3], blockvec[k + 3]);
      res += CMUL_FLOATS(deq2[(tmp1 >>  12) & 0x7][k + 4], blockvec[k + 4]);
      res += CMUL_FLOATS(deq2[(tmp1 >>  15) & 0x7][k + 5], blockvec[k + 5]);
      res += CMUL_FLOATS(deq2[(tmp1 >>  18) & 0x7][k + 6], blockvec[k + 6]);
      res += CMUL_FLOATS(deq2[(tmp1 >>  21) & 0x7][k + 7], blockvec[k + 7]);
      res += CMUL_FLOATS(deq2[(tmp1 >>  24) & 0x7][k + 8], blockvec[k + 8]);
      res += CMUL_FLOATS(deq2[(tmp1 >>  27) & 0x7][k + 9], blockvec[k + 9]);
      i += width;
      k += 10;
    }

    atomicAdd(&mul[b * width + col], res);
  }
}

//batched version (4-bit)
__global__ void VecQuant4MatMulKernelNUQPerChannelBatched_custom(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
) {

  int row = BLOCKHEIGHT4 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  __shared__ float blockvec[BLOCKWIDTH];

  //Modified dequant block
  __shared__ float deq2[16][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = (((row / 4) * 32) + off) * 16;
  for (int val = 0; val < 16; val += 1) {
    int lut_index = row_offset + (val & 0xf);
    deq2[val][off] = lookup_table[lut_index];
  }

  int i;
  float res;
  int k;
  unsigned int tmp;

  for (int b = 0; b < batch; ++b){
    i = width * row + col;
    res = 0;
    k = 0;

    __syncthreads();
    blockvec[threadIdx.x] = vec[b * vec_height + (row / BLOCKHEIGHT4) * BLOCKWIDTH + threadIdx.x];
    __syncthreads();

    while (k < BLOCKWIDTH) {
      tmp = as_unsigned(mat[i]);

      res += CMUL_FLOATS(deq2[(tmp >>  0) & 0xf][k + 0], blockvec[k + 0]);
      res += CMUL_FLOATS(deq2[(tmp >>  4) & 0xf][k + 1], blockvec[k + 1]);
      res += CMUL_FLOATS(deq2[(tmp >>  8) & 0xf][k + 2], blockvec[k + 2]);
      res += CMUL_FLOATS(deq2[(tmp >>  12) & 0xf][k + 3], blockvec[k + 3]);
      res += CMUL_FLOATS(deq2[(tmp >>  16) & 0xf][k + 4], blockvec[k + 4]);
      res += CMUL_FLOATS(deq2[(tmp >>  20) & 0xf][k + 5], blockvec[k + 5]);
      res += CMUL_FLOATS(deq2[(tmp >>  24) & 0xf][k + 6], blockvec[k + 6]);
      res += CMUL_FLOATS(deq2[(tmp >>  28) & 0xf][k + 7], blockvec[k + 7]);

      i += width;
      k += 8;
    }

    atomicAdd(&mul[b * width + col], res);
  }
}

template <typename scalar_t>
__global__ void SPMV_ATOMIC_custom(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0;
        int start_elem = rows[row];
        int end_elem = rows[row+1];
        for (int i = start_elem; i < end_elem; i++) {
            dot += CMUL_FLOATS(mat[i], vec[cols[i]]);
        }
        atomicAdd(&mul[row], dot);
    }
}

template <typename scalar_t>
__global__ void SPMV_ATOMIC_BATCHED_custom(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows,
  int batch,
  int vec_height
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        int start_elem = rows[row];
        int end_elem = rows[row+1];
        for (int b = 0; b < batch; ++b){
            float dot = 0;
            for (int i = start_elem; i < end_elem; i++) {
                dot +=CMUL_FLOATS(mat[i], vec[b * vec_height + cols[i]]);
                // dot += mat[i] * vec[cols[i] * batch + b];
                // dot += mat[i] * vec[cols[i]];
            }
            atomicAdd(&mul[b * num_rows + row], dot);
            // atomicAdd(&mul[row * batch + b], dot);
            // atomicAdd(&mul[row], dot);
        }
    }
}

// Dense kernel for only a subset of rows
__global__ void DenseMatVecKernel_custom(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int height,
    int width
) {

  int row = BLOCKWIDTH * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[row + threadIdx.x];

  __syncthreads();

  int i = width * row + col;
  int k = 0;
  float res = 0;

  if (threadIdx.x < width) {
    while (k < BLOCKWIDTH) {
      res += CMUL_FLOATS(full_rows[i], blockvec[k]);
      k += 1;
      i += width;
    }

    int col_idx = full_row_indices[col];
    atomicAdd(&mul[col_idx], res);
  }
}


// Dense kernel for only a subset of rows
__global__ void DenseMatVecKernelBatched_custom(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int height,
    int width,
    int batch,
    int vec_height,
    int matwidth
) {

  int row = BLOCKWIDTH * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];

  for (int b = 0; b < batch; ++b){
    int i = width * row + col;
    int k = 0;
    float res = 0;

    __syncthreads();
    blockvec[threadIdx.x] = vec[b * vec_height + row + threadIdx.x];
    __syncthreads();

    if (threadIdx.x < width) {
      while (k < BLOCKWIDTH) {
        res += CMUL_FLOATS(full_rows[i], blockvec[k]);
        k += 1;
        i += width;
      }

      int col_idx = full_row_indices[col];
      atomicAdd(&mul[b * matwidth + col_idx], res);
    }
  }
}