#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

/**********************************************************************/
// FP32
void vecquant3matmul_nuq_perchannel_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
);
void vecquant4matmul_nuq_perchannel_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
);
void vecquant3matmul_nuq_perchannel_batched_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
);
void vecquant4matmul_nuq_perchannel_batched_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
);

void vecquant3matmul_spmv_nuq_perchannel_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
);
void vecquant4matmul_spmv_nuq_perchannel_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
);
void vecquant3matmul_spmv_nuq_perchannel_batched_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
);
void vecquant4matmul_spmv_nuq_perchannel_batched_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
);

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
);
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
);
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
);
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
);

void vecquant3matmul_nuq_perchannel(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_nuq_perchannel_cuda(vec, mat, mul, lookup_table);
}
void vecquant4matmul_nuq_perchannel(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_cuda(vec, mat, mul, lookup_table);
}
void vecquant3matmul_nuq_perchannel_batched(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_nuq_perchannel_batched_cuda(vec, mat, mul, lookup_table);
}
void vecquant4matmul_nuq_perchannel_batched(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_batched_cuda(vec, mat, mul, lookup_table);
}

void vecquant3matmul_spmv_nuq_perchannel(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_spmv_nuq_perchannel_cuda(rows, cols, mat, vec, mul, num_rows, mat3, lookup_table);
}
void vecquant4matmul_spmv_nuq_perchannel(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_spmv_nuq_perchannel_cuda(rows, cols, mat, vec, mul, num_rows, mat4, lookup_table);
}

void vecquant3matmul_spmv_nuq_perchannel_batched(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_spmv_nuq_perchannel_batched_cuda(rows, cols, mat, vec, mul, num_rows, mat3, lookup_table);
}
void vecquant4matmul_spmv_nuq_perchannel_batched(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_spmv_nuq_perchannel_batched_cuda(rows, cols, mat, vec, mul, num_rows, mat4, lookup_table);
}

void vecquant3matmul_spmv_hybrid_nuq_perchannel(
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
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_spmv_hybrid_nuq_perchannel_cuda(rows, cols, mat, vec, full_rows, full_row_indices, mul, num_rows, mat3, lookup_table);
}
void vecquant4matmul_spmv_hybrid_nuq_perchannel(
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
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_spmv_hybrid_nuq_perchannel_cuda(rows, cols, mat, vec, full_rows, full_row_indices, mul, num_rows, mat4, lookup_table);
}

void vecquant3matmul_spmv_hybrid_nuq_perchannel_batched(
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
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_spmv_hybrid_nuq_perchannel_batched_cuda(rows, cols, mat, vec, full_rows, full_row_indices, mul, num_rows, mat3, lookup_table);
}
void vecquant4matmul_spmv_hybrid_nuq_perchannel_batched(
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
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_spmv_hybrid_nuq_perchannel_batched_cuda(rows, cols, mat, vec, full_rows, full_row_indices, mul, num_rows, mat4, lookup_table);
}

/**********************************************************************/
// FP16

void vecquant3matmul_nuq_perchannel_fp16_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
);
void vecquant4matmul_nuq_perchannel_fp16_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
);
void vecquant3matmul_nuq_perchannel_batched_fp16_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
);
void vecquant4matmul_nuq_perchannel_batched_fp16_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
);

void vecquant3matmul_spmv_nuq_perchannel_fp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
);
void vecquant4matmul_spmv_nuq_perchannel_fp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
);
void vecquant3matmul_spmv_nuq_perchannel_batched_fp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
);
void vecquant4matmul_spmv_nuq_perchannel_batched_fp16_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
);

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
);
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
);
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
);
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
);

void vecquant3matmul_nuq_perchannel_fp16(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_nuq_perchannel_fp16_cuda(vec, mat, mul, lookup_table);
}
void vecquant4matmul_nuq_perchannel_fp16(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_fp16_cuda(vec, mat, mul, lookup_table);
}
void vecquant3matmul_nuq_perchannel_batched_fp16(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_nuq_perchannel_batched_fp16_cuda(vec, mat, mul, lookup_table);
}
void vecquant4matmul_nuq_perchannel_batched_fp16(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_batched_fp16_cuda(vec, mat, mul, lookup_table);
}

void vecquant3matmul_spmv_nuq_perchannel_fp16(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_spmv_nuq_perchannel_fp16_cuda(rows, cols, mat, vec, mul, num_rows, mat3, lookup_table);
}
void vecquant4matmul_spmv_nuq_perchannel_fp16(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_spmv_nuq_perchannel_fp16_cuda(rows, cols, mat, vec, mul, num_rows, mat4, lookup_table);
}

void vecquant3matmul_spmv_nuq_perchannel_batched_fp16(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_spmv_nuq_perchannel_batched_fp16_cuda(rows, cols, mat, vec, mul, num_rows, mat3, lookup_table);
}
void vecquant4matmul_spmv_nuq_perchannel_batched_fp16(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_spmv_nuq_perchannel_batched_fp16_cuda(rows, cols, mat, vec, mul, num_rows, mat4, lookup_table);
}

void vecquant3matmul_spmv_hybrid_nuq_perchannel_fp16(
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
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_spmv_hybrid_nuq_perchannel_fp16_cuda(rows, cols, mat, vec, full_rows, full_row_indices, mul, num_rows, mat3, lookup_table);
}
void vecquant4matmul_spmv_hybrid_nuq_perchannel_fp16(
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
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_spmv_hybrid_nuq_perchannel_fp16_cuda(rows, cols, mat, vec, full_rows, full_row_indices, mul, num_rows, mat4, lookup_table);
}

void vecquant3matmul_spmv_hybrid_nuq_perchannel_batched_fp16(
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
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_spmv_hybrid_nuq_perchannel_batched_fp16_cuda(rows, cols, mat, vec, full_rows, full_row_indices, mul, num_rows, mat3, lookup_table);
}
void vecquant4matmul_spmv_hybrid_nuq_perchannel_batched_fp16(
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
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_spmv_hybrid_nuq_perchannel_batched_fp16_cuda(rows, cols, mat, vec, full_rows, full_row_indices, mul, num_rows, mat4, lookup_table);
}

/**********************************************************************/
// CUSTOM

void vecquant3matmul_nuq_perchannel_custom_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
);
void vecquant4matmul_nuq_perchannel_custom_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
);
void vecquant3matmul_nuq_perchannel_batched_custom_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
);
void vecquant4matmul_nuq_perchannel_batched_custom_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
);

void vecquant3matmul_spmv_nuq_perchannel_custom_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
);
void vecquant4matmul_spmv_nuq_perchannel_custom_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
);
void vecquant3matmul_spmv_nuq_perchannel_batched_custom_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
);
void vecquant4matmul_spmv_nuq_perchannel_batched_custom_cuda(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
);

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
);
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
);
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
);
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
);

void vecquant3matmul_nuq_perchannel_custom(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_nuq_perchannel_custom_cuda(vec, mat, mul, lookup_table);
}
void vecquant4matmul_nuq_perchannel_custom(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_custom_cuda(vec, mat, mul, lookup_table);
}
void vecquant3matmul_nuq_perchannel_batched_custom(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_nuq_perchannel_batched_custom_cuda(vec, mat, mul, lookup_table);
}
void vecquant4matmul_nuq_perchannel_batched_custom(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_batched_custom_cuda(vec, mat, mul, lookup_table);
}

void vecquant3matmul_spmv_nuq_perchannel_custom(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_spmv_nuq_perchannel_custom_cuda(rows, cols, mat, vec, mul, num_rows, mat3, lookup_table);
}
void vecquant4matmul_spmv_nuq_perchannel_custom(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_spmv_nuq_perchannel_custom_cuda(rows, cols, mat, vec, mul, num_rows, mat4, lookup_table);
}

void vecquant3matmul_spmv_nuq_perchannel_batched_custom(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_spmv_nuq_perchannel_batched_custom_cuda(rows, cols, mat, vec, mul, num_rows, mat3, lookup_table);
}
void vecquant4matmul_spmv_nuq_perchannel_batched_custom(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_spmv_nuq_perchannel_batched_custom_cuda(rows, cols, mat, vec, mul, num_rows, mat4, lookup_table);
}

void vecquant3matmul_spmv_hybrid_nuq_perchannel_custom(
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
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_spmv_hybrid_nuq_perchannel_custom_cuda(rows, cols, mat, vec, full_rows, full_row_indices, mul, num_rows, mat3, lookup_table);
}
void vecquant4matmul_spmv_hybrid_nuq_perchannel_custom(
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
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_spmv_hybrid_nuq_perchannel_custom_cuda(rows, cols, mat, vec, full_rows, full_row_indices, mul, num_rows, mat4, lookup_table);
}

void vecquant3matmul_spmv_hybrid_nuq_perchannel_batched_custom(
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
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_spmv_hybrid_nuq_perchannel_batched_custom_cuda(rows, cols, mat, vec, full_rows, full_row_indices, mul, num_rows, mat3, lookup_table);
}
void vecquant4matmul_spmv_hybrid_nuq_perchannel_batched_custom(
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
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_spmv_hybrid_nuq_perchannel_batched_custom_cuda(rows, cols, mat, vec, full_rows, full_row_indices, mul, num_rows, mat4, lookup_table);
}

/**********************************************************************/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // fp32
  m.def("vecquant3matmul_nuq_perchannel", &vecquant3matmul_nuq_perchannel, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA)");
  m.def("vecquant4matmul_nuq_perchannel", &vecquant4matmul_nuq_perchannel, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA)");
  m.def("vecquant3matmul_nuq_perchannel_batched", &vecquant3matmul_nuq_perchannel_batched, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA)");
  m.def("vecquant4matmul_nuq_perchannel_batched", &vecquant4matmul_nuq_perchannel_batched, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA)");
  m.def("vecquant3matmul_spmv_nuq_perchannel", &vecquant3matmul_spmv_nuq_perchannel, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant4matmul_spmv_nuq_perchannel", &vecquant4matmul_spmv_nuq_perchannel, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant3matmul_spmv_nuq_perchannel_batched", &vecquant3matmul_spmv_nuq_perchannel_batched, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant4matmul_spmv_nuq_perchannel_batched", &vecquant4matmul_spmv_nuq_perchannel_batched, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant3matmul_spmv_hybrid_nuq_perchannel", &vecquant3matmul_spmv_hybrid_nuq_perchannel, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant4matmul_spmv_hybrid_nuq_perchannel", &vecquant4matmul_spmv_hybrid_nuq_perchannel, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant3matmul_spmv_hybrid_nuq_perchannel_batched", &vecquant3matmul_spmv_hybrid_nuq_perchannel_batched, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant4matmul_spmv_hybrid_nuq_perchannel_batched", &vecquant4matmul_spmv_hybrid_nuq_perchannel_batched, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");

  // fp16
  m.def("vecquant3matmul_nuq_perchannel_fp16", &vecquant3matmul_nuq_perchannel_fp16, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA)");
  m.def("vecquant4matmul_nuq_perchannel_fp16", &vecquant4matmul_nuq_perchannel_fp16, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA)");
  m.def("vecquant3matmul_nuq_perchannel_batched_fp16", &vecquant3matmul_nuq_perchannel_batched_fp16, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA)");
  m.def("vecquant4matmul_nuq_perchannel_batched_fp16", &vecquant4matmul_nuq_perchannel_batched_fp16, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA)");
  m.def("vecquant3matmul_spmv_nuq_perchannel_fp16", &vecquant3matmul_spmv_nuq_perchannel_fp16, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant4matmul_spmv_nuq_perchannel_fp16", &vecquant4matmul_spmv_nuq_perchannel_fp16, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant3matmul_spmv_nuq_perchannel_batched_fp16", &vecquant3matmul_spmv_nuq_perchannel_batched_fp16, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant4matmul_spmv_nuq_perchannel_batched_fp16", &vecquant4matmul_spmv_nuq_perchannel_batched_fp16, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant3matmul_spmv_hybrid_nuq_perchannel_fp16", &vecquant3matmul_spmv_hybrid_nuq_perchannel_fp16, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant4matmul_spmv_hybrid_nuq_perchannel_fp16", &vecquant4matmul_spmv_hybrid_nuq_perchannel_fp16, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant3matmul_spmv_hybrid_nuq_perchannel_batched_fp16", &vecquant3matmul_spmv_hybrid_nuq_perchannel_batched_fp16, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant4matmul_spmv_hybrid_nuq_perchannel_batched_fp16", &vecquant4matmul_spmv_hybrid_nuq_perchannel_batched_fp16, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");

  // custom
  m.def("vecquant3matmul_nuq_perchannel_custom", &vecquant3matmul_nuq_perchannel_custom, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA)");
  m.def("vecquant4matmul_nuq_perchannel_custom", &vecquant4matmul_nuq_perchannel_custom, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA)");
  m.def("vecquant3matmul_nuq_perchannel_batched_custom", &vecquant3matmul_nuq_perchannel_batched_custom, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA)");
  m.def("vecquant4matmul_nuq_perchannel_batched_custom", &vecquant4matmul_nuq_perchannel_batched_custom, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA)");
  m.def("vecquant3matmul_spmv_nuq_perchannel_custom", &vecquant3matmul_spmv_nuq_perchannel_custom, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant4matmul_spmv_nuq_perchannel_custom", &vecquant4matmul_spmv_nuq_perchannel_custom, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant3matmul_spmv_nuq_perchannel_batched_custom", &vecquant3matmul_spmv_nuq_perchannel_batched_custom, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant4matmul_spmv_nuq_perchannel_batched_custom", &vecquant4matmul_spmv_nuq_perchannel_batched_custom, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant3matmul_spmv_hybrid_nuq_perchannel_custom", &vecquant3matmul_spmv_hybrid_nuq_perchannel_custom, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant4matmul_spmv_hybrid_nuq_perchannel_custom", &vecquant4matmul_spmv_hybrid_nuq_perchannel_custom, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant3matmul_spmv_hybrid_nuq_perchannel_batched_custom", &vecquant3matmul_spmv_hybrid_nuq_perchannel_batched_custom, "Non-Uniform Vector 3-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
  m.def("vecquant4matmul_spmv_hybrid_nuq_perchannel_batched_custom", &vecquant4matmul_spmv_hybrid_nuq_perchannel_batched_custom, "Non-Uniform Vector 4-bit Quantized Matrix Multiplication w/ Per-Channel LUT (CUDA - Hybrid)");
}
