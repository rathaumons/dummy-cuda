#include <cuda_runtime.h>

__global__ void add_one_kernel(float* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        x[i] += 1.0f;
}

void add_one_cuda_kernel(float* x, int N) {
    add_one_kernel<<<(N + 255) / 256, 256>>>(x, N);
}
