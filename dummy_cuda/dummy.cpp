#include <torch/extension.h>

void add_one_cuda_kernel(float* x, int N);

torch::Tensor add_one_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    add_one_cuda_kernel(x.data_ptr<float>(), x.numel());
    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_one_cuda", &add_one_cuda, "Add one (CUDA)");
}
