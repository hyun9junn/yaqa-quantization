import os
import re
import subprocess

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def nvcc_version():
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    try:
        output = subprocess.check_output([nvcc, "--version"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None

    match = re.search(r"release\s+(\d+)\.(\d+)", output)
    if not match:
        return None
    return tuple(map(int, match.groups()))


def default_cuda_arch_list():
    # B200/Blackwell is SM 10.0 and needs CUDA 12.8+ for native cubins.
    version = nvcc_version()
    if version and version >= (12, 8):
        return "8.0;8.6;8.9;9.0;10.0+PTX"
    return "8.0;8.6;8.9;9.0"


os.environ.setdefault("TORCH_CUDA_ARCH_LIST", default_cuda_arch_list())

setup(
    name="qtip_kernels_cuda",
    ext_modules=[
        CUDAExtension(name="qtip_kernels",
                      sources=[
                          "src/wrapper.cpp", "src/inference.cu",
                          "src/qtip_torch.cu"
                      ],
                      extra_compile_args={
                          "cxx":
                          ["-O3", "--fast-math", "-lineinfo", "-std=c++17"],
                          "nvcc": [
                              "-O3", "--use_fast_math", "-lineinfo", "-keep",
                              "-std=c++17", "--ptxas-options=-v"
                          ],
                      })
    ],
    cmdclass={"build_ext": BuildExtension},
)
