#include "cuda_processing.h"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

__global__ void BgrToHsiKernel(const uint8_t *b, const uint8_t *g,
                               const uint8_t *r, float *h, float *s, float *i,
                               int width, int height) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    size_t idx = y * width + x;
    float blue = static_cast<float>(b[idx]) / 255.0F;
    float green = static_cast<float>(g[idx]) / 255.0F;
    float red = static_cast<float>(r[idx]) / 255.0F;
    float sum = red + green + blue;
    float intensity = sum / 3.0F;
    float min_rgb = fminf(red, fminf(green, blue));
    float saturation = 0.0F;
    if (sum > 1e-6F) {
      saturation = 1.0F - (3.0F * min_rgb) / sum;
    }
    saturation = fminf(fmaxf(saturation, 0.0F), 1.0F);
    float hue = 0.0F;
    if (saturation > 1e-6F) {
      float num = 0.5F * ((red - green) + (red - blue));
      float den =
          sqrtf((red - green) * (red - green) + (red - blue) * (green - blue));
      float cos_theta = 0.0F;
      if (den > 1e-6F) {
        cos_theta = fminf(fmaxf(num / den, -1.0F), 1.0F);
      }
      float theta = acosf(cos_theta);
      hue = (blue <= green) ? theta : (2.0F * CUDART_PI_F - theta);
      hue = hue * 180.0F / CUDART_PI_F;
    }
    h[idx] = hue;        // [0..360)
    s[idx] = saturation; // [0..1]
    i[idx] = intensity;  // [0..1]
  }
}

__global__ void BgrToGrayKernel(const uint8_t *b, const uint8_t *g,
                                const uint8_t *r, uint8_t *gray, int width,
                                int height) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    size_t idx = y * width + x;
    gray[idx] = static_cast<uint8_t>(0.299F * static_cast<float>(r[idx]) +
                                     0.587F * static_cast<float>(g[idx]) +
                                     0.114F * static_cast<float>(b[idx]));
  }
}

__device__ inline void HsiToRgb(const float HUE_DEG_IN, const float SAT_IN,
                                const float INT_IN, float &rOut, float &gOut,
                                float &bOut) {
  float s = fminf(fmaxf(SAT_IN, 0.0F), 1.0F);
  float i = fminf(fmaxf(INT_IN, 0.0F), 1.0F);
  if (s <= 1e-6F) {
    rOut = gOut = bOut = i;
    return;
  }
  float h = fmodf(HUE_DEG_IN, 360.0F);
  if (h < 0.0F) {
    h += 360.0F;
  }
  if (h < 120.0F) {
    float h_rad = h * (CUDART_PI_F / 180.0F);
    float denom = cosf(CUDART_PI_F / 3.0F - h_rad);
    denom = (fabsf(denom) < 1e-6F) ? (denom >= 0 ? 1e-6F : -1e-6F) : denom;
    float r = i * (1.0F + (s * cosf(h_rad) / denom));
    float b = i * (1.0F - s);
    float g = 3.0F * i - (r + b);
    rOut = fminf(fmaxf(r, 0.0F), 1.0F);
    gOut = fminf(fmaxf(g, 0.0F), 1.0F);
    bOut = fminf(fmaxf(b, 0.0F), 1.0F);
    return;
  }
  if (h < 240.0F) {
    float h2 = (h - 120.0F) * (CUDART_PI_F / 180.0F);
    float denom = cosf(CUDART_PI_F / 3.0F - h2);
    denom = (fabsf(denom) < 1e-6F) ? (denom >= 0 ? 1e-6F : -1e-6F) : denom;
    float r = i * (1.0F - s);
    float g = i * (1.0F + (s * cosf(h2) / denom));
    float b = 3.0F * i - (r + g);
    rOut = fminf(fmaxf(r, 0.0F), 1.0F);
    gOut = fminf(fmaxf(g, 0.0F), 1.0F);
    bOut = fminf(fmaxf(b, 0.0F), 1.0F);
    return;
  }
  {
    float h3 = (h - 240.0F) * (CUDART_PI_F / 180.0F);
    float denom = cosf(CUDART_PI_F / 3.0F - h3);
    denom = (fabsf(denom) < 1e-6F) ? (denom >= 0 ? 1e-6F : -1e-6F) : denom;
    float g = i * (1.0F - s);
    float b = i * (1.0F + (s * cosf(h3) / denom));
    float r = 3.0F * i - (g + b);
    rOut = fminf(fmaxf(r, 0.0F), 1.0F);
    gOut = fminf(fmaxf(g, 0.0F), 1.0F);
    bOut = fminf(fmaxf(b, 0.0F), 1.0F);
  }
}

__global__ void HsiToBgrKernel(const float *h, const float *s, const float *i,
                               uint8_t *b, uint8_t *g, uint8_t *r, int width,
                               int height) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    size_t idx = y * width + x;
    float rr;
    float gg;
    float bb;
    HsiToRgb(h[idx], s[idx], i[idx], rr, gg, bb);
    r[idx] = static_cast<uint8_t>(fminf(fmaxf(rr * 255.0F, 0.0F), 255.0F));
    g[idx] = static_cast<uint8_t>(fminf(fmaxf(gg * 255.0F, 0.0F), 255.0F));
    b[idx] = static_cast<uint8_t>(fminf(fmaxf(bb * 255.0F, 0.0F), 255.0F));
  }
}

__global__ void InRangeKernel(const uint8_t *src, uint8_t *dst, int width,
                              int height, uint8_t lower_bound,
                              uint8_t upper_bound) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    size_t idx = y * width + x;
    dst[idx] = (src[idx] >= lower_bound && src[idx] <= upper_bound) ? 255 : 0;
  }
}

__global__ void BitwiseAndKernel(const uint8_t *src1, const uint8_t *src2,
                                 uint8_t *dst, int width, int height) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    size_t idx = y * width + x;
    dst[idx] = src1[idx] & src2[idx];
  }
}

__global__ void ErodeKernel(const uint8_t *src, uint8_t *dst, int width,
                            int height, const uint8_t *kernel,
                            int kernel_size) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  int k_half = kernel_size / 2;
  if (x < width && y < height) {
    bool match = true;
    for (int ky = -k_half; ky <= k_half; ++ky) {
      for (int kx = -k_half; kx <= k_half; ++kx) {
        int ix = static_cast<int>(x) + kx;
        int iy = static_cast<int>(y) + ky;
        if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
          if (kernel[(ky + k_half) * kernel_size + (kx + k_half)] == 1 &&
              src[iy * width + ix] == 0) {
            match = false;
            break;
          }
        }
      }
      if (!match) {
        break;
      }
    }
    dst[y * width + x] = match ? 255 : 0;
  }
}

__global__ void DilateKernel(const uint8_t *src, uint8_t *dst, int width,
                             int height, const uint8_t *kernel,
                             int kernel_size) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  int k_half = kernel_size / 2;
  if (x < width && y < height) {
    bool match = false;
    for (int ky = -k_half; ky <= k_half; ++ky) {
      for (int kx = -k_half; kx <= k_half; ++kx) {
        int ix = static_cast<int>(x) + kx;
        int iy = static_cast<int>(y) + ky;
        if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
          if (kernel[(ky + k_half) * kernel_size + (kx + k_half)] == 1 &&
              src[iy * width + ix] == 255) {
            match = true;
            break;
          }
        }
      }
      if (match) {
        break;
      }
    }
    dst[y * width + x] = match ? 255 : 0;
  }
}

__constant__ int8_t d_kernel_sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int8_t d_kernel_sobel_y[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

__global__ void SobelKernel(const uint8_t *src, uint8_t *dst, int width,
                            int height) {
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int gx = 0;
    int gy = 0;
    for (int ky = -1; ky <= 1; ++ky) {
      for (int kx = -1; kx <= 1; ++kx) {
        int ix = min(max(static_cast<int>(x) + kx, 0), width - 1);
        int iy = min(max(static_cast<int>(y) + ky, 0), height - 1);
        gx += src[iy * width + ix] * d_kernel_sobel_x[(ky + 1) * 3 + (kx + 1)];
        gy += src[iy * width + ix] * d_kernel_sobel_y[(ky + 1) * 3 + (kx + 1)];
      }
    }
    int mag = min(
        static_cast<int>(sqrtf(static_cast<float>(gx * gx + gy * gy))), 255);
    dst[y * width + x] = static_cast<uint8_t>(mag);
  }
}
} // namespace

void CudaBgrToGray(const std::vector<CudaImage<uint8_t>> &bgr_image,
                   CudaImage<uint8_t> &gray_image) {
  if (bgr_image.size() != 3) {
    throw std::runtime_error("BGR image must have 3 channels.");
  }
  if (bgr_image[0].Empty() || bgr_image[1].Empty() || bgr_image[2].Empty()) {
    throw std::runtime_error("Input BGR images must not be empty.");
  }
  size_t width = bgr_image[0].Width();
  size_t height = bgr_image[0].Height();
  if (bgr_image[1].Width() != width || bgr_image[1].Height() != height ||
      bgr_image[2].Width() != width || bgr_image[2].Height() != height) {
    throw std::runtime_error("All BGR channels must have the same dimensions.");
  }
  gray_image.Create(width, height);
  dim3 block_size(16, 16);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y);
  BgrToGrayKernel<<<grid_size, block_size>>>(
      bgr_image[0].DeviceData(), bgr_image[1].DeviceData(),
      bgr_image[2].DeviceData(), gray_image.DeviceData(),
      static_cast<int>(width), static_cast<int>(height));
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("BgrToGrayKernel failed: ") +
                             cudaGetErrorString(err));
  }
}

void CudaBgrToHsi(const std::vector<CudaImage<uint8_t>> &bgr_image,
                  std::vector<CudaImage<float>> &hsi_image) {
  if (bgr_image.size() != 3) {
    throw std::runtime_error("BGR image must have 3 channels.");
  }
  if (bgr_image[0].Empty() || bgr_image[1].Empty() || bgr_image[2].Empty()) {
    throw std::runtime_error("Input BGR images must not be empty.");
  }
  size_t width = bgr_image[0].Width();
  size_t height = bgr_image[0].Height();
  if (bgr_image[1].Width() != width || bgr_image[1].Height() != height ||
      bgr_image[2].Width() != width || bgr_image[2].Height() != height) {
    throw std::runtime_error("All BGR channels must have the same dimensions.");
  }
  hsi_image.resize(3);
  for (auto &channel : hsi_image) {
    channel.Create(width, height);
  }
  dim3 block_size(16, 16);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y);
  BgrToHsiKernel<<<grid_size, block_size>>>(
      bgr_image[0].DeviceData(), bgr_image[1].DeviceData(),
      bgr_image[2].DeviceData(), hsi_image[0].DeviceData(),
      hsi_image[1].DeviceData(), hsi_image[2].DeviceData(),
      static_cast<int>(width), static_cast<int>(height));
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("BgrToHsiKernel failed: ") +
                             cudaGetErrorString(err));
  }
}

void CudaHsiToBgr(const std::vector<CudaImage<float>> &hsi_image,
                  std::vector<CudaImage<uint8_t>> &bgr_image) {
  if (hsi_image.size() != 3) {
    throw std::runtime_error("HSI image must have 3 channels.");
  }
  if (hsi_image[0].Empty() || hsi_image[1].Empty() || hsi_image[2].Empty()) {
    throw std::runtime_error("Input HSI images must not be empty.");
  }
  size_t width = hsi_image[0].Width();
  size_t height = hsi_image[0].Height();
  if (hsi_image[1].Width() != width || hsi_image[1].Height() != height ||
      hsi_image[2].Width() != width || hsi_image[2].Height() != height) {
    throw std::runtime_error("All HSI channels must have the same dimensions.");
  }
  bgr_image.resize(3);
  for (auto &channel : bgr_image) {
    channel.Create(width, height);
  }
  dim3 block_size(16, 16);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y);
  HsiToBgrKernel<<<grid_size, block_size>>>(
      hsi_image[0].DeviceData(), hsi_image[1].DeviceData(),
      hsi_image[2].DeviceData(), bgr_image[0].DeviceData(),
      bgr_image[1].DeviceData(), bgr_image[2].DeviceData(),
      static_cast<int>(width), static_cast<int>(height));
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("HsiToBgrKernel failed: ") +
                             cudaGetErrorString(err));
  }
}

void CudaInRange(const CudaImage<uint8_t> &src, const uint8_t LOWER_BOUND,
                 const uint8_t UPPER_BOUND, CudaImage<uint8_t> &dst) {
  if (src.Empty()) {
    throw std::runtime_error("Input image must not be empty.");
  }
  size_t width = src.Width();
  size_t height = src.Height();
  dst.Create(width, height);
  dim3 block_size(16, 16);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y);
  InRangeKernel<<<grid_size, block_size>>>(
      src.DeviceData(), dst.DeviceData(), static_cast<int>(width),
      static_cast<int>(height), LOWER_BOUND, UPPER_BOUND);
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("InRangeKernel failed: ") +
                             cudaGetErrorString(err));
  }
}

void CudaErosion(const CudaImage<uint8_t> &src,
                 const CudaImage<uint8_t> &kernel, CudaImage<uint8_t> &dst) {
  if (src.Empty()) {
    throw std::runtime_error("Input image must not be empty.");
  }
  if (kernel.Empty() || kernel.Width() != kernel.Height() ||
      kernel.Width() % 2 == 0) {
    throw std::runtime_error(
        "Kernel must be non-empty, square, and have an odd size.");
  }
  size_t width = src.Width();
  size_t height = src.Height();
  dst.Create(width, height);
  dim3 block_size(16, 16);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y);
  ErodeKernel<<<grid_size, block_size>>>(
      src.DeviceData(), dst.DeviceData(), static_cast<int>(width),
      static_cast<int>(height), kernel.DeviceData(),
      static_cast<int>(kernel.Width()));
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("ErodeKernel failed: ") +
                             cudaGetErrorString(err));
  }
}

void CudaDilation(const CudaImage<uint8_t> &src,
                  const CudaImage<uint8_t> &kernel, CudaImage<uint8_t> &dst) {
  if (src.Empty()) {
    throw std::runtime_error("Input image must not be empty.");
  }
  if (kernel.Empty() || kernel.Width() != kernel.Height() ||
      kernel.Width() % 2 == 0) {
    throw std::runtime_error(
        "Kernel must be non-empty, square, and have an odd size.");
  }
  size_t width = src.Width();
  size_t height = src.Height();
  dst.Create(width, height);
  dim3 block_size(16, 16);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y);
  DilateKernel<<<grid_size, block_size>>>(
      src.DeviceData(), dst.DeviceData(), static_cast<int>(width),
      static_cast<int>(height), kernel.DeviceData(),
      static_cast<int>(kernel.Width()));
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("DilateKernel failed: ") +
                             cudaGetErrorString(err));
  }
}

void CudaSobel(const CudaImage<uint8_t> &src, CudaImage<uint8_t> &dst) {
  if (src.Empty()) {
    throw std::runtime_error("Input image must not be empty.");
  }
  size_t width = src.Width();
  size_t height = src.Height();
  dst.Create(width, height);
  dim3 block_size(16, 16);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y);
  SobelKernel<<<grid_size, block_size>>>(src.DeviceData(), dst.DeviceData(),
                                         static_cast<int>(width),
                                         static_cast<int>(height));
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("SobelKernel failed: ") +
                             cudaGetErrorString(err));
  }
}

void CudaBitwiseAnd(const CudaImage<uint8_t> &src1,
                    const CudaImage<uint8_t> &src2, CudaImage<uint8_t> &dst) {
  if (src1.Empty() || src2.Empty()) {
    throw std::runtime_error("Input images must not be empty.");
  }
  size_t width = src1.Width();
  size_t height = src1.Height();
  if (src2.Width() != width || src2.Height() != height) {
    throw std::runtime_error("Input images must have the same dimensions.");
  }
  dst.Create(width, height);
  dim3 block_size(16, 16);
  dim3 grid_size((width + block_size.x - 1) / block_size.x,
                 (height + block_size.y - 1) / block_size.y);
  BitwiseAndKernel<<<grid_size, block_size>>>(
      src1.DeviceData(), src2.DeviceData(), dst.DeviceData(),
      static_cast<int>(width), static_cast<int>(height));
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("BitwiseAndKernel failed: ") +
                             cudaGetErrorString(err));
  }
}