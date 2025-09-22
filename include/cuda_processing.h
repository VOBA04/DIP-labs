#pragma once

#include "cuda_image.cuh"
#include <cstdint>
#include <vector>

void CudaBgrToGray(const std::vector<CudaImage<uint8_t>> &bgr_image,
                   CudaImage<uint8_t> &gray_image);

void CudaBgrToHsi(const std::vector<CudaImage<uint8_t>> &bgr_image,
                  std::vector<CudaImage<float>> &hsi_image);

void CudaHsiToBgr(const std::vector<CudaImage<float>> &hsi_image,
                  std::vector<CudaImage<uint8_t>> &bgr_image);

void CudaInRange(const CudaImage<uint8_t> &src, uint8_t LOWER_BOUND,
                 uint8_t UPPER_BOUND, CudaImage<uint8_t> &dst);

void CudaErosion(const CudaImage<uint8_t> &src,
                 const CudaImage<uint8_t> &kernel, CudaImage<uint8_t> &dst);
void CudaDilation(const CudaImage<uint8_t> &src,
                  const CudaImage<uint8_t> &kernel, CudaImage<uint8_t> &dst);

void CudaSobel(const CudaImage<uint8_t> &src, CudaImage<uint8_t> &dst);

void CudaBitwiseAnd(const CudaImage<uint8_t> &src1,
                    const CudaImage<uint8_t> &src2, CudaImage<uint8_t> &dst);