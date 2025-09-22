#pragma once

#include "image.h"
#include <cuda_runtime.h>
#include <opencv2/core/mat.hpp>

template <typename T> class CudaImage : public Image<T> {
private:
  T *d_data_ = nullptr;

public:
  CudaImage() = default;

  CudaImage(size_t width, size_t height) { this->Create(width, height); }

  CudaImage(size_t width, size_t height, T value)
      : Image<T>(width, height, value) {
    AllocateDevice();
    Upload();
  }

  CudaImage(size_t width, size_t height, const std::vector<T> &data)
      : Image<T>(width, height, data) {
    AllocateDevice();
    Upload();
  }

  explicit CudaImage(const Image<T> &host) {
    if (host.Empty()) {
      return;
    }
    this->Create(host.Width(), host.Height());
    std::memcpy(this->data_, host.Data(), this->size_);
    Upload();
  }

  explicit CudaImage(const cv::Mat &mat) : Image<T>(mat) {
    AllocateDevice();
    Upload();
  }

  CudaImage(const CudaImage &other) { CopyFrom(other); }

  CudaImage(CudaImage &&other) noexcept : Image<T>(std::move(other)) {
    d_data_ = other.d_data_;
    other.d_data_ = nullptr;
  }

  CudaImage &operator=(const CudaImage &other) {
    if (this != &other) {
      Release();
      CopyFrom(other);
    }
    return *this;
  }

  CudaImage &operator=(CudaImage &&other) noexcept {
    if (this != &other) {
      Release();
      Image<T>::operator=(std::move(other));
      d_data_ = other.d_data_;
      other.d_data_ = nullptr;
    }
    return *this;
  }

  ~CudaImage() override { Release(); }

  void Create(size_t width, size_t height) override {
    Image<T>::Create(width, height);
    AllocateDevice();
    Upload();
  }

  void Release() override {
    FreeDevice();
    Image<T>::Release();
  }

  T *DeviceData() { return d_data_; }
  const T *DeviceData() const { return d_data_; }

  void Upload() const {
    if (!this->data_ || !d_data_) {
      return;
    }
    cudaError_t err =
        cudaMemcpy(d_data_, this->data_, this->size_, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaMemcpy failed in CudaImage::Upload");
    }
  }

  void Download() const {
    if (!this->data_ || !d_data_) {
      return;
    }
    cudaError_t err =
        cudaMemcpy(this->data_, d_data_, this->size_, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaMemcpy failed in CudaImage::Download");
    }
  }

  Image<T> ToHost() const {
    Image<T> host;
    if (this->Empty()) {
      return host;
    }
    host.Create(this->width_, this->height_);
    Download();
    std::memcpy(host.Data(), this->data_, this->size_);
    return host;
  }

  cv::Mat ToCvMat(int cv_type) const {
    if (this->Empty()) {
      return cv::Mat();
    }
    Download();
    return cv::Mat(static_cast<int>(this->height_),
                   static_cast<int>(this->width_), cv_type, this->data_);
  }

  void FromCvMat(const cv::Mat &mat) {
    Image<T>::FromCvMat(mat);
    AllocateDevice();
    Upload();
  }

private:
  void AllocateDevice() {
    if (this->image_size_ == 0) {
      return;
    }
    FreeDevice();
    cudaError_t err = cudaMalloc(&d_data_, this->size_);
    if (err != cudaSuccess) {
      d_data_ = nullptr;
      throw std::runtime_error("cudaMalloc failed in CudaImage");
    }
  }

  void FreeDevice() {
    if (d_data_) {
      cudaFree(d_data_);
      d_data_ = nullptr;
    }
  }

  void CopyFrom(const CudaImage &other) {
    if (other.Empty()) {
      return;
    }
    Image<T>::Create(other.width_, other.height_);
    std::memcpy(this->data_, other.data_, this->size_);
    AllocateDevice();
    if (other.d_data_) {
      cudaMemcpy(d_data_, other.d_data_, this->size_, cudaMemcpyDeviceToDevice);
    } else {
      Upload();
    }
  }
};