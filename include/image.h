#pragma once

#include <algorithm>
#include <cstddef>
#include <opencv2/core/mat.hpp>
#include <stdexcept>
#include <string>
#include <vector>

void ShowImages(const std::vector<cv::Mat> &images,
                const std::string &window_name, size_t ROWS = 1);

#ifdef WITH_QT
void ShowImage3D(const cv::Mat &image);
#endif

void ShowMarkers(const cv::Mat &markers, const std::string &window_name);

template <typename T> class Image {
protected:
  size_t width_ = 0;
  size_t height_ = 0;
  size_t size_ = 0;
  size_t image_size_ = 0;
  T *data_ = nullptr;

public:
  Image() = default;

  Image(size_t width, size_t height) { Create(width, height); }

  Image(size_t width, size_t height, const T *data) {
    Create(width, height);
    if (data && data_) {
      std::copy(data, data + image_size_, data_);
    }
  }

  Image(size_t width, size_t height, T value) {
    Create(width, height);
    if (data_) {
      std::fill_n(data_, image_size_, value);
    }
  }

  Image(size_t width, size_t height, const std::vector<T> &data) {
    if (width == 0 || height == 0) {
      throw std::invalid_argument("Invalid image dimensions");
    }
    size_t expected_size =
        static_cast<size_t>(width) * static_cast<size_t>(height);
    if (data.size() != expected_size) {
      throw std::invalid_argument(
          "Data size does not match width*height in Image(vector) constructor");
    }
    Create(width, height);
    if (data_) {
      std::copy(data.begin(), data.end(), data_);
    }
  }

  explicit Image(const cv::Mat &mat) { FromCvMat(mat); }

  Image(const Image &other) { CopyFrom(other); }

  Image(Image &&other) noexcept
      : width_(other.width_), height_(other.height_), size_(other.size_),
        image_size_(other.image_size_), data_(other.data_) {
    other.width_ = other.height_ = 0;
    other.size_ = other.image_size_ = 0;
    other.data_ = nullptr;
  }

  Image &operator=(const Image &other) {
    if (this != &other) {
      FreeData();
      CopyFrom(other);
    }
    return *this;
  }

  Image &operator=(Image &&other) noexcept {
    if (this != &other) {
      FreeData();
      width_ = other.width_;
      height_ = other.height_;
      size_ = other.size_;
      image_size_ = other.image_size_;
      data_ = other.data_;
      other.width_ = other.height_ = 0;
      other.size_ = other.image_size_ = 0;
      other.data_ = nullptr;
    }
    return *this;
  }

  virtual ~Image() { FreeData(); }

  // Make virtual so derived classes can extend allocation lifecycle
  virtual void Create(size_t width, size_t height) {
    if (width == 0 || height == 0) {
      throw std::invalid_argument("Invalid image dimensions");
    }
    FreeData();
    width_ = width;
    height_ = height;
    image_size_ = static_cast<size_t>(width_) * static_cast<size_t>(height_);
    size_ = image_size_ * sizeof(T);
    data_ = static_cast<T *>(std::malloc(size_));
    if (!data_) {
      throw std::bad_alloc();
    }
    std::fill_n(data_, image_size_, static_cast<T>(0));
  }

  // Make virtual so derived classes can add extra cleanup
  virtual void Release() { FreeData(); }

  int Width() const { return width_; }
  int Height() const { return height_; }
  size_t Size() const { return image_size_; }
  size_t Bytes() const { return size_; }

  T *Data() { return data_; }
  const T *Data() const { return data_; }

  T &At(size_t x, size_t y) {
    CheckCoords(x, y);
    return data_[static_cast<size_t>(y) * width_ + x];
  }
  const T &At(size_t x, size_t y) const {
    CheckCoords(x, y);
    return data_[static_cast<size_t>(y) * width_ + x];
  }

  T &operator[](size_t idx) { return data_[idx]; }
  const T &operator[](size_t idx) const { return data_[idx]; }

  bool Empty() const { return data_ == nullptr || image_size_ == 0; }

  cv::Mat ToCvMat(int cv_type) const {
    if (Empty()) {
      return cv::Mat();
    }
    return cv::Mat(height_, width_, cv_type, const_cast<T *>(data_)).clone();
  }

  void FromCvMat(const cv::Mat &mat) {
    if (mat.empty()) {
      Release();
      return;
    }
    if (mat.channels() != 1) {
      throw std::invalid_argument(
          "cv::Mat with multiple channels not supported for this Image<T> "
          "method");
    }
    Create(mat.cols, mat.rows);
    CV_Assert((int)mat.elemSize1() == (int)sizeof(T));
    std::memcpy(data_, mat.data, size_);
  }

protected:
  void FreeData() {
    if (data_) {
      std::free(data_);
      data_ = nullptr;
    }
    width_ = height_ = 0;
    size_ = image_size_ = 0;
  }

private:
  void CheckCoords(int x, int y) const {
    if (x < 0 || y < 0 || x >= width_ || y >= height_) {
      throw std::out_of_range("Image coordinates out of range");
    }
  }

  void CopyFrom(const Image &other) {
    if (other.Empty()) {
      return;
    }
    width_ = other.width_;
    height_ = other.height_;
    image_size_ = other.image_size_;
    size_ = other.size_;
    data_ = static_cast<T *>(std::malloc(size_));
    if (!data_) {
      throw std::bad_alloc();
    }
    std::memcpy(data_, other.data_, size_);
  }
};

struct ObjectProperties {
  int number;
  float area;
  float perimeter;
  float elongation;
  cv::Vec3b color;
};

std::vector<ObjectProperties>
CalculateObjectsProperties(const cv::Mat &markers);

std::vector<ObjectProperties>
CalculateObjectsPropertiesCPU(const cv::Mat &markers);

void DisplayObjectProperties(const std::vector<ObjectProperties> &properties);

void ShowPCA(const cv::Mat &markers, const cv::Mat &image,
             const std::string &window_name);