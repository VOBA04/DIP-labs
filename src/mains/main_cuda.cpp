#include "cuda_image.cuh"
#include "cuda_processing.h"
#include "image.h"

#include <QApplication>
#include <QFileDialog>
#include <cstdint>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

const int ESC_KEY = 27;

int main(int argc, char *argv[]) {
  QApplication app(argc, argv);
  QString file_name = QFileDialog::getOpenFileName(
      nullptr, "Выберите изображение", "../../images/",
      "Images (*.jpg *.jpeg *.png *.bmp)");
  if (file_name.isEmpty()) {
    return 0;
  }

  cv::Mat img = cv::imread(file_name.toStdString());
  if (img.empty()) {
    std::cerr << "Не удалось загрузить изображение!" << std::endl;
    return -1;
  }

  ShowImages({img}, "Original Image");

  std::vector<cv::Mat> bgr_channels;
  cv::split(img, bgr_channels);
  std::vector<CudaImage<uint8_t>> cuda_bgr_channels;
  cuda_bgr_channels.reserve(bgr_channels.size());
  for (const auto &channel : bgr_channels) {
    cuda_bgr_channels.emplace_back(channel);
  }
  ShowImages({cuda_bgr_channels[0].ToCvMat(CV_8UC1),
              cuda_bgr_channels[1].ToCvMat(CV_8UC1),
              cuda_bgr_channels[2].ToCvMat(CV_8UC1)},
             "BGR Channels");

  std::vector<CudaImage<float>> cuda_hsi_channels(3);
  CudaBgrToHsi(cuda_bgr_channels, cuda_hsi_channels);
  CudaImage<uint8_t> hue8;
  CudaImage<uint8_t> sat8;
  CudaImage<uint8_t> int8img;
  CudaConvertFloatToUint8(cuda_hsi_channels[0], 360.0F, hue8);
  CudaConvertFloatToUint8(cuda_hsi_channels[1], 1.0F, sat8);
  CudaConvertFloatToUint8(cuda_hsi_channels[2], 1.0F, int8img);
  ShowImages(
      {hue8.ToCvMat(CV_8UC1), sat8.ToCvMat(CV_8UC1), int8img.ToCvMat(CV_8UC1)},
      "HSI Channels (normalized)");

  CudaImage<uint8_t> sat_mask;
  CudaImage<uint8_t> sat_mask2;
  CudaImage<uint8_t> sat_mask3;
  CudaInRange(sat8, 150, 255, sat_mask);
  const size_t KERNEL_SIZE_1 = 5;
  CudaImage<uint8_t> morph_kernel(KERNEL_SIZE_1, KERNEL_SIZE_1, 1);
  CudaDilation(sat_mask, morph_kernel, sat_mask2);
  CudaErosion(sat_mask2, morph_kernel, sat_mask3);
  CudaImage<uint8_t> sat_mask4;
  CudaImage<uint8_t> sat_mask5;
  CudaImage<uint8_t> sat_mask6;
  CudaImage<uint8_t> sat_mask7;
  const size_t KERNEL_SIZE_2 = 27;
  CudaImage<uint8_t> morph_kernel2(KERNEL_SIZE_2, KERNEL_SIZE_2, 1);
  CudaErosion(sat_mask3, morph_kernel2, sat_mask4);
  CudaErosion(sat_mask4, morph_kernel2, sat_mask5);
  CudaDilation(sat_mask5, morph_kernel2, sat_mask6);
  CudaDilation(sat_mask6, morph_kernel2, sat_mask7);
  ShowImages({sat_mask.ToCvMat(CV_8UC1), sat_mask2.ToCvMat(CV_8UC1),
              sat_mask3.ToCvMat(CV_8UC1), sat_mask4.ToCvMat(CV_8UC1),
              sat_mask5.ToCvMat(CV_8UC1), sat_mask6.ToCvMat(CV_8UC1),
              sat_mask7.ToCvMat(CV_8UC1)},
             "Masks", 2);

  std::vector<CudaImage<uint8_t>> cuda_figure_bgrs(3);
  for (int i = 0; i < 3; i++) {
    CudaBitwiseAnd(cuda_bgr_channels[i], sat_mask7, cuda_figure_bgrs[i]);
  }
  ShowImages({cuda_figure_bgrs[0].ToCvMat(CV_8UC1),
              cuda_figure_bgrs[1].ToCvMat(CV_8UC1),
              cuda_figure_bgrs[2].ToCvMat(CV_8UC1)},
             "Masked BGR");

  cv::Mat masked_image;
  cv::merge(std::vector<cv::Mat>{cuda_figure_bgrs[0].ToCvMat(CV_8UC1),
                                 cuda_figure_bgrs[1].ToCvMat(CV_8UC1),
                                 cuda_figure_bgrs[2].ToCvMat(CV_8UC1)},
            masked_image);
  ShowImages({masked_image}, "Figures");

  CudaImage<uint8_t> gray_image;
  CudaBgrToGray(cuda_figure_bgrs, gray_image);
  CudaImage<uint8_t> sobel_image;
  CudaSobel(gray_image, sobel_image);
  ShowImages({gray_image.ToCvMat(CV_8UC1), sobel_image.ToCvMat(CV_8UC1)},
             "Gray and Sobel");

  int key;
  do {
    key = cv::waitKey(0);
  } while (key != ESC_KEY);
  cv::destroyAllWindows();
  return 0;
}