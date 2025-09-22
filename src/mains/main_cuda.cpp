#include "cuda_image.cuh"
#include "cuda_processing.h"
#include "image.h"

#include <QApplication>
#include <QFileDialog>
#include <cstdint>
#include <iostream>
#include <opencv2/core.hpp>
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
  // Convert H,S,I to 8-bit for display
  cv::Mat hue_f = cuda_hsi_channels[0].ToCvMat(CV_32FC1);
  cv::Mat sat_f = cuda_hsi_channels[1].ToCvMat(CV_32FC1);
  cv::Mat int_f = cuda_hsi_channels[2].ToCvMat(CV_32FC1);
  cv::Mat hue_8u;
  cv::Mat sat_8u;
  cv::Mat int_8u;
  hue_f.convertTo(hue_8u, CV_8UC1, 255.0 / 360.0);
  sat_f.convertTo(sat_8u, CV_8UC1, 255.0);
  int_f.convertTo(int_8u, CV_8UC1, 255.0);
  ShowImages({hue_8u, sat_8u, int_8u}, "HSI Channels (normalized)");

  // Convert HSI back to BGR (8-bit) and display
  std::vector<CudaImage<uint8_t>> bgr_from_hsi(3);
  CudaHsiToBgr(cuda_hsi_channels, bgr_from_hsi);
  cv::Mat b_mat = bgr_from_hsi[0].ToCvMat(CV_8UC1);
  cv::Mat g_mat = bgr_from_hsi[1].ToCvMat(CV_8UC1);
  cv::Mat r_mat = bgr_from_hsi[2].ToCvMat(CV_8UC1);
  cv::Mat bgr_recon;
  cv::merge(std::vector<cv::Mat>{b_mat, g_mat, r_mat}, bgr_recon);
  ShowImages({bgr_recon}, "BGR reconstructed from HSI");

  int key;
  do {
    key = cv::waitKey(0);
  } while (key != ESC_KEY);
  cv::destroyAllWindows();
  return 0;
}