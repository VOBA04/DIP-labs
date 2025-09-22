#include "image.h"

#include <QApplication>
#include <QFileDialog>
#include <iostream>
#include <opencv2/core.hpp>
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

  cv::Mat hsv_image;
  cv::cvtColor(img, hsv_image, cv::COLOR_BGR2HSV);
  std::vector<cv::Mat> hsv_channels;
  cv::split(hsv_image, hsv_channels);
  ShowImages(hsv_channels, "HSV Channels");

  cv::Mat s_channel = hsv_channels[1];
  cv::Mat s_mask;
  std::vector<cv::Mat> mask_processing_steps;
  mask_processing_steps.push_back(s_channel.clone());
  cv::inRange(s_channel, cv::Scalar(182), cv::Scalar(255), s_mask);
  mask_processing_steps.push_back(s_mask.clone());
  cv::Mat morph_kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
  // Размыкание
  cv::erode(s_mask, s_mask, morph_kernel);
  mask_processing_steps.push_back(s_mask.clone());
  cv::dilate(s_mask, s_mask, morph_kernel);
  mask_processing_steps.push_back(s_mask.clone());
  cv::Mat morph_kernel_2 =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
  // Размыкание
  cv::erode(s_mask, s_mask, morph_kernel_2);
  mask_processing_steps.push_back(s_mask.clone());
  cv::dilate(s_mask, s_mask, morph_kernel_2);
  mask_processing_steps.push_back(s_mask.clone());
  cv::Mat morph_kernel_3 =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25));
  // Замыкание
  cv::dilate(s_mask, s_mask, morph_kernel_3);
  mask_processing_steps.push_back(s_mask.clone());
  cv::erode(s_mask, s_mask, morph_kernel_3);
  mask_processing_steps.push_back(s_mask.clone());
  ShowImages(mask_processing_steps, "S Channel and Mask", 2);

  cv::Mat h_figures;
  cv::Mat s_figures;
  cv::Mat v_figures;
  cv::bitwise_and(hsv_channels[0], s_mask, h_figures);
  cv::bitwise_and(hsv_channels[1], s_mask, s_figures);
  cv::bitwise_and(hsv_channels[2], s_mask, v_figures);
  ShowImages({h_figures, s_figures, v_figures}, "Figures Channels");
  cv::Mat figures;
  cv::merge(std::vector<cv::Mat>{h_figures, s_figures, v_figures}, figures);
  cv::cvtColor(figures, figures, cv::COLOR_HSV2BGR);
  ShowImages({figures}, "Figures");

  cv::Mat gray_figures;
  cv::cvtColor(figures, gray_figures, cv::COLOR_BGR2GRAY);
  ShowImages({gray_figures}, "Gray Figures");
  cv::Mat sobel_x;
  cv::Mat sobel_y;
  cv::Mat sobel_xy;
  cv::Mat sobel_kernel =
      (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
  cv::filter2D(gray_figures, sobel_x, CV_16S, sobel_kernel);
  cv::filter2D(gray_figures, sobel_y, CV_16S, sobel_kernel.t());
  cv::Mat abs_sobel_x;
  cv::Mat abs_sobel_y;
  cv::convertScaleAbs(sobel_x, abs_sobel_x);
  cv::convertScaleAbs(sobel_y, abs_sobel_y);
  cv::addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0, sobel_xy);
  ShowImages({sobel_xy}, "Sobel on Gray Figures");

  int key;
  do {
    key = cv::waitKey(0);
  } while (key != ESC_KEY);
  cv::destroyAllWindows();
  return 0;
}