#include "image.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#ifdef WITH_QT
#include <QApplication>
#include <QFileDialog>
#endif

const int ESC_KEY = 27;

int main(int argc, char *argv[]) {
  std::string image_path;
#ifdef WITH_QT
  QApplication app(argc, argv);
  QString file_name = QFileDialog::getOpenFileName(
      nullptr, "Выберите изображение", "../images/",
      "Images (*.jpg *.jpeg *.png *.bmp)");
  if (file_name.isEmpty()) {
    return 0;
  }
  image_path = file_name.toStdString();
#else
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
    return 1;
  }
  image_path = argv[1];
#endif

  cv::Mat img = cv::imread(image_path);
  if (img.empty()) {
    std::cerr << "Не удалось загрузить изображение!" << std::endl;
    return -1;
  }

  ShowImages({img}, "Original Image");

  cv::Mat hsv_image;
  cv::cvtColor(img, hsv_image, cv::COLOR_BGR2HSV);
  std::vector<cv::Mat> hsv_channels;
  cv::split(hsv_image, hsv_channels);
  // ShowImages(hsv_channels, "HSV Channels");

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
  // ShowImages(mask_processing_steps, "S Channel and Mask", 2);
  ShowImages({s_mask}, "S Channel Mask");

  cv::Mat h_figures;
  cv::Mat s_figures;
  cv::Mat v_figures;
  cv::bitwise_and(hsv_channels[0], s_mask, h_figures);
  cv::bitwise_and(hsv_channels[1], s_mask, s_figures);
  cv::bitwise_and(hsv_channels[2], s_mask, v_figures);
  // ShowImages({h_figures, s_figures, v_figures}, "Figures Channels");
  cv::Mat figures;
  cv::merge(std::vector<cv::Mat>{h_figures, s_figures, v_figures}, figures);
  cv::cvtColor(figures, figures, cv::COLOR_HSV2BGR);
  ShowImages({figures}, "Figures");

  cv::Mat gray_figures;
  cv::cvtColor(figures, gray_figures, cv::COLOR_BGR2GRAY);
  // ShowImages({gray_figures}, "Gray Figures");

  cv::Mat dist;
  cv::distanceTransform(s_mask, dist, cv::DIST_L2, 3);
  cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);
  cv::Mat dist_bin;
  cv::threshold(dist, dist_bin, 0.8, 1.0, cv::THRESH_BINARY);
  ShowImages({dist, dist_bin}, "Distance Transform");
  dist_bin.convertTo(dist_bin, CV_8U);

  cv::Mat markers;
  cv::connectedComponents(dist_bin, markers);

  cv::Mat markers_8u;
  markers.convertTo(markers_8u, CV_8U);

  cv::Mat morph_kernel_4 =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
  bool changed = true;
  while (changed) {
    changed = false;
    cv::Mat dilated;
    cv::dilate(markers_8u, dilated, morph_kernel_4);
    dilated.setTo(0, s_mask == 0);
    cv::Mat update_mask;
    cv::bitwise_and(markers_8u == 0, dilated != 0, update_mask);
    const int UPDATED_PIXELS = cv::countNonZero(update_mask);
    if (UPDATED_PIXELS == 0) {
      continue;
    }
    dilated.copyTo(markers_8u, update_mask);
    changed = true;
  }

  markers_8u.convertTo(markers, CV_32S);
  ShowMarkers(markers, "Markers");

  // cv::Mat sobel_x;
  // cv::Mat sobel_y;
  // cv::Mat sobel_xy;
  // cv::Mat sobel_kernel =
  //     (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
  // cv::filter2D(blurred_gray, sobel_x, CV_16S, sobel_kernel);
  // cv::filter2D(blurred_gray, sobel_y, CV_16S, sobel_kernel.t());
  // cv::Mat abs_sobel_x;
  // cv::Mat abs_sobel_y;
  // cv::convertScaleAbs(sobel_x, abs_sobel_x);
  // cv::convertScaleAbs(sobel_y, abs_sobel_y);
  // cv::addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0, sobel_xy);
  // cv::Mat sobel_binarized;
  // cv::threshold(sobel_xy, sobel_binarized, 10, 255, cv::THRESH_BINARY);
  // ShowImages({sobel_xy, sobel_binarized}, "Sobel on Gray Figures");

  // ShowImage3D(sobel_xy);

  int key;
  do {
    key = cv::waitKey(0);
  } while (key != ESC_KEY);
  cv::destroyAllWindows();
  return 0;
}