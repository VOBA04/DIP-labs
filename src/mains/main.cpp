#include "image.h"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/core/base.hpp>
#include <opencv2/core/types.hpp>
#include <ostream>
#include <sstream>
#ifndef WITH_QT
#include <limits>
#endif
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include <torch/torch.h>

#ifdef WITH_QT
#include <QApplication>
#include <QFileDialog>
#include <QInputDialog>
#include <QString>
#endif

const int ESC_KEY = 27;

namespace fs = std::filesystem;

#include "digitnet.h"

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

  std::vector<cv::Mat> masks;
  cv::Mat g_mask;
  cv::inRange(h_figures, 35, 95, g_mask);
  masks.push_back(g_mask.clone());
  cv::Mat kernel_1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::morphologyEx(g_mask, g_mask, cv::MORPH_ERODE, kernel_1);
  masks.push_back(g_mask.clone());
  cv::Mat kernel_2 =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 11));
  cv::morphologyEx(g_mask, g_mask, cv::MORPH_DILATE, kernel_2);
  masks.push_back(g_mask.clone());
  ShowImages(masks, "G Masks");

  cv::Mat markers;
  cv::connectedComponents(g_mask, markers);

  cv::Mat markers_8u;
  markers.convertTo(markers_8u, CV_8U);

  cv::Mat morph_kernel_4 =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
  bool changed = true;
  while (changed) {
    changed = false;
    cv::Mat dilated;
    cv::dilate(markers_8u, dilated, morph_kernel_4);
    dilated.setTo(0, g_mask == 0);
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

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(markers_8u, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  std::vector<cv::Mat> digits;
  for (size_t i = 0; i < contours.size(); i++) {
    std::vector<cv::Point> pts = contours[i];
    if (pts.size() < 5) {
      continue;
    }
    CV_Assert(pts.size() <=
              static_cast<size_t>(std::numeric_limits<int>::max()));
    cv::Mat data_pts = cv::Mat(static_cast<int>(pts.size()), 2, CV_64F);
    for (int j = 0; j < pts.size(); j++) {
      data_pts.at<double>(j, 0) = pts[j].x;
      data_pts.at<double>(j, 1) = pts[j].y;
    }
    cv::PCA pca_analysis(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Point2d center(pca_analysis.mean.at<double>(0, 0),
                       pca_analysis.mean.at<double>(0, 1));
    cv::Vec2d eigen_vec = pca_analysis.eigenvectors.row(0);
    double angle = atan2(eigen_vec[1], eigen_vec[0]) * 180.0 / CV_PI;
    angle -= 90.0;
    cv::Rect bbox = cv::boundingRect(pts);
    const int EXPAND_PERCENT = 15;
    int expand_x = static_cast<int>(bbox.width * EXPAND_PERCENT / 100.0);
    int expand_y = static_cast<int>(bbox.height * EXPAND_PERCENT / 100.0);
    int x = std::max(bbox.x - expand_x, 0);
    int y = std::max(bbox.y - expand_y, 0);
    int w = std::min(bbox.width + 2 * expand_x, markers_8u.cols - x);
    int h = std::min(bbox.height + 2 * expand_y, markers_8u.rows - y);
    cv::Rect expanded_bbox(x, y, w, h);
    cv::Mat roi = markers_8u(expanded_bbox);
    cv::Mat rot_mat = cv::getRotationMatrix2D(
        cv::Point2f(static_cast<float>(roi.cols) / 2.0F,
                    static_cast<float>(roi.rows) / 2.0F),
        angle, 1.0F);
    cv::Mat rotated;
    cv::warpAffine(roi, rotated, rot_mat, roi.size(), cv::INTER_CUBIC);
    cv::normalize(rotated, rotated, 0, 255, cv::NORM_MINMAX);
    cv::Moments m = cv::moments(rotated, true);
    double cy = m.m01 / m.m00;
    double center_y = static_cast<double>(rotated.rows) / 2.0;
    if (cy > center_y) {
      angle += 180.0;
      rot_mat = cv::getRotationMatrix2D(
          cv::Point2f(static_cast<float>(roi.cols) / 2.0F,
                      static_cast<float>(roi.rows) / 2.0F),
          angle, 1.0F);
      cv::warpAffine(roi, rotated, rot_mat, roi.size(), cv::INTER_CUBIC);
      cv::normalize(rotated, rotated, 0, 255, cv::NORM_MINMAX);
    }
    cv::threshold(rotated, rotated, 128, 255, cv::THRESH_BINARY);
    if (cv::countNonZero(rotated) > 10000) {
      digits.emplace_back(rotated.clone());
      continue;
    }
  }

  ShowPCA(markers_8u, img, "PCA Axes");
  ShowImages(digits, "Digit ROIs (PCA aligned)");

  auto dir = fs::directory_entry(MODELS);
  if (!dir.exists() || !dir.is_directory()) {
    std::cout << "Нет каталога с моделями. Сначала обучите модель\n";
    int key;
    do {
      key = cv::waitKey(0);
    } while (key != ESC_KEY);
    cv::destroyAllWindows();
    return 0;
  }
  std::string model_path;
#ifdef WITH_QT
  {
    QString chosen = QFileDialog::getOpenFileName(nullptr, "Выберите модель",
                                                  QString::fromUtf8(MODELS),
                                                  "Torch Model (*.pt)");
    if (chosen.isEmpty()) {
      std::cerr << "Модель не выбрана. Завершение." << std::endl;
      int key;
      do {
        key = cv::waitKey(0);
      } while (key != ESC_KEY);
      cv::destroyAllWindows();
      return 0;
    }
    model_path = chosen.toStdString();
  }
#else
  {
    std::cout << "Введите путь к модели (.pt)\nПример: " << MODELS
              << "/digit_model_mnist_5.pt\n> ";
    std::getline(std::cin, model_path);
    if (model_path.empty()) {
      model_path = std::string(MODELS) + "/digit_model_mnist_5.pt";
    }
  }
#endif

  try {
    torch::Device device(
        torch::cuda::is_available()
            ? torch::kCUDA
            : (torch::mps::is_available() ? torch::kMPS : torch::kCPU));
    auto model = std::make_shared<DigitNet>();
    torch::load(model, model_path);
    model->to(device);
    model->eval();
    torch::NoGradGuard no_grad;

    std::vector<int> preds(digits.size(), -1);
    std::vector<float> confs(digits.size(), 0.0F);
    for (size_t i = 0; i < digits.size(); ++i) {
      const cv::Mat &digit = digits[i];
      if (digit.empty()) {
        continue;
      }
      auto x = MatToTensor28x28(digit);
      if (device.is_cuda()) {
        x = x.to(torch::kCUDA, true);
      } else if (device.is_mps()) {
        x = x.to(torch::kMPS, false);
      }
      auto out = model->Forward(x);           // log-probs [1,10]
      auto probs = out.exp().to(torch::kCPU); // convert to probabilities
      int pred = probs.argmax(1).item<int>();
      auto prob = probs[0][pred].item<float>();
      preds[i] = pred;
      confs[i] = prob;
    }

    std::vector<cv::Mat> digits_annotated;
    digits_annotated.reserve(digits.size());
    for (size_t i = 0; i < digits.size(); ++i) {
      if (preds[i] >= 0) {
        std::ostringstream oss;
        oss << preds[i] << " (" << std::fixed << std::setprecision(1)
            << (confs[i] * 100.0F) << "%)";
        std::string text = oss.str();
        cv::Mat canvas = AddBottomText(digits[i], 30, text);
        digits_annotated.push_back(canvas);
      } else {
        digits_annotated.push_back(digits[i]);
      }
    }
    if (!digits_annotated.empty()) {
      ShowImages(digits_annotated, "Digit Predictions");
    }
  } catch (const std::exception &e) {
    std::cerr << "Ошибка при загрузке/инференсе модели: " << e.what()
              << std::endl;
  }

  int key;
  do {
    key = cv::waitKey(0);
  } while (key != ESC_KEY);
  cv::destroyAllWindows();
  return 0;
}