#include "image.h"

#include <iostream>
#ifndef WITH_QT
#include <limits>
#endif
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#ifdef WITH_QT
#include <QApplication>
#include <QFileDialog>
#include <QInputDialog>
#include <QString>
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

  auto properties = CalculateObjectsPropertiesCPU(markers);
  DisplayObjectProperties(properties);

  if (!properties.empty()) {
    int num_clusters = 3;
#ifdef WITH_QT
    bool ok = false;
    num_clusters =
        QInputDialog::getInt(nullptr, QStringLiteral("Количество кластеров"),
                             QStringLiteral("Введите K (1..%1):")
                                 .arg(static_cast<int>(properties.size())),
                             3, 1, static_cast<int>(properties.size()), 1, &ok);
    if (!ok) {
      num_clusters = 0;
    }
#else
    std::cout << "Введите количество кластеров K (1.." << properties.size()
              << "): ";
    int k_input = 3;
    if (!(std::cin >> k_input)) {
      std::cin.clear();
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      k_input = 3;
    }
    num_clusters =
        std::max(1, std::min(k_input, static_cast<int>(properties.size())));
#endif
    if (num_clusters >= 1) {
      cv::Mat features(static_cast<int>(properties.size()), 3, CV_32F);
      for (size_t i = 0; i < properties.size(); ++i) {
        features.at<float>(static_cast<int>(i), 0) = properties[i].area;
        features.at<float>(static_cast<int>(i), 1) = properties[i].perimeter;
        features.at<float>(static_cast<int>(i), 2) = properties[i].elongation;
      }
      cv::Mat labels;
      cv::kmeans(
          features, num_clusters, labels,
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                           100, 1.0),
          3, cv::KMEANS_PP_CENTERS);
      const std::vector<cv::Vec3b> BASE_COLORS = {
          cv::Vec3b(255, 0, 0),   // синий (BGR)
          cv::Vec3b(0, 0, 255),   // красный
          cv::Vec3b(0, 255, 0),   // зеленый
          cv::Vec3b(0, 255, 255), // желтый
          cv::Vec3b(255, 0, 255), // маджента
          cv::Vec3b(255, 255, 0), // циан
          cv::Vec3b(0, 128, 255), // оранжевый
          cv::Vec3b(128, 0, 128), // фиолетовый
          cv::Vec3b(128, 128, 0), // оливковый
          cv::Vec3b(128, 0, 0)    // бордовый
      };
      std::vector<cv::Vec3b> cluster_colors;
      cluster_colors.reserve(static_cast<size_t>(num_clusters));
      for (int i = 0; i < num_clusters; ++i) {
        cluster_colors.push_back(
            BASE_COLORS[static_cast<size_t>(i % BASE_COLORS.size())]);
      }
      cv::Mat cluster_img = cv::Mat::zeros(markers.size(), CV_8UC3);
      for (size_t i = 0; i < properties.size(); ++i) {
        int label = labels.at<int>(static_cast<int>(i), 0);
        int obj_num = properties[i].number;
        cv::Mat mask = (markers == obj_num);
        cluster_img.setTo(
            cluster_colors[static_cast<size_t>(label % cluster_colors.size())],
            mask);
      }
      ShowImages({cluster_img}, "Clusters");
    }
  }

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(markers_8u, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  std::vector<cv::Mat> digits(contours.size());
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
    digits[i] = rotated;
  }
  ShowImages(digits, "Digit ROIs (PCA aligned)");

  int key;
  do {
    key = cv::waitKey(0);
  } while (key != ESC_KEY);
  cv::destroyAllWindows();
  return 0;
}