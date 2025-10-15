#include "image.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#ifdef WITH_QT
#include <QAbstractItemView>
#include <QColor>
#include <QDialog>
#include <QDialogButtonBox>
#include <QHeaderView>
#include <QLabel>
#include <QMessageBox>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QWidget>
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkImageData.h>
#include <vtkImageDataGeometryFilter.h>
#include <vtkImageViewer2.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkWarpScalar.h>
#endif

void ShowImages(const std::vector<cv::Mat> &images,
                const std::string &window_name, const size_t ROWS) {
  if (images.empty()) {
    throw std::invalid_argument("No images to display");
  }
  if (ROWS == 0) {
    throw std::invalid_argument("Rows must be greater than 0");
  }
  const size_t COLS = (images.size() + ROWS - 1) / ROWS;
  const int MAT_TYPE = images.front().type();
  for (const auto &img : images) {
    if (img.empty()) {
      throw std::invalid_argument("One of the images is empty");
    }
    if (img.type() != MAT_TYPE) {
      throw std::invalid_argument("All images must have the same type");
    }
  }
  std::vector<int> row_heights(ROWS, 0);
  std::vector<int> col_widths(COLS, 0);
  for (size_t idx = 0; idx < images.size(); ++idx) {
    const size_t ROW_INDEX = idx / COLS;
    const size_t COL_INDEX = idx % COLS;
    const cv::Mat &img = images[idx];
    if (img.empty()) {
      throw std::invalid_argument("One of the images is empty");
    }
    row_heights[ROW_INDEX] = std::max(row_heights[ROW_INDEX], img.rows);
    col_widths[COL_INDEX] = std::max(col_widths[COL_INDEX], img.cols);
  }
  const size_t USED_ROWS = (images.size() + COLS - 1) / COLS;
  int total_height = 0;
  int total_width = 0;
  for (size_t r_i = 0; r_i < USED_ROWS; ++r_i) {
    total_height += row_heights[r_i];
  }
  for (size_t c_i = 0; c_i < COLS; ++c_i) {
    total_width += col_widths[c_i];
  }
  if (COLS > 1) {
    total_width += static_cast<int>(COLS - 1);
  }
  if (USED_ROWS > 1) {
    total_height += static_cast<int>(USED_ROWS - 1);
  }
  cv::Mat canvas(total_height, total_width, MAT_TYPE, cv::Scalar(0));
  std::vector<int> y_offsets(ROWS, 0);
  std::vector<int> x_offsets(COLS, 0);
  for (size_t r_i = 1; r_i < ROWS; ++r_i) {
    y_offsets[r_i] = y_offsets[r_i - 1] + row_heights[r_i - 1];
  }
  for (size_t c_i = 1; c_i < COLS; ++c_i) {
    x_offsets[c_i] = x_offsets[c_i - 1] + col_widths[c_i - 1];
  }
  for (size_t idx = 0; idx < images.size(); ++idx) {
    const size_t ROW_INDEX = idx / COLS;
    const size_t COL_INDEX = idx % COLS;
    const cv::Mat &src = images[idx];
    const int Y_VAL = y_offsets[ROW_INDEX];
    const int X_VAL = x_offsets[COL_INDEX];
    cv::Rect roi(X_VAL, Y_VAL, src.cols, src.rows);
    if (roi.x + roi.width <= canvas.cols && roi.y + roi.height <= canvas.rows) {
      src.copyTo(canvas(roi));
    }
  }
  cv::namedWindow(window_name, cv::WINDOW_NORMAL);
  cv::imshow(window_name, canvas);
}

#ifdef WITH_QT
void ShowImage3D(const cv::Mat &image) {
  if (image.empty()) {
    throw std::invalid_argument("Input image is empty");
  }
  if (image.type() != CV_8UC1) {
    throw std::invalid_argument(
        "Input image must be a single-channel grayscale image (CV_8UC1)");
  }
  vtkSmartPointer<vtkImageData> vtk_img = vtkSmartPointer<vtkImageData>::New();
  int width = image.cols;
  int height = image.rows;
  vtk_img->SetDimensions(width, height, 1);
  vtk_img->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
  unsigned char *vtk_ptr =
      static_cast<unsigned char *>(vtk_img->GetScalarPointer());
  memcpy(vtk_ptr, image.data,
         static_cast<size_t>(width) * static_cast<size_t>(height));
  vtkSmartPointer<vtkImageDataGeometryFilter> geometry_filter =
      vtkSmartPointer<vtkImageDataGeometryFilter>::New();
  geometry_filter->SetInputData(vtk_img);
  double min_val = 0.0;
  double max_val = 255.0;
  {
    double min_v = 0.0;
    double max_v = 0.0;
    cv::minMaxLoc(image, &min_v, &max_v);
    min_val = min_v;
    max_val = max_v > 0.0 ? max_v : 255.0;
  }
  double target_height = std::max(width, height) * 0.2;
  double scale_factor = target_height / max_val;
  vtkSmartPointer<vtkWarpScalar> warp = vtkSmartPointer<vtkWarpScalar>::New();
  warp->SetInputConnection(geometry_filter->GetOutputPort());
  warp->SetNormal(0.0, 0.0, 1.0);
  warp->UseNormalOn();
  warp->SetScaleFactor(scale_factor);
  vtkSmartPointer<vtkPolyDataMapper> mapper =
      vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(warp->GetOutputPort());
  mapper->ScalarVisibilityOff();
  vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
  actor->GetProperty()->SetColor(0.8, 0.8, 0.8);
  vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
  renderer->AddActor(actor);
  renderer->SetBackground(0.1, 0.1, 0.2);
  renderer->ResetCamera();
  renderer->GetActiveCamera()->Azimuth(45);
  renderer->GetActiveCamera()->Elevation(30);
  renderer->GetActiveCamera()->OrthogonalizeViewUp();
  vtkSmartPointer<vtkRenderWindow> render_window =
      vtkSmartPointer<vtkRenderWindow>::New();
  render_window->AddRenderer(renderer);
  render_window->SetWindowName("VTK 3D Gray Image");
  render_window->SetSize(900, 700);
  vtkSmartPointer<vtkRenderWindowInteractor> interactor =
      vtkSmartPointer<vtkRenderWindowInteractor>::New();
  interactor->SetRenderWindow(render_window);
  render_window->Render();
  interactor->Start();
}
#endif

void ShowMarkers(const cv::Mat &markers, const std::string &window_name) {
  if (markers.empty()) {
    throw std::invalid_argument("Input markers image is empty");
  }
  if (markers.type() != CV_32S) {
    throw std::invalid_argument("Input markers image must be of type CV_32S");
  }
  double min_val;
  double max_val;
  cv::minMaxLoc(markers, &min_val, &max_val);
  cv::Mat display;
  markers.convertTo(display, CV_8U, 255.0 / (max_val - min_val),
                    -min_val * 255.0 / (max_val - min_val));
  cv::applyColorMap(display, display, cv::COLORMAP_JET);
  cv::namedWindow(window_name, cv::WINDOW_NORMAL);
  cv::imshow(window_name, display);
}

std::vector<ObjectProperties>
CalculateObjectsProperties(const cv::Mat &markers) {
  std::vector<ObjectProperties> properties;
  double min_val;
  double max_val;
  minMaxLoc(markers, &min_val, &max_val);
  int max_label = static_cast<int>(max_val);
  double alpha = 0.0;
  double beta = 0.0;
  if (max_val != min_val) {
    alpha = 255.0 / (max_val - min_val);
    beta = -min_val * alpha;
  }
  for (int label = 1; label <= max_label; ++label) {
    cv::Mat mask = (markers == label);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL,
                 cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
      continue;
    }
    const auto &contour = contours[0];
    float area = static_cast<float>(cv::contourArea(contour));
    float perimeter = static_cast<float>(cv::arcLength(contour, true));
    float elongation = 1.0;
    if (contour.size() >= 5) {
      cv::RotatedRect ellipse = cv::fitEllipse(contour);
      float major_axis = std::max(ellipse.size.width, ellipse.size.height);
      float minor_axis = std::min(ellipse.size.width, ellipse.size.height);
      if (minor_axis > 0) {
        elongation = major_axis / minor_axis;
      }
    }
    int scaled_label = 0;
    if (max_val != min_val) {
      scaled_label = cv::saturate_cast<uchar>(alpha * label + beta);
    }
    cv::Mat label_mat(1, 1, CV_8UC1, cv::Scalar(scaled_label));
    cv::Mat color_mat;
    cv::applyColorMap(label_mat, color_mat, cv::COLORMAP_JET);
    cv::Vec3b color = color_mat.at<cv::Vec3b>(0, 0);
    ObjectProperties prop = {label, area, perimeter, elongation, color};
    properties.push_back(prop);
  }
  return properties;
}

std::vector<ObjectProperties>
CalculateObjectsPropertiesCPU(const cv::Mat &markers) {
  // CPU-only implementation: no OpenCV algorithms, only basic loops
  if (markers.empty()) {
    throw std::invalid_argument("Input markers image is empty");
  }
  if (markers.type() != CV_32S) {
    throw std::invalid_argument("Input markers image must be of type CV_32S");
  }
  int width = markers.cols;
  int height = markers.rows;
  int min_label = std::numeric_limits<int>::max();
  int max_label = std::numeric_limits<int>::min();
  for (int y = 0; y < height; ++y) {
    const int *row = markers.ptr<int>(y);
    for (int x = 0; x < width; ++x) {
      const int V = row[x];
      if (V < min_label) {
        min_label = V;
      }
      if (V > max_label) {
        max_label = V;
      }
    }
  }
  if (max_label < 1) {
    return {};
  }
  struct Acc {
    uint64_t count = 0;
    uint64_t perimeter = 0;
    double sumx = 0.0;
    double sumy = 0.0;
    double sumxx = 0.0;
    double sumyy = 0.0;
    double sumxy = 0.0;
  };
  std::vector<Acc> acc(static_cast<size_t>(max_label + 1));
  auto is_out_of_bounds = [&](int nx, int ny) -> bool {
    return nx < 0 || ny < 0 || nx >= width || ny >= height;
  };
  for (int y = 0; y < height; ++y) {
    const int *row = markers.ptr<int>(y);
    for (int x = 0; x < width; ++x) {
      int label_val = row[x];
      if (label_val <= 0) {
        continue;
      }
      Acc &a = acc[static_cast<size_t>(label_val)];
      a.count += 1;
      a.sumx += static_cast<double>(x);
      a.sumy += static_cast<double>(y);
      a.sumxx += static_cast<double>(x) * static_cast<double>(x);
      a.sumyy += static_cast<double>(y) * static_cast<double>(y);
      a.sumxy += static_cast<double>(x) * static_cast<double>(y);
      {
        int nx = x;
        int ny = y - 1;
        if (is_out_of_bounds(nx, ny)) {
          a.perimeter += 1;
        } else {
          int neighbor_label = markers.ptr<int>(ny)[nx];
          if (neighbor_label != label_val) {
            a.perimeter += 1;
          }
        }
      }
      {
        int nx = x;
        int ny = y + 1;
        if (is_out_of_bounds(nx, ny)) {
          a.perimeter += 1;
        } else {
          int neighbor_label = markers.ptr<int>(ny)[nx];
          if (neighbor_label != label_val) {
            a.perimeter += 1;
          }
        }
      }
      {
        int nx = x - 1;
        int ny = y;
        if (is_out_of_bounds(nx, ny)) {
          a.perimeter += 1;
        } else {
          int neighbor_label = markers.ptr<int>(ny)[nx];
          if (neighbor_label != label_val) {
            a.perimeter += 1;
          }
        }
      }
      {
        int nx = x + 1;
        int ny = y;
        if (is_out_of_bounds(nx, ny)) {
          a.perimeter += 1;
        } else {
          int neighbor_label = markers.ptr<int>(ny)[nx];
          if (neighbor_label != label_val) {
            a.perimeter += 1;
          }
        }
      }
    }
  }
  auto clamp01 = [](double t) {
    if (t < 0.0) {
      return 0.0;
    }
    if (t > 1.0) {
      return 1.0;
    }
    return t;
  };
  auto jet_rgb = [&](double v) -> cv::Vec3b {
    double r = clamp01(1.5 - std::fabs(4.0 * v - 3.0));
    double g = clamp01(1.5 - std::fabs(4.0 * v - 2.0));
    double b = clamp01(1.5 - std::fabs(4.0 * v - 1.0));
    auto r8 = static_cast<unsigned char>(std::round(r * 255.0));
    auto g8 = static_cast<unsigned char>(std::round(g * 255.0));
    auto b8 = static_cast<unsigned char>(std::round(b * 255.0));
    return {b8, g8, r8};
  };
  double alpha = 0.0;
  double beta = 0.0;
  if (max_label != min_label) {
    alpha = 255.0 / static_cast<double>(max_label - min_label);
    beta = -static_cast<double>(min_label) * alpha;
  }
  std::vector<ObjectProperties> properties;
  properties.reserve(static_cast<size_t>(max_label));
  for (int label = 1; label <= max_label; ++label) {
    const Acc &a = acc[static_cast<size_t>(label)];
    if (a.count == 0) {
      continue;
    }
    auto area_f = static_cast<float>(a.count);
    auto perimeter_f = static_cast<float>(a.perimeter);
    float elongation = 1.0F;
    if (a.count >= 2) {
      auto n = static_cast<double>(a.count);
      double mx = a.sumx / n;
      double my = a.sumy / n;
      double varx = a.sumxx / n - mx * mx;
      double vary = a.sumyy / n - my * my;
      double covxy = a.sumxy / n - mx * my;
      double tr = varx + vary;
      double disc = tr * tr - 4.0 * (varx * vary - covxy * covxy);
      double root = disc > 0.0 ? std::sqrt(disc) : 0.0;
      double l1 = 0.5 * (tr + root);
      double l2 = 0.5 * (tr - root);
      double eps = 1e-12;
      if (l2 > eps) {
        elongation = static_cast<float>(std::sqrt(l1 / l2));
      } else {
        elongation = 1.0F;
      }
    }
    int scaled_label = 0;
    if (max_label != min_label) {
      double sval = alpha * static_cast<double>(label) + beta;
      if (sval < 0.0) {
        sval = 0.0;
      }
      if (sval > 255.0) {
        sval = 255.0;
      }
      scaled_label = static_cast<int>(std::round(sval));
    }
    double val = static_cast<double>(scaled_label) / 255.0;
    cv::Vec3b color_bgr = jet_rgb(val);
    properties.push_back(
        ObjectProperties{label, area_f, perimeter_f, elongation, color_bgr});
  }
  return properties;
}

void DisplayObjectProperties(const std::vector<ObjectProperties> &properties) {
#ifdef WITH_QT
  if (properties.empty()) {
    QMessageBox::information(nullptr, QStringLiteral("Параметры сегментов"),
                             QStringLiteral("Сегменты не найдены."));
    return;
  }
  QDialog dialog;
  dialog.setWindowTitle(QStringLiteral("Параметры сегментов"));
  dialog.resize(520, 360);

  auto *layout = new QVBoxLayout(&dialog);

  auto *label = new QLabel(QStringLiteral("Найдено %1 сегментов")
                               .arg(static_cast<int>(properties.size())));
  label->setAlignment(Qt::AlignLeft);
  layout->addWidget(label);

  auto *table =
      new QTableWidget(static_cast<int>(properties.size()), 5, &dialog);
  table->setHorizontalHeaderLabels(
      {QStringLiteral("№"), QStringLiteral("Площадь"),
       QStringLiteral("Периметр"), QStringLiteral("Удлинение"),
       QStringLiteral("Цвет")});
  table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
  table->verticalHeader()->setVisible(false);
  table->setSelectionMode(QAbstractItemView::NoSelection);
  table->setEditTriggers(QAbstractItemView::NoEditTriggers);

  for (int row = 0; row < static_cast<int>(properties.size()); ++row) {
    const auto &prop = properties[static_cast<size_t>(row)];
    table->setItem(row, 0, new QTableWidgetItem(QString::number(prop.number)));
    table->setItem(row, 1,
                   new QTableWidgetItem(QString::number(
                       static_cast<double>(prop.area), 'f', 2)));
    table->setItem(row, 2,
                   new QTableWidgetItem(QString::number(
                       static_cast<double>(prop.perimeter), 'f', 2)));
    table->setItem(row, 3,
                   new QTableWidgetItem(QString::number(
                       static_cast<double>(prop.elongation), 'f', 2)));
    auto *color_item = new QTableWidgetItem;
    QColor segment_color(prop.color[2], prop.color[1], prop.color[0]);
    color_item->setData(Qt::DecorationRole, segment_color);
    color_item->setFlags(Qt::ItemIsEnabled);
    table->setItem(row, 4, color_item);
  }

  layout->addWidget(table);

  auto *buttons =
      new QDialogButtonBox(QDialogButtonBox::Ok, Qt::Horizontal, &dialog);
  QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog,
                   &QDialog::accept);
  layout->addWidget(buttons);

  dialog.exec();
#else
  if (properties.empty()) {
    std::cout << "Segment properties: no objects found." << std::endl;
    return;
  }

  std::cout << "Segment properties (count = " << properties.size() << ")"
            << std::endl;
  std::cout << std::left << std::setw(6) << "#" << std::setw(12) << "Area"
            << std::setw(12) << "Perimeter" << std::setw(12) << "Elongation"
            << std::setw(12) << "Color(B,G,R)" << std::endl;
  std::cout << std::string(54, '-') << std::endl;
  std::cout.setf(std::ios::fixed);
  std::cout << std::setprecision(2);
  for (const auto &prop : properties) {
    std::ostringstream color_stream;
    color_stream << static_cast<int>(prop.color[0]) << ','
                 << static_cast<int>(prop.color[1]) << ','
                 << static_cast<int>(prop.color[2]);
    std::cout << std::left << std::setw(6) << prop.number << std::setw(12)
              << prop.area << std::setw(12) << prop.perimeter << std::setw(12)
              << prop.elongation << std::setw(12) << color_stream.str()
              << std::endl;
  }
  std::cout.unsetf(std::ios::fixed);
  std::cout << std::defaultfloat;
#endif
}

// Helper to draw PCA axes for a single contour onto a BGR canvas
static void DrawPCAAxesOnContour(cv::Mat &canvas,
                                 const std::vector<cv::Point> &cnt) {
  if (cnt.size() < 5) {
    return;
  }
  CV_Assert(cnt.size() <= static_cast<size_t>(std::numeric_limits<int>::max()));
  cv::Mat data_pts(static_cast<int>(cnt.size()), 2, CV_64F);
  for (int i = 0; i < data_pts.rows; ++i) {
    data_pts.at<double>(i, 0) =
        static_cast<double>(cnt[static_cast<size_t>(i)].x);
    data_pts.at<double>(i, 1) =
        static_cast<double>(cnt[static_cast<size_t>(i)].y);
  }
  cv::PCA pca_analysis(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);
  double cx = pca_analysis.mean.at<double>(0, 0);
  double cy = pca_analysis.mean.at<double>(0, 1);
  cv::Vec2d v1 = pca_analysis.eigenvectors.row(0);
  cv::Vec2d v2 = pca_analysis.eigenvectors.row(1);

  double min1 = 0.0;
  double max1 = 0.0;
  double min2 = 0.0;
  double max2 = 0.0;
  bool first = true;
  for (const auto &p : cnt) {
    double dx = static_cast<double>(p.x) - cx;
    double dy = static_cast<double>(p.y) - cy;
    double t1 = dx * v1[0] + dy * v1[1];
    double t2 = dx * v2[0] + dy * v2[1];
    if (first) {
      min1 = max1 = t1;
      min2 = max2 = t2;
      first = false;
    } else {
      if (t1 < min1) {
        min1 = t1;
      }
      if (t1 > max1) {
        max1 = t1;
      }
      if (t2 < min2) {
        min2 = t2;
      }
      if (t2 > max2) {
        max2 = t2;
      }
    }
  }

  auto to_pt = [&](const cv::Point2d &pt) {
    return cv::Point(static_cast<int>(std::round(pt.x)),
                     static_cast<int>(std::round(pt.y)));
  };

  cv::Point2d c(cx, cy);
  cv::Point p1a = to_pt(c + cv::Point2d(v1[0], v1[1]) * max1);
  cv::Point p1b = to_pt(c + cv::Point2d(v1[0], v1[1]) * min1);
  cv::Point p2a = to_pt(c + cv::Point2d(v2[0], v2[1]) * max2);
  cv::Point p2b = to_pt(c + cv::Point2d(v2[0], v2[1]) * min2);

  cv::circle(canvas, to_pt(c), 3, cv::Scalar(0, 255, 255), cv::FILLED,
             cv::LINE_AA);
  cv::line(canvas, p1a, p1b, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
  cv::line(canvas, p2a, p2b, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
}

void ShowPCA(const cv::Mat &markers, const cv::Mat &image,
             const std::string &window_name) {
  if (markers.empty()) {
    throw std::invalid_argument("Input image is empty");
  }
  if (markers.type() != CV_8UC1) {
    throw std::invalid_argument(
        "Input image must be single-channel CV_8U (grayscale)");
  }
  if (image.empty()) {
    throw std::invalid_argument("Input image is empty");
  }
  if (image.type() != CV_8UC3) {
    throw std::invalid_argument(
        "Input image must be 3-channel CV_8UC3 (BGR color)");
  }
  if (markers.size() != image.size()) {
    throw std::invalid_argument("Input images must have the same size");
  }

  // Find separate figures by contours
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(markers, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  auto to_pt = [&](const cv::Point2d &p) {
    return cv::Point(static_cast<int>(std::round(p.x)),
                     static_cast<int>(std::round(p.y)));
  };

  cv::Mat canvas = image.clone();
  for (const auto &cnt : contours) {
    if (cnt.size() < 5) {
      continue; // PCA needs enough points
    }

    // Build data matrix for PCA from contour points
    CV_Assert(cnt.size() <=
              static_cast<size_t>(std::numeric_limits<int>::max()));
    cv::Mat data_pts(static_cast<int>(cnt.size()), 2, CV_64F);
    for (int i = 0; i < data_pts.rows; i++) {
      data_pts.at<double>(i, 0) =
          static_cast<double>(cnt[static_cast<size_t>(i)].x);
      data_pts.at<double>(i, 1) =
          static_cast<double>(cnt[static_cast<size_t>(i)].y);
    }

    cv::PCA pca_analysis(data_pts, cv::Mat(), cv::PCA::DATA_AS_ROW);
    double cx = pca_analysis.mean.at<double>(0, 0);
    double cy = pca_analysis.mean.at<double>(0, 1);
    cv::Vec2d v1 = pca_analysis.eigenvectors.row(0); // principal dir
    cv::Vec2d v2 = pca_analysis.eigenvectors.row(1); // secondary dir

    // Compute extents along each axis by projecting contour points
    double min1 = 0.0;
    double max1 = 0.0;
    double min2 = 0.0;
    double max2 = 0.0;
    bool first = true;
    for (const auto &p : cnt) {
      const double DX = static_cast<double>(p.x) - cx;
      const double DY = static_cast<double>(p.y) - cy;
      const double T1 = DX * v1[0] + DY * v1[1];
      const double T2 = DX * v2[0] + DY * v2[1];
      if (first) {
        min1 = max1 = T1;
        min2 = max2 = T2;
        first = false;
      } else {
        if (T1 < min1) {
          min1 = T1;
        }
        if (T1 > max1) {
          max1 = T1;
        }
        if (T2 < min2) {
          min2 = T2;
        }
        if (T2 > max2) {
          max2 = T2;
        }
      }
    }

    // Endpoints for axes spanning the figure
    cv::Point2d c(cx, cy);
    cv::Point p1a = to_pt(c + cv::Point2d(v1[0], v1[1]) * max1);
    cv::Point p1b = to_pt(c + cv::Point2d(v1[0], v1[1]) * min1);
    cv::Point p2a = to_pt(c + cv::Point2d(v2[0], v2[1]) * max2);
    cv::Point p2b = to_pt(c + cv::Point2d(v2[0], v2[1]) * min2);

    // Draw centroid and axes for this figure
    cv::line(canvas, p1a, p1b, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    cv::line(canvas, p2a, p2b, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    cv::circle(canvas, to_pt(c), 3, cv::Scalar(0, 255, 255), cv::FILLED,
               cv::LINE_AA);
  }

  cv::namedWindow(window_name, cv::WINDOW_NORMAL);
  cv::imshow(window_name, canvas);
}