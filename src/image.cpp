#include "image.h"
#include <algorithm>
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