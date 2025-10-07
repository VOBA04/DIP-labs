#include "image.h"
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#ifdef WITH_QT
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

  // Преобразуем 2D пиксельные значения в высоту (Z) через WarpScalar
  vtkSmartPointer<vtkImageDataGeometryFilter> geometry_filter =
      vtkSmartPointer<vtkImageDataGeometryFilter>::New();
  geometry_filter->SetInputData(vtk_img);

  // Масштабируем высоту по максимальному значению интенсивности
  double min_val = 0.0;
  double max_val = 255.0;
  {
    double min_v = 0.0;
    double max_v = 0.0;
    cv::minMaxLoc(image, &min_v, &max_v);
    min_val = min_v;
    max_val = max_v > 0.0 ? max_v : 255.0;
  }
  double target_height = std::max(width, height) * 0.2; // 20% от размера сетки
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