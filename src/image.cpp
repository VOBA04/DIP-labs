#include "image.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

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