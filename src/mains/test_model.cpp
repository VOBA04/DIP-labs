// Тестирование модели распознавания цифр на выборке MNIST
// (./external/mnist/mnist/test) Пользователь задаёт путь к модели (.pt) из
// папки ./models: через argv[1] или диалог Qt (если собран WITH_QT)

#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <torch/torch.h>
#include <utility>
#include <vector>

#include "digitnet.h"
#include "image.h"

#ifdef WITH_QT
#include <QApplication>
#include <QFileDialog>
#include <QString>
#endif

namespace fs = std::filesystem;

// Загрузка тестовой выборки MNIST из EXTERNAL "/mnist/mnist/test"
static auto LoadMnistTest(const std::string &root)
    -> std::vector<std::pair<torch::Tensor, int>> {
  std::vector<std::pair<torch::Tensor, int>> data;
  if (!fs::exists(root) || !fs::is_directory(root)) {
    throw std::runtime_error("MNIST test directory not found: " + root);
  }
  for (const auto &file : fs::directory_iterator(root)) {
    if (!file.is_regular_file()) {
      continue;
    }
    auto fname = file.path().filename().string();
    size_t i = 0;
    while (i < fname.size() &&
           (std::isdigit(static_cast<unsigned char>(fname[i])) != 0)) {
      ++i;
    }
    int label = (i > 0) ? std::stoi(fname.substr(0, i)) : -1;
    if (label < 0 || label > 9) {
      continue;
    }
    auto tensor = LoadImage(file.path().string());
    data.emplace_back(tensor, label);
  }
  return data;
}

static void PrintProgress(size_t current, size_t total) {
  const int BAR_WIDTH = 30;
  double ratio = (total != 0U)
                     ? static_cast<double>(current) / static_cast<double>(total)
                     : 1.0;
  int filled = static_cast<int>(ratio * BAR_WIDTH);
  int pct = static_cast<int>(ratio * 100.0);
  std::cout << "\r[";
  for (int i = 0; i < BAR_WIDTH; ++i) {
    std::cout << (i < filled ? '#' : '.');
  }
  std::cout << "]  " << std::setw(3) << pct << "%  (" << current << "/" << total
            << ")" << std::flush;
}

auto main(int argc, char *argv[]) -> int {
  std::string model_path;

#ifdef WITH_QT
  QApplication app(argc, argv);
  if (argc >= 2) {
    model_path = argv[1];
  } else {
    QString filters = "Torch Model (*.pt)";
    QString start_dir = QString::fromUtf8(MODELS);
    QString chosen = QFileDialog::getOpenFileName(
        nullptr, QObject::tr("Выберите модель"), start_dir, filters);
    if (chosen.isEmpty()) {
      return 0;
    }
    model_path = chosen.toStdString();
  }
#else
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_model.pt>" << std::endl;
    std::cerr << "Example: " << argv[0] << " " << MODELS
              << "/digit_model_mnist_5.pt" << std::endl;
    return 1;
  }
  model_path = argv[1];
#endif

  try {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                     : torch::kCPU);
    std::cout << "Device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
    const std::string TEST_ROOT = std::string(EXTERNAL) + "/mnist/mnist/test";
    std::cout << "Loading MNIST test set from: " << TEST_ROOT << std::endl;
    auto dataset = LoadMnistTest(TEST_ROOT);
    if (dataset.empty()) {
      std::cerr << "Test set is empty. Check path and data." << std::endl;
      return 2;
    }
    const auto N = static_cast<int64_t>(dataset.size());
    std::cout << "Samples: " << N << std::endl;
    std::vector<torch::Tensor> imgs;
    imgs.reserve(N);
    std::vector<int64_t> labels;
    labels.reserve(N);
    for (const auto &p : dataset) {
      imgs.push_back(p.first);
      labels.push_back(p.second);
    }
    auto inputs_cpu = torch::cat(imgs, 0).contiguous(); // [N,1,28,28]
    auto labels_cpu =
        torch::from_blob(labels.data(), {N}, torch::kLong).clone();
    if (device.is_cuda()) {
      inputs_cpu = inputs_cpu.pin_memory();
      labels_cpu = labels_cpu.pin_memory();
    }

    // Модель
    auto model = std::make_shared<DigitNet>();
    torch::load(model, model_path);
    model->to(device);
    model->eval();

    torch::NoGradGuard no_grad;
    const int64_t BATCH = 512;
    double total_loss = 0.0;
    int64_t correct = 0;

    for (int64_t start = 0; start < N; start += BATCH) {
      const int64_t BS = std::min<int64_t>(BATCH, N - start);
      auto x = inputs_cpu.narrow(0, start, BS);
      auto y = labels_cpu.narrow(0, start, BS);
      if (device.is_cuda()) {
        x = x.to(torch::kCUDA, true);
        y = y.to(torch::kCUDA, true);
      }
      auto out = model->Forward(x);
      auto loss = torch::nll_loss(out, y);
      total_loss +=
          static_cast<double>(loss.item<float>()) * static_cast<double>(BS);
      auto pred = out.argmax(1);
      correct += pred.eq(y).sum().item<int64_t>();

      PrintProgress(static_cast<size_t>(std::min<int64_t>(start + BS, N)),
                    static_cast<size_t>(N));
    }
    PrintProgress(static_cast<size_t>(N), static_cast<size_t>(N));
    std::cout << "\n";

    const double AVG_LOSS = total_loss / static_cast<double>(N);
    const double ACC =
        100.0 * static_cast<double>(correct) / static_cast<double>(N);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Test Loss: " << AVG_LOSS
              << " | Accuracy: " << std::setprecision(2) << ACC << "%\n";
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 3;
  }

  return 0;
}
