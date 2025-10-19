#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
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
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QLabel>
#include <QSpinBox>
#include <QStringList>
#endif

namespace fs = std::filesystem;

auto LoadPrintedDigitsDataset(const std::string &root)
    -> std::vector<std::pair<torch::Tensor, int>> {
  std::vector<std::pair<torch::Tensor, int>> data;
  for (const auto &dir : fs::directory_iterator(root)) {
    if (!dir.is_directory()) {
      continue;
    }
    const auto NAME = dir.path().filename().string();
    int label = -1;
    try {
      label = std::stoi(NAME);
    } catch (...) {
      continue;
    }
    if (label < 0 || label > 9) {
      continue;
    }
    for (const auto &file : fs::directory_iterator(dir)) {
      if (!file.is_regular_file()) {
        continue;
      }
      auto tensor = LoadImage(file.path().string());
      data.emplace_back(tensor, label);
    }
  }
  return data;
}

auto LoadMnistDataset(const std::string &root)
    -> std::vector<std::pair<torch::Tensor, int>> {
  std::vector<std::pair<torch::Tensor, int>> data;
  for (const auto &file : fs::directory_iterator(root)) {
    if (!file.is_regular_file()) {
      continue;
    }
    auto fname = file.path().filename().string();
    size_t i = 0;
    while (i < fname.size() &&
           (std::isdigit(static_cast<unsigned char>(fname[i])) != 0)) {
      i++;
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

// Hybrid dataset: digit '0' from MNIST, digits '1'-'9' from printed_digits
auto LoadHybridDataset(const std::string &mnist_root,
                       const std::string &printed_root)
    -> std::vector<std::pair<torch::Tensor, int>> {
  std::vector<std::pair<torch::Tensor, int>> data;

  // Load only '0' class from MNIST
  for (const auto &file : fs::directory_iterator(mnist_root)) {
    if (!file.is_regular_file()) {
      continue;
    }
    auto fname = file.path().filename().string();
    size_t i = 0;
    while (i < fname.size() &&
           (std::isdigit(static_cast<unsigned char>(fname[i])) != 0)) {
      i++;
    }
    int label = (i > 0) ? std::stoi(fname.substr(0, i)) : -1;
    if (label != 0) {
      continue;
    }
    auto tensor = LoadImage(file.path().string());
    data.emplace_back(tensor, 0);
  }

  // Load only classes '1'..'9' from printed_digits
  for (const auto &dir : fs::directory_iterator(printed_root)) {
    if (!dir.is_directory()) {
      continue;
    }
    const auto NAME = dir.path().filename().string();
    int label = -1;
    try {
      label = std::stoi(NAME);
    } catch (...) {
      continue;
    }
    if (label < 1 || label > 9) {
      continue;
    }
    for (const auto &file : fs::directory_iterator(dir)) {
      if (!file.is_regular_file()) {
        continue;
      }
      auto tensor = LoadImage(file.path().string());
      data.emplace_back(tensor, label);
    }
  }

  return data;
}

void PrintProgressBar(int epoch, int total_epochs, size_t current,
                      size_t total) {
  const int BAR_WIDTH = 30;
  double ratio = static_cast<double>(current) / static_cast<double>(total);
  int filled = static_cast<int>(ratio * BAR_WIDTH);
  int pct = static_cast<int>(ratio * 100.0);
  std::cout << "\rEpoch " << (epoch + 1) << "/" << total_epochs << "  [";
  for (int i = 0; i < BAR_WIDTH; ++i) {
    std::cout << (i < filled ? '#' : '.');
  }
  std::cout << "]  " << std::setw(3) << pct << "%  (" << current << "/" << total
            << ")" << std::flush;
}

auto main(int argc, char *argv[]) -> int {
  std::string dataset_name;
  int epochs = 0;

#ifdef WITH_QT
  std::unique_ptr<QApplication> app_ptr;
  int qt_argc = 1;
  char prog[] = "train";
  char *qt_argv[] = {prog, nullptr};
  app_ptr = std::make_unique<QApplication>(qt_argc, qt_argv);

  QDialog dialog;
  dialog.setWindowTitle("Train options");
  QFormLayout layout(&dialog);

  QComboBox dataset_box;
  dataset_box.addItems({"printed_digits", "mnist", "mnist0_printed_digits"});
  layout.addRow(new QLabel("Dataset:"), &dataset_box);

  QSpinBox epoch_box;
  epoch_box.setRange(1, 100);
  epoch_box.setValue(5);
  layout.addRow(new QLabel("Epochs:"), &epoch_box);

  QDialogButtonBox buttons(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
  layout.addRow(&buttons);
  QObject::connect(&buttons, &QDialogButtonBox::accepted, &dialog,
                   &QDialog::accept);
  QObject::connect(&buttons, &QDialogButtonBox::rejected, &dialog,
                   &QDialog::reject);

  if (dialog.exec() != QDialog::Accepted) {
    std::cerr << "Cancelled by user." << std::endl;
    return 1;
  }

  dataset_name = dataset_box.currentText().toStdString();
  epochs = epoch_box.value();
#else
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <dataset_name> <epochs>"
              << std::endl;
    return 1;
  }
  dataset_name = argv[1];
  epochs = std::stoi(argv[2]);
#endif

  torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                   : torch::kCPU);
  std::cout << "Device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
  // Enable cuDNN benchmark for faster convolutions on fixed-size inputs
  torch::globalContext().setBenchmarkCuDNN(true);

  std::vector<std::pair<torch::Tensor, int>> dataset;
  if (dataset_name == "printed_digits") {
    std::cout << "Loading printed digits dataset..." << std::endl;
    dataset = LoadPrintedDigitsDataset(EXTERNAL "/printed_digits/assets");
  } else if (dataset_name == "mnist") {
    std::cout << "Loading MNIST dataset..." << std::endl;
    dataset = LoadMnistDataset(EXTERNAL "/mnist/mnist/train");
  } else if (dataset_name == "mnist0_printed_digits") {
    std::cout
        << "Loading hybrid dataset (0 from MNIST, 1-9 from printed_digits)..."
        << std::endl;
    dataset = LoadHybridDataset(EXTERNAL "/mnist/mnist/train",
                                EXTERNAL "/printed_digits/assets");
  } else {
    std::cerr << "Unknown dataset: " << dataset_name << std::endl;
    return 1;
  }
  std::cout << "Dataset loaded with " << dataset.size() << " samples."
            << std::endl;

  // Stack all samples into one tensor [N,1,28,28] and labels [N]
  const auto N = static_cast<int64_t>(dataset.size());
  std::vector<torch::Tensor> imgs;
  imgs.reserve(N);
  std::vector<int64_t> labels_vec;
  labels_vec.reserve(N);
  for (int64_t i = 0; i < N; ++i) {
    imgs.push_back(dataset[i].first); // [1,1,28,28]
    labels_vec.push_back(static_cast<int64_t>(dataset[i].second));
  }
  auto inputs_cpu = torch::cat(imgs, 0).contiguous(); // [N,1,28,28]
  auto labels_cpu =
      torch::from_blob(labels_vec.data(), {N}, torch::kLong).clone();
  if (device.is_cuda()) {
    inputs_cpu = inputs_cpu.pin_memory();
    labels_cpu = labels_cpu.pin_memory();
  }

  torch::manual_seed(42);
  auto model = std::make_shared<DigitNet>();
  model->to(device);
  model->train();
  torch::optim::Adam optimizer(model->parameters(),
                               torch::optim::AdamOptions(0.001));

  const int64_t BATCH_SIZE = 256; // 128/256, если хватает VRAM
  std::cout << "Starting training for " << epochs << " epochs..." << std::endl;

  for (int epoch = 0; epoch < epochs; epoch++) {
    float total_loss = 0.0F;
    int64_t correct = 0;
    auto perm = torch::randperm(N, torch::TensorOptions().dtype(torch::kLong));

    // Сформируем перемешанные непрерывные батчи для более быстрых копирований
    auto inputs_shuf = inputs_cpu.index_select(0, perm).contiguous();
    auto labels_shuf = labels_cpu.index_select(0, perm).contiguous();
    if (device.is_cuda()) {
      inputs_shuf = inputs_shuf.pin_memory();
      labels_shuf = labels_shuf.pin_memory();
    }

    for (int64_t start = 0; start < N; start += BATCH_SIZE) {
      const int64_t BS = std::min<int64_t>(BATCH_SIZE, N - start);

      auto batch_x = inputs_shuf.narrow(0, start, BS);
      auto batch_y = labels_shuf.narrow(0, start, BS);

      if (device.is_cuda()) {
        batch_x = batch_x.to(torch::kCUDA, true);
        batch_y = batch_y.to(torch::kCUDA, true);
      }
      auto output = model->Forward(batch_x); // logits or log-probs
      auto loss = torch::nll_loss(output, batch_y);

      optimizer.zero_grad();
      loss.backward();
      optimizer.step();

      total_loss += loss.item<float>() * static_cast<float>(BS);
      auto predicted = output.argmax(1);
      correct += predicted.eq(batch_y).sum().item<int64_t>();

      // Обновляем прогресс реже (раз в ~10 батчей)
      if (((start / BATCH_SIZE) % 10) == 0) {
        size_t seen = static_cast<size_t>(std::min<int64_t>(start + BS, N));
        PrintProgressBar(epoch, epochs, seen, static_cast<size_t>(N));
      }
    }
    PrintProgressBar(epoch, epochs, static_cast<size_t>(N),
                     static_cast<size_t>(N));
    std::cout << '\n';
    std::cout << "Epoch " << (epoch + 1)
              << " | Loss: " << (total_loss / static_cast<float>(N))
              << " | Accuracy: "
              << (100.0 * static_cast<double>(correct) / static_cast<double>(N))
              << "%\n";
  }

  torch::save(model, MODELS "/digit_model_" + dataset_name + "_" +
                         std::to_string(epochs) + ".pt");
  return 0;
}