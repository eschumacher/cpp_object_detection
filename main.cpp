#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"

namespace {

constexpr void TFLITE_MINIMAL_CHECK(bool check) {
  if (!check) {
    std::cerr << "Minimal check failed!\n";
    exit(1);
  }
}

constexpr bool valid_args(int argc) {
  if (argc < 2) {
    std::cerr << "No model provided (arg1)!\n";
    return false;
  }

  if (argc != 3) {
    std::cerr << "No source image provided (arg2)!\n";
    return false;
  }

  return true;
}

}  // namespace

int main(int argc, char **argv) {
  if (!valid_args(argc)) {
    return 1;
  }

  std::string model_file(argv[1]);
  std::string image_file(argv[2]);

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  std::cout << "=== Pre-invoke Interpreter State ===\n";
  tflite::PrintInterpreterState(interpreter.get());

  // Load image
  cv::Mat image{cv::imread(image_file, cv::IMREAD_COLOR)};
  if (image.empty()) {
    std::cerr << "Failed to read image!\n";
    return 1;
  }

  // Convert color due to default opencv format as bgr
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);

  // Resize image (currently hardcoded for sample model input shape)
  cv::Mat resized_image;
  cv::resize(image_rgb, resized_image, cv::Size(320, 320), 0, 0,
             cv::INTER_AREA);

  // Fill input buffers with image
  auto *input = interpreter->typed_input_tensor<uint8_t>(0);
  std::copy(resized_image.datastart, resized_image.dataend, input);

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  std::cout << "\n\n=== Post-invoke Interpreter State ===\n";
  tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  auto *output_boxes = interpreter->typed_output_tensor<float>(0);
  auto *output_classes = interpreter->typed_output_tensor<float>(1);
  auto *output_scores = interpreter->typed_output_tensor<float>(2);
  int num_detections =
      static_cast<int>(*interpreter->typed_output_tensor<float>(3));

  // Parse detections
  std::cout << "Detections: " << num_detections << "\n";
  for (int i = 0; i < num_detections; ++i) {
    std::cout << "Detection " << i << ":\n";

    auto score = output_scores[i];
    std::cout << "Score: " << score << "\n";
    if (score < 0.4F) {
      std::cout << "Score does not meet threshold.\n";
      break;
    }

    auto *boxes = &output_boxes[static_cast<ptrdiff_t>(i * 4)];
    std::cout << "Box: " << boxes[0] << ", " << boxes[1] << ", " << boxes[2]
              << ", " << boxes[3] << "\n";

    auto type = output_classes[i];
    std::cout << "Class: " << type << "\n";

    cv::Rect bbox{cv::Point2i(static_cast<int>(boxes[1] * image.cols),
                              static_cast<int>(boxes[0] * image.rows)),
                  cv::Point2i(static_cast<int>(boxes[3] * image.cols),
                              static_cast<int>(boxes[2] * image.rows))};
    cv::rectangle(image, bbox, {0, 255, 0}, 3);
  }

  cv::imwrite("output.jpg", image);

  std::cout << "Minimal tflite supported!\n";
}
