#include <cstring>
#include <iostream>
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

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "No model provided (please pass as arg)!\n";
    return 1;
  }

  std::string model_file(argv[1]);

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

  // Fill input buffers
  // TODO: use sample image
  auto *input = interpreter->typed_input_tensor<uint8_t>(0);
  memset(input, 1, 320 * 320 * 3);

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  std::cout << "\n\n=== Post-invoke Interpreter State ===\n";
  tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  // TODO: parse output
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`

  std::cout << "Minimal tflite supported!\n";
}
