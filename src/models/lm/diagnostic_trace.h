#pragma once

#include <bit>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "inferx/ops/execution_context.h"

namespace inferx::lm {
// Explicit eager-only diagnostic. Disabled unless a trace directory is set;
// synchronization and file I/O must never be enabled in throughput measurements.
class DiagnosticTrace {
 public:
  explicit DiagnosticTrace(ops::ExecutionContext& ctx) : ctx_(ctx) {
    const char* directory = std::getenv("INFERX_DIAGNOSTIC_TRACE_DIR");
    if (!directory) return;
    static thread_local int next_step = 0;
    const int step = next_step++;
    const char* selected = std::getenv("INFERX_DIAGNOSTIC_TRACE_STEP");
    if (selected && step != std::stoi(selected)) return;
    auto capturing = ctx.runtime().IsCapturing(ctx.stream());
    if (!capturing.ok() || *capturing) throw std::runtime_error("tensor tracing requires eager execution");
    std::ostringstream name;
    name << "step_" << std::setfill('0') << std::setw(6) << step;
    directory_ = std::filesystem::path(directory) / name.str();
    if (!std::filesystem::create_directories(directory_))
      throw std::runtime_error("trace output already exists: " + directory_.string());
  }
  bool enabled() const { return !directory_.empty(); }
  void Write(const std::string& name, const Tensor& tensor) {
    if (directory_.empty()) return;
    auto status = ctx_.runtime().SynchronizeStream(ctx_.stream());
    if (!status.ok()) throw std::runtime_error(status.ToString());
    const int64_t width = tensor.Numel() / tensor.Dim(0);
    auto row = tensor.Slice(tensor.Dim(0) - 1, tensor.Dim(0));
    if (!row.ok()) throw std::runtime_error(row.status().ToString());
    std::vector<uint16_t> bits(width);
    if (tensor.GetDataType() != DataType::kBFloat16)
      throw std::runtime_error("trace expects BF16 activations");
    status = ctx_.runtime().Copy(bits.data(), row->Data(), bits.size() * 2, CopyKind::kDeviceToHost);
    if (!status.ok()) throw std::runtime_error(status.ToString());
    std::vector<float> values(width);
    for (int64_t i = 0; i < width; ++i) values[i] = std::bit_cast<float>(uint32_t(bits[i]) << 16);
    std::ofstream file(directory_ / (name + ".f32"), std::ios::binary);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
  }
 private:
  ops::ExecutionContext& ctx_;
  std::filesystem::path directory_;
};
}  // namespace inferx::lm
