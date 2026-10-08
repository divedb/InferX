#include "inferx/models/loading/quant_plan.h"

#include <optional>

#include "nlohmann/json.hpp"

namespace inferx::models {

StatusOr<std::optional<QuantPlan>> QuantPlan::FromConfigJson(const std::string& json) {
  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(json);
  } catch (const nlohmann::json::exception& e) {
    return InvalidArgumentError("invalid config.json: ", e.what());
  }
  if (!parsed.is_object() || !parsed.contains("quantization_config")) {
    return std::nullopt;
  }
  const auto& section = parsed.at("quantization_config");
  if (!section.is_object()) return std::nullopt;
  const std::string method = section.value("quant_method", std::string());
  if (method.empty()) return std::nullopt;

  QuantPlan plan;
  if (method == "gptq") {
    plan.format = QuantPlan::Format::kGptq;
    const std::string pack_dtype = section.value("pack_dtype", std::string("int32"));
    const std::string checkpoint_format =
        section.value("checkpoint_format", std::string("gptq"));
    if (pack_dtype != "int32" || checkpoint_format != "gptq") {
      return UnimplementedError("unsupported GPTQ packing: pack_dtype=", pack_dtype,
                                ", checkpoint_format=", checkpoint_format);
    }
  } else if (method == "awq") {
    plan.format = QuantPlan::Format::kAwq;
    const std::string version = section.value("version", std::string("gemm"));
    if (version != "gemm") {
      return UnimplementedError("unsupported AWQ packing version: ", version);
    }
  } else if (method == "mxfp4") {
    // gpt-oss: experts are fused MXFP4 tensors the loader unpacks by name.
    return std::nullopt;
  } else {
    return UnimplementedError("unsupported checkpoint quantization method: ", method);
  }

  plan.bits = section.value("bits", 4);
  plan.group_size = section.value("group_size", int64_t{128});
  if (plan.bits != 4 && plan.bits != 8) {
    return UnimplementedError("unsupported ", method, " bit width: ", plan.bits);
  }
  if (plan.group_size <= 0) {
    return InvalidArgumentError("quantization group_size must be positive");
  }
  return plan;
}

}  // namespace inferx::models
