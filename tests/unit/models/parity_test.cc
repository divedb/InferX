/// \file
/// \brief Full-model numerical parity: synthetic checkpoints per family run
///        through Model::Load -> Forward and compared against an independent
///        float32 reference implementation of the same translated config.
///
/// The reference consumes the SAME DecoderConfig the family translator
/// produces (dims, windows, routing policy) but executes the forward in
/// plain fp32 host code written from each family's reference math. The
/// engine rounds activations to bf16 at every kernel boundary, so parity is
/// asserted on relative L2 error of the logits plus top-1 agreement.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/cache/kv_block_pool.h"
#include "inferx/cache/recurrent_state_pool.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/tensor.h"
#include "inferx/models/causal/decoder_config.h"
#include "inferx/models/model.h"
#include "inferx/models/model_registry.h"
#include "inferx/dist/comm.h"
#include "inferx/ops/execution_context.h"
#include "inferx/ops/moe.h"

namespace inferx {
namespace {

constexpr int64_t kHidden = 16;
constexpr int64_t kHeads = 4;
constexpr int64_t kKv = 2;
constexpr int64_t kDim = 8;
constexpr int64_t kVocab = 40;

const std::vector<int32_t> kTokens{3, 17, 5, 29, 1};

// ---------------------------------------------------------------- fixture IO

struct FTensor {
  std::vector<int64_t> shape;
  std::vector<float> values;
};
using Fixture = std::map<std::string, FTensor>;

struct TempDir {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() / ("inferx_parity_" + std::to_string(::getpid()));
  TempDir() { std::filesystem::create_directories(path); }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

FTensor T(std::vector<int64_t> shape, const std::string& name, float scale = 0.1f) {
  int64_t numel = 1;
  for (auto d : shape) numel *= d;
  std::mt19937 rng(std::hash<std::string>{}(name));
  std::uniform_real_distribution<float> dist(-scale, scale);
  std::vector<float> values(numel);
  for (auto& v : values) v = dist(rng);
  return FTensor{std::move(shape), std::move(values)};
}

/// Writes F32 tensors plus optional raw U8 tensors (name, shape, bytes).
void WriteCheckpoint(const TempDir& dir, const Fixture& tensors,
                     const std::vector<std::tuple<std::string, std::vector<int64_t>,
                                                  std::vector<uint8_t>>>& raw = {}) {
  std::string header = "{";
  std::vector<std::byte> blob;
  auto append = [&](const std::string& name, const std::vector<int64_t>& shape, const void* data,
                    int64_t bytes, const char* dtype) {
    const int64_t begin = static_cast<int64_t>(blob.size());
    const auto* raw_bytes = static_cast<const std::byte*>(data);
    blob.insert(blob.end(), raw_bytes, raw_bytes + bytes);
    if (header.size() > 1) header += ",";
    header += "\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":[";
    for (size_t d = 0; d < shape.size(); ++d) {
      if (d) header += ",";
      header += std::to_string(shape[d]);
    }
    header += "],\"data_offsets\":[" + std::to_string(begin) + "," +
              std::to_string(static_cast<int64_t>(blob.size())) + "]}";
  };
  for (const auto& [name, tensor] : tensors) {
    append(name, tensor.shape, tensor.values.data(), tensor.values.size() * sizeof(float), "F32");
  }
  for (const auto& [name, shape, bytes] : raw) {
    append(name, shape, bytes.data(), bytes.size(), "U8");
  }
  header += "}";
  const uint64_t hlen = header.size();
  std::vector<std::byte> file(8 + hlen + blob.size(), std::byte{0});
  std::memcpy(file.data(), &hlen, 8);
  std::memcpy(file.data() + 8, header.data(), hlen);
  std::memcpy(file.data() + 8 + hlen, blob.data(), blob.size());
  std::ofstream out((dir.path / "model.safetensors").string(), std::ios::binary);
  out.write(reinterpret_cast<const char*>(file.data()), file.size());
}

// ------------------------------------------------------------- engine driver

class Engine {
 public:
  Engine() {
    auto runtime = RuntimeFor(DeviceId::Cuda(0));
    runtime_ = *runtime;
    runtime_->Activate().ok();
    stream_ = runtime_->CreateStream().value();
  }

  std::vector<float> Forward(const std::string& dir, const std::vector<int32_t>& tokens) {
    auto model = Model::Load(dir, DeviceId::Cuda(0), /*max_tokens=*/64, /*max_seqs=*/2);
    EXPECT_TRUE(model.ok()) << model.status();
    if (!model.ok()) return {};
    const auto requirements = (*model)->StateRequirements();
    std::vector<KvLayout> layouts;
    std::vector<RecurrentStateSpec> recurrent_specs;
    for (const auto& spec : requirements) {
      if (const auto* paged = std::get_if<PagedKvStateSpec>(&spec)) {
        layouts.push_back(paged->layout);
      } else {
        recurrent_specs.push_back(std::get<RecurrentStateSpec>(spec));
      }
    }
    auto pool = KvBlockPool::Create(8, 8, layouts, DeviceId::Cuda(0));
    EXPECT_TRUE(pool.ok());
    std::optional<RecurrentStatePool> rpool;
    if (!recurrent_specs.empty()) {
      auto made = RecurrentStatePool::Create(recurrent_specs, 2, DeviceId::Cuda(0));
      EXPECT_TRUE(made.ok());
      rpool = std::move(*made);
    }
    ModelState state;
    state.paged_kv = &*pool;
    state.recurrent = rpool.has_value() ? &*rpool : nullptr;
    int64_t paged = 0, rec = 0;
    for (const auto& spec : requirements) {
      if (std::holds_alternative<PagedKvStateSpec>(spec)) {
        state.layers.push_back(PagedKvState{paged++});
      } else {
        state.layers.push_back(RecurrentState{rec++});
      }
    }
    ops::ExecutionContext ctx(*runtime_, stream_);
    dist::SingleRankComm comm;
    if (rpool.has_value()) rpool->ResetSlot(ctx, 0).ok();

    const int64_t n = tokens.size();
    const DeviceId device = DeviceId::Cuda(0);
    auto i32 = [&](int64_t count) {
      return Tensor::Empty(DataType::kInt32, Shape({count}), device).value();
    };
    Tensor token_ids = i32(n), positions = i32(n), batch_indices = i32(n);
    Tensor qo = i32(2), kv = i32(2), last = i32(1), logit_rows = i32(1), blocks = i32(1);
    std::vector<int32_t> pos(n);
    for (int i = 0; i < n; ++i) pos[i] = i;
    std::vector<int32_t> zeros(n, 0);
    const std::vector<int32_t> qo_h{0, static_cast<int32_t>(n)};
    const std::vector<int32_t> kv_h{0, 1};
    const std::vector<int32_t> last_h{static_cast<int32_t>(n)};
    const std::vector<int32_t> rows_h{static_cast<int32_t>(n - 1)};
    runtime_->Copy(token_ids.Data(), tokens.data(), n * 4, CopyKind::kHostToDevice).ok();
    runtime_->Copy(positions.Data(), pos.data(), n * 4, CopyKind::kHostToDevice).ok();
    runtime_->Copy(batch_indices.Data(), zeros.data(), n * 4, CopyKind::kHostToDevice).ok();
    runtime_->Copy(qo.Data(), qo_h.data(), 8, CopyKind::kHostToDevice).ok();
    runtime_->Copy(kv.Data(), kv_h.data(), 8, CopyKind::kHostToDevice).ok();
    runtime_->Copy(last.Data(), last_h.data(), 4, CopyKind::kHostToDevice).ok();
    runtime_->Copy(logit_rows.Data(), rows_h.data(), 4, CopyKind::kHostToDevice).ok();
    runtime_->Copy(blocks.Data(), std::vector<int32_t>{0}.data(), 4, CopyKind::kHostToDevice).ok();
    std::optional<Tensor> slots;
    if (rpool.has_value()) {
      slots = i32(1);
      runtime_->Copy(slots->Data(), std::vector<int32_t>{0}.data(), 4,
                     CopyKind::kHostToDevice).ok();
    }
    ModelInput input{token_ids,
                     AttentionBatch{positions, batch_indices, qo, kv, blocks, last,
                                    absl::MakeConstSpan(qo_h), absl::MakeConstSpan(kv_h),
                                    static_cast<int>(n), 1},
                     logit_rows};
    input.attention.recurrent_indices = std::move(slots);
    auto logits = (*model)->Forward(input, state, ctx, comm);
    EXPECT_TRUE(logits.ok()) << logits.status();
    if (!logits.ok()) return {};
    std::vector<uint16_t> raw(logits->Numel());
    runtime_->SynchronizeStream(stream_).ok();
    runtime_->Copy(raw.data(), logits->Data(), raw.size() * 2, CopyKind::kDeviceToHost).ok();
    std::vector<float> out(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
      const uint32_t bits = static_cast<uint32_t>(raw[i]) << 16;
      std::memcpy(&out[i], &bits, 4);
    }
    return out;
  }

 private:
  DeviceRuntime* runtime_ = nullptr;
  Stream stream_;
};

// ------------------------------------------------------------- fp32 reference

bool RefTrace() {
  static const bool on = std::getenv("INFERX_PARITY_TRACE") != nullptr;
  return on;
}
void Dump(const char* stage, const float* v, int64_t n) {
  if (!RefTrace()) return;
  double norm = 0;
  for (int64_t i = 0; i < n; ++i) norm += double(v[i]) * double(v[i]);
  fprintf(stderr, "%-36s n=%3lld norm=%10.4f first3=[%.4f %.4f %.4f]\n", stage,
          static_cast<long long>(n), std::sqrt(norm), v[0], v[1], v[2]);
}

namespace ref {

float RmsNorm(const float* x, int64_t dim, float eps, bool plus_one, const float* weight,
              float* out) {
  float sum = 0;
  for (int64_t i = 0; i < dim; ++i) sum += x[i] * x[i];
  const float inv = 1.0f / std::sqrt(sum / dim + eps);
  for (int64_t i = 0; i < dim; ++i) {
    out[i] = x[i] * inv * (plus_one ? 1.0f + weight[i] : weight[i]);
  }
  return inv;
}

float InvFreq(int64_t i, const components::RotaryConfig& rotary) {
  const float inv = std::pow(rotary.theta, -2.0f * i / rotary.dim);
  if (rotary.type == "linear") return inv / rotary.factor;
  if (rotary.type == "llama3") {
    const float wavelen = 6.283185307179586f / inv;
    const float orig = static_cast<float>(rotary.original_max_position);
    const float low = orig / rotary.low_freq_factor, high = orig / rotary.high_freq_factor;
    if (wavelen > low) return inv / rotary.factor;
    if (wavelen < high) return inv;
    const float t =
        (orig / wavelen - rotary.low_freq_factor) / (rotary.high_freq_factor - rotary.low_freq_factor);
    return (1 - t) * inv / rotary.factor + t * inv;
  }
  if (rotary.type == "yarn") {
    const float orig = static_cast<float>(rotary.original_max_position);
    const auto correction = [&](float rotations) {
      return rotary.dim * std::log(orig / (rotations * 6.283185307179586f)) /
             (2 * std::log(rotary.theta));
    };
    const float lo = correction(rotary.beta_fast), hi = correction(rotary.beta_slow);
    const float span = hi - lo;
    float ramp = span > 0 ? std::min(std::max((static_cast<float>(i) - lo) / span, 0.0f), 1.0f) : 1.0f;
    if (rotary.truncate) {
      // floor/ceil applied by the engine; mirror per index.
      ramp = span > 0 ? std::min(std::max((static_cast<float>(i) - std::floor(lo)) /
                                              (std::ceil(hi) - std::floor(lo)),
                                          0.0f),
                                  1.0f)
                      : 1.0f;
    }
    return (inv / rotary.factor) * ramp + inv * (1 - ramp);
  }
  return inv;
}

void Rope(float* x, const components::RotaryConfig& rotary, int64_t pos, float attn_scale) {
  const int64_t half = rotary.dim / 2;
  for (int64_t i = 0; i < half; ++i) {
    const float angle = pos * InvFreq(i, rotary) * attn_scale;
    const float c = std::cos(angle), s = std::sin(angle);
    const float x1 = x[i], x2 = x[i + half];
    x[i] = x1 * c - x2 * s;
    x[i + half] = x2 * c + x1 * s;
  }
}

float Activation(int act, float x, float alpha, float limit, bool is_gate) {
  if (!is_gate) {
    // Only gpt-oss transforms the up half (clamp + shift).
    if (act == static_cast<int>(ops::Activation::kSiluOai)) {
      return std::min(std::max(x, -limit), limit) + 1.0f;
    }
    return x;
  }
  if (act == static_cast<int>(ops::Activation::kGeluTanh)) {
    const float t = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(t));
  }
  if (act == static_cast<int>(ops::Activation::kSiluOai)) {
    const float g = std::min(x, limit);
    return g / (1.0f + std::exp(-alpha * g));
  }
  return x / (1.0f + std::exp(-x));
}

/// y[out] = x @ w^T (+ bias).
void Matvec(const float* w, const float* x, int64_t out, int64_t in, float* y,
            const float* bias = nullptr) {
  for (int64_t o = 0; o < out; ++o) {
    float sum = bias ? bias[o] : 0.0f;
    for (int64_t i = 0; i < in; ++i) sum += w[o * in + i] * x[i];
    y[o] = sum;
  }
}

}  // namespace ref

/// \brief Step-by-step fp32 decoder matching the translated config exactly.
class ReferenceModel {
 public:
  ReferenceModel(causal::DecoderConfig config, Fixture weights)
      : config_(std::move(config)), w_(std::move(weights)) {
    for (const auto& block : config_.blocks) {
      keys_.emplace_back();
      values_.emplace_back();
      if (const auto* g = std::get_if<components::GatedDeltaNetConfig>(&block.mixer)) {
        gdn_.push_back(GdnState{*g});
      } else {
        gdn_.push_back(GdnState{});
      }
    }
  }

  std::vector<float> Forward(const std::vector<int32_t>& tokens) {
    const int64_t hidden = config_.model.hidden_size;
    last_tokens_ = tokens.size();
    std::vector<float> x(hidden);
    const auto& embed = w_.at("model.embed_tokens.weight").values;
    const int64_t last = tokens.back();
    std::memcpy(x.data(), &embed[tokens[0] * hidden], hidden * sizeof(float));
    // Time-major: every token streams through every layer, filling caches.
    std::vector<float> all_x(tokens.size() * hidden);
    for (size_t t = 0; t < tokens.size(); ++t) {
      std::vector<float> hidden_state(hidden);
      std::memcpy(hidden_state.data(), &embed[tokens[t] * hidden], hidden * sizeof(float));
      if (config_.embedding_scale != 1.0f) {
        for (auto& v : hidden_state) v *= config_.embedding_scale;
      }
      for (size_t layer = 0; layer < config_.blocks.size(); ++layer) {
        Step(layer, t, hidden_state);
      }
      if (t + 1 == tokens.size() && RefTrace()) {
        Dump("embedding", hidden_state.data(), hidden);
      }
      std::memcpy(&all_x[t * hidden], hidden_state.data(), hidden * sizeof(float));
    }
    (void)x;
    (void)last;
    std::vector<float> normed(hidden);
    ref::RmsNorm(&all_x[(tokens.size() - 1) * hidden], hidden, config_.final_norm.eps,
                 config_.final_norm.plus_one, W("model.norm.weight"), normed.data());
    Dump("final_norm", normed.data(), hidden);
    std::vector<float> logits(config_.model.vocab_size);
    if (config_.model.tie_word_embeddings) {
      ref::Matvec(embed.data(), normed.data(), config_.model.vocab_size, hidden, logits.data());
    } else {
      ref::Matvec(W("lm_head.weight"), normed.data(), config_.model.vocab_size, hidden,
                  logits.data());
    }
    return logits;
  }

 private:
  const float* W(const std::string& name) const { return w_.at(name).values.data(); }

  struct GdnState {
    components::GatedDeltaNetConfig config;
    std::vector<float> state;   // [vh * dk * dv]
    std::vector<float> conv;    // [conv_dim * (k-1)]
    std::vector<float> q_hist;  // conv inputs of this sequence, grows per step.
  };

  void Step(size_t layer, size_t t, std::vector<float>& hidden) {
    const auto& block = config_.blocks[layer];
    trace_ = RefTrace() && t + 1 == last_tokens_;
    const std::string p = "model.layers." + std::to_string(layer) + ".";
    const int64_t hidden_dim = config_.model.hidden_size;
    std::vector<float> normed(hidden_dim), mixed(hidden_dim);
    ref::RmsNorm(hidden.data(), hidden_dim, block.norm.eps, block.norm.plus_one,
                 W(p + "input_layernorm.weight"), normed.data());
    if (trace_) Dump("layer_N.input_norm", normed.data(), hidden_dim);

    if (const auto* a = std::get_if<components::AttentionConfig>(&block.mixer)) {
      AttentionStep(*a, p, block.norm.eps, layer, t, normed, mixed);
      if (trace_) Dump("layer_N.o_proj", mixed.data(), hidden_dim);
    } else if (const auto* m = std::get_if<components::MlaConfig>(&block.mixer)) {
      MlaStep(*m, p, block.norm.eps, layer, t, normed, mixed);
    } else {
      GdnStep(std::get<components::GatedDeltaNetConfig>(block.mixer), p, layer, t, normed, mixed);
    }

    if (block.residual == components::ResidualStyle::kOutputNorm) {
      ref::RmsNorm(mixed.data(), hidden_dim, block.mixer_out_norm.eps, block.mixer_out_norm.plus_one,
                   W(p + "post_attention_layernorm.weight"), mixed.data());
      if (trace_) Dump("layer_N.mixer_out_norm", mixed.data(), hidden_dim);
    }
    for (int64_t i = 0; i < hidden_dim; ++i) hidden[i] += mixed[i];
    if (trace_) Dump("layer_N.residual", hidden.data(), hidden_dim);

    ref::RmsNorm(hidden.data(), hidden_dim, block.norm.eps, block.norm.plus_one,
                 W(p + (block.residual == components::ResidualStyle::kOutputNorm
                            ? std::string("pre_feedforward_layernorm.weight")
                            : std::string("post_attention_layernorm.weight"))),
                 normed.data());
    if (const auto* dense = std::get_if<components::SwiGluConfig>(&block.feed_forward)) {
      DenseStep(*dense, p, normed, mixed);
      if (trace_) Dump("layer_N.down_proj", mixed.data(), hidden_dim);
    } else {
      MoeStep(std::get<components::MoeConfig>(block.feed_forward), p, normed, mixed);
      if (trace_) Dump("layer_N.moe_out", mixed.data(), hidden_dim);
    }
    if (block.residual == components::ResidualStyle::kOutputNorm) {
      ref::RmsNorm(mixed.data(), hidden_dim, block.feed_forward_out_norm.eps,
                   block.feed_forward_out_norm.plus_one,
                   W(p + "post_feedforward_layernorm.weight"), mixed.data());
      if (trace_) Dump("layer_N.feed_forward_out_norm", mixed.data(), hidden_dim);
    }
    for (int64_t i = 0; i < hidden_dim; ++i) hidden[i] += mixed[i];
  }

  void AttentionStep(const components::AttentionConfig& a, const std::string& p, float eps,
                     size_t layer, size_t t, const std::vector<float>& normed,
                     std::vector<float>& mixed) {
    const int64_t qw = a.query_heads * a.head_dim, kvw = a.kv_heads * a.head_dim;
    const bool gated = a.output_gate == components::OutputGate::kSigmoid;
    std::vector<float> q(qw), k(kvw), v(kvw), gate(gated ? qw : 0);
    {
      std::vector<float> qkv(gated ? 2 * qw : qw);
      ref::Matvec(W(p + "self_attn.q_proj.weight"), normed.data(), qkv.size(),
                  config_.model.hidden_size, qkv.data());
      if (a.qkv_bias) {
        const auto& qb = W(p + "self_attn.q_proj.bias");
        for (size_t i = 0; i < qkv.size(); ++i) qkv[i] += qb[i];
      }
      std::vector<float> kv(2 * kvw);
      ref::Matvec(W(p + "self_attn.k_proj.weight"), normed.data(), kvw,
                  config_.model.hidden_size, kv.data());
      ref::Matvec(W(p + "self_attn.v_proj.weight"), normed.data(), kvw,
                  config_.model.hidden_size, &kv[kvw]);
      if (a.qkv_bias) {
        const auto& kb = W(p + "self_attn.k_proj.bias");
        const auto& vb = W(p + "self_attn.v_proj.bias");
        for (int64_t i = 0; i < kvw; ++i) {
          kv[i] += kb[i];
          kv[kvw + i] += vb[i];
        }
      }
      if (gated) {
        std::memcpy(q.data(), qkv.data(), qw * sizeof(float));
        std::memcpy(gate.data(), &qkv[qw], qw * sizeof(float));
      } else {
        std::memcpy(q.data(), qkv.data(), qw * sizeof(float));
      }
      std::memcpy(k.data(), kv.data(), kvw * sizeof(float));
      std::memcpy(v.data(), &kv[kvw], kvw * sizeof(float));
    }
    std::vector<float> head(a.head_dim), normed_head(a.head_dim);
    for (int64_t h = 0; h < a.query_heads; ++h) {
      std::memcpy(head.data(), &q[h * a.head_dim], a.head_dim * sizeof(float));
      if (a.qk_norm) {
        ref::RmsNorm(head.data(), a.head_dim, eps, a.qk_norm_plus_one,
                     W(p + "self_attn.q_norm.weight"), normed_head.data());
        std::memcpy(head.data(), normed_head.data(), a.head_dim * sizeof(float));
      }
      ref::Rope(head.data(), a.rotary, static_cast<int64_t>(t),
                a.rotary.Params().scaling.attention_scale);
      std::memcpy(&q[h * a.head_dim], head.data(), a.head_dim * sizeof(float));
    }
    for (int64_t h = 0; h < a.kv_heads; ++h) {
      std::memcpy(head.data(), &k[h * a.head_dim], a.head_dim * sizeof(float));
      if (a.qk_norm) {
        ref::RmsNorm(head.data(), a.head_dim, eps, a.qk_norm_plus_one,
                     W(p + "self_attn.k_norm.weight"), normed_head.data());
        std::memcpy(head.data(), normed_head.data(), a.head_dim * sizeof(float));
      }
      ref::Rope(head.data(), a.rotary, static_cast<int64_t>(t),
                a.rotary.Params().scaling.attention_scale);
      std::memcpy(&k[h * a.head_dim], head.data(), a.head_dim * sizeof(float));
    }
    if (trace_) Dump("layer_N.k_rope", k.data(), kvw);
    if (trace_) Dump("layer_N.q_rope", q.data(), qw);
    keys_[layer].push_back(k);
    values_[layer].push_back(v);

    const float scale =
        a.scale_override > 0 ? a.scale_override : 1.0f / std::sqrt(float(a.head_dim));
    const int64_t ratio = a.query_heads / a.kv_heads;
    const float* sinks = nullptr;
    auto sink_it = w_.find(p + "self_attn.sinks");
    if (sink_it != w_.end()) sinks = sink_it->second.values.data();
    std::vector<float> out(qw);
    for (int64_t h = 0; h < a.query_heads; ++h) {
      const int64_t kv_head = h / ratio;
      std::vector<float> scores(t + 1, -1e30f);
      for (size_t s = 0; s <= t; ++s) {
        if (a.sliding_window > 0 &&
            static_cast<int64_t>(s) < static_cast<int64_t>(t) - (a.sliding_window - 1)) {
          continue;
        }
        float dot = 0;
        for (int64_t d = 0; d < a.head_dim; ++d) {
          dot += q[h * a.head_dim + d] * keys_[layer][s][kv_head * a.head_dim + d];
        }
        scores[s] = dot * scale;
      }
      float m = -1e30f;
      for (float sc : scores) m = std::max(m, sc);
      float denom = 0;
      if (sinks != nullptr) {
        m = std::max(m, sinks[h]);
        denom += std::exp(sinks[h] - m);
      }
      std::vector<float> weights2(t + 1, 0.0f);
      for (size_t s = 0; s <= t; ++s) {
        if (scores[s] < -1e29f) continue;
        weights2[s] = std::exp(scores[s] - m);
        denom += weights2[s];
      }
      for (int64_t d = 0; d < a.head_dim; ++d) {
        float acc = 0;
        for (size_t s = 0; s <= t; ++s) {
          if (weights2[s] == 0) continue;
          acc += weights2[s] / denom * values_[layer][s][kv_head * a.head_dim + d];
        }
        out[h * a.head_dim + d] = acc;
      }
    }
    if (gated) {
      for (int64_t i = 0; i < qw; ++i) out[i] /= (1.0f + std::exp(-gate[i]));
    }
    if (trace_) Dump("layer_N.attention", out.data(), qw);
    ref::Matvec(W(p + "self_attn.o_proj.weight"), out.data(), config_.model.hidden_size, qw,
                mixed.data());
    if (a.output_bias) {
      const auto& ob = W(p + "self_attn.o_proj.bias");
      for (int64_t i = 0; i < config_.model.hidden_size; ++i) mixed[i] += ob[i];
    }
  }

  void MlaStep(const components::MlaConfig& a, const std::string& p, float eps, size_t layer,
               size_t t, const std::vector<float>& normed, std::vector<float>& mixed) {
    const int64_t hidden = config_.model.hidden_size;
    const int64_t hd = a.head_dim();
    std::vector<float> qa(a.q_lora_rank);
    ref::Matvec(W(p + "self_attn.q_a_proj.weight"), normed.data(), a.q_lora_rank, hidden, qa.data());
    std::vector<float> qa_normed(a.q_lora_rank);
    ref::RmsNorm(qa.data(), a.q_lora_rank, eps, false, W(p + "self_attn.q_a_layernorm.weight"),
                 qa_normed.data());
    std::vector<float> q(a.query_heads * hd);
    ref::Matvec(W(p + "self_attn.q_b_proj.weight"), qa_normed.data(), a.query_heads * hd,
                a.q_lora_rank, q.data());
    std::vector<float> kv_a(a.kv_lora_rank + a.qk_rope_head_dim);
    ref::Matvec(W(p + "self_attn.kv_a_proj_with_mqa.weight"), normed.data(),
                a.kv_lora_rank + a.qk_rope_head_dim, hidden, kv_a.data());
    std::vector<float> latent(kv_a.begin(), kv_a.begin() + a.kv_lora_rank);
    std::vector<float> rope_part(kv_a.begin() + a.kv_lora_rank, kv_a.end());
    std::vector<float> latent_normed(a.kv_lora_rank);
    ref::RmsNorm(latent.data(), a.kv_lora_rank, eps, false,
                 W(p + "self_attn.kv_a_layernorm.weight"), latent_normed.data());
    std::vector<float> up(a.query_heads * (a.qk_nope_head_dim + a.v_head_dim));
    ref::Matvec(W(p + "self_attn.kv_b_proj.weight"), latent_normed.data(),
                a.query_heads * (a.qk_nope_head_dim + a.v_head_dim), a.kv_lora_rank, up.data());
    std::vector<float> k(a.query_heads * hd), v(a.query_heads * hd, 0.0f);
    for (int64_t h = 0; h < a.query_heads; ++h) {
      for (int64_t d = 0; d < a.qk_rope_head_dim; ++d) {
        k[h * hd + d] = rope_part[d];
      }
      for (int64_t d = 0; d < a.qk_nope_head_dim; ++d) {
        k[h * hd + a.qk_rope_head_dim + d] =
            up[h * (a.qk_nope_head_dim + a.v_head_dim) + d];
      }
      for (int64_t d = 0; d < a.v_head_dim; ++d) {
        v[h * hd + d] = up[h * (a.qk_nope_head_dim + a.v_head_dim) + a.qk_nope_head_dim + d];
      }
    }
    // RoPE over the leading rope slice of every head (q and k share layout).
    for (int64_t h = 0; h < a.query_heads; ++h) {
      components::RotaryConfig per_head = a.rotary;
      (void)per_head;
      ref::Rope(&q[h * hd], a.rotary, static_cast<int64_t>(t),
                a.rotary.Params().scaling.attention_scale);
    }
    // k rope slice: only the rope part rotates; replicate per head by rotating
    // the shared slice then re-inserting.
    {
      std::vector<float> rope_copy = rope_part;
      ref::Rope(rope_copy.data(), a.rotary, static_cast<int64_t>(t),
                a.rotary.Params().scaling.attention_scale);
      for (int64_t h = 0; h < a.query_heads; ++h) {
        for (int64_t d = 0; d < a.qk_rope_head_dim; ++d) k[h * hd + d] = rope_copy[d];
      }
    }
    keys_[layer].push_back(k);
    values_[layer].push_back(v);
    const float scale =
        a.scale_override > 0 ? a.scale_override : 1.0f / std::sqrt(float(hd));
    std::vector<float> out(a.query_heads * hd);
    for (int64_t h = 0; h < a.query_heads; ++h) {
      std::vector<float> scores(t + 1, -1e30f);
      for (size_t s = 0; s <= t; ++s) {
        float dot = 0;
        for (int64_t d = 0; d < hd; ++d) {
          dot += q[h * hd + d] * keys_[layer][s][h * hd + d];
        }
        scores[s] = dot * scale;
      }
      float m = *std::max_element(scores.begin(), scores.end());
      float denom = 0;
      std::vector<float> weights2(t + 1);
      for (size_t s = 0; s <= t; ++s) {
        weights2[s] = std::exp(scores[s] - m);
        denom += weights2[s];
      }
      for (int64_t d = 0; d < a.v_head_dim; ++d) {
        float acc = 0;
        for (size_t s = 0; s <= t; ++s) acc += weights2[s] / denom * values_[layer][s][h * hd + d];
        out[h * hd + d] = acc;
      }
    }
    // The value tail beyond v_head_dim is zero, matching the padded
    // projection the engine loads.
    std::vector<float> padded(a.query_heads * hd, 0.0f);
    for (int64_t h = 0; h < a.query_heads; ++h) {
      std::memcpy(&padded[h * hd], &out[h * hd], a.v_head_dim * sizeof(float));
    }
    ref::Matvec(W(p + "self_attn.o_proj.weight"), padded.data(), hidden, a.query_heads * hd,
                mixed.data());
  }

  void GdnStep(const components::GatedDeltaNetConfig& a, const std::string& p, size_t layer,
               size_t t, const std::vector<float>& normed, std::vector<float>& mixed) {
    const int64_t hidden = config_.model.hidden_size;
    const int64_t kh = a.key_heads, vh = a.value_heads, kd = a.key_dim, vd = a.value_dim;
    const int64_t q_total = kh * kd, v_total = vh * vd;
    GdnState& st = gdn_[layer];
    if (st.state.empty()) st.state.assign(vh * kd * vd, 0.0f);
    const int64_t conv_dim = 2 * q_total + v_total;
    if (st.conv.empty()) st.conv.assign(conv_dim * (a.conv_kernel_size - 1), 0.0f);

    // Per-group [q | k | v | z] projection (checkpoint order).
    const int64_t nvg = vh / kh;
    std::vector<float> proj(2 * q_total + 2 * v_total);
    ref::Matvec(W(p + "self_attn.in_proj_qkvz.weight"), normed.data(), proj.size(), hidden,
                proj.data());
    std::vector<float> ba(2 * vh);
    ref::Matvec(W(p + "self_attn.in_proj_ba.weight"), normed.data(), 2 * vh, hidden, ba.data());
    std::vector<float> conv_in(conv_dim);
    for (int64_t g = 0; g < kh; ++g) {
      const float* src = &proj[g * (2 * kd + 2 * nvg * vd)];
      std::memcpy(&conv_in[g * kd], src, kd * sizeof(float));
      std::memcpy(&conv_in[q_total + g * kd], src + kd, kd * sizeof(float));
      std::memcpy(&conv_in[2 * q_total + g * nvg * vd], src + 2 * kd, nvg * vd * sizeof(float));
    }
    std::vector<float> z(v_total);
    for (int64_t g = 0; g < kh; ++g) {
      const float* src = &proj[g * (2 * kd + 2 * nvg * vd) + 2 * kd + nvg * vd];
      std::memcpy(&z[g * nvg * vd], src, nvg * vd * sizeof(float));
    }

    // Causal depthwise conv over the ring state, then silu.
    const auto& conv_w = W(p + "self_attn.conv1d.weight");
    std::vector<float> window(a.conv_kernel_size);
    for (int64_t c = 0; c < conv_dim; ++c) {
      for (int64_t j = 0; j < a.conv_kernel_size - 1; ++j) {
        window[j] = st.conv[c * (a.conv_kernel_size - 1) + j];
      }
      window[a.conv_kernel_size - 1] = conv_in[c];
      float sum = 0;
      for (int64_t j = 0; j < a.conv_kernel_size; ++j) {
        sum += conv_w[c * a.conv_kernel_size + j] * window[j];
      }
      conv_in[c] = sum / (1.0f + std::exp(-sum));
      for (int64_t j = 0; j + 1 < a.conv_kernel_size - 1; ++j) {
        st.conv[c * (a.conv_kernel_size - 1) + j] =
            st.conv[c * (a.conv_kernel_size - 1) + j + 1];
      }
      if (a.conv_kernel_size > 1) {
        st.conv[c * (a.conv_kernel_size - 1) + a.conv_kernel_size - 2] = window[a.conv_kernel_size - 1];
      }
    }

    std::vector<float> beta(vh), g(vh);
    for (int64_t h = 0; h < vh; ++h) {
      const int64_t base = (h / nvg) * 2 * nvg + h % nvg;
      const float b = ba[base];
      const float av = ba[base + nvg];
      const float dt_bias = W(p + "self_attn.dt_bias.weight")[h];
      const float a_log = W(p + "self_attn.A_log.weight")[h];
      const float sp = av + dt_bias > 20 ? av + dt_bias : std::log(1 + std::exp(av + dt_bias));
      beta[h] = 1.0f / (1.0f + std::exp(-b));
      g[h] = -std::exp(a_log) * sp;
    }

    std::vector<float> y(v_total);
    const float qscale = 1.0f / std::sqrt(float(kd));
    for (int64_t h = 0; h < vh; ++h) {
      const int64_t khead = h / nvg;
      std::vector<float> k(kd), q(kd);
      float kn = 0, qn = 0;
      for (int64_t d = 0; d < kd; ++d) {
        k[d] = conv_in[q_total + khead * kd + d];
        q[d] = conv_in[khead * kd + d];
        kn += k[d] * k[d];
        qn += q[d] * q[d];
      }
      const float ki = 1.0f / std::sqrt(kn + 1e-6f);
      const float qi = 1.0f / std::sqrt(qn + 1e-6f) * qscale;
      for (int64_t d = 0; d < kd; ++d) {
        k[d] *= ki;
        q[d] *= qi;
      }
      const float decay = std::exp(g[h]);
      const float b = beta[h];
      std::vector<float> kv_mem(vd, 0.0f);
      for (int64_t d = 0; d < kd; ++d) {
        for (int64_t c = 0; c < vd; ++c) {
          st.state[(h * kd + d) * vd + c] *= decay;
        }
      }
      for (int64_t d = 0; d < kd; ++d) {
        for (int64_t c = 0; c < vd; ++c) {
          kv_mem[c] += st.state[(h * kd + d) * vd + c] * k[d];
        }
      }
      for (int64_t d = 0; d < kd; ++d) {
        for (int64_t c = 0; c < vd; ++c) {
          const float value = conv_in[2 * q_total + h * vd + c];
          st.state[(h * kd + d) * vd + c] += k[d] * b * (value - kv_mem[c]);
        }
      }
      for (int64_t c = 0; c < vd; ++c) {
        float acc = 0;
        for (int64_t d = 0; d < kd; ++d) {
          acc += st.state[(h * kd + d) * vd + c] * q[d];
        }
        y[h * vd + c] = acc;
      }
    }

    // Gated norm per head: rmsnorm(y) * weight * silu(z).
    for (int64_t h = 0; h < vh; ++h) {
      std::vector<float> normed_head(vd);
      ref::RmsNorm(&y[h * vd], vd, 1e-6f, false, W(p + "self_attn.norm.weight"),
                   normed_head.data());
      for (int64_t c = 0; c < vd; ++c) {
        const float gate = z[h * vd + c];
        y[h * vd + c] = normed_head[c] * (gate / (1.0f + std::exp(-gate)));
      }
    }
    ref::Matvec(W(p + "self_attn.out_proj.weight"), y.data(), config_.model.hidden_size, v_total,
                mixed.data());
  }

  void DenseStep(const components::SwiGluConfig& dense, const std::string& p,
                 const std::vector<float>& normed, std::vector<float>& mixed) {
    const int64_t hidden = config_.model.hidden_size;
    const int64_t inter = dense.intermediate_size;
    std::vector<float> gate(inter), up(inter);
    ref::Matvec(W(p + "mlp.gate_proj.weight"), normed.data(), inter, hidden, gate.data());
    ref::Matvec(W(p + "mlp.up_proj.weight"), normed.data(), inter, hidden, up.data());
    const int act = static_cast<int>(dense.activation);
    for (int64_t i = 0; i < inter; ++i) {
      float value = ref::Activation(act, gate[i], dense.oai_alpha, dense.oai_limit, true) *
                    ref::Activation(act, up[i], dense.oai_alpha, dense.oai_limit, false);
      if (dense.activation == ops::Activation::kSiluOai) {
        value = ref::Activation(act, gate[i], dense.oai_alpha, dense.oai_limit, true) *
                (std::min(std::max(up[i], -dense.oai_limit), dense.oai_limit) + 1.0f);
      }
      gate[i] = value;
    }
    if (trace_) Dump("layer_N.silu", gate.data(), inter);
    ref::Matvec(W(p + "mlp.down_proj.weight"), gate.data(), hidden, inter, mixed.data());
  }

  void MoeStep(const components::MoeConfig& moe, const std::string& p,
               const std::vector<float>& normed, std::vector<float>& mixed) {
    const int64_t hidden = config_.model.hidden_size;
    std::vector<float> logits(moe.num_experts);
    ref::Matvec(W(p + "mlp.gate.weight"), normed.data(), moe.num_experts, hidden, logits.data());
    const float* router_bias = nullptr;
    auto rb = w_.find(p + "mlp.gate.bias");
    if (rb != w_.end()) router_bias = rb->second.values.data();
    const float* correction = nullptr;
    auto cb = w_.find(p + "mlp.e_score_correction_bias.weight");
    if (cb != w_.end()) correction = cb->second.values.data();
    if (router_bias != nullptr) {
      for (int64_t e = 0; e < moe.num_experts; ++e) logits[e] += router_bias[e];
    }
    std::vector<float> scores(moe.num_experts);
    for (int64_t e = 0; e < moe.num_experts; ++e) {
      scores[e] = moe.routing.scoring == ops::RouterScoring::kSigmoidGroupTopk
                      ? 1.0f / (1.0f + std::exp(-logits[e]))
                      : logits[e];
    }
    std::vector<bool> selectable(moe.num_experts, true);
    if (moe.routing.scoring == ops::RouterScoring::kSigmoidGroupTopk && moe.routing.group_count > 1) {
      const int64_t per = moe.num_experts / moe.routing.group_count;
      std::vector<float> group_scores(moe.routing.group_count);
      for (int64_t g = 0; g < moe.routing.group_count; ++g) {
        float best = -1e30f, second = -1e30f;
        for (int64_t i = 0; i < per; ++i) {
          float s = scores[g * per + i];
          if (correction) s += correction[g * per + i];
          if (s > best) {
            second = best;
            best = s;
          } else if (s > second) {
            second = s;
          }
        }
        group_scores[g] = best + second;
      }
      std::fill(selectable.begin(), selectable.end(), false);
      for (int kept = 0; kept < moe.routing.group_topk; ++kept) {
        int64_t best_g = -1;
        float best = -1e30f;
        for (int64_t g = 0; g < moe.routing.group_count; ++g) {
          if (group_scores[g] > best) {
            best = group_scores[g];
            best_g = g;
          }
        }
        if (best_g < 0) break;
        group_scores[best_g] = -1e30f;
        for (int64_t i = 0; i < per; ++i) selectable[best_g * per + i] = true;
      }
    }
    std::vector<int64_t> chosen;
    std::vector<float> selected;
    std::vector<bool> used(moe.num_experts, false);
    for (int k = 0; k < moe.experts_per_token; ++k) {
      int64_t best_e = -1;
      float best = -1e30f;
      for (int64_t e = 0; e < moe.num_experts; ++e) {
        float s = scores[e];
        if (correction && moe.routing.scoring == ops::RouterScoring::kSigmoidGroupTopk) {
          s += correction[e];
        }
        if (selectable[e] && !used[e] && s > best) {
          best = s;
          best_e = e;
        }
      }
      if (best_e < 0) break;
      used[best_e] = true;
      chosen.push_back(best_e);
      selected.push_back(moe.routing.scoring == ops::RouterScoring::kSigmoidGroupTopk
                             ? scores[best_e]
                             : logits[best_e] - (router_bias ? router_bias[best_e] : 0.0f));
    }
    if (moe.routing.scoring == ops::RouterScoring::kSoftmaxTopkRenorm) {
      float m = *std::max_element(selected.begin(), selected.end());
      float sum = 0;
      for (auto& v : selected) {
        v = std::exp(v - m);
        sum += v;
      }
      for (auto& v : selected) v /= sum;
    } else if (moe.routing.normalize) {
      float sum = 0;
      for (float v : selected) sum += v;
      for (auto& v : selected) v /= (sum + 1e-20f);
    }
    for (auto& v : selected) v *= moe.routing.routing_scale;

    std::fill(mixed.begin(), mixed.end(), 0.0f);
    const int64_t inter = moe.intermediate_size;
    for (int k = 0; k < static_cast<int>(chosen.size()); ++k) {
      const std::string ep = p + "mlp.experts." + std::to_string(chosen[k]) + ".";
      std::vector<float> gate(inter), up(inter), act(inter);
      ref::Matvec(W(ep + "gate_proj.weight"), normed.data(), inter, hidden, gate.data());
      ref::Matvec(W(ep + "up_proj.weight"), normed.data(), inter, hidden, up.data());
      for (int64_t i = 0; i < inter; ++i) {
        act[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
      }
      std::vector<float> out(hidden);
      ref::Matvec(W(ep + "down_proj.weight"), act.data(), hidden, inter, out.data());
      for (int64_t i = 0; i < hidden; ++i) mixed[i] += selected[k] * out[i];
    }
    if (moe.shared_intermediate_size > 0) {
      const std::string sp = p + "mlp.shared_expert.";
      std::vector<float> gate(moe.shared_intermediate_size), up(moe.shared_intermediate_size);
      ref::Matvec(W(sp + "gate_proj.weight"), normed.data(), moe.shared_intermediate_size, hidden,
                  gate.data());
      ref::Matvec(W(sp + "up_proj.weight"), normed.data(), moe.shared_intermediate_size, hidden,
                  up.data());
      for (int64_t i = 0; i < moe.shared_intermediate_size; ++i) {
        gate[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
      }
      std::vector<float> out(hidden);
      ref::Matvec(W(sp + "down_proj.weight"), gate.data(), hidden,
                  moe.shared_intermediate_size, out.data());
      if (moe.gate_shared_expert) {
        std::vector<float> sg(1);
        ref::Matvec(W(p + "mlp.shared_expert_gate.weight"), normed.data(), 1, hidden, sg.data());
        const float factor = 1.0f / (1.0f + std::exp(-sg[0]));
        for (int64_t i = 0; i < hidden; ++i) mixed[i] += factor * out[i];
      } else {
        for (int64_t i = 0; i < hidden; ++i) mixed[i] += out[i];
      }
    }
  }

  causal::DecoderConfig config_;
  Fixture w_;
  size_t last_tokens_ = 0;
  bool trace_ = false;
  std::vector<std::vector<std::vector<float>>> keys_, values_;
  std::vector<GdnState> gdn_;
};

// ------------------------------------------------------------------ fixtures

struct FamilyFixture {
  std::string config_json;
  Fixture tensors;
};

const std::string kCommon =
    "\"vocab_size\":40,\"max_position_embeddings\":128,\"rms_norm_eps\":0.00001,"
    "\"rope_theta\":10000.0";

std::string Json(const std::string& model_type, const std::vector<std::string>& extra) {
  std::string json = "{\"model_type\":\"" + model_type + "\"";
  for (const auto& field : extra) json += "," + field;
  return json + "," + kCommon + "}";
}

void WriteConfig(const TempDir& dir, const std::string& json) {
  std::ofstream(dir.path / "config.json") << json;
}


void CompareLogits(const std::vector<float>& got, const std::vector<float>& want,
                   double tol, const char* family) {
  ASSERT_EQ(got.size(), want.size()) << family;
  double got_norm = 0, diff = 0;
  int64_t top_got = 0, top_want = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    got_norm += double(got[i]) * double(got[i]);
    diff += double(got[i] - want[i]) * double(got[i] - want[i]);
    if (got[i] > got[top_got]) top_got = i;
    if (want[i] > want[top_want]) top_want = i;
  }
  const double relative = std::sqrt(diff / std::max(got_norm, 1e-9));
  EXPECT_LT(relative, tol) << family << " relative error " << relative;
  EXPECT_EQ(top_got, top_want) << family << " top token differs (relative " << relative << ")";
}

void AddDenseLayer(Fixture& tensors, const std::string& prefix, int64_t hidden, int64_t inter,
                   const std::vector<std::string>& extra_norms = {}) {
  tensors[prefix + "input_layernorm.weight"] = T({hidden}, prefix + "input_layernorm", 0.4f);
  tensors[prefix + "post_attention_layernorm.weight"] =
      T({hidden}, prefix + "post_attention_layernorm", 0.4f);
  if (inter > 0) {
    tensors[prefix + "mlp.gate_proj.weight"] = T({inter, hidden}, prefix + "gate");
    tensors[prefix + "mlp.up_proj.weight"] = T({inter, hidden}, prefix + "up");
    tensors[prefix + "mlp.down_proj.weight"] = T({hidden, inter}, prefix + "down");
  }
  for (const auto& name : extra_norms) {
    tensors[prefix + name] = T({hidden}, prefix + name, 0.4f);
  }
}

causal::DecoderConfig TranslateOrDie(const std::string& model_type, const std::string& json) {
  CheckpointConfig identity;
  identity.model_type = model_type;
  auto config = TranslateFamilyConfig(identity, json);
  if (!config.ok()) {
    ADD_FAILURE() << config.status().ToString();
    return causal::DecoderConfig{};
  }
  return *config;
}

// ------------------------------------------------------------------- tests

TEST(ParityTest, Gemma3SandwichWindowsAndGelu) {
  TempDir dir;
  const std::string json = Json(
      "gemma3_text",
      {"\"hidden_size\":16,\"intermediate_size\":24,\"num_hidden_layers\":2,"
       "\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":8,"
       "\"hidden_act\":\"gelu_pytorch_tanh\",\"attention_bias\":true,"
       "\"query_pre_attn_scalar\":16,\"sliding_window\":4,\"sliding_window_pattern\":2,"
       "\"rope_scaling\":{\"rope_type\":\"linear\",\"factor\":4.0},"
       "\"rope_local_base_freq\":5000,\"tie_word_embeddings\":true"});
  WriteConfig(dir, json);
  Fixture tensors;
  tensors["model.embed_tokens.weight"] = T({kVocab, kHidden}, "embed", 0.2f);
  tensors["model.norm.weight"] = T({kHidden}, "final_norm", 0.4f);
  for (int layer = 0; layer < 2; ++layer) {
    const std::string p = "model.layers." + std::to_string(layer) + ".";
    AddDenseLayer(tensors, p, kHidden, 24,
                  {"pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"});
    tensors[p + "self_attn.q_proj.weight"] = T({kHeads * kDim, kHidden}, p + "q");
    tensors[p + "self_attn.q_proj.bias"] = T({kHeads * kDim}, p + "qb", 0.05f);
    tensors[p + "self_attn.k_proj.weight"] = T({kKv * kDim, kHidden}, p + "k");
    tensors[p + "self_attn.k_proj.bias"] = T({kKv * kDim}, p + "kb", 0.05f);
    tensors[p + "self_attn.v_proj.weight"] = T({kKv * kDim, kHidden}, p + "v");
    tensors[p + "self_attn.v_proj.bias"] = T({kKv * kDim}, p + "vb", 0.05f);
    tensors[p + "self_attn.o_proj.weight"] = T({kHidden, kHeads * kDim}, p + "o");
    tensors[p + "self_attn.o_proj.bias"] = T({kHidden}, p + "ob", 0.05f);
    tensors[p + "self_attn.q_norm.weight"] = T({kDim}, p + "qn", 0.3f);
    tensors[p + "self_attn.k_norm.weight"] = T({kDim}, p + "kn", 0.3f);
  }
  WriteCheckpoint(dir, tensors);

  Engine engine;
  const auto got = engine.Forward(dir.path.string(), kTokens);
  ReferenceModel reference(TranslateOrDie("gemma3_text", json), tensors);
  const auto want = reference.Forward(kTokens);
  CompareLogits(got, want, 0.12, "gemma3");
}

TEST(ParityTest, MixtralSoftmaxRoutedExperts) {
  TempDir dir;
  const std::string json =
      Json("mixtral", {"\"hidden_size\":16,\"intermediate_size\":20,\"num_hidden_layers\":2,"
                        "\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":8,"
                        "\"sliding_window\":5,\"use_sliding_window\":true,"
                        "\"num_local_experts\":4,\"num_experts_per_tok\":2,"
                        "\"norm_topk_prob\":true"});
  WriteConfig(dir, json);
  Fixture tensors;
  tensors["model.embed_tokens.weight"] = T({kVocab, kHidden}, "embed", 0.2f);
  tensors["model.norm.weight"] = T({kHidden}, "final_norm", 0.4f);
  tensors["lm_head.weight"] = T({kVocab, kHidden}, "head", 0.2f);
  for (int layer = 0; layer < 2; ++layer) {
    const std::string p = "model.layers." + std::to_string(layer) + ".";
    AddDenseLayer(tensors, p, kHidden, 0);
    tensors[p + "mlp.gate.weight"] = T({4, kHidden}, p + "router", 0.3f);
    for (int e = 0; e < 4; ++e) {
      const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
      tensors[ep + "gate_proj.weight"] = T({20, kHidden}, ep + "g");
      tensors[ep + "up_proj.weight"] = T({20, kHidden}, ep + "u");
      tensors[ep + "down_proj.weight"] = T({kHidden, 20}, ep + "d");
    }
    tensors[p + "self_attn.q_proj.weight"] = T({kHeads * kDim, kHidden}, p + "q");
    tensors[p + "self_attn.k_proj.weight"] = T({kKv * kDim, kHidden}, p + "k");
    tensors[p + "self_attn.v_proj.weight"] = T({kKv * kDim, kHidden}, p + "v");
    tensors[p + "self_attn.o_proj.weight"] = T({kHidden, kHeads * kDim}, p + "o");
  }
  WriteCheckpoint(dir, tensors);
  Engine engine;
  const auto got = engine.Forward(dir.path.string(), kTokens);
  ReferenceModel reference(TranslateOrDie("mixtral", json), tensors);
  CompareLogits(got, reference.Forward(kTokens), 0.12, "mixtral");
}

TEST(ParityTest, DeepseekV3MlaAndGroupedRouting) {
  TempDir dir;
  const std::string json = Json(
      "deepseek_v3",
      {"\"hidden_size\":16,\"num_hidden_layers\":2,\"num_attention_heads\":4,"
       "\"num_key_value_heads\":4,\"head_dim\":8,\"intermediate_size\":20,"
       "\"q_lora_rank\":8,\"kv_lora_rank\":6,\"qk_nope_head_dim\":4,\"qk_rope_head_dim\":4,"
       "\"v_head_dim\":6,\"first_k_dense_replace\":1,\"n_routed_experts\":4,"
       "\"num_experts_per_tok\":2,\"n_group\":2,\"topk_group\":1,\"norm_topk_prob\":true,"
       "\"routed_scaling_factor\":1.25,\"moe_intermediate_size\":12"});
  WriteConfig(dir, json);
  Fixture tensors;
  tensors["model.embed_tokens.weight"] = T({kVocab, kHidden}, "embed", 0.2f);
  tensors["model.norm.weight"] = T({kHidden}, "final_norm", 0.4f);
  tensors["lm_head.weight"] = T({kVocab, kHidden}, "head", 0.2f);
  for (int layer = 0; layer < 2; ++layer) {
    const std::string p = "model.layers." + std::to_string(layer) + ".";
    AddDenseLayer(tensors, p, kHidden, layer == 0 ? 20 : 0);
    if (layer > 0) {
      tensors[p + "mlp.gate.weight"] = T({4, kHidden}, p + "router", 0.3f);
      tensors[p + "mlp.e_score_correction_bias.weight"] = T({4}, p + "corr", 0.05f);
      for (int e = 0; e < 4; ++e) {
        const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
        tensors[ep + "gate_proj.weight"] = T({12, kHidden}, ep + "g");
        tensors[ep + "up_proj.weight"] = T({12, kHidden}, ep + "u");
        tensors[ep + "down_proj.weight"] = T({kHidden, 12}, ep + "d");
      }
    }
    tensors[p + "self_attn.q_a_proj.weight"] = T({8, kHidden}, p + "qa");
    tensors[p + "self_attn.q_a_layernorm.weight"] = T({8}, p + "qan", 0.4f);
    tensors[p + "self_attn.q_b_proj.weight"] = T({4 * 8, 8}, p + "qb");
    tensors[p + "self_attn.kv_a_proj_with_mqa.weight"] = T({10, kHidden}, p + "kva");
    tensors[p + "self_attn.kv_a_layernorm.weight"] = T({6}, p + "kvan", 0.4f);
    tensors[p + "self_attn.kv_b_proj.weight"] = T({4 * 10, 6}, p + "kvb");
    tensors[p + "self_attn.o_proj.weight"] = T({kHidden, 4 * 6}, p + "o");
  }
  WriteCheckpoint(dir, tensors);
  Engine engine;
  const auto got = engine.Forward(dir.path.string(), kTokens);
  // The engine zero-pads each head's o_proj columns to head_dim at load;
  // mirror that so the reference multiplies the same matrix.
  Fixture reference_tensors = tensors;
  for (int layer = 0; layer < 2; ++layer) {
    const std::string p = "model.layers." + std::to_string(layer) + ".";
    const auto& o = tensors.at(p + "self_attn.o_proj.weight");
    std::vector<float> padded(16 * 32, 0.0f);
    for (int64_t r = 0; r < 16; ++r) {
      for (int64_t h = 0; h < 4; ++h) {
        for (int64_t d = 0; d < 6; ++d) {
          padded[r * 32 + h * 8 + d] = o.values[r * 24 + h * 6 + d];
        }
      }
    }
    reference_tensors[p + "self_attn.o_proj.weight"] = FTensor{{16, 32}, std::move(padded)};
  }
  ReferenceModel reference(TranslateOrDie("deepseek_v3", json), reference_tensors);
  CompareLogits(got, reference.Forward(kTokens), 0.15, "deepseek_v3");
}

TEST(ParityTest, Qwen3NextHybridGdnAndGatedAttention) {
  TempDir dir;
  const std::string json = Json(
      "qwen3_next",
      {"\"hidden_size\":16,\"intermediate_size\":20,\"num_hidden_layers\":2,"
       "\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":8,"
       "\"layer_types\":[\"linear_attention\",\"full_attention\"],"
       "\"full_attention_interval\":1,\"linear_num_key_heads\":1,"
       "\"linear_num_value_heads\":2,\"linear_key_head_dim\":4,\"linear_value_head_dim\":4,"
       "\"linear_conv_kernel_dim\":3,\"decoder_sparse_step\":1,\"num_experts\":4,"
       "\"num_experts_per_tok\":2,\"moe_intermediate_size\":12,\"norm_topk_prob\":true,"
       "\"shared_expert_intermediate_size\":12,\"mlp_only_layers\":[]"});
  WriteConfig(dir, json);
  Fixture tensors;
  tensors["model.embed_tokens.weight"] = T({kVocab, kHidden}, "embed", 0.15f);
  tensors["model.norm.weight"] = T({kHidden}, "final_norm", 0.4f);
  tensors["lm_head.weight"] = T({kVocab, kHidden}, "head", 0.15f);
  const int64_t q_total = 1 * 4, v_total = 2 * 4;
  for (int layer = 0; layer < 2; ++layer) {
    const std::string p = "model.layers." + std::to_string(layer) + ".";
    AddDenseLayer(tensors, p, kHidden, 0);
    tensors[p + "mlp.gate.weight"] = T({4, kHidden}, p + "router", 0.3f);
    for (int e = 0; e < 4; ++e) {
      const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
      tensors[ep + "gate_proj.weight"] = T({12, kHidden}, ep + "g");
      tensors[ep + "up_proj.weight"] = T({12, kHidden}, ep + "u");
      tensors[ep + "down_proj.weight"] = T({kHidden, 12}, ep + "d");
    }
    tensors[p + "mlp.shared_expert.gate_proj.weight"] = T({12, kHidden}, p + "sg");
    tensors[p + "mlp.shared_expert.up_proj.weight"] = T({12, kHidden}, p + "su");
    tensors[p + "mlp.shared_expert.down_proj.weight"] = T({kHidden, 12}, p + "sd");
    tensors[p + "mlp.shared_expert_gate.weight"] = T({1, kHidden}, p + "sgate", 0.3f);
    if (layer == 0) {
      tensors[p + "self_attn.in_proj_qkvz.weight"] =
          T({2 * q_total + 2 * v_total, kHidden}, p + "qkvz");
      tensors[p + "self_attn.in_proj_ba.weight"] = T({2 * 2, kHidden}, p + "ba");
      tensors[p + "self_attn.conv1d.weight"] = FTensor{{{2 * q_total + v_total, 1, 3}}, T({2 * q_total + v_total, 3}, p + "conv", 0.3f).values};
      tensors[p + "self_attn.A_log.weight"] = T({2}, p + "alog", 0.1f);
      tensors[p + "self_attn.dt_bias.weight"] = T({2}, p + "dtb", 0.1f);
      tensors[p + "self_attn.norm.weight"] = T({4}, p + "gnorm", 0.4f);
      tensors[p + "self_attn.out_proj.weight"] = T({kHidden, v_total}, p + "out");
    } else {
      // Gated attention: q rows are [q_h | gate_h] per head, interleaved.
      const int64_t qw = kHeads * kDim;
      std::vector<float> q_rows(2 * qw * kHidden);
      const auto q_plain = T({qw, kHidden}, p + "q").values;
      const auto gate_plain = T({qw, kHidden}, p + "gate", 0.3f).values;
      for (int64_t h = 0; h < kHeads; ++h) {
        for (int64_t d = 0; d < kDim; ++d) {
          for (int64_t c = 0; c < kHidden; ++c) {
            q_rows[((2 * h) * kDim + d) * kHidden + c] = q_plain[(h * kDim + d) * kHidden + c];
            q_rows[((2 * h + 1) * kDim + d) * kHidden + c] =
                gate_plain[(h * kDim + d) * kHidden + c];
          }
        }
      }
      tensors[p + "self_attn.q_proj.weight"] =
          FTensor{{2 * qw, kHidden}, std::move(q_rows)};
      tensors[p + "self_attn.k_proj.weight"] = T({kKv * kDim, kHidden}, p + "k");
      tensors[p + "self_attn.v_proj.weight"] = T({kKv * kDim, kHidden}, p + "v");
      tensors[p + "self_attn.o_proj.weight"] = T({kHidden, qw}, p + "o");
      tensors[p + "self_attn.q_norm.weight"] = T({kDim}, p + "qn", 0.3f);
      tensors[p + "self_attn.k_norm.weight"] = T({kDim}, p + "kn", 0.3f);
    }
  }
  WriteCheckpoint(dir, tensors);
  Engine engine;
  const auto got = engine.Forward(dir.path.string(), kTokens);
  // The reference reads de-interleaved weights: split the fixture's q rows.
  Fixture reference_tensors = tensors;
  {
    const std::string p = "model.layers.1.";
    const auto& fused = tensors.at(p + "self_attn.q_proj.weight").values;
    const int64_t qw = kHeads * kDim;
    std::vector<float> plain(2 * qw * kHidden);
    for (int64_t h = 0; h < kHeads; ++h) {
      for (int64_t d = 0; d < kDim; ++d) {
        for (int64_t c = 0; c < kHidden; ++c) {
          plain[(h * kDim + d) * kHidden + c] = fused[((2 * h) * kDim + d) * kHidden + c];
          plain[(qw + h * kDim + d) * kHidden + c] =
              fused[((2 * h + 1) * kDim + d) * kHidden + c];
        }
      }
    }
    reference_tensors[p + "self_attn.q_proj.weight"] =
        FTensor{{qw, kHidden}, std::vector<float>(plain.begin(), plain.begin() + qw * kHidden)};
    reference_tensors[p + "self_attn.gate_proj.weight"] =
        FTensor{{qw, kHidden},
                std::vector<float>(plain.begin() + qw * kHidden, plain.end())};
  }
  ReferenceModel reference(TranslateOrDie("qwen3_next", json), reference_tensors);
  CompareLogits(got, reference.Forward(kTokens), 0.15, "qwen3_next");
}

}  // namespace
}  // namespace inferx