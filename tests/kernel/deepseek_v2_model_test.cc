/// DeepseekV2Model, against a host reference of the whole stack.
///
/// The M9 layer suites pin MLA and MoE individually; what a model class can
/// still get wrong is the *composition* -- norms in the wrong place, a
/// residual dropped, the dense/MoE branch inverted, the per-sequence batch
/// slicing misaligned. There is no DeepSeek checkpoint on this box, so like
/// the layer suites this compares against the definition: a synthetic
/// checkpoint small enough that a float reference of the entire model runs in
/// milliseconds, written to disk in the real safetensors + config.json format
/// so the loader's name and shape mapping is exercised too.

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "absl/strings/str_cat.h"
#include "inferx/core/cuda_utils.h"
#include "inferx/model/deepseek_v2.h"
#include "inferx/model/forward_batch.h"

namespace inferx {
namespace {

using model::DeepseekV2Model;
using model::ForwardBatch;

// --- Tiny model shape --------------------------------------------------------

constexpr int64_t kHidden = 32;
constexpr int64_t kDenseInter = 48;
constexpr int64_t kLayers = 2;  // layer 0 dense, layer 1 MoE
constexpr int64_t kHeads = 2;
constexpr int64_t kVocab = 64;
constexpr int64_t kLatent = 32;
constexpr int64_t kNope = 16;
constexpr int64_t kRope = 8;
constexpr int64_t kVDim = 16;
constexpr int64_t kQk = kNope + kRope;
constexpr int64_t kExperts = 4;
constexpr int64_t kTopK = 2;
constexpr int64_t kMoeInter = 16;
constexpr int64_t kSharedInter = 16;  // n_shared_experts 1 × moe_inter
constexpr float kTheta = 10000.0f;
constexpr float kEps = 1e-6f;

// The DeepSeek key spellings on purpose: the checkpoint fixture exercises the
// same alias path (n_routed_experts, n_shared_experts, type: null q_lora_rank)
// a real download would.
constexpr char kConfigJson[] = R"({
  "architectures": ["DeepseekV2ForCausalLM"],
  "hidden_size": 32,
  "intermediate_size": 48,
  "num_hidden_layers": 2,
  "num_attention_heads": 2,
  "num_key_value_heads": 2,
  "vocab_size": 64,
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000,
  "tie_word_embeddings": false,
  "attention_bias": false,
  "torch_dtype": "bfloat16",
  "kv_lora_rank": 32,
  "q_lora_rank": null,
  "qk_nope_head_dim": 16,
  "qk_rope_head_dim": 8,
  "v_head_dim": 16,
  "n_routed_experts": 4,
  "num_experts_per_tok": 2,
  "moe_intermediate_size": 16,
  "n_shared_experts": 1,
  "norm_topk_prob": false,
  "first_k_dense_replace": 1,
  "moe_layer_freq": 1,
  "routed_scaling_factor": 1.0,
  "scoring_func": "softmax",
  "topk_method": "greedy"
})";

// --- Deterministic host tensors ----------------------------------------------

float Fill(int64_t a, int64_t b, float salt) {
  return 0.4f * std::sin(0.61f * static_cast<float>(a) + salt) *
         std::cos(0.23f * static_cast<float>(b) - salt);
}

std::vector<float> RoundTrip(const std::vector<float>& host) {
  std::vector<float> out(host.size());
  for (size_t i = 0; i < host.size(); ++i) {
    out[i] = __bfloat162float(__float2bfloat16(host[i]));
  }
  return out;
}

std::vector<float> RandomTensor(size_t n, float salt) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = Fill(static_cast<int64_t>(i), static_cast<int64_t>(i % 17), salt);
  }
  return RoundTrip(v);
}

std::vector<float> NormWeight(size_t n, float salt) {
  std::vector<float> v = RandomTensor(n, salt);
  for (float& x : v) x = 1.0f + 0.1f * x;
  return RoundTrip(v);
}

// --- The synthetic checkpoint ------------------------------------------------

// Every tensor the loader reads, as floats already rounded through bf16 so the
// host reference computes with exactly what the device sees.
struct HostWeights {
  std::vector<float> embed, final_norm, lm_head;

  struct Layer {
    std::vector<float> input_norm, post_norm;
    std::vector<float> q_proj, kv_a, kv_a_norm, kv_b, o;
    // Dense (layer 0).
    std::vector<float> gate, up, down;
    // MoE (layer 1).
    std::vector<float> router;
    std::vector<std::vector<float>> e_gate, e_up, e_down;
    std::vector<float> s_gate, s_up, s_down;
  };
  std::vector<Layer> layers;
};

HostWeights MakeWeights() {
  HostWeights w;
  w.embed = RandomTensor(static_cast<size_t>(kVocab * kHidden), 0.13f);
  w.final_norm = NormWeight(static_cast<size_t>(kHidden), 0.29f);
  w.lm_head = RandomTensor(static_cast<size_t>(kVocab * kHidden), 0.43f);

  for (int64_t i = 0; i < kLayers; ++i) {
    HostWeights::Layer l;
    const float s = 1.0f + static_cast<float>(i);

    l.input_norm = NormWeight(static_cast<size_t>(kHidden), 0.51f * s);
    l.post_norm = NormWeight(static_cast<size_t>(kHidden), 0.67f * s);
    l.q_proj =
        RandomTensor(static_cast<size_t>(kHeads * kQk * kHidden), 0.87f * s);
    l.kv_a = RandomTensor(static_cast<size_t>((kLatent + kRope) * kHidden),
                          1.03f * s);
    l.kv_a_norm = NormWeight(static_cast<size_t>(kLatent), 1.21f * s);
    l.kv_b = RandomTensor(
        static_cast<size_t>(kHeads * (kNope + kVDim) * kLatent), 1.39f * s);
    l.o = RandomTensor(static_cast<size_t>(kHidden * kHeads * kVDim),
                       1.57f * s);

    if (i == 0) {
      l.gate = RandomTensor(static_cast<size_t>(kDenseInter * kHidden), 1.73f);
      l.up = RandomTensor(static_cast<size_t>(kDenseInter * kHidden), 1.91f);
      l.down = RandomTensor(static_cast<size_t>(kHidden * kDenseInter), 2.09f);
    } else {
      l.router = RandomTensor(static_cast<size_t>(kExperts * kHidden), 2.27f);
      for (int64_t e = 0; e < kExperts; ++e) {
        const float es = 2.41f + 0.37f * static_cast<float>(e);
        l.e_gate.push_back(
            RandomTensor(static_cast<size_t>(kMoeInter * kHidden), es));
        l.e_up.push_back(
            RandomTensor(static_cast<size_t>(kMoeInter * kHidden), es + 0.11f));
        l.e_down.push_back(
            RandomTensor(static_cast<size_t>(kHidden * kMoeInter), es + 0.23f));
      }
      l.s_gate =
          RandomTensor(static_cast<size_t>(kSharedInter * kHidden), 3.51f);
      l.s_up =
          RandomTensor(static_cast<size_t>(kSharedInter * kHidden), 3.67f);
      l.s_down =
          RandomTensor(static_cast<size_t>(kHidden * kSharedInter), 3.83f);
    }

    w.layers.push_back(std::move(l));
  }

  return w;
}

// Writes config.json and a single-file model.safetensors holding bf16 copies
// of every host tensor, in the format the real loader reads.
class TempCheckpoint {
 public:
  TempCheckpoint() {
    char tmpl[] = "/tmp/inferx_dsv2_XXXXXX";
    dir_ = ::mkdtemp(tmpl) ? std::string(tmpl) : std::string();
  }

  ~TempCheckpoint() {
    if (dir_.empty()) return;
    for (const std::string& f : files_) ::unlink(f.c_str());
    ::rmdir(dir_.c_str());
  }

  const std::string& dir() const { return dir_; }

  void Add(const std::string& name, const std::vector<int64_t>& shape,
           const std::vector<float>& values) {
    Entry e;
    e.name = name;
    e.shape = shape;
    e.data.resize(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      const __nv_bfloat16 b = __float2bfloat16(values[i]);
      std::memcpy(&e.data[i], &b, sizeof(uint16_t));
    }
    entries_.push_back(std::move(e));
  }

  void Write(const std::string& config_json) {
    {
      const std::string path = dir_ + "/config.json";
      std::ofstream out(path);
      out << config_json;
      files_.push_back(path);
    }

    std::string header = "{";
    size_t offset = 0;
    for (size_t i = 0; i < entries_.size(); ++i) {
      const Entry& e = entries_[i];
      const size_t bytes = e.data.size() * sizeof(uint16_t);
      absl::StrAppend(&header, i == 0 ? "" : ",", "\"", e.name,
                      "\":{\"dtype\":\"BF16\",\"shape\":[");
      for (size_t d = 0; d < e.shape.size(); ++d) {
        absl::StrAppend(&header, d == 0 ? "" : ",", e.shape[d]);
      }
      absl::StrAppend(&header, "],\"data_offsets\":[", offset, ",",
                      offset + bytes, "]}");
      offset += bytes;
    }
    header += "}";

    const std::string path = dir_ + "/model.safetensors";
    std::ofstream out(path, std::ios::binary);
    const uint64_t header_len = header.size();
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const Entry& e : entries_) {
      out.write(reinterpret_cast<const char*>(e.data.data()),
                static_cast<std::streamsize>(e.data.size() * sizeof(uint16_t)));
    }
    files_.push_back(path);
  }

 private:
  struct Entry {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<uint16_t> data;
  };

  std::string dir_;
  std::vector<Entry> entries_;
  std::vector<std::string> files_;
};

void WriteCheckpoint(TempCheckpoint& ckpt, const HostWeights& w) {
  ckpt.Add("model.embed_tokens.weight", {kVocab, kHidden}, w.embed);
  ckpt.Add("model.norm.weight", {kHidden}, w.final_norm);
  ckpt.Add("lm_head.weight", {kVocab, kHidden}, w.lm_head);

  for (int64_t i = 0; i < kLayers; ++i) {
    const HostWeights::Layer& l = w.layers[static_cast<size_t>(i)];
    const std::string p = absl::StrCat("model.layers.", i, ".");

    ckpt.Add(p + "input_layernorm.weight", {kHidden}, l.input_norm);
    ckpt.Add(p + "post_attention_layernorm.weight", {kHidden}, l.post_norm);
    ckpt.Add(p + "self_attn.q_proj.weight", {kHeads * kQk, kHidden}, l.q_proj);
    ckpt.Add(p + "self_attn.kv_a_proj_with_mqa.weight",
             {kLatent + kRope, kHidden}, l.kv_a);
    ckpt.Add(p + "self_attn.kv_a_layernorm.weight", {kLatent}, l.kv_a_norm);
    ckpt.Add(p + "self_attn.kv_b_proj.weight",
             {kHeads * (kNope + kVDim), kLatent}, l.kv_b);
    ckpt.Add(p + "self_attn.o_proj.weight", {kHidden, kHeads * kVDim}, l.o);

    if (i == 0) {
      ckpt.Add(p + "mlp.gate_proj.weight", {kDenseInter, kHidden}, l.gate);
      ckpt.Add(p + "mlp.up_proj.weight", {kDenseInter, kHidden}, l.up);
      ckpt.Add(p + "mlp.down_proj.weight", {kHidden, kDenseInter}, l.down);
    } else {
      ckpt.Add(p + "mlp.gate.weight", {kExperts, kHidden}, l.router);
      for (int64_t e = 0; e < kExperts; ++e) {
        const std::string ep = absl::StrCat(p, "mlp.experts.", e, ".");
        ckpt.Add(ep + "gate_proj.weight", {kMoeInter, kHidden},
                 l.e_gate[static_cast<size_t>(e)]);
        ckpt.Add(ep + "up_proj.weight", {kMoeInter, kHidden},
                 l.e_up[static_cast<size_t>(e)]);
        ckpt.Add(ep + "down_proj.weight", {kHidden, kMoeInter},
                 l.e_down[static_cast<size_t>(e)]);
      }
      ckpt.Add(p + "mlp.shared_experts.gate_proj.weight",
               {kSharedInter, kHidden}, l.s_gate);
      ckpt.Add(p + "mlp.shared_experts.up_proj.weight",
               {kSharedInter, kHidden}, l.s_up);
      ckpt.Add(p + "mlp.shared_experts.down_proj.weight",
               {kHidden, kSharedInter}, l.s_down);
    }
  }

  ckpt.Write(kConfigJson);
}

// --- Host reference ----------------------------------------------------------

std::vector<float> MatMulT(const std::vector<float>& x,
                           const std::vector<float>& w, int64_t m, int64_t k,
                           int64_t n) {
  std::vector<float> y(static_cast<size_t>(m * n), 0.0f);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int64_t p = 0; p < k; ++p) {
        acc += x[static_cast<size_t>(i * k + p)] *
               w[static_cast<size_t>(j * k + p)];
      }
      y[static_cast<size_t>(i * n + j)] = acc;
    }
  }
  return y;
}

std::vector<float> RmsNormed(const std::vector<float>& x,
                             const std::vector<float>& weight, int64_t rows,
                             int64_t width) {
  std::vector<float> out(x);
  for (int64_t r = 0; r < rows; ++r) {
    float sq = 0.0f;
    for (int64_t c = 0; c < width; ++c) {
      const float v = out[static_cast<size_t>(r * width + c)];
      sq += v * v;
    }
    const float inv = 1.0f / std::sqrt(sq / static_cast<float>(width) + kEps);
    for (int64_t c = 0; c < width; ++c) {
      out[static_cast<size_t>(r * width + c)] *=
          inv * weight[static_cast<size_t>(c)];
    }
  }
  return out;
}

void RopeTail(std::vector<float>& x, int64_t rows, int64_t width,
              int64_t rope_dim, int64_t rows_per_position) {
  const int64_t half = rope_dim / 2;
  for (int64_t r = 0; r < rows; ++r) {
    const float pos = static_cast<float>(r / rows_per_position);
    float* row = x.data() + r * width + (width - rope_dim);
    for (int64_t j = 0; j < half; ++j) {
      const float inv_freq = std::pow(
          kTheta, -2.0f * static_cast<float>(j) / static_cast<float>(rope_dim));
      const float angle = pos * inv_freq;
      const float s = std::sin(angle);
      const float cs = std::cos(angle);
      const float lo = row[j];
      const float hi = row[j + half];
      row[j] = lo * cs - hi * s;
      row[j + half] = hi * cs + lo * s;
    }
  }
}

std::vector<float> SiluMlp(const std::vector<float>& x,
                           const std::vector<float>& gate,
                           const std::vector<float>& up,
                           const std::vector<float>& down, int64_t tokens,
                           int64_t width, int64_t inter) {
  const std::vector<float> g = MatMulT(x, gate, tokens, width, inter);
  const std::vector<float> u = MatMulT(x, up, tokens, width, inter);
  std::vector<float> act(static_cast<size_t>(tokens * inter));
  for (size_t i = 0; i < act.size(); ++i) {
    act[i] = (g[i] / (1.0f + std::exp(-g[i]))) * u[i];
  }
  return MatMulT(act, down, tokens, inter, width);
}

// MLA attention for one full prompt at positions 0..tokens-1, the definition
// with no cache, mirroring mla_test's reference with q_lora_rank == 0.
std::vector<float> ReferenceAttention(const HostWeights::Layer& l,
                                      const std::vector<float>& x,
                                      int64_t tokens) {
  std::vector<float> q = MatMulT(x, l.q_proj, tokens, kHidden, kHeads * kQk);
  RopeTail(q, tokens * kHeads, kQk, kRope, kHeads);

  std::vector<float> kv_a =
      MatMulT(x, l.kv_a, tokens, kHidden, kLatent + kRope);
  RopeTail(kv_a, tokens, kLatent + kRope, kRope, 1);

  std::vector<float> latent(static_cast<size_t>(tokens * kLatent));
  std::vector<float> k_rope(static_cast<size_t>(tokens * kRope));
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t d = 0; d < kLatent; ++d) {
      latent[static_cast<size_t>(t * kLatent + d)] =
          kv_a[static_cast<size_t>(t * (kLatent + kRope) + d)];
    }
    for (int64_t d = 0; d < kRope; ++d) {
      k_rope[static_cast<size_t>(t * kRope + d)] =
          kv_a[static_cast<size_t>(t * (kLatent + kRope) + kLatent + d)];
    }
  }

  {
    std::vector<float> normed = RmsNormed(latent, l.kv_a_norm, tokens, kLatent);
    latent.swap(normed);
  }

  const std::vector<float> kv =
      MatMulT(latent, l.kv_b, tokens, kLatent, kHeads * (kNope + kVDim));

  std::vector<float> attn(static_cast<size_t>(tokens * kHeads * kVDim), 0.0f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(kQk));

  for (int64_t i = 0; i < tokens; ++i) {
    for (int64_t hd = 0; hd < kHeads; ++hd) {
      std::vector<float> scores(static_cast<size_t>(i + 1));
      for (int64_t j = 0; j <= i; ++j) {
        float dot = 0.0f;
        for (int64_t d = 0; d < kNope; ++d) {
          dot += q[static_cast<size_t>((i * kHeads + hd) * kQk + d)] *
                 kv[static_cast<size_t>((j * kHeads + hd) * (kNope + kVDim) +
                                        d)];
        }
        for (int64_t d = 0; d < kRope; ++d) {
          dot += q[static_cast<size_t>((i * kHeads + hd) * kQk + kNope + d)] *
                 k_rope[static_cast<size_t>(j * kRope + d)];
        }
        scores[static_cast<size_t>(j)] = dot * scale;
      }

      float max_v = -INFINITY;
      for (float s : scores) max_v = std::max(max_v, s);
      float sum = 0.0f;
      for (float& s : scores) {
        s = std::exp(s - max_v);
        sum += s;
      }

      for (int64_t d = 0; d < kVDim; ++d) {
        float acc = 0.0f;
        for (int64_t j = 0; j <= i; ++j) {
          acc += scores[static_cast<size_t>(j)] *
                 kv[static_cast<size_t>((j * kHeads + hd) * (kNope + kVDim) +
                                        kNope + d)];
        }
        attn[static_cast<size_t>((i * kHeads + hd) * kVDim + d)] = acc / sum;
      }
    }
  }

  return MatMulT(attn, l.o, tokens, kHeads * kVDim, kHidden);
}

// DeepSeek's MoE: softmax over all experts, greedy top-k without
// renormalization, plus the ungated shared MLP.
std::vector<float> ReferenceMoe(const HostWeights::Layer& l,
                                const std::vector<float>& x, int64_t tokens) {
  const std::vector<float> logits =
      MatMulT(x, l.router, tokens, kHidden, kExperts);

  std::vector<float> out(static_cast<size_t>(tokens * kHidden), 0.0f);

  for (int64_t t = 0; t < tokens; ++t) {
    std::vector<float> probs(static_cast<size_t>(kExperts));
    float max_v = -INFINITY;
    for (int64_t e = 0; e < kExperts; ++e) {
      max_v = std::max(max_v, logits[static_cast<size_t>(t * kExperts + e)]);
    }
    float sum = 0.0f;
    for (int64_t e = 0; e < kExperts; ++e) {
      probs[static_cast<size_t>(e)] =
          std::exp(logits[static_cast<size_t>(t * kExperts + e)] - max_v);
      sum += probs[static_cast<size_t>(e)];
    }
    for (float& p : probs) p /= sum;

    std::vector<float> masked(probs);
    const std::vector<float> row(x.begin() + static_cast<ptrdiff_t>(t * kHidden),
                                 x.begin() +
                                     static_cast<ptrdiff_t>((t + 1) * kHidden));
    for (int64_t slot = 0; slot < kTopK; ++slot) {
      int64_t best = -1;
      float best_v = -INFINITY;
      for (int64_t e = 0; e < kExperts; ++e) {
        if (masked[static_cast<size_t>(e)] > best_v) {
          best_v = masked[static_cast<size_t>(e)];
          best = e;
        }
      }
      masked[static_cast<size_t>(best)] = -INFINITY;

      const std::vector<float> y = SiluMlp(
          row, l.e_gate[static_cast<size_t>(best)],
          l.e_up[static_cast<size_t>(best)],
          l.e_down[static_cast<size_t>(best)], 1, kHidden, kMoeInter);
      for (int64_t d = 0; d < kHidden; ++d) {
        out[static_cast<size_t>(t * kHidden + d)] +=
            probs[static_cast<size_t>(best)] * y[static_cast<size_t>(d)];
      }
    }
  }

  const std::vector<float> shared =
      SiluMlp(x, l.s_gate, l.s_up, l.s_down, tokens, kHidden, kSharedInter);
  for (size_t i = 0; i < out.size(); ++i) out[i] += shared[i];

  return out;
}

std::vector<float> ReferenceModel(const HostWeights& w,
                                  const std::vector<int32_t>& token_ids) {
  const int64_t tokens = static_cast<int64_t>(token_ids.size());

  std::vector<float> x(static_cast<size_t>(tokens * kHidden));
  for (int64_t t = 0; t < tokens; ++t) {
    for (int64_t d = 0; d < kHidden; ++d) {
      x[static_cast<size_t>(t * kHidden + d)] =
          w.embed[static_cast<size_t>(token_ids[static_cast<size_t>(t)] *
                                          kHidden +
                                      d)];
    }
  }

  for (int64_t i = 0; i < kLayers; ++i) {
    const HostWeights::Layer& l = w.layers[static_cast<size_t>(i)];

    const std::vector<float> normed = RmsNormed(x, l.input_norm, tokens, kHidden);
    const std::vector<float> attn = ReferenceAttention(l, normed, tokens);
    for (size_t j = 0; j < x.size(); ++j) x[j] += attn[j];

    const std::vector<float> normed2 = RmsNormed(x, l.post_norm, tokens, kHidden);
    const std::vector<float> ffn =
        i == 0 ? SiluMlp(normed2, l.gate, l.up, l.down, tokens, kHidden,
                         kDenseInter)
               : ReferenceMoe(l, normed2, tokens);
    for (size_t j = 0; j < x.size(); ++j) x[j] += ffn[j];
  }

  const std::vector<float> normed = RmsNormed(x, w.final_norm, tokens, kHidden);
  return MatMulT(normed, w.lm_head, tokens, kHidden, kVocab);
}

// --- The tests ---------------------------------------------------------------

class DeepseekV2ModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";
  }
};

// One prefill batch over an attached cache: sequential slots in freshly
// allocated blocks, logits for every position.
ForwardBatch PrefillBatch(DeepseekV2Model& m,
                          const std::vector<int32_t>& token_ids) {
  const int64_t tokens = static_cast<int64_t>(token_ids.size());
  const int64_t block_size = m.kv_pool()->block_size();
  const int64_t blocks = (tokens + block_size - 1) / block_size;

  ForwardBatch batch;
  batch.num_seqs = 1;
  batch.max_blocks_per_seq = blocks;
  for (int64_t b = 0; b < blocks; ++b) {
    auto block = m.kv_pool()->AllocateBlock();
    EXPECT_TRUE(block.ok()) << block.status();
    batch.block_table.push_back(*block);
  }
  for (int64_t t = 0; t < tokens; ++t) {
    batch.token_ids.push_back(token_ids[static_cast<size_t>(t)]);
    batch.positions.push_back(static_cast<int32_t>(t));
    batch.seq_of_token.push_back(0);
    batch.slots.push_back(
        batch.block_table[static_cast<size_t>(t / block_size)] *
            static_cast<int32_t>(block_size) +
        static_cast<int32_t>(t % block_size));
    batch.logits_indices.push_back(static_cast<int32_t>(t));
  }
  return batch;
}

void ExpectClose(const std::vector<float>& got, const std::vector<float>& want,
                 double tolerance, const char* what) {
  ASSERT_EQ(got.size(), want.size()) << what;

  double scale = 0.0;
  for (const float v : want) scale = std::max<double>(scale, std::abs(v));
  ASSERT_GT(scale, 0.0) << "degenerate reference for " << what;

  double worst = 0.0;
  size_t worst_i = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double err = std::abs(static_cast<double>(got[i]) - want[i]);
    if (err > worst) {
      worst = err;
      worst_i = i;
    }
  }

  EXPECT_LT(worst / scale, tolerance)
      << what << ": worst error " << worst << " at " << worst_i
      << " against scale " << scale;
}

TEST_F(DeepseekV2ModelTest, PrefillMatchesTheHostReference) {
  const HostWeights w = MakeWeights();
  TempCheckpoint ckpt;
  ASSERT_FALSE(ckpt.dir().empty());
  WriteCheckpoint(ckpt, w);

  auto m = DeepseekV2Model::Load(ckpt.dir());
  ASSERT_TRUE(m.ok()) << m.status();
  ASSERT_TRUE(m->AttachKvCache(/*num_blocks=*/4, /*block_size=*/4).ok());

  const std::vector<int32_t> prompt{5, 17, 3, 42, 63, 0, 9};
  const ForwardBatch batch = PrefillBatch(*m, prompt);

  std::vector<float> got;
  ASSERT_TRUE(m->Step(batch, &got).ok());

  const std::vector<float> want = ReferenceModel(w, prompt);

  // Two full layers plus embedding and head in bf16 against a float
  // reference: the layer suites hold 2-3%, and the composition roughly
  // doubles the depth.
  ExpectClose(got, want, 0.05, "prefill logits");
}

TEST_F(DeepseekV2ModelTest, DecodingTokenByTokenEqualsPrefillingAtOnce) {
  const HostWeights w = MakeWeights();
  TempCheckpoint ckpt;
  ASSERT_FALSE(ckpt.dir().empty());
  WriteCheckpoint(ckpt, w);

  auto m = DeepseekV2Model::Load(ckpt.dir());
  ASSERT_TRUE(m.ok()) << m.status();

  const std::vector<int32_t> prompt{11, 2, 30, 8, 55, 21};

  // Prefill in one call.
  ASSERT_TRUE(m->AttachKvCache(4, 4).ok());
  std::vector<float> prefilled;
  ASSERT_TRUE(m->Step(PrefillBatch(*m, prompt), &prefilled).ok());
  const std::vector<float> last_prefill(
      prefilled.end() - kVocab, prefilled.end());

  // Same tokens one at a time, through a fresh cache.
  ASSERT_TRUE(m->AttachKvCache(4, 4).ok());
  std::vector<int32_t> blocks;
  for (int64_t b = 0; b < 2; ++b) {
    auto block = m->kv_pool()->AllocateBlock();
    ASSERT_TRUE(block.ok()) << block.status();
    blocks.push_back(*block);
  }

  std::vector<float> last_decode;
  for (size_t t = 0; t < prompt.size(); ++t) {
    ForwardBatch step;
    step.num_seqs = 1;
    step.max_blocks_per_seq = 2;
    step.block_table = blocks;
    step.token_ids = {prompt[t]};
    step.positions = {static_cast<int32_t>(t)};
    step.seq_of_token = {0};
    step.slots = {blocks[t / 4] * 4 + static_cast<int32_t>(t % 4)};
    step.logits_indices = {0};
    step.decode_only = t > 0;

    ASSERT_TRUE(m->Step(step, &last_decode).ok());
  }

  ExpectClose(last_decode, last_prefill, 0.02, "decode vs prefill");
}

TEST_F(DeepseekV2ModelTest, ForwardMatchesThePagedStep) {
  const HostWeights w = MakeWeights();
  TempCheckpoint ckpt;
  ASSERT_FALSE(ckpt.dir().empty());
  WriteCheckpoint(ckpt, w);

  auto m = DeepseekV2Model::Load(ckpt.dir());
  ASSERT_TRUE(m.ok()) << m.status();

  const std::vector<int32_t> prompt{7, 40, 1, 60};

  // The reference path builds its own temporary pool -- no AttachKvCache.
  std::vector<float> forward;
  ASSERT_TRUE(m->Forward(prompt, &forward).ok());

  ASSERT_TRUE(m->AttachKvCache(4, 4).ok());
  std::vector<float> stepped;
  ASSERT_TRUE(m->Step(PrefillBatch(*m, prompt), &stepped).ok());

  // Same kernels, same shapes, fresh caches: the two must agree to bf16
  // noise, not merely to reference tolerance.
  ExpectClose(forward, stepped, 1e-3, "Forward vs Step");
}

TEST_F(DeepseekV2ModelTest, BatchedSequencesMatchTheirSoloRuns) {
  // Two sequences of different lengths in one Step must produce exactly what
  // each produces alone -- the per-sequence MLA loop's slicing is what is
  // under test, and a misaligned range would corrupt both plausibly.
  const HostWeights w = MakeWeights();
  TempCheckpoint ckpt;
  ASSERT_FALSE(ckpt.dir().empty());
  WriteCheckpoint(ckpt, w);

  auto m = DeepseekV2Model::Load(ckpt.dir());
  ASSERT_TRUE(m.ok()) << m.status();

  const std::vector<int32_t> a{5, 17, 3, 42};
  const std::vector<int32_t> b{33, 9, 61};

  auto solo = [&](const std::vector<int32_t>& prompt) {
    EXPECT_TRUE(m->AttachKvCache(4, 4).ok());
    std::vector<float> logits;
    EXPECT_TRUE(m->Step(PrefillBatch(*m, prompt), &logits).ok());
    return logits;
  };

  const std::vector<float> solo_a = solo(a);
  const std::vector<float> solo_b = solo(b);

  // Both sequences in one batch.
  ASSERT_TRUE(m->AttachKvCache(4, 4).ok());
  ForwardBatch batch;
  batch.num_seqs = 2;
  batch.max_blocks_per_seq = 1;
  for (int64_t s = 0; s < 2; ++s) {
    auto block = m->kv_pool()->AllocateBlock();
    ASSERT_TRUE(block.ok()) << block.status();
    batch.block_table.push_back(*block);
  }
  auto append = [&](const std::vector<int32_t>& prompt, int32_t seq) {
    for (size_t t = 0; t < prompt.size(); ++t) {
      batch.token_ids.push_back(prompt[t]);
      batch.positions.push_back(static_cast<int32_t>(t));
      batch.seq_of_token.push_back(seq);
      batch.slots.push_back(
          batch.block_table[static_cast<size_t>(seq)] * 4 +
          static_cast<int32_t>(t));
      batch.logits_indices.push_back(
          static_cast<int32_t>(batch.token_ids.size() - 1));
    }
  };
  append(a, 0);
  append(b, 1);

  std::vector<float> combined;
  ASSERT_TRUE(m->Step(batch, &combined).ok());

  const std::vector<float> combined_a(
      combined.begin(),
      combined.begin() + static_cast<ptrdiff_t>(a.size() * kVocab));
  const std::vector<float> combined_b(
      combined.begin() + static_cast<ptrdiff_t>(a.size() * kVocab),
      combined.end());

  ExpectClose(combined_a, solo_a, 2e-2, "sequence A batched vs solo");
  ExpectClose(combined_b, solo_b, 2e-2, "sequence B batched vs solo");
}

}  // namespace
}  // namespace inferx
