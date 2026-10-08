// Teacher-forced numerical diagnostic; the CLI-independent body of
// `inferx diagnostic replay-logits`. Never use this path for throughput timing.
#include <algorithm>
#include <bit>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <random>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "inferx/core/status_util.h"
#include "inferx/diagnostic/replay_logits.h"
#include "inferx/dist/comm.h"
#include "inferx/models/model.h"
#include "nlohmann/json.hpp"

namespace inferx::diagnostic {
namespace {

using Json = nlohmann::json;

void Replay(const ReplayLogitsParams& params) {
  if (params.chunk_size <= 0) throw std::runtime_error("positive chunk size required");
  Json fixture;
  std::ifstream(params.fixture_path) >> fixture;
  if (fixture.at("cases").empty()) throw std::runtime_error("fixture has no cases");
  const DeviceId device = DeviceId::Cuda(0);
  auto* runtime = Take(RuntimeFor(device));
  Check(runtime->Activate());
  const Stream stream = Take(runtime->CreateStream());
  struct StreamGuard {
    DeviceRuntime* runtime;
    Stream stream;
    ~StreamGuard() { (void)runtime->DestroyStream(stream); }
  } guard{runtime, stream};
  models::ModelLoadOptions options;
  options.directory = params.model_dir;
  options.device = device;
  options.max_tokens = params.chunk_size;
  options.max_seqs = 1;
  auto model = Take(models::LoadModel(options));
  const auto& info = model->info();
  const auto& requirements = model->requirements().layers;
  if (requirements.empty() || !std::holds_alternative<models::PagedKvStateSpec>(requirements[0]))
    throw std::runtime_error("replay requires a paged-attention model");
  const int num_layers = static_cast<int>(requirements.size());
  const KvLayout layout = std::get<models::PagedKvStateSpec>(requirements[0]).layout;
  const int page_size = 16;
  int max_length = 0;
  for (const auto& entry : fixture.at("cases")) {
    const auto tokens = entry.at("tokens").get<std::vector<int32_t>>();
    const int prompt_length = entry.at("prompt_length");
    if (tokens.empty() ||
        tokens.size() > static_cast<size_t>(info.max_position_embeddings) ||
        prompt_length <= 0 || static_cast<size_t>(prompt_length) > tokens.size())
      throw std::runtime_error("invalid replay prefix length");
    for (int token : tokens)
      if (token < 0 || token >= info.vocab_size) throw std::runtime_error("invalid token ID");
    max_length = std::max(max_length, static_cast<int>(tokens.size()));
  }
  const int blocks = (max_length + page_size - 1) / page_size;
  std::vector<int32_t> physical_pages(blocks);
  std::iota(physical_pages.begin(), physical_pages.end(), 0);
  if (params.page_order == "reverse") std::reverse(physical_pages.begin(), physical_pages.end());
  if (params.page_order == "shuffle") {
    std::mt19937 rng(20260922);
    std::shuffle(physical_pages.begin(), physical_pages.end(), rng);
  }
  auto pool = Take(KvBlockPool::Create(num_layers, blocks, page_size, layout, device));
  models::ModelState state;
  state.paged_kv = &pool;
  for (int i = 0; i < num_layers; ++i) state.layers.push_back(models::PagedKvState{i});
  ops::OpContext ctx(*runtime, stream);
  dist::SingleRankComm comm;
  const auto upload = [&](const std::vector<int32_t>& values) {
    auto tensor = Take(Tensor::Empty(DataType::kInt32,
        Shape({static_cast<int64_t>(values.size())}), device));
    Check(runtime->Copy(tensor.Data(), values.data(), values.size() * sizeof(int32_t),
                        CopyKind::kHostToDevice));
    return tensor;
  };
  std::ofstream file(params.output, std::ios::binary);
  file.exceptions(std::ios::badbit | std::ios::failbit);
  int case_index = 0;
  for (const auto& entry : fixture.at("cases")) {
    const auto tokens = entry.at("tokens").get<std::vector<int32_t>>();
    const int prompt_length = entry.at("prompt_length");
    std::optional<Tensor> logits;
    for (int start = 0; start < static_cast<int>(tokens.size());) {
      const int count = start < prompt_length ? std::min(params.chunk_size, prompt_length - start) : 1;
      const int end = start + count;
      std::vector<int32_t> positions(count), qo{0, count};
      std::iota(positions.begin(), positions.end(), start);
      const int used_blocks = (end + page_size - 1) / page_size;
      std::vector<int32_t> kv{0, used_blocks}, pages(used_blocks);
      for (int i = 0; i < used_blocks; ++i) pages[i] = physical_pages[i];
      const Tensor token_ids = upload(std::vector<int32_t>(tokens.begin() + start, tokens.begin() + end));
      const Tensor logit_rows = upload({count - 1});
      models::ModelInputs input{
          token_ids,
          {upload(positions), upload(std::vector<int32_t>(count, 0)),
           upload(qo), upload(kv), upload(pages),
           upload({(end - 1) % page_size + 1}), qo, kv, count, 1}};
      models::ForwardContext fctx{ctx, state, comm};
      auto hidden = Take(model->Forward(input, fctx));
      logits = Take(model->ComputeLogits(hidden, models::LogitSelection{logit_rows}, fctx)).values;
      // Keep all input allocations alive until the forward has finished.
      Check(runtime->SynchronizeStream(stream));
      start = end;
    }
    if (logits->GetDataType() != DataType::kBFloat16 || logits->Numel() != info.vocab_size)
      throw std::runtime_error("unexpected replay logit dtype or shape");
    std::vector<uint16_t> bits(info.vocab_size);
    Check(runtime->Copy(bits.data(), logits->Data(), bits.size() * sizeof(uint16_t),
                        CopyKind::kDeviceToHost));
    std::vector<float> values(bits.size());
    for (size_t i = 0; i < bits.size(); ++i)
      values[i] = std::bit_cast<float>(static_cast<uint32_t>(bits[i]) << 16);
    file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    if (!params.cache_dir.empty()) {
      const auto dir = std::filesystem::path(params.cache_dir) / std::to_string(case_index);
      if (!std::filesystem::create_directories(dir)) throw std::runtime_error("cache dump directory exists");
      for (int layer = 0; layer < num_layers; ++layer) {
        for (bool key : {true, false}) {
          Tensor cache = Take(key ? pool.KeyCache(layer) : pool.ValueCache(layer));
          std::vector<uint16_t> raw(cache.Numel());
          Check(runtime->Copy(raw.data(), cache.Data(), cache.NBytes(), CopyKind::kDeviceToHost));
          std::ofstream dump(dir / ("layer_" + std::to_string(layer) + (key ? ".k.bf16" : ".v.bf16")),
                             std::ios::binary);
          dump.exceptions(std::ios::badbit | std::ios::failbit);
          const size_t width = layout.kv_heads * layout.head_dim;
          for (size_t pos = 0; pos < tokens.size(); ++pos) {
            const size_t offset = (physical_pages[pos / page_size] * page_size + pos % page_size) * width;
            dump.write(reinterpret_cast<const char*>(raw.data() + offset), width * 2);
          }
        }
      }
    }
    ++case_index;
  }
  file.close();
}

}  // namespace

Status RunReplayLogits(const ReplayLogitsParams& params) {
  return Guarded([&] { Replay(params); });
}

}  // namespace inferx::diagnostic
