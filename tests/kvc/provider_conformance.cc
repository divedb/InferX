// Provider conformance suite for the fixed opaque-component profile.
//
// usage: provider_conformance <provider.so> [mode]
//
// Modes:
//   (none)  synchronous provider (reference host provider)
//   async   delayed-completion fixture: pending states, deadlines,
//           busy commits, poisoned transactions, failed dependencies
//   cuda    tiered CUDA provider opened with a small deterministic budget
//
// The suite drives providers only through the consumer facade.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

#include "kvc/cache.h"
#include "kvc/extensions/stats.h"
#include "test_util.h"

namespace {

using namespace kvc;
using namespace kvc_test;

enum class Mode { kDefault, kAsync, kCuda };

Mode g_mode = Mode::kDefault;
const char* g_provider_path = nullptr;

// Provider options for each mode.
KvcDescriptor OpenOptions(std::vector<unsigned char>& payload) {
  KvcDescriptor options{};
  if (g_mode == Mode::kAsync) {
    options.schema = {{"async.fixture", 13}, 1, 0};
    payload = {40, 0, 0, 0,   // delay_ms = 40
               0,  0, 0, 0};  // fail_store disabled by default
  } else if (g_mode == Mode::kCuda) {
    options.schema = {{"kvc.cuda.options", 15}, 1, 0};
    auto put64 = [&payload](uint64_t v) {
      for (int i = 0; i < 8; ++i) payload.push_back(static_cast<unsigned char>(v >> (i * 8)));
    };
    auto put32 = [&payload](uint32_t v) {
      for (int i = 0; i < 4; ++i) payload.push_back(static_cast<unsigned char>(v >> (i * 8)));
    };
    put64(48ull << 20);  // gpu_budget = 48 MiB
    put64(2ull << 20);   // host_budget = 2 MiB (tiering is covered by the
                         // dedicated CUDA test; here it must merely work)
    put32(0);            // device ordinal 0
    put32(0);            // reserved
  }
  options.canonical_payload = {payload.data(), payload.size()};
  return options;
}

// One configured session plus everything its configuration borrows.
struct Harness {
  std::unique_ptr<KVCacheProvider> provider;
  ConfigBuilder::BuiltConfig config;
  ConfigBuilder builder;

  static kvc::Result<std::unique_ptr<Harness>> Make() {
    auto harness = std::make_unique<Harness>();
    std::vector<unsigned char> payload;
    auto opened = OpenProvider({g_provider_path, OpenOptions(payload)});
    if (!opened) return std::unexpected(opened.error());
    harness->provider = std::move(*opened);
    harness->config = harness->builder.Build();
    if (auto status = harness->provider->Configure(harness->config.config); !status.Ok())
      return std::unexpected(status);
    return harness;
  }

  KVCacheControl control() const { return provider->Control(); }
  KVCacheData data() const { return provider->Data(); }

  // Waits for a transfer to succeed in every mode.
  Status WaitGood(const TransferHandle& handle) const {
    auto status = handle.Wait();
    return status;
  }
};

// Publishes blocks with deterministic patterns through one transaction.
bool Publish(Harness& harness, const std::vector<KvcBlock>& blocks, uint64_t seed_base = 0) {
  const auto& components = harness.builder.components;
  const auto& layers = harness.builder.layers;
  uint64_t row = 0;
  for (const auto& component : components) row += component.bytes_per_block;
  auto source = HostBuffer::Import(harness.data(), blocks.size() * layers.size() * row);
  CHECK(source);
  uint64_t offset = 0;
  for (const auto& block : blocks) {
    for (uint32_t layer : layers) {
      for (const auto& component : components) {
        FillPattern(source->bytes.data() + offset, block, layer, component.id,
                    component.bytes_per_block);
        offset += component.bytes_per_block;
      }
    }
  }
  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(seed_base ? seed_base : blocks.front().logical_index + 100);
  request.blocks = blocks.data();
  request.block_count = blocks.size();
  auto write = harness.control().BeginWrite(request);
  CHECK(write);
  CHECK(write->Valid());
  BindingBuilder builder;
  for (const auto& block : blocks) builder.AddBlock(block);
  for (uint32_t layer : layers) builder.AddLayer(layer);
  builder.Bind(source->region, harness.builder);
  auto transfer_request = builder.Request();
  transfer_request.selection.group = harness.builder.group_id;
  auto stored = harness.data().Store(*write, transfer_request);
  CHECK(stored);
  CHECK(harness.WaitGood(*stored).Ok());
  auto committed = harness.control().Commit(*write);
  CHECK(committed);
  CHECK_EQ(committed->blocks.size(), blocks.size());
  return true;
}

// Looks up the longest candidate covering [0, catalog_count).
LookupResult LookupAll(Harness& harness, const std::vector<KvcBlock>& catalog,
                       uint64_t token_count) {
  LookupOne one(catalog, token_count);
  auto result = harness.control().Lookup(one.request);
  CHECK(result);
  return std::move(*result);
}

// ---------------------------------------------------------------------------
// Shared tests
// ---------------------------------------------------------------------------

void TestCapabilitiesAndConfiguration() {
  auto made = Harness::Make();
  CHECK(made);
  auto& harness = **made;
  auto capabilities = harness.provider->GetCapabilities();
  CHECK(capabilities);
  CHECK((capabilities->layouts & (1u << KVC_OPAQUE_BYTES)) != 0);
  CHECK((capabilities->memory_types & (1u << KVC_HOST)) != 0);
  bool opaque = false;
  for (const auto& schema : capabilities->schemas)
    opaque = opaque || schema.name == "kvc.opaque";
  CHECK(opaque);
  auto installed = harness.provider->Configuration();
  CHECK(installed);
  CHECK_EQ((*installed)->group_count, 1u);
  CHECK_EQ((*installed)->groups[0].tokens_per_block, 16u);

  // Double configure is rejected.
  auto again = harness.builder.Build();
  CHECK_EQ(harness.provider->Configure(again.config).code, KVC_INVALID_ARGUMENT);

  // Invalid variants: rebuild with a broken group description.
  ConfigBuilder broken_builder = harness.builder;
  broken_builder.tokens_per_block = 0;
  auto broken = broken_builder.Build();
  CHECK_EQ(harness.provider->Configure(broken.config).code, KVC_INVALID_ARGUMENT);

  CHECK(harness.provider->Shutdown().Ok());
}

void TestPublicationRoundTrip() {
  auto made = Harness::Make();
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 11),
                               BlockAt(0, 1, tokens, tokens, 12),
                               BlockAt(0, 2, 7, tokens, 13)};  // Partial final block.
  CHECK(Publish(harness, blocks, 5001));

  auto result = LookupAll(harness, blocks, 2 * tokens + 7);
  CHECK(result.read.Valid());
  CHECK_EQ(result.prefix.token_count, 2 * tokens + 7);
  auto manifest = result.read.Blocks();
  CHECK(manifest);
  CHECK_EQ(manifest->size(), blocks.size());

  // Load everything into a fresh host buffer and verify every slot.
  const auto& components = harness.builder.components;
  const auto& layers = harness.builder.layers;
  uint64_t row = 0;
  for (const auto& component : components) row += component.bytes_per_block;
  auto destination = HostBuffer::Import(harness.data(), blocks.size() * layers.size() * row);
  CHECK(destination);
  BindingBuilder builder;
  for (const auto& block : blocks) builder.AddBlock(block);
  for (uint32_t layer : layers) builder.AddLayer(layer);
  builder.Bind(destination->region, harness.builder);
  auto request = builder.Request();
  request.selection.group = harness.builder.group_id;
  auto loaded = harness.data().Load(result.read, request);
  CHECK(loaded);
  CHECK(harness.WaitGood(*loaded).Ok());
  uint64_t offset = 0;
  for (const auto& block : blocks)
    for (uint32_t layer : layers)
      for (const auto& component : components) {
        CHECK_MSG(CheckPattern(destination->bytes.data() + offset, block, layer, component.id,
                               component.bytes_per_block),
                  "loaded payload mismatch");
        offset += component.bytes_per_block;
      }
}

void TestLayerwiseStoresAndIncomplete() {
  auto made = Harness::Make();
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 21)};
  const auto& components = harness.builder.components;
  uint64_t row = 0;
  for (const auto& component : components) row += component.bytes_per_block;

  auto source = HostBuffer::Import(harness.data(), blocks.size() * 2 * row);
  CHECK(source);
  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(6001);
  request.blocks = blocks.data();
  request.block_count = blocks.size();
  auto write = harness.control().BeginWrite(request);
  CHECK(write);

  // Commit before any store: incomplete.
  CHECK_EQ(harness.control().Commit(*write).error().code, KVC_INCOMPLETE);
  // Query reports the allocated state.
  auto query = harness.control().QueryWrite(TxIdOf(6001));
  CHECK(query);
  CHECK_EQ(query->state, KVC_ALLOCATED);

  // Store one layer at a time.
  uint64_t offset = 0;
  for (uint32_t layer : harness.builder.layers) {
    for (const auto& block : blocks)
      for (const auto& component : components) {
        FillPattern(source->bytes.data() + offset, block, layer, component.id,
                    component.bytes_per_block);
        offset += component.bytes_per_block;
      }
  }
  offset = 0;
  for (uint32_t layer : harness.builder.layers) {
    BindingBuilder builder;
    for (const auto& block : blocks) builder.AddBlock(block);
    builder.AddLayer(layer);
    builder.Bind(source->region, harness.builder, offset);
    auto store_request = builder.Request();
    store_request.selection.group = harness.builder.group_id;
    auto stored = harness.data().Store(*write, store_request);
    CHECK(stored);
    CHECK(harness.WaitGood(*stored).Ok());
    offset += row;
    if (layer == harness.builder.layers.front()) {
      auto writing = harness.control().QueryWrite(TxIdOf(6001));
      CHECK(writing);
      CHECK_EQ(writing->state, KVC_WRITING);
      // Not all layers stored yet.
      CHECK_EQ(harness.control().Commit(*write).error().code, KVC_INCOMPLETE);
    }
  }
  auto committed = harness.control().Commit(*write);
  CHECK(committed);
  // Repeated commit returns the same receipt.
  auto again = harness.control().Commit(*write);
  CHECK(again);
  CHECK_EQ(again->blocks.size(), committed->blocks.size());
  CHECK(DigestEq(again->blocks[0].key, committed->blocks[0].key));
}

void TestCommitAndTransactionSemantics() {
  auto made = Harness::Make();
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;

  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 31)};
  CHECK(Publish(harness, blocks, 7001));
  auto status = harness.control().QueryWrite(TxIdOf(7001));
  CHECK(status);
  CHECK_EQ(status->state, KVC_COMMITTED);
  CHECK_EQ(status->blocks.size(), 1u);

  // Unknown transaction.
  CHECK_EQ(harness.control().QueryWrite(TxIdOf(9999)).error().code, KVC_NOT_FOUND);
  // Aborting a committed transaction is refused.
  CHECK_EQ(harness.control().Abort(TxIdOf(7001)).code, KVC_ALREADY_COMMITTED);

  // Reusing a transaction id conflicts.
  std::vector<KvcBlock> other{BlockAt(0, 5, tokens, tokens, 32)};
  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(7001);
  request.blocks = other.data();
  request.block_count = other.size();
  CHECK_EQ(harness.control().BeginWrite(request).error().code, KVC_CONFLICT);

  // Reserving an already-committed key is refused.
  request.transaction = TxIdOf(7002);
  request.blocks = blocks.data();
  CHECK_EQ(harness.control().BeginWrite(request).error().code, KVC_ALREADY_EXISTS);

  // A live reservation also blocks the key.
  std::vector<KvcBlock> reserved{BlockAt(0, 6, tokens, tokens, 33)};
  request.transaction = TxIdOf(7003);
  request.blocks = reserved.data();
  auto writer = harness.control().BeginWrite(request);
  CHECK(writer);
  request.transaction = TxIdOf(7004);
  CHECK_EQ(harness.control().BeginWrite(request).error().code, KVC_ALREADY_EXISTS);
  // Storing into a foreign session's write handle is stale.
  auto second = Harness::Make();
  CHECK(second);
  KvcTransferRequest foreign{};
  foreign.struct_size = sizeof(foreign);
  CHECK_EQ((*second)->data().Store(*writer, foreign).error().code, KVC_STALE_HANDLE);
  writer = std::unexpected(Status{});
  auto aborted = harness.control().QueryWrite(TxIdOf(7003));
  CHECK(aborted);
  CHECK_EQ(aborted->state, KVC_WRITE_ABORTED);
}

void TestLookupSemantics() {
  auto made = Harness::Make();
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 41),
                               BlockAt(0, 1, tokens, tokens, 42),
                               BlockAt(0, 2, tokens, tokens, 43)};
  CHECK(Publish(harness, blocks, 8001));

  // Candidates of increasing length; the longest available wins.
  std::vector<KvcPrefix> prefixes{PrefixOf(1, tokens), PrefixOf(2, 2 * tokens),
                                  PrefixOf(3, 3 * tokens)};
  std::vector<KvcRange> ranges{{0, 1}, {0, 2}, {0, 3}};
  std::vector<KvcCandidate> candidates;
  for (size_t i = 0; i < 3; ++i) candidates.push_back(KvcCandidate{prefixes[i], &ranges[i], 1});
  KvcLookupRequest request{};
  request.struct_size = sizeof(request);
  request.catalog = blocks.data();
  request.catalog_count = blocks.size();
  request.candidates = candidates.data();
  request.candidate_count = candidates.size();
  auto result = harness.control().Lookup(request);
  CHECK(result);
  CHECK(result->read.Valid());
  CHECK_EQ(result->prefix.token_count, 3 * tokens);

  // Removing a middle block makes only the shortest candidate satisfiable:
  // availability is not monotonic.
  CHECK(harness.control().Remove(blocks[1].key).Ok());
  auto shorter = harness.control().Lookup(request);
  CHECK(shorter);
  CHECK(shorter->read.Valid());
  CHECK_EQ(shorter->prefix.token_count, tokens);

  // Republishing the middle block restores the long candidate.
  std::vector<KvcBlock> again{BlockAt(0, 1, tokens, tokens, 42)};
  CHECK(Publish(harness, again, 8002));
  auto restored = harness.control().Lookup(request);
  CHECK(restored);
  CHECK(restored->read.Valid());
  CHECK_EQ(restored->prefix.token_count, 3 * tokens);

  // Structural validation: duplicate catalog keys.
  std::vector<KvcBlock> duplicate{blocks[0], blocks[0]};
  KvcLookupRequest dup_request{};
  dup_request.struct_size = sizeof(dup_request);
  dup_request.catalog = duplicate.data();
  dup_request.catalog_count = duplicate.size();
  CHECK_EQ(harness.control().Lookup(dup_request).error().code, KVC_INVALID_ARGUMENT);

  std::vector<KvcRange> overflow{KvcRange{2, 5}};
  std::vector<KvcPrefix> prefix{PrefixOf(9, tokens)};
  std::vector<KvcCandidate> bad_range{KvcCandidate{prefix[0], overflow.data(), 1}};
  KvcLookupRequest range_request{};
  range_request.struct_size = sizeof(range_request);
  range_request.catalog = blocks.data();
  range_request.catalog_count = blocks.size();
  range_request.candidates = bad_range.data();
  range_request.candidate_count = 1;
  CHECK_EQ(harness.control().Lookup(range_request).error().code, KVC_INVALID_RANGE);

  // Empty candidate list is a normal miss.
  KvcLookupRequest miss_request{};
  miss_request.struct_size = sizeof(miss_request);
  miss_request.catalog = blocks.data();
  miss_request.catalog_count = blocks.size();
  auto miss = harness.control().Lookup(miss_request);
  CHECK(miss);
  CHECK(!miss->read.Valid());

  // Metadata disagreement against stored state.
  std::vector<KvcBlock> mismatched{blocks[0]};
  mismatched[0].token_count = tokens - 1;
  std::vector<KvcRange> all{{0, 1}};
  std::vector<KvcPrefix> hit_prefix{PrefixOf(4, tokens)};
  std::vector<KvcCandidate> hit{KvcCandidate{hit_prefix[0], all.data(), 1}};
  KvcLookupRequest mismatch_request{};
  mismatch_request.struct_size = sizeof(mismatch_request);
  mismatch_request.catalog = mismatched.data();
  mismatch_request.catalog_count = 1;
  mismatch_request.candidates = hit.data();
  mismatch_request.candidate_count = 1;
  CHECK_EQ(harness.control().Lookup(mismatch_request).error().code, KVC_INCOMPATIBLE_MODEL);
}

void TestRemoveAndPinnedGenerations() {
  auto made = Harness::Make();
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 51)};
  CHECK(Publish(harness, blocks, 9001));

  // Pin the generation, then retire it.
  auto pinned = LookupAll(harness, blocks, tokens);
  CHECK(pinned.read.Valid());
  CHECK(harness.control().Remove(blocks[0].key).Ok());
  // Idempotent.
  CHECK(harness.control().Remove(blocks[0].key).Ok());

  // New lookups miss.
  auto miss = LookupAll(harness, blocks, tokens);
  CHECK(!miss.read.Valid());

  // The pinned generation still loads.
  const auto& components = harness.builder.components;
  const auto& layers = harness.builder.layers;
  uint64_t row = 0;
  for (const auto& component : components) row += component.bytes_per_block;
  auto destination = HostBuffer::Import(harness.data(), row * layers.size());
  CHECK(destination);
  BindingBuilder builder;
  builder.AddBlock(blocks[0]);
  for (uint32_t layer : layers) builder.AddLayer(layer);
  builder.Bind(destination->region, harness.builder);
  auto request = builder.Request();
  request.selection.group = harness.builder.group_id;
  auto loaded = harness.data().Load(pinned.read, request);
  CHECK(loaded);
  CHECK(harness.WaitGood(*loaded).Ok());
  uint64_t offset = 0;
  for (uint32_t layer : layers)
    for (const auto& component : components) {
      CHECK(CheckPattern(destination->bytes.data() + offset, blocks[0], layer, component.id,
                         component.bytes_per_block));
      offset += component.bytes_per_block;
    }

  // A new generation may occupy the retired key.
  std::vector<KvcBlock> fresh{BlockAt(0, 0, tokens, tokens, 52)};
  CHECK(Publish(harness, fresh, 9002));
  auto second = LookupAll(harness, fresh, tokens);
  CHECK(second.read.Valid());
  auto destination2 = HostBuffer::Import(harness.data(), row * layers.size());
  CHECK(destination2);
  BindingBuilder builder2;
  builder2.AddBlock(fresh[0]);
  for (uint32_t layer : layers) builder2.AddLayer(layer);
  builder2.Bind(destination2->region, harness.builder);
  auto request2 = builder2.Request();
  request2.selection.group = harness.builder.group_id;
  auto loaded2 = harness.data().Load(second.read, request2);
  CHECK(loaded2);
  CHECK(harness.WaitGood(*loaded2).Ok());
  offset = 0;
  for (uint32_t layer : layers)
    for (const auto& component : components) {
      CHECK(CheckPattern(destination2->bytes.data() + offset, fresh[0], layer, component.id,
                         component.bytes_per_block));
      offset += component.bytes_per_block;
    }
}

void TestAbortSemantics() {
  auto made = Harness::Make();
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 61),
                               BlockAt(0, 1, tokens, tokens, 62)};
  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(10001);
  request.blocks = blocks.data();
  request.block_count = blocks.size();
  auto write = harness.control().BeginWrite(request);
  CHECK(write);

  // Store one block, then abort.
  const auto& components = harness.builder.components;
  uint64_t row = 0;
  for (const auto& component : components) row += component.bytes_per_block;
  auto source = HostBuffer::Import(harness.data(), row * harness.builder.layers.size());
  CHECK(source);
  // Full layer-0 coverage of block 0 only.
  uint64_t offset = 0;
  for (const auto& component : components) {
    FillPattern(source->bytes.data() + offset, blocks[0], harness.builder.layers[0],
                component.id, component.bytes_per_block);
    offset += component.bytes_per_block;
  }
  BindingBuilder builder;
  builder.AddBlock(blocks[0]);
  builder.AddLayer(harness.builder.layers[0]);
  builder.Bind(source->region, harness.builder);
  auto store_request = builder.Request();
  store_request.selection.group = harness.builder.group_id;
  auto stored = harness.data().Store(*write, store_request);
  CHECK(stored);
  CHECK(harness.WaitGood(*stored).Ok());

  CHECK(harness.control().Abort(TxIdOf(10001)).Ok());
  // Idempotent abort.
  CHECK(harness.control().Abort(TxIdOf(10001)).Ok());
  CHECK_EQ(harness.control().Commit(*write).error().code, KVC_ABORTED);
  CHECK_EQ(harness.data().Store(*write, store_request).error().code, KVC_ABORTED);
  auto query = harness.control().QueryWrite(TxIdOf(10001));
  CHECK(query);
  CHECK_EQ(query->state, KVC_WRITE_ABORTED);
  auto miss = LookupAll(harness, blocks, 2 * tokens);
  CHECK(!miss.read.Valid());

  // Handle destruction abandons an unfinished transaction.
  std::vector<KvcBlock> more{BlockAt(0, 3, tokens, tokens, 63)};
  KvcWriteRequest second{};
  second.struct_size = sizeof(second);
  second.transaction = TxIdOf(10002);
  second.blocks = more.data();
  second.block_count = more.size();
  {
    auto abandoned = harness.control().BeginWrite(second);
    CHECK(abandoned);
  }  // Destruction requests abandonment.
  auto state = harness.control().QueryWrite(TxIdOf(10002));
  CHECK(state);
  CHECK_EQ(state->state, KVC_WRITE_ABORTED);
}

void TestBindingValidation() {
  auto made = Harness::Make();
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 71)};
  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(11001);
  request.blocks = blocks.data();
  request.block_count = blocks.size();
  auto write = harness.control().BeginWrite(request);
  CHECK(write);

  const auto& layers = harness.builder.layers;
  const auto& components = harness.builder.components;
  uint64_t row = 0;
  for (const auto& component : components) row += component.bytes_per_block;
  auto source = HostBuffer::Import(harness.data(), 4 * row * layers.size());
  CHECK(source);

  BindingBuilder builder;
  builder.AddBlock(blocks[0]);
  for (uint32_t layer : layers) builder.AddLayer(layer);
  builder.Bind(source->region, harness.builder);
  auto full = builder.Request();
  full.selection.group = harness.builder.group_id;

  // Incomplete coverage: drop one binding.
  std::vector<KvcBinding> partial(builder.bindings.begin(), builder.bindings.end() - 1);
  auto dropped = full;
  dropped.bindings = partial.data();
  dropped.binding_count = partial.size();
  CHECK_EQ(harness.data().Store(*write, dropped).error().code, KVC_INVALID_ARGUMENT);

  // Overlapping write inside one request: same binding count, one slot
  // bound twice (leaving another slot uncovered by content, not count).
  std::vector<KvcBinding> doubled(builder.bindings);
  doubled[1] = doubled[0];
  auto overlap = full;
  overlap.bindings = doubled.data();
  overlap.binding_count = doubled.size();
  CHECK_EQ(harness.data().Store(*write, overlap).error().code, KVC_CONFLICT);

  // Tensor layout is outside this profile.
  std::vector<KvcBinding> tensored(builder.bindings);
  tensored[0].layout = KVC_STRIDED_TENSOR;
  auto as_tensor = full;
  as_tensor.bindings = tensored.data();
  CHECK_EQ(harness.data().Store(*write, as_tensor).error().code, KVC_UNSUPPORTED);

  // Wrong component size.
  std::vector<KvcBinding> sized(builder.bindings);
  sized[0].byte_count = sized[0].byte_count - 1;
  auto wrong_size = full;
  wrong_size.bindings = sized.data();
  CHECK_EQ(harness.data().Store(*write, wrong_size).error().code, KVC_INVALID_RANGE);

  // Out-of-region binding.
  std::vector<KvcBinding> beyond(builder.bindings);
  beyond[0].byte_offset = source->bytes.size() - beyond[0].byte_count + 1;
  auto out_of_range = full;
  out_of_range.bindings = beyond.data();
  CHECK_EQ(harness.data().Store(*write, out_of_range).error().code, KVC_INVALID_RANGE);

  // Layer outside the group.
  auto bad_layer = full;
  std::vector<uint32_t> layers_copy(layers);
  layers_copy[0] = 99;
  bad_layer.selection.layers = layers_copy.data();
  CHECK_EQ(harness.data().Store(*write, bad_layer).error().code, KVC_INVALID_RANGE);

  // Block outside the manifest.
  auto bad_block = full;
  std::vector<KvcSlice> slices(builder.slices);
  slices[0].key = DigestOf(123456);
  bad_block.selection.blocks = slices.data();
  CHECK_EQ(harness.data().Store(*write, bad_block).error().code, KVC_INVALID_RANGE);

  // Partial slice of an opaque block.
  auto partial_slice = full;
  std::vector<KvcSlice> half(builder.slices);
  half[0].first_token = 1;
  half[0].token_count = tokens - 1;
  partial_slice.selection.blocks = half.data();
  CHECK_EQ(harness.data().Store(*write, partial_slice).error().code, KVC_UNSUPPORTED);

  // Foreign region.
  auto other = Harness::Make();
  CHECK(other);
  auto foreign = HostBuffer::Import((*other)->data(), row);
  CHECK(foreign);
  std::vector<KvcBinding> foreign_bindings(builder.bindings);
  foreign_bindings[0].region = foreign->region.NativeHandle();
  auto foreign_request = full;
  foreign_request.bindings = foreign_bindings.data();
  CHECK_EQ(harness.data().Store(*write, foreign_request).error().code, KVC_STALE_HANDLE);

  // The transaction survived every rejected submission: a full store and
  // commit still succeed.
  auto stored = harness.data().Store(*write, full);
  CHECK(stored);
  CHECK(harness.WaitGood(*stored).Ok());
  CHECK(harness.control().Commit(*write));

  // A read-only region cannot be a load destination.
  auto readonly = HostBuffer::Import(harness.data(), row * layers.size(), KVC_READ_ONLY);
  CHECK(readonly);
  auto hit = LookupAll(harness, blocks, tokens);
  CHECK(hit.read.Valid());
  BindingBuilder load_builder;
  load_builder.AddBlock(blocks[0]);
  for (uint32_t layer : layers) load_builder.AddLayer(layer);
  load_builder.Bind(readonly->region, harness.builder);
  auto load_request = load_builder.Request();
  load_request.selection.group = harness.builder.group_id;
  CHECK_EQ(harness.data().Load(hit.read, load_request).error().code, KVC_INVALID_ARGUMENT);
}

void TestImportValidation() {
  auto made = Harness::Make();
  CHECK(made);
  auto& harness = **made;
  std::vector<unsigned char> bytes(128);
  std::string runtime = "cpu";
  std::string identifier = "0";

  KvcMemoryInfo info{};
  info.struct_size = sizeof(info);
  info.memory_type = KVC_HOST;
  info.access = KVC_READ_WRITE;
  info.locator_kind = KVC_LOCAL_ADDRESS;
  info.local_address = reinterpret_cast<uintptr_t>(bytes.data());
  info.byte_size = 0;
  info.device_runtime = {runtime.data(), runtime.size()};
  info.device_identifier = {identifier.data(), identifier.size()};
  auto zero = harness.data().ImportMemory(info, std::make_shared<int>(1));
  CHECK_EQ(zero.error().code, KVC_INVALID_ARGUMENT);

  info.byte_size = bytes.size();
  info.locator_kind = KVC_EXTERNAL_MEMORY;
  auto external = harness.data().ImportMemory(info, std::make_shared<int>(1));
  CHECK_EQ(external.error().code, KVC_UNSUPPORTED);

  info.locator_kind = KVC_LOCAL_ADDRESS;
  std::string wrong_runtime = "cuda";
  info.device_runtime = {wrong_runtime.data(), wrong_runtime.size()};
  auto wrong = harness.data().ImportMemory(info, std::make_shared<int>(1));
  CHECK_EQ(wrong.error().code, KVC_UNSUPPORTED);

  // Device imports depend on the provider (host rejects, cuda accepts), so
  // only the metadata round-trip is checked for accepted imports.
  std::string cpu = "cpu";
  info.device_runtime = {cpu.data(), cpu.size()};
  auto good = harness.data().ImportMemory(info, std::make_shared<int>(1));
  CHECK(good);
  auto described = good->Info();
  CHECK(described);
  CHECK_EQ(described->byte_size, bytes.size());
  CHECK_EQ(described->memory_type, KVC_HOST);
}

void TestBudgetExhaustion() {
  // Blocks of 32 MiB exhaust every provider's budget: the host and fixture
  // providers cap at 64 MiB, and the CUDA mode opens with 48 MiB.
  ConfigBuilder builder;
  builder.layers = {0};
  builder.components = {{0, 32ull << 20}};
  std::vector<unsigned char> payload;
  auto opened = OpenProvider({g_provider_path, OpenOptions(payload)});
  CHECK(opened);
  KVCacheProvider& provider = **opened;
  auto config = builder.Build();
  CHECK(provider.Configure(config.config).Ok());

  const uint64_t tokens = builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 81),
                               BlockAt(0, 1, tokens, tokens, 82),
                               BlockAt(0, 2, tokens, tokens, 83)};
  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(12001);
  request.blocks = blocks.data();
  request.block_count = blocks.size();
  auto control = provider.Control();
  CHECK_EQ(control.BeginWrite(request).error().code, KVC_RESOURCE_EXHAUSTED);
  // A smaller manifest fits; releasing the handle lets shutdown drain.
  {
    request.block_count = 1;
    auto write = control.BeginWrite(request);
    CHECK(write);
  }
  CHECK(provider.Shutdown().Ok());
}

void TestShutdownLifecycle() {
  auto made = Harness::Make();
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 91)};
  CHECK(Publish(harness, blocks, 13001));
  {
    auto pinned = LookupAll(harness, blocks, tokens);
    CHECK(pinned.read.Valid());
    // Live owners keep the session busy.
    CHECK_EQ(harness.provider->Shutdown().code, KVC_BUSY);
    // Closing rejects new lookups with an explicit status.
    LookupOne one(blocks, tokens);
    auto rejected = harness.control().Lookup(one.request);
    CHECK_EQ(rejected.error().code, KVC_SHUTTING_DOWN);
  }
  CHECK(harness.provider->Shutdown().Ok());
  // After shutdown, new work is refused.
  std::vector<KvcBlock> fresh{BlockAt(0, 9, tokens, tokens, 92)};
  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(13002);
  request.blocks = fresh.data();
  request.block_count = fresh.size();
  CHECK_EQ(harness.control().BeginWrite(request).error().code, KVC_SHUTTING_DOWN);
  CHECK_EQ(harness.control().Remove(blocks[0].key).code, KVC_SHUTTING_DOWN);
}

// Helper: poll result is structurally sane for every provider.
bool polyled_check(const Result<TransferStatus>& polled) {
  if (!polled) return false;
  const uint32_t state = polled->state;
  return state == KVC_PENDING || state == KVC_SUCCEEDED || state == KVC_FAILED ||
         state == KVC_TRANSFER_CANCELLED;
}

void TestTransferPollWait() {
  auto made = Harness::Make();
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 101)};
  CHECK(Publish(harness, blocks, 14001));
  auto hit = LookupAll(harness, blocks, tokens);
  CHECK(hit.read.Valid());

  const auto& layers = harness.builder.layers;
  const auto& components = harness.builder.components;
  uint64_t row = 0;
  for (const auto& component : components) row += component.bytes_per_block;
  auto destination = HostBuffer::Import(harness.data(), row * layers.size());
  CHECK(destination);
  BindingBuilder builder;
  builder.AddBlock(blocks[0]);
  for (uint32_t layer : layers) builder.AddLayer(layer);
  builder.Bind(destination->region, harness.builder);
  auto request = builder.Request();
  request.selection.group = harness.builder.group_id;
  auto loaded = harness.data().Load(hit.read, request);
  CHECK(loaded);
  auto polled = loaded->Poll();
  CHECK(polyled_check(polled));
  CHECK_EQ(harness.WaitGood(*loaded).code, KVC_OK);
  uint64_t offset = 0;
  for (uint32_t layer : layers)
    for (const auto& component : components) {
      CHECK(CheckPattern(destination->bytes.data() + offset, blocks[0], layer, component.id,
                         component.bytes_per_block));
      offset += component.bytes_per_block;
    }
  // Cancel is optional: providers without the feature report Unsupported.
  auto cancelled = loaded->Cancel();
  CHECK(cancelled.code == KVC_UNSUPPORTED || cancelled.code == KVC_OK);
}

void TestStatsExtension() {
  auto made = Harness::Make();
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 161)};
  CHECK(Publish(harness, blocks, 16001));
  auto hit = LookupAll(harness, blocks, tokens);
  CHECK(hit.read.Valid());
  const auto& layers = harness.builder.layers;
  const auto& components = harness.builder.components;
  uint64_t row = 0;
  for (const auto& component : components) row += component.bytes_per_block;
  auto destination = HostBuffer::Import(harness.data(), row * layers.size());
  CHECK(destination);
  BindingBuilder builder;
  builder.AddBlock(blocks[0]);
  for (uint32_t layer : layers) builder.AddLayer(layer);
  builder.Bind(destination->region, harness.builder);
  auto request = builder.Request();
  request.selection.group = harness.builder.group_id;
  auto loaded = harness.data().Load(hit.read, request);
  CHECK(loaded);
  CHECK(harness.WaitGood(*loaded).Ok());

  KvcSchema schema{};
  schema.name = {"kvc.stats", 9};
  schema.major = 1;
  auto extension = harness.provider->QueryExtension(schema);
  if (!extension) return;  // The extension is optional per provider.
  auto* api = static_cast<const KvcStatsApi*>(extension->Table());
  CHECK(api && api->get);
  KvcStatsRecord record{};
  record.struct_size = sizeof(record);
  auto status = api->get(extension->NativeHandle(), &record);
  CHECK_EQ(status.code, KVC_OK);
  CHECK(record.lookups >= 1);
  CHECK(record.lookup_hits >= 1);
  CHECK(record.lookup_hit_blocks >= 1);
  CHECK(record.blocks_committed >= 1);
  CHECK(record.transactions_committed >= 1);
  CHECK(record.stores >= 1);
  CHECK(record.store_bytes > 0);
  CHECK(record.loads >= 1);
  CHECK(record.load_bytes > 0);
  CHECK(record.gpu_bytes_in_use + record.host_bytes_in_use > 0);
}

void TestConcurrentReaders() {
  auto made = Harness::Make();
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 111),
                               BlockAt(0, 1, tokens, tokens, 112),
                               BlockAt(0, 2, tokens, tokens, 113)};
  CHECK(Publish(harness, blocks, 15001));

  const auto& layers = harness.builder.layers;
  const auto& components = harness.builder.components;
  uint64_t row = 0;
  for (const auto& component : components) row += component.bytes_per_block;

  std::atomic<int> failures{0};
  auto reader = [&harness, &blocks, &layers, &components, row, &failures] {
    for (int i = 0; i < 8; ++i) {
      auto result = LookupAll(harness, blocks, 3 * blocks.front().token_count);
      if (!result.read.Valid()) {
        ++failures;
        return;
      }
      auto destination =
          HostBuffer::Import(harness.data(), row * layers.size() * blocks.size());
      if (!destination) {
        ++failures;
        return;
      }
      BindingBuilder builder;
      for (const auto& block : blocks) builder.AddBlock(block);
      for (uint32_t layer : layers) builder.AddLayer(layer);
      builder.Bind(destination->region, harness.builder);
      auto request = builder.Request();
      request.selection.group = harness.builder.group_id;
      auto loaded = harness.data().Load(result.read, request);
      if (!loaded || !harness.WaitGood(*loaded).Ok()) {
        ++failures;
        return;
      }
      uint64_t offset = 0;
      for (const auto& block : blocks)
        for (uint32_t layer : layers)
          for (const auto& component : components) {
            if (!CheckPattern(destination->bytes.data() + offset, block, layer, component.id,
                              component.bytes_per_block)) {
              ++failures;
              return;
            }
            offset += component.bytes_per_block;
          }
    }
  };
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) threads.emplace_back(reader);
  for (auto& thread : threads) thread.join();
  CHECK_EQ(failures.load(), 0);
}

// ---------------------------------------------------------------------------
// Async-mode tests (delayed-completion fixture with failure injection)
// ---------------------------------------------------------------------------

namespace async {

// Opens a fixture with a chosen failure index.
std::unique_ptr<Harness> MakeFailing(uint32_t fail_store) {
  auto harness = std::make_unique<Harness>();
  std::vector<unsigned char> payload{40, 0, 0, 0, 0, 0, 0, 0};
  payload[4] = static_cast<unsigned char>(fail_store);
  payload[5] = static_cast<unsigned char>(fail_store >> 8);
  payload[6] = static_cast<unsigned char>(fail_store >> 16);
  payload[7] = static_cast<unsigned char>(fail_store >> 24);
  KvcDescriptor options{};
  options.schema = {{"async.fixture", 13}, 1, 0};
  options.canonical_payload = {payload.data(), payload.size()};
  auto opened = OpenProvider({g_provider_path, options});
  CHECK(opened);
  harness->provider = std::move(*opened);
  harness->config = harness->builder.Build();
  CHECK(harness->provider->Configure(harness->config.config).Ok());
  return harness;
}

void TestPendingDeadlineAndCompletion() {
  auto made = MakeFailing(0);
  CHECK(made);
  auto& harness = *made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 121)};
  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(16001);
  request.blocks = blocks.data();
  request.block_count = 1;
  auto write = harness.control().BeginWrite(request);
  CHECK(write);

  const auto& layers = harness.builder.layers;
  const auto& components = harness.builder.components;
  uint64_t row = 0;
  for (const auto& component : components) row += component.bytes_per_block;
  auto source = HostBuffer::Import(harness.data(), row * layers.size());
  CHECK(source);
  uint64_t offset = 0;
  for (uint32_t layer : layers)
    for (const auto& component : components) {
      FillPattern(source->bytes.data() + offset, blocks[0], layer, component.id,
                  component.bytes_per_block);
      offset += component.bytes_per_block;
    }
  BindingBuilder builder;
  builder.AddBlock(blocks[0]);
  for (uint32_t layer : layers) builder.AddLayer(layer);
  builder.Bind(source->region, harness.builder);
  auto store_request = builder.Request();
  store_request.selection.group = harness.builder.group_id;
  auto stored = harness.data().Store(*write, store_request);
  CHECK(stored);

  // Freshly accepted work is pending (40 ms delay).
  auto polled = stored->Poll();
  CHECK(polled);
  CHECK_EQ(polled->state, KVC_PENDING);
  // A short wait times out without completing the transfer.
  CHECK_EQ(stored->Wait(1000000u /* 1 ms */).code, KVC_DEADLINE_EXCEEDED);
  auto still = stored->Poll();
  CHECK(still);
  CHECK_EQ(still->state, KVC_PENDING);
  // Commit observes the pending store.
  CHECK_EQ(harness.control().Commit(*write).error().code, KVC_BUSY);
  // Unbounded wait completes.
  CHECK(stored->Wait().Ok());
  auto finished = stored->Poll();
  CHECK(finished);
  CHECK_EQ(finished->state, KVC_SUCCEEDED);
  CHECK(harness.control().Commit(*write));
}

void TestDroppedHandleStillCompletes() {
  auto made = MakeFailing(0);
  CHECK(made);
  auto& harness = *made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 131)};
  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(17001);
  request.blocks = blocks.data();
  request.block_count = 1;
  auto write = harness.control().BeginWrite(request);
  CHECK(write);
  const auto& layers = harness.builder.layers;
  const auto& components = harness.builder.components;
  uint64_t row = 0;
  for (const auto& component : components) row += component.bytes_per_block;
  auto source = HostBuffer::Import(harness.data(), row * layers.size());
  CHECK(source);
  uint64_t offset = 0;
  for (uint32_t layer : layers)
    for (const auto& component : components) {
      FillPattern(source->bytes.data() + offset, blocks[0], layer, component.id,
                  component.bytes_per_block);
      offset += component.bytes_per_block;
    }
  BindingBuilder builder;
  builder.AddBlock(blocks[0]);
  for (uint32_t layer : layers) builder.AddLayer(layer);
  builder.Bind(source->region, harness.builder);
  auto store_request = builder.Request();
  store_request.selection.group = harness.builder.group_id;
  {
    auto dropped = harness.data().Store(*write, store_request);
    CHECK(dropped);
    CHECK(dropped->Poll());
  }  // Handle dropped: the work still runs to completion.
  for (int i = 0; i < 4000; ++i) {
    if (harness.control().Commit(*write)) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  CHECK_MSG(false, "commit never succeeded after dropping the store handle");
}

void TestFailedStorePoisonsTransaction() {
  auto made = MakeFailing(1);  // The first accepted store fails in flight.
  CHECK(made);
  auto& harness = *made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 141)};
  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(18001);
  request.blocks = blocks.data();
  request.block_count = 1;
  auto write = harness.control().BeginWrite(request);
  CHECK(write);
  const auto& layers = harness.builder.layers;
  const auto& components = harness.builder.components;
  uint64_t row = 0;
  for (const auto& component : components) row += component.bytes_per_block;
  auto source = HostBuffer::Import(harness.data(), row * layers.size());
  CHECK(source);
  BindingBuilder builder;
  builder.AddBlock(blocks[0]);
  for (uint32_t layer : layers) builder.AddLayer(layer);
  builder.Bind(source->region, harness.builder);
  auto store_request = builder.Request();
  store_request.selection.group = harness.builder.group_id;
  auto stored = harness.data().Store(*write, store_request);
  CHECK(stored);
  auto failed = stored->Wait();
  CHECK_EQ(failed.code, KVC_TRANSPORT_ERROR);
  auto polled = stored->Poll();
  CHECK(polled);
  CHECK_EQ(polled->state, KVC_FAILED);
  // The poisoned transaction refuses further work.
  CHECK_EQ(harness.control().Commit(*write).error().code, KVC_ABORTED);
  CHECK_EQ(harness.data().Store(*write, store_request).error().code, KVC_ABORTED);
  CHECK(harness.control().Abort(TxIdOf(18001)).Ok());
  auto query = harness.control().QueryWrite(TxIdOf(18001));
  CHECK(query);
  CHECK_EQ(query->state, KVC_WRITE_ABORTED);
}

void TestFailedDependency() {
  auto made = MakeFailing(1);
  CHECK(made);
  auto& harness = *made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 151)};
  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(19001);
  request.blocks = blocks.data();
  request.block_count = 1;
  auto write = harness.control().BeginWrite(request);
  CHECK(write);
  const auto& layers = harness.builder.layers;
  const auto& components = harness.builder.components;
  uint64_t row = 0;
  for (const auto& component : components) row += component.bytes_per_block;
  auto source = HostBuffer::Import(harness.data(), row * layers.size());
  CHECK(source);
  auto source2 = HostBuffer::Import(harness.data(), row * layers.size());
  CHECK(source2);
  BindingBuilder builder;
  builder.AddBlock(blocks[0]);
  for (uint32_t layer : layers) builder.AddLayer(layer);
  builder.Bind(source->region, harness.builder);
  auto store_request = builder.Request();
  store_request.selection.group = harness.builder.group_id;
  auto doomed = harness.data().Store(*write, store_request);
  CHECK(doomed);

  // A second store depending on the doomed one is accepted while the
  // dependency is pending and must fail with it.
  BindingBuilder second;
  second.AddBlock(blocks[0]);
  for (uint32_t layer : layers) second.AddLayer(layer);
  // Different source region so the slots differ... same slots would be an
  // overlap; use a different block for the dependent store.
  std::vector<KvcBlock> other{BlockAt(0, 1, tokens, tokens, 152)};
  KvcWriteRequest second_tx{};
  second_tx.struct_size = sizeof(second_tx);
  second_tx.transaction = TxIdOf(19002);
  second_tx.blocks = other.data();
  second_tx.block_count = 1;
  auto second_write = harness.control().BeginWrite(second_tx);
  CHECK(second_write);
  BindingBuilder dependent_builder;
  dependent_builder.AddBlock(other[0]);
  for (uint32_t layer : layers) dependent_builder.AddLayer(layer);
  dependent_builder.Bind(source2->region, harness.builder);
  KvcObject* deps[1] = {doomed->NativeHandle()};
  auto dependent_request = dependent_builder.Request(deps, 1);
  dependent_request.selection.group = harness.builder.group_id;
  auto dependent = harness.data().Store(*second_write, dependent_request);
  CHECK(dependent);
  CHECK_EQ(dependent->Wait().code, KVC_ABORTED);

  // Once terminal, a failed dependency rejects new submissions outright.
  CHECK_EQ(doomed->Wait().code, KVC_TRANSPORT_ERROR);
  CHECK_EQ(harness.data().Store(*second_write, dependent_request).error().code, KVC_ABORTED);
}

void TestShutdownBusyWhilePending() {
  auto made = MakeFailing(0);
  CHECK(made);
  auto& harness = *made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 161)};
  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(20001);
  request.blocks = blocks.data();
  request.block_count = 1;
  auto write = harness.control().BeginWrite(request);
  CHECK(write);
  const auto& layers = harness.builder.layers;
  const auto& components = harness.builder.components;
  uint64_t row = 0;
  for (const auto& component : components) row += component.bytes_per_block;
  auto source = HostBuffer::Import(harness.data(), row * layers.size());
  CHECK(source);
  BindingBuilder builder;
  builder.AddBlock(blocks[0]);
  for (uint32_t layer : layers) builder.AddLayer(layer);
  builder.Bind(source->region, harness.builder);
  auto store_request = builder.Request();
  store_request.selection.group = harness.builder.group_id;
  auto stored = harness.data().Store(*write, store_request);
  CHECK(stored);

  // Pending work and live owners keep the session busy.
  CHECK_EQ(harness.provider->Shutdown().code, KVC_BUSY);
  // Completion observation remains available while closing.
  CHECK(stored->Wait().Ok());
  CHECK(harness.control().Commit(*write));
  // Handles still alive: still busy.
  CHECK_EQ(harness.provider->Shutdown().code, KVC_BUSY);
}

}  // namespace async

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <provider.so> [async|cuda]\n", argv[0]);
    return 2;
  }
  g_provider_path = argv[1];
  if (argc > 2 && std::strcmp(argv[2], "async") == 0) g_mode = Mode::kAsync;
  if (argc > 2 && std::strcmp(argv[2], "cuda") == 0) g_mode = Mode::kCuda;

  TEST(TestCapabilitiesAndConfiguration);
  TEST(TestPublicationRoundTrip);
  TEST(TestLayerwiseStoresAndIncomplete);
  TEST(TestCommitAndTransactionSemantics);
  TEST(TestLookupSemantics);
  TEST(TestRemoveAndPinnedGenerations);
  TEST(TestAbortSemantics);
  TEST(TestBindingValidation);
  TEST(TestImportValidation);
  TEST(TestBudgetExhaustion);
  TEST(TestShutdownLifecycle);
  TEST(TestTransferPollWait);
  TEST(TestStatsExtension);
  TEST(TestConcurrentReaders);
  if (g_mode == Mode::kAsync) {
    TEST(async::TestPendingDeadlineAndCompletion);
    TEST(async::TestDroppedHandleStillCompletes);
    TEST(async::TestFailedStorePoisonsTransaction);
    TEST(async::TestFailedDependency);
    TEST(async::TestShutdownBusyWhilePending);
  }
  return ExitCode();
}
