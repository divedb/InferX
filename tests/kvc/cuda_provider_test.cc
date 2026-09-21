// CUDA tiered-provider tests: device-resident storage, device-to-device
// transfers with event completion, LRU offload to the host tier, promotion
// back to the GPU, eviction to misses, and pin interactions.
//
// usage: cuda_provider_test <libkvc_cuda.so>
//
// Budgets are set through the provider's options record so every scenario
// is deterministic regardless of the installed GPU.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "kvc/cache.h"
#include "kvc/extensions/stats.h"
#include "test_util.h"

namespace {

using namespace kvc;
using namespace kvc_test;

const char* kProviderPath = nullptr;

// ---- Options encoding -----------------------------------------------------

struct Options {
  uint64_t gpu_budget;
  uint64_t host_budget;
};

KvcDescriptor Encode(const Options& options, std::vector<unsigned char>& payload) {
  auto put64 = [&payload](uint64_t v) {
    for (int i = 0; i < 8; ++i) payload.push_back(static_cast<unsigned char>(v >> (i * 8)));
  };
  auto put32 = [&payload](uint32_t v) {
    for (int i = 0; i < 4; ++i) payload.push_back(static_cast<unsigned char>(v >> (i * 8)));
  };
  put64(options.gpu_budget);
  put64(options.host_budget);
  put32(0);
  put32(0);
  KvcDescriptor descriptor{};
  descriptor.schema = {{"kvc.cuda.options", 15}, 1, 0};
  descriptor.canonical_payload = {payload.data(), payload.size()};
  return descriptor;
}

// ---- Device memory ---------------------------------------------------------

struct DeviceBuffer {
  void* memory = nullptr;
  size_t size = 0;
  MemoryRegion region;

  static kvc::Result<DeviceBuffer> Import(const KVCacheData& data, size_t bytes,
                                          uint32_t access = KVC_READ_WRITE) {
    DeviceBuffer buffer;
    if (cudaMalloc(&buffer.memory, bytes) != cudaSuccess || !buffer.memory)
      return std::unexpected(Status{KVC_RESOURCE_EXHAUSTED, "cudaMalloc failed"});
    // cudaMemset is asynchronous on the null stream; producer readiness is
    // an engine-side obligation, so synchronize before the buffer can be
    // submitted to provider transfers on a different stream.
    if (cudaMemset(buffer.memory, 0, bytes) != cudaSuccess ||
        cudaStreamSynchronize(nullptr) != cudaSuccess)
      return std::unexpected(Status{KVC_TRANSPORT_ERROR, "cudaMemset failed"});
    buffer.size = bytes;
    std::string runtime = "cuda";
    std::string identifier = "0";
    KvcMemoryInfo info{};
    info.struct_size = sizeof(info);
    info.memory_type = KVC_DEVICE;
    info.access = access;
    info.locator_kind = KVC_LOCAL_ADDRESS;
    info.local_address = reinterpret_cast<uintptr_t>(buffer.memory);
    info.byte_size = bytes;
    info.device_runtime = {runtime.data(), runtime.size()};
    info.device_identifier = {identifier.data(), identifier.size()};
    auto region = data.ImportMemory(info, std::make_shared<int>(7));
    if (!region) {
      cudaFree(buffer.memory);
      return std::unexpected(region.error());
    }
    buffer.region = std::move(*region);
    return buffer;
  }

  bool Upload(const unsigned char* host) const {
    return cudaMemcpy(memory, host, size, cudaMemcpyHostToDevice) == cudaSuccess;
  }
  bool Download(unsigned char* host) const {
    return cudaMemcpy(host, memory, size, cudaMemcpyDeviceToHost) == cudaSuccess;
  }

  // A user-declared destructor suppresses implicit moves, so returning a
  // DeviceBuffer by value would copy it and free the pointer twice. Moves
  // are therefore explicit and leave the source empty.
  DeviceBuffer() = default;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : memory(other.memory), size(other.size), region(std::move(other.region)) {
    other.memory = nullptr;
    other.size = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Release();
      memory = other.memory;
      size = other.size;
      region = std::move(other.region);
      other.memory = nullptr;
      other.size = 0;
    }
    return *this;
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() { Release(); }

  void Release() {
    region = MemoryRegion();
    if (memory) {
      cudaFree(memory);
      memory = nullptr;
    }
    size = 0;
  }
};

// ---- Harness ---------------------------------------------------------------

struct Harness {
  std::unique_ptr<KVCacheProvider> provider;
  ConfigBuilder::BuiltConfig config;
  ConfigBuilder builder;

  static kvc::Result<std::unique_ptr<Harness>> Make(const Options& options) {
    auto harness = std::make_unique<Harness>();
    std::vector<unsigned char> payload;
    auto opened = OpenProvider({kProviderPath, Encode(options, payload)});
    if (!opened) return std::unexpected(opened.error());
    harness->provider = std::move(*opened);
    harness->config = harness->builder.Build();
    if (auto status = harness->provider->Configure(harness->config.config); !status.Ok())
      return std::unexpected(status);
    return harness;
  }

  KVCacheControl control() const { return provider->Control(); }
  KVCacheData data() const { return provider->Data(); }
};

uint64_t BlockBytes(const ConfigBuilder& builder) {
  uint64_t row = 0;
  for (const auto& component : builder.components) row += component.bytes_per_block;
  return row * builder.layers.size();
}

// Fills a host staging buffer for the given blocks (block-major layout).
void FillHost(std::vector<unsigned char>& bytes, const std::vector<KvcBlock>& blocks,
              const ConfigBuilder& builder) {
  uint64_t offset = 0;
  for (const auto& block : blocks)
    for (uint32_t layer : builder.layers)
      for (const auto& component : builder.components) {
        FillPattern(bytes.data() + offset, block, layer, component.id,
                    component.bytes_per_block);
        offset += component.bytes_per_block;
      }
}

bool CheckHost(const std::vector<unsigned char>& bytes, const std::vector<KvcBlock>& blocks,
               const ConfigBuilder& builder) {
  uint64_t offset = 0;
  for (const auto& block : blocks)
    for (uint32_t layer : builder.layers)
      for (const auto& component : builder.components) {
        if (!CheckPattern(bytes.data() + offset, block, layer, component.id,
                          component.bytes_per_block))
          return false;
        offset += component.bytes_per_block;
      }
  return true;
}

// Publishes blocks from a device-resident source buffer.
bool PublishFromDevice(Harness& harness, const std::vector<KvcBlock>& blocks, uint64_t tx) {
  const uint64_t payload = blocks.size() * BlockBytes(harness.builder);
  std::vector<unsigned char> staging(payload);
  FillHost(staging, blocks, harness.builder);
  auto source = DeviceBuffer::Import(harness.data(), payload);
  CHECK(source);
  CHECK(source->Upload(staging.data()));

  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(tx);
  request.blocks = blocks.data();
  request.block_count = blocks.size();
  auto write = harness.control().BeginWrite(request);
  if (!write)
    std::fprintf(stderr, "begin failed: code=%u diag=%s\n", write.error().code,
                 write.error().diagnostic.c_str());
  CHECK(write);
  BindingBuilder builder;
  for (const auto& block : blocks) builder.AddBlock(block);
  for (uint32_t layer : harness.builder.layers) builder.AddLayer(layer);
  builder.Bind(source->region, harness.builder);
  auto store_request = builder.Request();
  store_request.selection.group = harness.builder.group_id;
  auto stored = harness.data().Store(*write, store_request);
  CHECK(stored);
  CHECK(stored->Wait().Ok());
  CHECK(harness.control().Commit(*write));
  return true;
}

// Loads pinned blocks into a device destination and verifies the bytes.
bool LoadAndCheckDevice(Harness& harness, const ReadHandle& read,
                        const std::vector<KvcBlock>& blocks) {
  const uint64_t payload = blocks.size() * BlockBytes(harness.builder);
  auto destination = DeviceBuffer::Import(harness.data(), payload);
  CHECK(destination);
  BindingBuilder builder;
  for (const auto& block : blocks) builder.AddBlock(block);
  for (uint32_t layer : harness.builder.layers) builder.AddLayer(layer);
  builder.Bind(destination->region, harness.builder);
  auto request = builder.Request();
  request.selection.group = harness.builder.group_id;
  auto loaded = harness.data().Load(read, request);
  CHECK(loaded);
  CHECK(loaded->Wait().Ok());
  std::vector<unsigned char> staging(payload);
  CHECK(destination->Download(staging.data()));
  CHECK_MSG(CheckHost(staging, blocks, harness.builder), "device roundtrip payload mismatch");
  return true;
}

LookupOne MakeLookup(const std::vector<KvcBlock>& catalog, uint64_t token_count) {
  return LookupOne(catalog, token_count);
}

// ---- Tests -----------------------------------------------------------------

void TestDeviceRoundTrip() {
  const uint64_t block = BlockBytes(ConfigBuilder{});
  auto made = Harness::Make({4 * block, 1ull << 30});
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 211),
                               BlockAt(0, 1, tokens, tokens, 212)};
  CHECK(PublishFromDevice(harness, blocks, 30001));

  auto one = MakeLookup(blocks, 2 * tokens);
  auto result = harness.control().Lookup(one.request);
  CHECK(result);
  CHECK(result->read.Valid());
  CHECK(LoadAndCheckDevice(harness, result->read, blocks));
}

void TestDeviceToHostLoad() {
  const uint64_t block = BlockBytes(ConfigBuilder{});
  auto made = Harness::Make({4 * block, 1ull << 30});
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 221)};
  CHECK(PublishFromDevice(harness, blocks, 30002));

  auto one = MakeLookup(blocks, tokens);
  auto result = harness.control().Lookup(one.request);
  CHECK(result);
  CHECK(result->read.Valid());
  // Load into host memory: exercises the device-to-host path.
  auto destination =
      HostBuffer::Import(harness.data(), blocks.size() * BlockBytes(harness.builder));
  CHECK(destination);
  BindingBuilder builder;
  builder.AddBlock(blocks[0]);
  for (uint32_t layer : harness.builder.layers) builder.AddLayer(layer);
  builder.Bind(destination->region, harness.builder);
  auto request = builder.Request();
  request.selection.group = harness.builder.group_id;
  auto loaded = harness.data().Load(result->read, request);
  CHECK(loaded);
  CHECK(loaded->Wait().Ok());
  CHECK(CheckHost(destination->bytes, blocks, harness.builder));
}

void TestOffloadAndRestore() {
  // One block of GPU budget: every additional publication writes the
  // least-recently-used block back to the host tier.
  const uint64_t block = BlockBytes(ConfigBuilder{});
  auto made = Harness::Make({1 * block, 64 * block});
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> b0{BlockAt(0, 0, tokens, tokens, 231)};
  std::vector<KvcBlock> b1{BlockAt(0, 1, tokens, tokens, 232)};
  std::vector<KvcBlock> b2{BlockAt(0, 2, tokens, tokens, 233)};
  CHECK(PublishFromDevice(harness, b0, 30003));
  CHECK(PublishFromDevice(harness, b1, 30004));  // b0 offloaded
  CHECK(PublishFromDevice(harness, b2, 30005));  // b1 offloaded

  // Everything remains discoverable and loadable from either tier.
  for (const auto* group : {&b0, &b1, &b2}) {
    auto one = MakeLookup(*group, tokens);
    auto result = harness.control().Lookup(one.request);
    CHECK(result);
    CHECK(result->read.Valid());
    CHECK(LoadAndCheckDevice(harness, result->read, *group));
  }
  // All three together still hit.
  std::vector<KvcBlock> all;
  all.insert(all.end(), b0.begin(), b0.end());
  all.insert(all.end(), b1.begin(), b1.end());
  all.insert(all.end(), b2.begin(), b2.end());
  auto one = MakeLookup(all, 3 * tokens);
  auto result = harness.control().Lookup(one.request);
  CHECK(result);
  CHECK(result->read.Valid());
}

void TestEvictionToMissAndRecreation() {
  // One GPU block and one host block: the third publication evicts the
  // first block entirely (host budget drops it).
  const uint64_t block = BlockBytes(ConfigBuilder{});
  auto made = Harness::Make({1 * block, 1 * block});
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> b0{BlockAt(0, 0, tokens, tokens, 241)};
  std::vector<KvcBlock> b1{BlockAt(0, 1, tokens, tokens, 242)};
  std::vector<KvcBlock> b2{BlockAt(0, 2, tokens, tokens, 243)};
  CHECK(PublishFromDevice(harness, b0, 30006));
  CHECK(PublishFromDevice(harness, b1, 30007));  // b0 -> host tier
  CHECK(PublishFromDevice(harness, b2, 30008));  // b0 dropped (host full)

  // b0 is gone; b1 and b2 remain discoverable.
  auto b0_lookup = MakeLookup(b0, tokens);
  auto miss = harness.control().Lookup(b0_lookup.request);
  CHECK(miss);
  CHECK(!miss->read.Valid());
  for (const auto* group : {&b1, &b2}) {
    auto one = MakeLookup(*group, tokens);
    auto result = harness.control().Lookup(one.request);
    CHECK(result);
    CHECK(result->read.Valid());
    CHECK(LoadAndCheckDevice(harness, result->read, *group));
  }

  // The evicted key can be published again.
  CHECK(PublishFromDevice(harness, b0, 30009));
  auto again = MakeLookup(b0, tokens);
  auto hit = harness.control().Lookup(again.request);
  CHECK(hit);
  CHECK(hit->read.Valid());
}

void TestPromotionRestoresToGpu() {
  // Two GPU blocks, roomy host tier: publish three blocks (b0 offloaded),
  // retire b1 to free device budget, then load b0 alone: the provider must
  // restore it to device memory (budget available) and serve device-side.
  const uint64_t block = BlockBytes(ConfigBuilder{});
  auto made = Harness::Make({2 * block, 64 * block});
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> b0{BlockAt(0, 0, tokens, tokens, 251)};
  std::vector<KvcBlock> b1{BlockAt(0, 1, tokens, tokens, 252)};
  std::vector<KvcBlock> b2{BlockAt(0, 2, tokens, tokens, 253)};
  CHECK(PublishFromDevice(harness, b0, 30010));
  CHECK(PublishFromDevice(harness, b1, 30011));
  CHECK(PublishFromDevice(harness, b2, 30012));  // b0 offloaded to host

  // Retire b1: unpinned and unreferenced, its device slot is reclaimed.
  CHECK(harness.control().Remove(b1[0].key).Ok());
  auto b1_lookup = MakeLookup(b1, tokens);
  auto retired = harness.control().Lookup(b1_lookup.request);
  CHECK(retired);
  CHECK(!retired->read.Valid());

  // b0 now fits on the device again; loading it promotes it back.
  auto one = MakeLookup(b0, tokens);
  auto result = harness.control().Lookup(one.request);
  CHECK(result);
  CHECK(result->read.Valid());
  CHECK(LoadAndCheckDevice(harness, result->read, b0));

  // A further write must still respect the budget. The pinned, promoted b0
  // cannot be offloaded, so a single-block publication evicts b2 and fits.
  std::vector<KvcBlock> b3{BlockAt(0, 3, tokens, tokens, 254)};
  CHECK(PublishFromDevice(harness, b3, 30013));
}

void TestPinsPreventEviction() {
  const uint64_t block = BlockBytes(ConfigBuilder{});
  auto made = Harness::Make({1 * block, 1 * block});
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> b0{BlockAt(0, 0, tokens, tokens, 261)};
  CHECK(PublishFromDevice(harness, b0, 30014));

  // Pin b0, then try to publish a new block: nothing is evictable.
  auto one = MakeLookup(b0, tokens);
  auto pinned = harness.control().Lookup(one.request);
  CHECK(pinned);
  CHECK(pinned->read.Valid());

  std::vector<KvcBlock> b1{BlockAt(0, 1, tokens, tokens, 262)};
  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(30015);
  request.blocks = b1.data();
  request.block_count = b1.size();
  auto refused = harness.control().BeginWrite(request);
  CHECK_EQ(refused.error().code, KVC_RESOURCE_EXHAUSTED);

  // The pinned generation still loads.
  CHECK(LoadAndCheckDevice(harness, pinned->read, b0));

  // Releasing the pin lets the publication proceed.
  pinned->read = ReadHandle();
  auto write = harness.control().BeginWrite(request);
  CHECK(write);
  write = std::unexpected(Status{});
}

void TestHostBudgetPinsSurvive() {
  // GPU: two blocks, host: one block. Pin a block that was offloaded to
  // the host tier; a further offload must fail rather than drop it.
  const uint64_t block = BlockBytes(ConfigBuilder{});
  auto made = Harness::Make({2 * block, 1 * block});
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> b0{BlockAt(0, 0, tokens, tokens, 271)};
  std::vector<KvcBlock> b1{BlockAt(0, 1, tokens, tokens, 272)};
  std::vector<KvcBlock> b2{BlockAt(0, 2, tokens, tokens, 273)};
  CHECK(PublishFromDevice(harness, b0, 30017));
  CHECK(PublishFromDevice(harness, b1, 30018));
  // Pin both: b0 (LRU) would be the offload victim.
  std::vector<KvcBlock> both;
  both.insert(both.end(), b0.begin(), b0.end());
  both.insert(both.end(), b1.begin(), b1.end());
  auto one = MakeLookup(both, 2 * tokens);
  auto pinned = harness.control().Lookup(one.request);
  CHECK(pinned);
  CHECK(pinned->read.Valid());

  // Publishing a third block needs GPU room; both victims are pinned and
  // neither host-tier block can be dropped, so the reservation fails.
  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(30019);
  request.blocks = b2.data();
  request.block_count = b2.size();
  CHECK_EQ(harness.control().BeginWrite(request).error().code, KVC_RESOURCE_EXHAUSTED);

  // The pinned blocks remain readable.
  CHECK(LoadAndCheckDevice(harness, pinned->read, both));
}

void TestDependenciesBetweenDeviceTransfers() {
  const uint64_t block = BlockBytes(ConfigBuilder{});
  auto made = Harness::Make({4 * block, 1ull << 30});
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 281)};
  CHECK(PublishFromDevice(harness, blocks, 30020));

  auto one = MakeLookup(blocks, tokens);
  auto result = harness.control().Lookup(one.request);
  CHECK(result);
  CHECK(result->read.Valid());

  // Two dependent loads into different device destinations.
  auto first = DeviceBuffer::Import(harness.data(), BlockBytes(harness.builder));
  auto second = DeviceBuffer::Import(harness.data(), BlockBytes(harness.builder));
  CHECK(first);
  CHECK(second);
  BindingBuilder builder;
  builder.AddBlock(blocks[0]);
  for (uint32_t layer : harness.builder.layers) builder.AddLayer(layer);
  builder.Bind(first->region, harness.builder);
  auto first_request = builder.Request();
  first_request.selection.group = harness.builder.group_id;
  auto load1 = harness.data().Load(result->read, first_request);
  CHECK(load1);

  BindingBuilder builder2;
  builder2.AddBlock(blocks[0]);
  for (uint32_t layer : harness.builder.layers) builder2.AddLayer(layer);
  builder2.Bind(second->region, harness.builder);
  KvcObject* deps[1] = {load1->NativeHandle()};
  auto second_request = builder2.Request(deps, 1);
  second_request.selection.group = harness.builder.group_id;
  auto load2 = harness.data().Load(result->read, second_request);
  CHECK(load2);
  CHECK(load2->Wait().Ok());
  CHECK(load1->Wait().Ok());

  std::vector<unsigned char> staging(BlockBytes(harness.builder));
  CHECK(first->Download(staging.data()));
  CHECK(CheckHost(staging, blocks, harness.builder));
  CHECK(second->Download(staging.data()));
  CHECK(CheckHost(staging, blocks, harness.builder));
}

void TestLayerwiseDeviceStores() {
  const uint64_t block = BlockBytes(ConfigBuilder{});
  auto made = Harness::Make({4 * block, 1ull << 30});
  CHECK(made);
  auto& harness = **made;
  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 291)};
  const uint64_t row = BlockBytes(harness.builder) / harness.builder.layers.size();

  auto source = DeviceBuffer::Import(harness.data(), 2 * row);
  CHECK(source);
  std::vector<unsigned char> staging(2 * row);
  FillHost(staging, blocks, harness.builder);
  CHECK(source->Upload(staging.data()));

  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(30021);
  request.blocks = blocks.data();
  request.block_count = blocks.size();
  auto write = harness.control().BeginWrite(request);
  CHECK(write);
  uint64_t offset = 0;
  for (uint32_t layer : harness.builder.layers) {
    BindingBuilder builder;
    builder.AddBlock(blocks[0]);
    builder.AddLayer(layer);
    builder.Bind(source->region, harness.builder, offset);
    auto store_request = builder.Request();
    store_request.selection.group = harness.builder.group_id;
    auto stored = harness.data().Store(*write, store_request);
    CHECK(stored);
    CHECK(stored->Wait().Ok());
    offset += row;
  }
  CHECK(harness.control().Commit(*write));

  auto one = MakeLookup(blocks, tokens);
  auto result = harness.control().Lookup(one.request);
  CHECK(result);
  CHECK(result->read.Valid());
  CHECK(LoadAndCheckDevice(harness, result->read, blocks));
}

void TestStatsExtension() {
  // GPU holds one block and the host tier holds one block: publishing three
  // blocks offloads and then evicts, and loads exercise restore paths.
  const uint64_t block = BlockBytes(ConfigBuilder{});
  auto made = Harness::Make({1 * block, 1 * block});
  CHECK(made);
  auto& harness = **made;
  KvcSchema schema{};
  schema.name = {"kvc.stats", 9};
  schema.major = 1;
  auto extension = harness.provider->QueryExtension(schema);
  CHECK(extension);
  auto* api = static_cast<const KvcStatsApi*>(extension->Table());

  auto snapshot = [&]() -> KvcStatsRecord {
    KvcStatsRecord record{};
    record.struct_size = sizeof(record);
    CHECK_EQ(api->get(extension->NativeHandle(), &record).code, KVC_OK);
    return record;
  };

  const uint64_t tokens = harness.builder.tokens_per_block;
  std::vector<KvcBlock> b0{BlockAt(0, 0, tokens, tokens, 271)};
  std::vector<KvcBlock> b1{BlockAt(0, 1, tokens, tokens, 272)};
  std::vector<KvcBlock> b2{BlockAt(0, 2, tokens, tokens, 273)};
  CHECK(PublishFromDevice(harness, b0, 30030));
  CHECK(PublishFromDevice(harness, b1, 30031));  // b0 -> host tier.
  CHECK(PublishFromDevice(harness, b2, 30032));  // b0 dropped (host full).

  const KvcStatsRecord after_publish = snapshot();
  CHECK_EQ(after_publish.lookups, 0u);  // No lookups before this snapshot.
  CHECK_EQ(after_publish.blocks_committed, 3u);
  CHECK_EQ(after_publish.transactions_committed, 3u);
  CHECK_EQ(after_publish.stores, 3u);
  CHECK_EQ(after_publish.store_bytes, 3 * block);
  CHECK(after_publish.offloads >= 1);
  CHECK_EQ(after_publish.offload_bytes, after_publish.offloads * block);
  CHECK(after_publish.blocks_evicted >= 1);
  CHECK(after_publish.gpu_bytes_in_use <= 1 * block);
  CHECK_EQ(after_publish.gpu_budget, 1 * block);
  CHECK_EQ(after_publish.host_budget, 1 * block);

  // Loading the surviving blocks exercises restore paths; device loads
  // either promote (when budget allows) or copy from the host tier.
  for (const auto* group : {&b1, &b2}) {
    auto one = MakeLookup(*group, tokens);
    auto result = harness.provider->Control().Lookup(one.request);
    CHECK(result);
    CHECK(result->read.Valid());
    CHECK(LoadAndCheckDevice(harness, result->read, *group));
  }
  const KvcStatsRecord after_load = snapshot();
  CHECK(after_load.loads >= 2);
  CHECK(after_load.load_bytes >= 2 * block);
  CHECK(after_load.lookup_hits >= 2);
}

void TestForeignRegionRejected() {
  const uint64_t block = BlockBytes(ConfigBuilder{});
  Options options{4 * block, 1ull << 30};
  auto made = Harness::Make(options);
  CHECK(made);
  auto other = Harness::Make(options);
  CHECK(other);

  std::vector<unsigned char> bytes(256);
  std::string runtime = "cpu";
  std::string identifier = "1";  // Wrong CPU identifier.
  KvcMemoryInfo info{};
  info.struct_size = sizeof(info);
  info.memory_type = KVC_HOST;
  info.access = KVC_READ_WRITE;
  info.locator_kind = KVC_LOCAL_ADDRESS;
  info.local_address = reinterpret_cast<uintptr_t>(bytes.data());
  info.byte_size = bytes.size();
  info.device_runtime = {runtime.data(), runtime.size()};
  info.device_identifier = {identifier.data(), identifier.size()};
  CHECK_EQ((*made)->data().ImportMemory(info, std::make_shared<int>(1)).error().code,
           KVC_UNSUPPORTED);

  // Device memory from another session cannot bind here.
  auto foreign_device = DeviceBuffer::Import((*other)->data(), block);
  CHECK(foreign_device);
  const uint64_t tokens = (*made)->builder.tokens_per_block;
  std::vector<KvcBlock> fresh{BlockAt(0, 9, tokens, tokens, 292)};
  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(30022);
  request.blocks = fresh.data();
  request.block_count = fresh.size();
  auto write = (*made)->control().BeginWrite(request);
  CHECK(write);
  BindingBuilder builder;
  builder.AddBlock(fresh[0]);
  for (uint32_t layer : (*made)->builder.layers) builder.AddLayer(layer);
  builder.Bind(foreign_device->region, (*made)->builder);
  auto store_request = builder.Request();
  store_request.selection.group = (*made)->builder.group_id;
  CHECK_EQ((*made)->data().Store(*write, store_request).error().code, KVC_STALE_HANDLE);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <libkvc_cuda.so>\n", argv[0]);
    return 2;
  }
  kProviderPath = argv[1];

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0) {
    std::fprintf(stderr, "no CUDA device available; skipping\n");
    return 0;  // Let CI without GPUs pass; the conformance suite covers CPU paths.
  }

  TEST(TestDeviceRoundTrip);
  TEST(TestDeviceToHostLoad);
  TEST(TestOffloadAndRestore);
  TEST(TestEvictionToMissAndRecreation);
  TEST(TestPromotionRestoresToGpu);
  TEST(TestPinsPreventEviction);
  TEST(TestHostBudgetPinsSurvive);
  TEST(TestDependenciesBetweenDeviceTransfers);
  TEST(TestLayerwiseDeviceStores);
  TEST(TestStatsExtension);
  TEST(TestForeignRegionRejected);
  return ExitCode();
}
