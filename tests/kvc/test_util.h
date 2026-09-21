// Shared helpers for the KV-cache provider tests.
//
// Test identities are derived from a fixed digest function over the block
// record fields. That is sufficient for conformance testing against any
// provider implementing the opaque profile; production callers must derive
// identities from the canonical identity scheme instead (see
// docs/extensions/kvcache.md).

#ifndef KVC_TESTS_TEST_UTIL_H_
#define KVC_TESTS_TEST_UTIL_H_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kvc/cache.h"

namespace kvc_test {

// ---- Minimal check framework (the kvc tests do not link googletest). ----

inline int& Failures() {
  static int count = 0;
  return count;
}
inline const char*& CurrentTest() {
  static const char* name = "";
  return name;
}

inline void Report(const char* file, int line, const std::string& message) {
  std::fprintf(stderr, "%s:%d: FAILURE in %s: %s\n", file, line, CurrentTest(),
               message.c_str());
  ++Failures();
}

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) ::kvc_test::Report(__FILE__, __LINE__, "check failed: " #cond); \
  } while (false)

#define CHECK_MSG(cond, msg)                                    \
  do {                                                          \
    if (!(cond)) ::kvc_test::Report(__FILE__, __LINE__, (msg)); \
  } while (false)

#define CHECK_EQ(actual, expected)                                                       \
  do {                                                                                   \
    auto a_ = (actual);                                                                  \
    auto b_ = (expected);                                                                \
    if (!(a_ == b_))                                                                     \
      ::kvc_test::Report(__FILE__, __LINE__, "check failed: " #actual " == " #expected); \
  } while (false)

template <class Fn>
inline void RunTest(const char* name, Fn&& fn) {
  CurrentTest() = name;
  const int before = Failures();
  fn();
  std::printf("[ %s ] %s\n", Failures() == before ? "OK" : "FAILED", name);
}

#define TEST(fn) ::kvc_test::RunTest(#fn, fn)

inline int ExitCode() { return Failures() == 0 ? 0 : 1; }

// ---- Digest helpers ----------------------------------------------------

inline uint64_t Mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  return h;
}

inline KvcDigest DigestOf(uint64_t a, uint64_t b = 0, uint64_t c = 0, uint64_t d = 0) {
  uint64_t h = 0xcbf29ce484222325ull;
  h = Mix(h, a);
  h = Mix(h, b);
  h = Mix(h, c);
  h = Mix(h, d);
  KvcDigest digest{};
  for (int i = 0; i < 4; ++i) {
    uint64_t word = h + i * 0x100000001b3ull;
    h = Mix(h, word);
    std::memcpy(digest.bytes + i * 8, &word, sizeof(word));
  }
  return digest;
}

inline KvcTransactionId TxIdOf(uint64_t value) {
  KvcTransactionId id{};
  std::memcpy(id.bytes, &value, sizeof(value));
  for (size_t i = sizeof(value); i < sizeof(id.bytes); ++i)
    id.bytes[i] = static_cast<unsigned char>(value >> ((i % 8) * 8));
  return id;
}

inline bool DigestEq(const KvcDigest& a, const KvcDigest& b) {
  return std::memcmp(a.bytes, b.bytes, sizeof(a.bytes)) == 0;
}

// ---- Configuration builder ---------------------------------------------

struct ComponentDef {
  uint32_t id;
  uint64_t bytes_per_block;
};

// Owns the borrowed arrays inside a KvcConfig.
struct ConfigBuilder {
  uint32_t group_id = 0;
  uint32_t tokens_per_block = 16;
  std::vector<uint32_t> layers{0, 1};
  std::vector<ComponentDef> components{{0, 64}, {1, 96}};
  KvcDigest cache_namespace = DigestOf(1);
  KvcDigest model_identity = DigestOf(2);
  std::string partition_name = "kvc.test.partition";
  std::vector<unsigned char> partition_payload{0x01};

  struct BuiltConfig {
    KvcConfig config{};
    std::vector<uint32_t> layers;
    std::vector<KvcComponent> components;
    std::string partition_name;
    std::vector<unsigned char> partition_payload;
    std::vector<KvcGroup> groups_storage;
  };

  BuiltConfig Build() const {
    BuiltConfig built;
    built.config.struct_size = sizeof(KvcConfig);
    built.config.cache_namespace = cache_namespace;
    built.config.model_identity = model_identity;
    built.layers = layers;
    built.components.reserve(components.size());
    for (const auto& component : components)
      built.components.push_back(KvcComponent{component.id,
                                              KVC_OPAQUE_BYTES,
                                              {{{"kvc.bytes", 9}, 1, 0}, {nullptr, 0}},
                                              component.bytes_per_block});
    built.partition_name = partition_name;
    built.partition_payload = partition_payload;
    static const unsigned char kEmpty[1] = {0};
    KvcGroup group{};
    group.id = group_id;
    group.tokens_per_block = tokens_per_block;
    group.semantics = KvcDescriptor{{{"kvc.opaque", 10}, 1, 0}, {kEmpty, 0}};
    group.layers = built.layers.data();
    group.layer_count = built.layers.size();
    group.components = built.components.data();
    group.component_count = built.components.size();
    built.groups_storage.assign(1, group);
    built.config.groups = built.groups_storage.data();
    built.config.group_count = 1;
    built.config.logical_partition.schema = {
        {built.partition_name.data(), built.partition_name.size()}, 0, 0};
    built.config.logical_partition.canonical_payload = {built.partition_payload.data(),
                                                        built.partition_payload.size()};
    return built;
  }
};

// ---- Block records ------------------------------------------------------

inline KvcBlock BlockAt(uint32_t group, uint64_t logical_index, uint64_t token_count,
                        uint64_t tokens_per_block, uint64_t seed) {
  KvcBlock block{};
  block.key = DigestOf(seed, logical_index, token_count, 7);
  block.group = group;
  block.logical_index = logical_index;
  block.first_token = logical_index * tokens_per_block;
  block.token_count = token_count;
  block.dependency = DigestOf(seed, logical_index, 99, 3);
  return block;
}

inline KvcPrefix PrefixOf(uint64_t seed, uint64_t token_count) {
  return KvcPrefix{DigestOf(seed, 5, 6, 8), token_count};
}

// Deterministic payload: every byte depends on (block, layer, component).
inline void FillPattern(unsigned char* data, const KvcBlock& block, uint32_t layer,
                        uint32_t component, uint64_t byte_count) {
  for (uint64_t i = 0; i < byte_count; ++i) {
    data[i] = static_cast<unsigned char>(block.key.bytes[i % 32] ^ (layer * 41) ^
                                         (component * 29) ^ (i * 2654435761u >> 24));
  }
}

inline bool CheckPattern(const unsigned char* data, const KvcBlock& block, uint32_t layer,
                         uint32_t component, uint64_t byte_count) {
  for (uint64_t i = 0; i < byte_count; ++i) {
    const unsigned char expected = static_cast<unsigned char>(
        block.key.bytes[i % 32] ^ (layer * 41) ^ (component * 29) ^ (i * 2654435761u >> 24));
    if (data[i] != expected) return false;
  }
  return true;
}

// ---- Host memory + region import ----------------------------------------

struct HostBuffer {
  std::vector<unsigned char> bytes;
  kvc::MemoryRegion region;

  static kvc::Result<HostBuffer> Import(const kvc::KVCacheData& data, uint64_t size,
                                        uint32_t access = KVC_READ_WRITE,
                                        uint32_t memory_type = KVC_HOST) {
    HostBuffer buffer;
    buffer.bytes.assign(size, 0);
    std::string runtime = "cpu";
    std::string identifier = "0";
    KvcMemoryInfo info{};
    info.struct_size = sizeof(info);
    info.memory_type = memory_type;
    info.access = access;
    info.locator_kind = KVC_LOCAL_ADDRESS;
    info.local_address = reinterpret_cast<uintptr_t>(buffer.bytes.data());
    info.byte_size = size;
    info.device_runtime = {runtime.data(), runtime.size()};
    info.device_identifier = {identifier.data(), identifier.size()};
    auto region = data.ImportMemory(info, std::make_shared<int>(42));
    if (!region) return std::unexpected(region.error());
    buffer.region = std::move(*region);
    return buffer;
  }
};

// ---- Span and binding construction --------------------------------------

// A lookup request for one candidate covering the whole catalog, owning
// everything the request borrows.
struct LookupOne {
  std::vector<KvcRange> ranges;
  std::vector<KvcPrefix> prefixes;
  std::vector<KvcCandidate> candidates;
  KvcLookupRequest request{};

  explicit LookupOne(const std::vector<KvcBlock>& catalog, uint64_t token_count) {
    ranges.push_back(KvcRange{0, catalog.size()});
    prefixes.push_back(PrefixOf(7777, token_count));
    candidates.push_back(KvcCandidate{prefixes[0], ranges.data(), 1});
    request.struct_size = sizeof(request);
    request.catalog = catalog.data();
    request.catalog_count = catalog.size();
    request.candidates = candidates.data();
    request.candidate_count = 1;
  }
};

struct BindingBuilder {
  std::vector<KvcSlice> slices;
  std::vector<uint32_t> layers;
  std::vector<KvcBinding> bindings;

  void AddBlock(const KvcBlock& block) {
    slices.push_back(KvcSlice{block.key, 0, block.token_count});
  }
  void AddLayer(uint32_t layer) { layers.push_back(layer); }

  // Adds one binding per (block, layer, component) of the span.
  void Bind(const kvc::MemoryRegion& region, const ConfigBuilder& config, uint64_t offset = 0) {
    bindings.clear();
    for (const auto& slice : slices) {
      for (uint32_t layer : layers) {
        for (const auto& component : config.components) {
          KvcBinding binding{};
          binding.block = slice;
          binding.layer = layer;
          binding.component = component.id;
          binding.region = region.NativeHandle();
          binding.byte_offset = offset;
          binding.byte_count = component.bytes_per_block;
          binding.layout = KVC_OPAQUE_BYTES;
          binding.rank = 0;
          offset += component.bytes_per_block;
          bindings.push_back(binding);
        }
      }
    }
  }

  KvcTransferRequest Request(KvcObject* const* deps = nullptr, uint64_t dep_count = 0) const {
    KvcTransferRequest request{};
    request.struct_size = sizeof(request);
    request.selection.group = 0;  // Filled by callers that use other groups.
    request.selection.layers = layers.data();
    request.selection.layer_count = layers.size();
    request.selection.blocks = slices.data();
    request.selection.block_count = slices.size();
    request.bindings = bindings.data();
    request.binding_count = bindings.size();
    request.dependencies = deps;
    request.dependency_count = dep_count;
    return request;
  }
};

}  // namespace kvc_test

#endif  // KVC_TESTS_TEST_UTIL_H_
