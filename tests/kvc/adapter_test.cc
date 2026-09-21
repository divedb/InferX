// Engine integration boundary test: importing owned InferX Storage into a
// provider as transfer memory, plus the borrowed-storage rejection path.
//
// usage: adapter_test <libkvc_host.so> [libkvc_cuda.so]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(INFERX_TEST_CUDA)
#include <cuda_runtime.h>
#endif

#include "inferx/cache/provider_memory.h"
#include "kvc/cache.h"
#include "test_util.h"

namespace {

using namespace kvc_test;
using namespace kvc;
using namespace inferx;

void TestBorrowedStorageRejected(const char* path) {
  auto opened = OpenProvider({path, {}});
  CHECK(opened);
  auto& provider = **opened;
  ConfigBuilder builder;
  auto config = builder.Build();
  CHECK(provider.Configure(config.config).Ok());

  std::vector<unsigned char> bytes(256);
  StoragePtr borrowed = Storage::Borrow(bytes.data(), bytes.size(), DeviceId::Cpu());
  auto rejected = ImportProviderMemory(provider.Data(), borrowed);
  CHECK(!rejected);
  CHECK_EQ(rejected.error().code, KVC_INVALID_ARGUMENT);
  CHECK(provider.Shutdown().Ok());
}

void TestHostStorageRoundTrip(const char* path) {
  auto opened = OpenProvider({path, {}});
  CHECK(opened);
  auto& provider = **opened;
  ConfigBuilder builder;
  auto config = builder.Build();
  CHECK(provider.Configure(config.config).Ok());

  auto allocated = Storage::Allocate(4096, DeviceId::Cpu());
  CHECK(allocated.ok());
  StoragePtr storage = std::move(*allocated);
  auto* raw_storage = reinterpret_cast<unsigned char*>(storage->Data());
  auto imported = ImportProviderMemory(provider.Data(), storage);
  CHECK(imported);
  CHECK_EQ(imported->Info()->memory_type, KVC_HOST);
  // The adapter retained the storage: dropping our handle must not affect
  // the region.
  storage.Reset();

  const uint64_t tokens = builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 311)};
  // Everything that retains provider objects is scoped so the final
  // shutdown observes a drained session.
  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(40001);
  request.blocks = blocks.data();
  request.block_count = blocks.size();
  auto write = provider.Control().BeginWrite(request);
  CHECK(write);
  BindingBuilder binder;
  binder.AddBlock(blocks[0]);
  for (uint32_t layer : builder.layers) binder.AddLayer(layer);
  binder.Bind(*imported, builder);
  auto store_request = binder.Request();
  store_request.selection.group = builder.group_id;
  // Fill the engine storage with the pattern the block will carry.
  uint64_t fill = 0;
  for (uint32_t layer : builder.layers)
    for (const auto& component : builder.components) {
      FillPattern(raw_storage + fill, blocks[0], layer, component.id,
                  component.bytes_per_block);
      fill += component.bytes_per_block;
    }
  auto stored = provider.Data().Store(*write, store_request);
  CHECK(stored);
  CHECK(stored->Wait().Ok());
  CHECK(provider.Control().Commit(*write));

  LookupOne one(blocks, tokens);
  auto hit = provider.Control().Lookup(one.request);
  CHECK(hit);
  CHECK(hit->read.Valid());
  auto destination = HostBuffer::Import(provider.Data(), 4096);
  CHECK(destination);
  BindingBuilder loader;
  loader.AddBlock(blocks[0]);
  for (uint32_t layer : builder.layers) loader.AddLayer(layer);
  loader.Bind(destination->region, builder);
  auto load_request = loader.Request();
  load_request.selection.group = builder.group_id;
  auto loaded = provider.Data().Load(hit->read, load_request);
  CHECK(loaded);
  CHECK(loaded->Wait().Ok());
  // The loaded copy must equal the engine storage the block was stored from.
  uint64_t offset = 0;
  for (uint32_t layer : builder.layers)
    for (const auto& component : builder.components) {
      CHECK(CheckPattern(raw_storage + offset, blocks[0], layer, component.id,
                         component.bytes_per_block));
      CHECK(CheckPattern(destination->bytes.data() + offset, blocks[0], layer, component.id,
                         component.bytes_per_block));
      offset += component.bytes_per_block;
    }
  imported = MemoryRegion();
  destination = Result<HostBuffer>{};
  write = Result<WriteHandle>{};
  stored = Result<TransferHandle>{};
  loaded = Result<TransferHandle>{};
  hit = Result<LookupResult>{};
  CHECK(provider.Shutdown().Ok());
}

#if defined(INFERX_TEST_CUDA)
void TestCudaStorageRoundTrip(const char* path) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0) return;
  auto opened = OpenProvider({path, {}});
  CHECK(opened);
  auto& provider = **opened;
  ConfigBuilder builder;
  auto config = builder.Build();
  CHECK(provider.Configure(config.config).Ok());

  auto allocated = Storage::Allocate(4096, DeviceId::Cuda(0));
  CHECK(allocated.ok());
  StoragePtr storage = std::move(*allocated);
  auto imported = ImportProviderMemory(provider.Data(), storage);
  CHECK(imported);
  CHECK_EQ(imported->Info()->memory_type, KVC_DEVICE);
  storage.Reset();

  const uint64_t tokens = builder.tokens_per_block;
  std::vector<KvcBlock> blocks{BlockAt(0, 0, tokens, tokens, 321)};
  std::vector<unsigned char> staging(4096, 0);
  uint64_t offset = 0;
  for (uint32_t layer : builder.layers)
    for (const auto& component : builder.components) {
      FillPattern(staging.data() + offset, blocks[0], layer, component.id,
                  component.bytes_per_block);
      offset += component.bytes_per_block;
    }
  CHECK(cudaMemcpy(reinterpret_cast<void*>(imported->Info()->local_address), staging.data(),
                   4096, cudaMemcpyHostToDevice) == cudaSuccess);
  CHECK(cudaStreamSynchronize(nullptr) == cudaSuccess);

  KvcWriteRequest request{};
  request.struct_size = sizeof(request);
  request.transaction = TxIdOf(40002);
  request.blocks = blocks.data();
  request.block_count = blocks.size();
  auto write = provider.Control().BeginWrite(request);
  CHECK(write);
  BindingBuilder binder;
  binder.AddBlock(blocks[0]);
  for (uint32_t layer : builder.layers) binder.AddLayer(layer);
  binder.Bind(*imported, builder);
  auto store_request = binder.Request();
  store_request.selection.group = builder.group_id;
  auto stored = provider.Data().Store(*write, store_request);
  CHECK(stored);
  CHECK(stored->Wait().Ok());
  CHECK(provider.Control().Commit(*write));

  LookupOne one(blocks, tokens);
  auto hit = provider.Control().Lookup(one.request);
  CHECK(hit);
  CHECK(hit->read.Valid());
  auto destination = HostBuffer::Import(provider.Data(), 4096);
  CHECK(destination);
  BindingBuilder loader;
  loader.AddBlock(blocks[0]);
  for (uint32_t layer : builder.layers) loader.AddLayer(layer);
  loader.Bind(destination->region, builder);
  auto load_request = loader.Request();
  load_request.selection.group = builder.group_id;
  auto loaded = provider.Data().Load(hit->read, load_request);
  CHECK(loaded);
  CHECK(loaded->Wait().Ok());
  offset = 0;
  for (uint32_t layer : builder.layers)
    for (const auto& component : builder.components) {
      CHECK(CheckPattern(destination->bytes.data() + offset, blocks[0], layer, component.id,
                         component.bytes_per_block));
      offset += component.bytes_per_block;
    }
  imported = MemoryRegion();
  destination = Result<HostBuffer>{};
  write = Result<WriteHandle>{};
  stored = Result<TransferHandle>{};
  loaded = Result<TransferHandle>{};
  hit = Result<LookupResult>{};
  CHECK(provider.Shutdown().Ok());
}
#endif

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <libkvc_host.so> [libkvc_cuda.so]\n", argv[0]);
    return 2;
  }
  RunTest("BorrowedStorageRejected", [path = argv[1]] { TestBorrowedStorageRejected(path); });
  RunTest("HostStorageRoundTrip", [path = argv[1]] { TestHostStorageRoundTrip(path); });
#if defined(INFERX_TEST_CUDA)
  if (const char* cuda_path = std::getenv("KVC_ADAPTER_CUDA"))
    RunTest("CudaStorageRoundTrip", [cuda_path] { TestCudaStorageRoundTrip(cuda_path); });
#endif
  return ExitCode();
}
