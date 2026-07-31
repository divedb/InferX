#include "inferx/model/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <utility>

#include "absl/strings/str_cat.h"
#include "inferx/common/json.h"
#include "inferx/core/storage.h"

namespace inferx::model {
namespace {

// The header is a little-endian u64 length followed by that many bytes of JSON.
// Data begins immediately after, and every tensor's offsets are relative to
// that point rather than to the file.
constexpr size_t kHeaderLenBytes = 8;

// A sanity bound, not a format limit. Qwen2.5-3B's header is 30 KB; anything
// past this means the length prefix was misread and we are about to allocate
// something absurd from a corrupt file.
constexpr size_t kMaxHeaderBytes = 256u << 20;

Status SystemError(std::string_view what, std::string_view path) {
  return InternalError(what, " '", path, "': ", std::strerror(errno));
}

/// One mmap'd file, unmapped on destruction.
class MappedFile {
 public:
  static StatusOr<std::unique_ptr<MappedFile>> Open(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return SystemError("cannot open", path);

    struct stat st {};
    if (::fstat(fd, &st) != 0) {
      const Status s = SystemError("cannot stat", path);
      ::close(fd);
      return s;
    }

    const size_t size = static_cast<size_t>(st.st_size);
    if (size < kHeaderLenBytes) {
      ::close(fd);
      return InvalidArgumentError("'", path, "' is too small (", size,
                                  " bytes) to be a safetensors file");
    }

    void* addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);

    // The descriptor is not needed once the mapping exists, and holding 434
    // of them open across a sharded checkpoint would be pointless.
    ::close(fd);

    if (addr == MAP_FAILED) return SystemError("cannot mmap", path);

    // Advisory: weights are read once, in order, at load. Telling the kernel so
    // lets it read ahead instead of faulting page by page.
    ::madvise(addr, size, MADV_SEQUENTIAL);

    return std::unique_ptr<MappedFile>(new MappedFile(addr, size, path));
  }

  ~MappedFile() {
    if (addr_ != nullptr) ::munmap(addr_, size_);
  }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const std::byte* data() const { return static_cast<const std::byte*>(addr_); }
  size_t size() const { return size_; }
  const std::string& path() const { return path_; }

 private:
  MappedFile(void* addr, size_t size, std::string path)
      : addr_(addr), size_(size), path_(std::move(path)) {}

  void* addr_ = nullptr;
  size_t size_ = 0;
  std::string path_;
};

StatusOr<std::string> ReadWholeFile(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return SystemError("cannot open", path);

  std::string out;
  char buf[65536];

  for (;;) {
    const ssize_t n = ::read(fd, buf, sizeof(buf));
    if (n < 0) {
      const Status s = SystemError("cannot read", path);
      ::close(fd);
      return s;
    }
    if (n == 0) break;
    out.append(buf, static_cast<size_t>(n));
  }

  ::close(fd);
  return out;
}

bool FileExists(const std::string& path) {
  struct stat st {};
  return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string JoinPath(std::string_view dir, std::string_view name) {
  std::string out(dir);
  if (!out.empty() && out.back() != '/') out.push_back('/');
  out.append(name);
  return out;
}

}  // namespace

StatusOr<DataType> SafetensorsDataType(std::string_view name) {
  // The format's own spelling, which is not ours. Only the types a checkpoint
  // can actually contain are listed; anything else is rejected rather than
  // guessed, because a wrong dtype here reinterprets the whole tensor.
  if (name == "F16")  return DataType::kFloat16;
  if (name == "BF16") return DataType::kBFloat16;
  if (name == "F32")  return DataType::kFloat;
  if (name == "F64")  return DataType::kDouble;
  if (name == "I8")   return DataType::kInt8;
  if (name == "U8")   return DataType::kUInt8;
  if (name == "I16")  return DataType::kInt16;
  if (name == "U16")  return DataType::kUInt16;
  if (name == "I32")  return DataType::kInt32;
  if (name == "U32")  return DataType::kUInt32;
  if (name == "I64")  return DataType::kInt64;
  if (name == "U64")  return DataType::kUInt64;
  if (name == "BOOL") return DataType::kBool;
  if (name == "F8_E4M3") return DataType::kFloat8E4M3FN;
  if (name == "F8_E5M2") return DataType::kFloat8E5M2;

  return InvalidArgumentError("unsupported safetensors dtype '", name, "'");
}

struct Checkpoint::Impl {
  std::vector<std::unique_ptr<MappedFile>> shards;
  std::vector<std::string> shard_paths;
  // Where each shard's tensor data starts, i.e. 8 + header length.
  std::vector<size_t> data_offsets;
  absl::flat_hash_map<std::string, TensorEntry> entries;

  /// Parses one shard's header and merges its entries into the flat map.
  Status AddShard(const std::string& path) {
    INFERX_ASSIGN_OR_RETURN(std::unique_ptr<MappedFile> file,
                            MappedFile::Open(path));

    uint64_t header_len = 0;
    std::memcpy(&header_len, file->data(), kHeaderLenBytes);

    if (header_len > kMaxHeaderBytes) {
      return InvalidArgumentError("'", path, "' declares a ", header_len,
                                  "-byte header, which exceeds the ",
                                  kMaxHeaderBytes, "-byte sanity limit");
    }

    if (kHeaderLenBytes + header_len > file->size()) {
      return InvalidArgumentError("'", path, "' declares a ", header_len,
                                  "-byte header but the file is only ",
                                  file->size(), " bytes");
    }

    const std::string_view header_text(
        reinterpret_cast<const char*>(file->data() + kHeaderLenBytes),
        header_len);

    INFERX_ASSIGN_OR_RETURN(const JsonValue header, ParseJson(header_text));
    INFERX_ASSIGN_OR_RETURN(const auto* fields, header.AsObject());

    const size_t data_start = kHeaderLenBytes + header_len;
    const size_t data_bytes = file->size() - data_start;
    const int shard_index = static_cast<int>(shards.size());

    for (const auto& [name, value] : *fields) {
      // Publisher metadata (`{"format": "pt"}`), not a tensor.
      if (name == "__metadata__") continue;

      INFERX_ASSIGN_OR_RETURN(const auto* fields_of, value.AsObject());
      (void)fields_of;

      INFERX_ASSIGN_OR_RETURN(const std::string_view dtype_name,
                              value.RequiredString("dtype"));
      INFERX_ASSIGN_OR_RETURN(const DataType dtype,
                              SafetensorsDataType(dtype_name));

      const JsonValue* shape_value = value.Find("shape");
      if (shape_value == nullptr) {
        return InvalidArgumentError("tensor '", name, "' has no shape");
      }
      INFERX_ASSIGN_OR_RETURN(const auto* dims_json, shape_value->AsArray());

      std::vector<int64_t> dims;
      dims.reserve(dims_json->size());
      for (const JsonValue& d : *dims_json) {
        INFERX_ASSIGN_OR_RETURN(const int64_t extent, d.AsInt());
        if (extent < 0) {
          return InvalidArgumentError("tensor '", name, "' has a negative "
                                      "extent ", extent);
        }
        dims.push_back(extent);
      }

      const JsonValue* offsets_value = value.Find("data_offsets");
      if (offsets_value == nullptr) {
        return InvalidArgumentError("tensor '", name, "' has no data_offsets");
      }
      INFERX_ASSIGN_OR_RETURN(const auto* offsets, offsets_value->AsArray());

      if (offsets->size() != 2) {
        return InvalidArgumentError("tensor '", name, "' has ", offsets->size(),
                                    " data_offsets, expected 2");
      }

      INFERX_ASSIGN_OR_RETURN(const int64_t begin, (*offsets)[0].AsInt());
      INFERX_ASSIGN_OR_RETURN(const int64_t end, (*offsets)[1].AsInt());

      if (begin < 0 || end < begin) {
        return InvalidArgumentError("tensor '", name, "' has inverted offsets [",
                                    begin, ", ", end, ")");
      }

      // Bounds-checked against the mapping before anything is allowed to point
      // into it. This is the check that turns a corrupt or truncated download
      // into an error at load instead of a segfault mid-forward-pass.
      if (static_cast<size_t>(end) > data_bytes) {
        return InvalidArgumentError("tensor '", name, "' ends at ", end,
                                    " but '", path, "' has only ", data_bytes,
                                    " bytes of tensor data");
      }

      TensorEntry entry;
      entry.dtype = dtype;
      entry.shape = Shape(absl::MakeConstSpan(dims));
      entry.begin = static_cast<size_t>(begin);
      entry.end = static_cast<size_t>(end);
      entry.shard = shard_index;

      // The header says how many bytes it occupies and the shape says how many
      // it should. Disagreement means one of them is being misread, and it is
      // worth failing on rather than trusting either.
      const size_t expected =
          static_cast<size_t>(DataTypeByteSize(dtype, entry.shape.Numel()));
      if (entry.nbytes() != expected) {
        return InvalidArgumentError(
            "tensor '", name, "' spans ", entry.nbytes(), " bytes but its ",
            DataTypeName(dtype), " shape ", entry.shape.ToString(),
            " needs ", expected);
      }

      if (!entries.emplace(name, std::move(entry)).second) {
        return InvalidArgumentError("tensor '", name,
                                    "' appears in more than one shard");
      }
    }

    shard_paths.push_back(path);
    data_offsets.push_back(data_start);
    shards.push_back(std::move(file));

    return OkStatus();
  }
};

StatusOr<Checkpoint> Checkpoint::OpenFile(std::string_view path) {
  auto impl = std::make_unique<Impl>();
  INFERX_RETURN_IF_ERROR(impl->AddShard(std::string(path)));
  return Checkpoint(std::move(impl));
}

StatusOr<Checkpoint> Checkpoint::Open(std::string_view dir) {
  const std::string index_path = JoinPath(dir, "model.safetensors.index.json");
  const std::string single_path = JoinPath(dir, "model.safetensors");

  auto impl = std::make_unique<Impl>();

  if (FileExists(index_path)) {
    INFERX_ASSIGN_OR_RETURN(const std::string text, ReadWholeFile(index_path));
    INFERX_ASSIGN_OR_RETURN(const JsonValue index, ParseJson(text));

    const JsonValue* weight_map = index.Find("weight_map");
    if (weight_map == nullptr) {
      return InvalidArgumentError("'", index_path, "' has no weight_map");
    }
    INFERX_ASSIGN_OR_RETURN(const auto* mapping, weight_map->AsObject());

    // The index maps every tensor to its shard; we only need the distinct set
    // of shards, since each shard's own header is authoritative about what it
    // contains. Sorted so shard indices are stable across runs, which keeps
    // error messages and test expectations reproducible.
    std::vector<std::string> files;
    for (const auto& [tensor_name, file_value] : *mapping) {
      INFERX_ASSIGN_OR_RETURN(const std::string_view file, file_value.AsString());
      (void)tensor_name;
      files.emplace_back(file);
    }

    std::sort(files.begin(), files.end());
    files.erase(std::unique(files.begin(), files.end()), files.end());

    if (files.empty()) {
      return InvalidArgumentError("'", index_path, "' lists no shards");
    }

    for (const std::string& file : files) {
      const std::string shard = JoinPath(dir, file);
      if (!FileExists(shard)) {
        return NotFoundError("shard '", shard, "' named by ", index_path,
                             " does not exist");
      }
      INFERX_RETURN_IF_ERROR(impl->AddShard(shard));
    }

    return Checkpoint(std::move(impl));
  }

  if (FileExists(single_path)) {
    INFERX_RETURN_IF_ERROR(impl->AddShard(single_path));
    return Checkpoint(std::move(impl));
  }

  return NotFoundError("no model.safetensors or model.safetensors.index.json "
                       "in '", dir, "'");
}

Checkpoint::Checkpoint(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Checkpoint::~Checkpoint() = default;
Checkpoint::Checkpoint(Checkpoint&&) noexcept = default;
Checkpoint& Checkpoint::operator=(Checkpoint&&) noexcept = default;

const TensorEntry* Checkpoint::FindEntry(std::string_view name) const {
  const auto it = impl_->entries.find(name);
  return it == impl_->entries.end() ? nullptr : &it->second;
}

bool Checkpoint::Contains(std::string_view name) const {
  return FindEntry(name) != nullptr;
}

size_t Checkpoint::size() const { return impl_->entries.size(); }

const std::vector<std::string>& Checkpoint::shard_paths() const {
  return impl_->shard_paths;
}

size_t Checkpoint::TotalBytes() const {
  size_t total = 0;
  for (const auto& [name, entry] : impl_->entries) {
    (void)name;
    total += entry.nbytes();
  }
  return total;
}

std::vector<std::string> Checkpoint::Names() const {
  std::vector<std::string> names;
  names.reserve(impl_->entries.size());

  for (const auto& [name, entry] : impl_->entries) {
    (void)entry;
    names.push_back(name);
  }

  std::sort(names.begin(), names.end());
  return names;
}

StatusOr<Tensor> Checkpoint::Get(std::string_view name) const {
  const TensorEntry* entry = FindEntry(name);

  if (entry == nullptr) {
    return NotFoundError("checkpoint has no tensor '", name, "' (it has ",
                         impl_->entries.size(), " tensors)");
  }

  const MappedFile& file = *impl_->shards[static_cast<size_t>(entry->shard)];
  const size_t data_start = impl_->data_offsets[static_cast<size_t>(entry->shard)];

  // Borrowed, not copied: the bytes stay in the mapping and the OS pages them
  // in on demand. This is the whole point of the class (T17) -- a 3B checkpoint
  // becomes one mapping and a handle per tensor.
  //
  // const_cast because Storage's interface is not const-aware and the mapping
  // is PROT_READ. Nothing downstream writes to a checkpoint tensor; the upload
  // path reads from it into device memory.
  std::byte* const base =
      const_cast<std::byte*>(file.data()) + data_start + entry->begin;

  StoragePtr storage = Storage::Borrow(base, entry->nbytes(), DeviceId::Cpu());

  return Tensor::FromStorage(std::move(storage), 0, entry->dtype, entry->shape);
}

StatusOr<Tensor> Checkpoint::GetChecked(std::string_view name,
                                        const Shape& expected) const {
  INFERX_ASSIGN_OR_RETURN(Tensor t, Get(name));

  if (t.shape() != expected) {
    return InvalidArgumentError("tensor '", name, "' has shape ",
                                t.shape().ToString(), ", expected ",
                                expected.ToString());
  }

  return t;
}

}  // namespace inferx::model
