#include "inferx/core/storage.h"

#include "absl/strings/str_cat.h"

namespace inferx {

StatusOr<StoragePtr> Storage::Allocate(size_t bytes, Allocator* allocator) {
  if (allocator == nullptr) return InvalidArgumentError("null allocator");

  INFERX_ASSIGN_OR_RETURN(void* p, allocator->Allocate(bytes));

  return StoragePtr(new Storage(p, bytes, allocator->Device(), allocator));
}

StatusOr<StoragePtr> Storage::Allocate(size_t bytes, DeviceId device) {
  INFERX_ASSIGN_OR_RETURN(Allocator * alloc, AllocatorFor(device));

  return Allocate(bytes, alloc);
}

StoragePtr Storage::Borrow(void* data, size_t bytes, DeviceId device) {
  return StoragePtr(new Storage(data, bytes, device, nullptr));
}

Storage::~Storage() {
  if (allocator_ != nullptr) {
    // TODO(gc): a failure here means a real bug (double free or a foreign
    // pointer) and is currently discarded. This is the destructor site the
    // logger discussion is about.
    allocator_->Deallocate(data_).IgnoreError();
  }

  data_ = nullptr;
  size_ = 0;
}

std::string Storage::ToString() const {
  return absl::StrCat("Storage(", size_, "B, ", device_.ToString(), ", ",
                      IsBorrowed() ? "borrowed" : allocator_->Name(), ")");
}

}  // namespace inferx
