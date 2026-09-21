//==- intrusive_ref_cnt_ptr.h - Smart Refcounting Pointer --*- C++ -*-==//
//
// Adapted for InferX from the LLVM Project's llvm/ADT/IntrusiveRefCntPtr.h,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// InferX adaptations: everything lives in namespace inferx, the accessors use
// the project's camelCase method names (Get/UseCount/Reset), and a
// makeIntrusiveRefCnt factory is provided. The reference-counting semantics
// are unchanged from upstream.
//
//===----------------------------------------------------------------------===//

#ifndef INFERX_CORE_INTRUSIVE_REF_CNT_PTR_H_
#define INFERX_CORE_INTRUSIVE_REF_CNT_PTR_H_

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace inferx {

/// \brief A CRTP base class for objects with a non-atomic reference count.
///
/// A class derives from `RefCountedBase<T>` to gain `Retain()`/`Release()`
/// support for use with `IntrusiveRefCntPtr`. Not thread-safe; use
/// `ThreadSafeRefCountedBase` when references cross threads.
template <class Derived>
class RefCountedBase {
 public:
  RefCountedBase() = default;
  RefCountedBase(const RefCountedBase&) : RefCount(0) {}
  RefCountedBase& operator=(const RefCountedBase&) {
    // A copy assignment does not borrow or release references: the count
    // tracks handles to *this* object, which assignment does not change.
    return *this;
  }
  ~RefCountedBase() {
    // The destructor asserts in debug builds when references remain.
    assert(RefCount == 0 &&
           "Destruction occurred when there are still references to this.");
  }

  void Retain() const { ++RefCount; }
  void Release() const {
    assert(RefCount > 0 && "Releasing a reference that does not exist.");
    if (--RefCount == 0) {
      delete static_cast<const Derived*>(this);
    }
  }

  /// \brief Returns the current reference count; approximate under
  ///        concurrency.
  uint32_t UseCount() const { return RefCount; }

 private:
  mutable uint32_t RefCount = 0;
};

/// \brief A CRTP base class for objects with an atomic reference count.
///
/// The thread-safe counterpart of `RefCountedBase`: `Release()` deletes the
/// object when the last reference drops, whatever thread that happens on.
template <class Derived>
class ThreadSafeRefCountedBase {
  mutable std::atomic<uint32_t> RefCount;

 protected:
  ThreadSafeRefCountedBase() : RefCount(0) {}
  ThreadSafeRefCountedBase(const ThreadSafeRefCountedBase&) : RefCount(0) {}
  ThreadSafeRefCountedBase& operator=(const ThreadSafeRefCountedBase&) {
    return *this;
  }
#ifndef NDEBUG
  ~ThreadSafeRefCountedBase() {
    // Default the destructor in release builds. A trivial destructor may
    // enable better codegen.
    assert(RefCount.load(std::memory_order_relaxed) == 0 &&
           "Destruction occurred when there are still references to this.");
  }
#else
  ~ThreadSafeRefCountedBase() = default;
#endif

 public:
  void Retain() const {
    RefCount.fetch_add(1, std::memory_order_relaxed);
  }

  void Release() const {
    if (RefCount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete static_cast<const Derived*>(this);
    }
  }

  /// \brief Returns the current reference count; approximate under
  ///        concurrency.
  uint32_t UseCount() const {
    return RefCount.load(std::memory_order_relaxed);
  }
};

/// \brief A reference-counting pointer to an object with intrusive Retain and
///        Release methods.
///
/// `IntrusiveRefCntPtr` is a single-pointer handle: copying it retains,
/// destroying it releases, and `sizeof(IntrusiveRefCntPtr<T>) == sizeof(T*)`.
/// The pointee controls its own destruction through `Release()`, which is what
/// lets `Tensor` stay a single word while still sharing ownership.
template <class T>
class IntrusiveRefCntPtr {
 public:
  /// \brief Constructs a null pointer.
  IntrusiveRefCntPtr() = default;
  /// \brief Constructs a null pointer from nullptr.
  IntrusiveRefCntPtr(std::nullptr_t) {}
  /// \brief Constructs a pointer that retains `obj`, which may be null.
  explicit IntrusiveRefCntPtr(T* obj) : Obj(obj) {
    if (Obj) Obj->Retain();
  }

  /// \brief Copy-constructs, retaining the pointee.
  IntrusiveRefCntPtr(const IntrusiveRefCntPtr& other) : Obj(other.Obj) {
    if (Obj) Obj->Retain();
  }

  /// \brief Move-constructs, leaving `other` null.
  IntrusiveRefCntPtr(IntrusiveRefCntPtr&& other) noexcept
      : Obj(other.Obj) {
    other.Obj = nullptr;
  }

  /// \brief Releases the pointee, if any.
  ~IntrusiveRefCntPtr() { Release(); }

  /// \brief Copy-assigns, retaining the new and releasing the old pointee.
  IntrusiveRefCntPtr& operator=(const IntrusiveRefCntPtr& other) {
    if (this != &other) {
      RetainAndReplace(other.Obj);
    }
    return *this;
  }

  /// \brief Move-assigns, releasing the old pointee and emptying `other`.
  IntrusiveRefCntPtr& operator=(IntrusiveRefCntPtr&& other) noexcept {
    if (this != &other) {
      Release();
      Obj = other.Obj;
      other.Obj = nullptr;
    }
    return *this;
  }

  /// \brief Releases the pointee and takes ownership of `obj`, retaining it.
  IntrusiveRefCntPtr& operator=(T* obj) {
    RetainAndReplace(obj);
    return *this;
  }

  /// \brief Releases the pointee and becomes null.
  IntrusiveRefCntPtr& operator=(std::nullptr_t) {
    Release();
    return *this;
  }

  /// \brief Returns the raw pointer without changing the reference count.
  T* Get() const { return Obj; }

  /// \brief Dereferences the pointee; must not be null.
  T& operator*() const { return *Get(); }

  /// \brief Returns the raw pointer for member access; must not be null.
  T* operator->() const { return Get(); }

  /// \brief True when the pointer is non-null.
  explicit operator bool() const { return Obj != nullptr; }

  /// \brief Releases the current pointee and takes `obj`, retaining it.
  void Reset(T* obj = nullptr) { RetainAndReplace(obj); }

  /// \brief Returns the current reference count; approximate under
  ///        concurrency.
  uint32_t UseCount() const { return Obj ? Obj->UseCount() : 0; }

  /// \brief Swaps two pointers without touching either reference count.
  void Swap(IntrusiveRefCntPtr& other) noexcept { std::swap(Obj, other.Obj); }

  /// \brief Compares the raw pointers.
  friend bool operator==(const IntrusiveRefCntPtr& a,
                         const IntrusiveRefCntPtr& b) {
    return a.Get() == b.Get();
  }
  friend bool operator!=(const IntrusiveRefCntPtr& a,
                         const IntrusiveRefCntPtr& b) {
    return !(a == b);
  }
  friend bool operator==(const IntrusiveRefCntPtr& a, std::nullptr_t) {
    return a.Get() == nullptr;
  }
  friend bool operator!=(const IntrusiveRefCntPtr& a, std::nullptr_t) {
    return a.Get() != nullptr;
  }
  friend bool operator==(std::nullptr_t, const IntrusiveRefCntPtr& b) {
    return nullptr == b.Get();
  }
  friend bool operator!=(std::nullptr_t, const IntrusiveRefCntPtr& b) {
    return nullptr != b.Get();
  }

 private:
  /// \brief Retains `obj` first, then releases the old pointee: safe even when
  ///        `obj` aliases the current pointee or shares ownership with another
  ///        handle whose release would destroy it.
  void RetainAndReplace(T* obj) {
    if (obj) obj->Retain();
    Release();
    Obj = obj;
  }

  /// \brief Releases the pointee, if any, and clears the raw pointer.
  void Release() {
    if (Obj) {
      T* tmp = Obj;
      Obj = nullptr;
      tmp->Release();
    }
  }

  T* Obj = nullptr;
};

/// \brief Constructs an object of type `T` and returns it wrapped in an
///        `IntrusiveRefCntPtr`.
///
/// \param args Arguments forwarded to T's constructor.
/// \return     A new reference to the constructed object.
template <typename T, typename... Args>
IntrusiveRefCntPtr<T> makeIntrusiveRefCnt(Args&&... args) {
  return IntrusiveRefCntPtr<T>(new T(std::forward<Args>(args)...));
}

}  // namespace inferx

#endif  // INFERX_CORE_INTRUSIVE_REF_CNT_PTR_H_
