//==- llvm/ADT/IntrusiveRefCntPtr.h - Smart Refcounting Pointer --*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the RefCountedBase, ThreadSafeRefCountedBase, and
/// IntrusiveRefCntPtr classes.
///
/// IntrusiveRefCntPtr is a smart pointer to an object which maintains a
/// reference count.  (ThreadSafe)RefCountedBase is a mixin class that adds a
/// refcount member variable and methods for updating the refcount.  An object
/// that inherits from (ThreadSafe)RefCountedBase deletes itself when its
/// refcount hits zero.
///
/// For example:
///
/// ```
///   class MyClass : public RefCountedBase<MyClass> {};
///
///   void foo() {
///     // Constructing an IntrusiveRefCntPtr increases the pointee's refcount
///     // by 1 (from 0 in this case).
///     IntrusiveRefCntPtr<MyClass> Ptr1(new MyClass());
///
///     // Copying an IntrusiveRefCntPtr increases the pointee's refcount by 1.
///     IntrusiveRefCntPtr<MyClass> Ptr2(Ptr1);
///
///     // Constructing an IntrusiveRefCntPtr has no effect on the object's
///     // refcount.  After a move, the moved-from pointer is null.
///     IntrusiveRefCntPtr<MyClass> Ptr3(std::move(Ptr1));
///     assert(Ptr1 == nullptr);
///
///     // Clearing an IntrusiveRefCntPtr decreases the pointee's refcount by 1.
///     Ptr2.reset();
///
///     // The object deletes itself when we return from the function, because
///     // Ptr3's destructor decrements its refcount to 0.
///   }
/// ```
///
/// You can use IntrusiveRefCntPtr with isa<T>(), dyn_cast<T>(), etc.:
///
/// ```
///   IntrusiveRefCntPtr<MyClass> Ptr(new MyClass());
///   OtherClass *Other = dyn_cast<OtherClass>(Ptr);  // Ptr.Get() not required
/// ```
///
/// IntrusiveRefCntPtr works with any class that
///
///  - inherits from (ThreadSafe)RefCountedBase,
///  - has Retain() and Release() methods, or
///  - specializes IntrusiveRefCntPtrInfo.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <memory>

namespace inferx {

/// A CRTP mixin class that adds reference counting to a type.
///
/// The lifetime of an object which inherits from RefCountedBase is managed by
/// calls to Release() and Retain(), which increment and decrement the object's
/// refcount, respectively.  When a Release() call decrements the refcount to 0,
/// the object deletes itself.
template <class Derived>
class RefCountedBase {
  mutable unsigned RefCount = 0;

 protected:
  RefCountedBase() = default;
  RefCountedBase(const RefCountedBase&) {}
  RefCountedBase& operator=(const RefCountedBase&) = delete;

#ifndef NDEBUG
  ~RefCountedBase() {
    assert(RefCount == 0 &&
           "Destruction occurred when there are still references to this.");
  }
#else
  // Default the destructor in release builds, A trivial destructor may enable
  // better codegen.
  ~RefCountedBase() = default;
#endif

 public:
  unsigned UseCount() const { return RefCount; }

  void Retain() const { ++RefCount; }

  void Release() const {
    assert(RefCount > 0 && "Reference count is already zero.");
    if (--RefCount == 0) delete static_cast<const Derived*>(this);
  }
};

/// A thread-safe version of \c RefCountedBase.
template <typename Derived>
class ThreadSafeRefCountedBase {
 protected:
  ThreadSafeRefCountedBase() = default;
  ThreadSafeRefCountedBase(const ThreadSafeRefCountedBase&) {}
  ThreadSafeRefCountedBase& operator=(const ThreadSafeRefCountedBase&) = delete;

#ifndef NDEBUG
  ~ThreadSafeRefCountedBase() {
    assert(ref_count_ == 0 &&
           "Destruction occurred when there are still references to this.");
  }
#else
  // Default the destructor in release builds, A trivial destructor may enable
  // better codegen.
  ~ThreadSafeRefCountedBase() = default;
#endif

 public:
  unsigned UseCount() const {
    return ref_count_.load(std::memory_order_relaxed);
  }

  void Retain() const { ref_count_.fetch_add(1, std::memory_order_relaxed); }

  void Release() const {
    int new_ref_count = ref_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;

    assert(new_ref_count >= 0 && "Reference count was already zero.");

    if (new_ref_count == 0) delete static_cast<const Derived*>(this);
  }

 private:
  mutable std::atomic<int> ref_count_{0};
};

/// Class you can specialize to provide custom retain/release functionality for
/// a type.
///
/// Usually specializing this class is not necessary, as IntrusiveRefCntPtr
/// works with any type which defines Retain() and Release() functions -- you
/// can define those functions yourself if RefCountedBase doesn't work for you.
///
/// One case when you might want to specialize this type is if you have
///  - Foo.h defines type Foo and includes Bar.h, and
///  - Bar.h uses IntrusiveRefCntPtr<Foo> in inline functions.
///
/// Because Foo.h includes Bar.h, Bar.h can't include Foo.h in order to pull in
/// the declaration of Foo.  Without the declaration of Foo, normally Bar.h
/// wouldn't be able to use IntrusiveRefCntPtr<Foo>, which wants to call
/// T::Retain and T::Release.
///
/// To resolve this, Bar.h could include a third header, FooFwd.h, which
/// forward-declares Foo and specializes IntrusiveRefCntPtrInfo<Foo>.  Then
/// Bar.h could use IntrusiveRefCntPtr<Foo>, although it still couldn't call any
/// functions on Foo itself, because Foo would be an incomplete type.
template <typename T>
struct IntrusiveRefCntPtrInfo {
  static unsigned UseCount(const T* obj) { return obj->UseCount(); }
  static void Retain(T* obj) { obj->Retain(); }
  static void Release(T* obj) { obj->Release(); }
};

/// A smart pointer to a reference-counted object that inherits from
/// RefCountedBase or ThreadSafeRefCountedBase.
///
/// This class increments its pointee's reference count when it is created, and
/// decrements its refcount when it's destroyed (or is changed to point to a
/// different object).
template <typename T>
class IntrusiveRefCntPtr {
 public:
  using element_type = T;

  explicit IntrusiveRefCntPtr() = default;
  IntrusiveRefCntPtr(T* obj) : obj_(obj) { Retain(); }
  IntrusiveRefCntPtr(const IntrusiveRefCntPtr& other) : obj_(other.obj_) {
    Retain();
  }

  IntrusiveRefCntPtr(IntrusiveRefCntPtr&& other) : obj_(other.obj_) {
    other.obj_ = nullptr;
  }

  template <class X,
            std::enable_if_t<std::is_convertible<X*, T*>::value, bool> = true>
  IntrusiveRefCntPtr(IntrusiveRefCntPtr<X> other) : obj_(other.Get()) {
    other.obj_ = nullptr;
  }

  template <class X,
            std::enable_if_t<std::is_convertible<X*, T*>::value, bool> = true>
  IntrusiveRefCntPtr(std::unique_ptr<X> other) : obj_(other.release()) {
    Retain();
  }

  ~IntrusiveRefCntPtr() { Release(); }

  IntrusiveRefCntPtr& operator=(IntrusiveRefCntPtr other) {
    Swap(other);

    return *this;
  }

  T& operator*() const { return *obj_; }
  T* operator->() const { return obj_; }
  T* Get() const { return obj_; }
  explicit operator bool() const { return obj_; }

  void Swap(IntrusiveRefCntPtr& other) {
    T* tmp = other.obj_;
    other.obj_ = obj_;
    obj_ = tmp;
  }

  void Reset() {
    Release();
    obj_ = nullptr;
  }

  void ResetWithoutRelease() { obj_ = nullptr; }

  unsigned UseCount() const {
    return obj_ ? IntrusiveRefCntPtrInfo<T>::UseCount(obj_) : 0;
  }

 private:
  template <typename X>
  friend class IntrusiveRefCntPtr;

  void Retain() {
    if (obj_) IntrusiveRefCntPtrInfo<T>::Retain(obj_);
  }

  void Release() {
    if (obj_) IntrusiveRefCntPtrInfo<T>::Release(obj_);
  }

  T* obj_ = nullptr;
};

template <class T, class U>
inline bool operator==(const IntrusiveRefCntPtr<T>& A,
                       const IntrusiveRefCntPtr<U>& B) {
  return A.Get() == B.Get();
}

template <class T, class U>
inline bool operator!=(const IntrusiveRefCntPtr<T>& A,
                       const IntrusiveRefCntPtr<U>& B) {
  return A.Get() != B.Get();
}

template <class T, class U>
inline bool operator==(const IntrusiveRefCntPtr<T>& A, U* B) {
  return A.Get() == B;
}

template <class T, class U>
inline bool operator!=(const IntrusiveRefCntPtr<T>& A, U* B) {
  return A.Get() != B;
}

template <class T, class U>
inline bool operator==(T* A, const IntrusiveRefCntPtr<U>& B) {
  return A == B.Get();
}

template <class T, class U>
inline bool operator!=(T* A, const IntrusiveRefCntPtr<U>& B) {
  return A != B.Get();
}

template <class T>
bool operator==(std::nullptr_t, const IntrusiveRefCntPtr<T>& B) {
  return !B;
}

template <class T>
bool operator==(const IntrusiveRefCntPtr<T>& A, std::nullptr_t B) {
  return B == A;
}

template <class T>
bool operator!=(std::nullptr_t A, const IntrusiveRefCntPtr<T>& B) {
  return !(A == B);
}

template <class T>
bool operator!=(const IntrusiveRefCntPtr<T>& A, std::nullptr_t B) {
  return !(A == B);
}

// Make IntrusiveRefCntPtr work with dyn_cast, isa, and the other idioms from
// Casting.h.
template <typename From>
struct simplify_type;

template <class T>
struct simplify_type<IntrusiveRefCntPtr<T>> {
  using SimpleType = T*;

  static SimpleType getSimplifiedValue(IntrusiveRefCntPtr<T>& Val) {
    return Val.Get();
  }
};

template <class T>
struct simplify_type<const IntrusiveRefCntPtr<T>> {
  using SimpleType = /*const*/ T*;

  static SimpleType getSimplifiedValue(const IntrusiveRefCntPtr<T>& Val) {
    return Val.Get();
  }
};

/// Factory function for creating intrusive ref counted pointers.
template <typename T, typename... Args>
IntrusiveRefCntPtr<T> makeIntrusiveRefCnt(Args&&... A) {
  return IntrusiveRefCntPtr<T>(new T(std::forward<Args>(A)...));
}

}  // namespace inferx
