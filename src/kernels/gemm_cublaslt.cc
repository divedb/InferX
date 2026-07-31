#include <cublasLt.h>

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "inferx/core/cuda_utils.h"
#include "inferx/kernels/gemm.h"

namespace inferx::kernels {
namespace {

// Spelled out rather than calling cublasGetStatusName/String, which live in
// libcublas -- linking the whole of cuBLAS to render two strings would widen the
// dependency for no benefit, and cuBLASLt is the only part of it we want.
const char* CublasStatusName(cublasStatus_t s) {
  switch (s) {
    case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
    default:                             return "CUBLAS_STATUS_UNKNOWN";
  }
}

// Mirrors CudaErrorToStatus: the status code carries what the caller can do
// about it, and the message carries where it happened.
Status CublasErrorToStatus(cublasStatus_t s, const char* expr, const char* file,
                           int line) {
  if (s == CUBLAS_STATUS_SUCCESS) return OkStatus();

  const std::string detail = absl::StrCat(CublasStatusName(s), " [", expr,
                                          " at ", file, ":", line, "]");

  switch (s) {
    case CUBLAS_STATUS_ALLOC_FAILED:
      return ResourceExhaustedError(detail);
    case CUBLAS_STATUS_NOT_INITIALIZED:
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return FailedPreconditionError(detail);
    case CUBLAS_STATUS_INVALID_VALUE:
      return InvalidArgumentError(detail);
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return UnimplementedError(detail);
    default:
      return InternalError(detail);
  }
}

#define INFERX_CUBLAS_RETURN_IF_ERROR(expr)                                 \
  do {                                                                      \
    cublasStatus_t _inferx_bl_status = (expr);                              \
    if (_inferx_bl_status != CUBLAS_STATUS_SUCCESS) [[unlikely]] {          \
      return ::inferx::kernels::CublasErrorToStatus(_inferx_bl_status,      \
                                                    #expr, __FILE__,        \
                                                    __LINE__);              \
    }                                                                       \
  } while (0)

struct ShapeKey {
  int64_t m, n, k;

  friend bool operator==(const ShapeKey& a, const ShapeKey& b) {
    return a.m == b.m && a.n == b.n && a.k == b.k;
  }

  template <typename H>
  friend H AbslHashValue(H h, const ShapeKey& s) {
    return H::combine(std::move(h), s.m, s.n, s.k);
  }
};

// One cached plan: the descriptors and the algorithm the heuristic chose.
//
// The descriptors are cached alongside the algorithm rather than rebuilt per
// call because they are the other half of what the heuristic was asked about --
// an algorithm is only valid for the layouts it was selected for, so caching
// one without the other would be caching half a decision.
struct Plan {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a = nullptr;  // weights,     col-major [k, n]
  cublasLtMatrixLayout_t b = nullptr;  // activations, col-major [k, m]
  cublasLtMatrixLayout_t c = nullptr;  // output,      col-major [n, m]
  cublasLtMatmulAlgo_t algo{};

  ~Plan() {
    // Destruction order does not matter to cuBLASLt, and each destroy tolerates
    // a null handle, so a partially-built plan cleans up correctly.
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (desc) cublasLtMatmulDescDestroy(desc);
  }

  Plan() = default;
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;
};

}  // namespace

struct CublasLtGemm::Impl {
  cublasLtHandle_t lt = nullptr;
  DeviceBuffer workspace;
  absl::flat_hash_map<ShapeKey, std::unique_ptr<Plan>> plans;

  ~Impl() {
    // Plans hold cuBLASLt descriptors and must be gone before the handle is.
    plans.clear();
    if (lt) cublasLtDestroy(lt);
  }

  StatusOr<const Plan*> GetOrBuild(int64_t m, int64_t n, int64_t k);
};

StatusOr<const Plan*> CublasLtGemm::Impl::GetOrBuild(int64_t m, int64_t n,
                                                     int64_t k) {
  const ShapeKey key{m, n, k};

  if (auto it = plans.find(key); it != plans.end()) return it->second.get();

  auto plan = std::make_unique<Plan>();

  // FP16 in, FP32 accumulate. The scale type is FP32 to match, which is what
  // makes alpha/beta below floats rather than halves.
  INFERX_CUBLAS_RETURN_IF_ERROR(
      cublasLtMatmulDescCreate(&plan->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  // The row-major-to-column-major mapping, which is the only subtle part here.
  //
  // Everything we own is row-major; cuBLASLt is column-major. A row-major
  // [r, c] buffer *is* a column-major [c, r] buffer with no data movement, so
  // rather than transposing anything we state the problem in the transposed
  // frame and let the output land in the layout we already wanted:
  //
  //   want (row-major)  y[m, n] = x[m, k] · w[n, k]ᵀ
  //   state (col-major) yᵀ[n, m] = wᵀ[n, k] · xᵀ[k, m]
  //
  // w is row-major [n, k], i.e. column-major [k, n], and OP_T turns that into
  // the [n, k] operand. x is row-major [m, k], i.e. column-major [k, m], which
  // is already the [k, m] operand, so OP_N. The result is column-major [n, m],
  // which is row-major [m, n] -- y, exactly as the caller declared it.
  const cublasOperation_t op_t = CUBLAS_OP_T;
  const cublasOperation_t op_n = CUBLAS_OP_N;

  INFERX_CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
      plan->desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)));
  INFERX_CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
      plan->desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

  INFERX_CUBLAS_RETURN_IF_ERROR(
      cublasLtMatrixLayoutCreate(&plan->a, CUDA_R_16F, k, n, k));
  INFERX_CUBLAS_RETURN_IF_ERROR(
      cublasLtMatrixLayoutCreate(&plan->b, CUDA_R_16F, k, m, k));
  INFERX_CUBLAS_RETURN_IF_ERROR(
      cublasLtMatrixLayoutCreate(&plan->c, CUDA_R_16F, n, m, n));

  cublasLtMatmulPreference_t pref = nullptr;
  INFERX_CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceCreate(&pref));

  const size_t ws_bytes = workspace.size();
  const cublasStatus_t pref_status = cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
      sizeof(ws_bytes));

  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  const cublasStatus_t heur_status =
      pref_status == CUBLAS_STATUS_SUCCESS
          ? cublasLtMatmulAlgoGetHeuristic(lt, plan->desc, plan->a, plan->b,
                                           plan->c, plan->c, pref, 1,
                                           &heuristic, &returned)
          : pref_status;

  // Destroyed before the error checks below so that neither early return leaks
  // it; the preference object is not needed once the heuristic has answered.
  cublasLtMatmulPreferenceDestroy(pref);

  if (heur_status != CUBLAS_STATUS_SUCCESS) {
    return CublasErrorToStatus(heur_status, "cublasLtMatmulAlgoGetHeuristic",
                               __FILE__, __LINE__);
  }

  if (returned == 0) {
    return UnimplementedError(
        "cuBLASLt has no algorithm for m=", m, " n=", n, " k=", k,
        " (f16 in, f32 accumulate, workspace ", ws_bytes, " B)");
  }

  plan->algo = heuristic.algo;

  const Plan* raw = plan.get();
  plans.emplace(key, std::move(plan));

  return raw;
}

StatusOr<CublasLtGemm> CublasLtGemm::Create(size_t workspace_bytes) {
  auto impl = std::make_unique<Impl>();

  INFERX_CUBLAS_RETURN_IF_ERROR(cublasLtCreate(&impl->lt));

  INFERX_ASSIGN_OR_RETURN(
      impl->workspace,
      DeviceBuffer::Allocate(workspace_bytes, DeviceId::Cuda(0)));

  return CublasLtGemm(std::move(impl));
}

CublasLtGemm::CublasLtGemm(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

CublasLtGemm::~CublasLtGemm() = default;
CublasLtGemm::CublasLtGemm(CublasLtGemm&&) noexcept = default;
CublasLtGemm& CublasLtGemm::operator=(CublasLtGemm&&) noexcept = default;

size_t CublasLtGemm::PlanCacheSize() const { return impl_->plans.size(); }

Status CublasLtGemm::Warm(int64_t m, int64_t n, int64_t k) {
  if (m <= 0 || n <= 0 || k <= 0) {
    return InvalidArgumentError("GEMM shape must be positive, got m=", m,
                                " n=", n, " k=", k);
  }

  INFERX_ASSIGN_OR_RETURN(const Plan* plan, impl_->GetOrBuild(m, n, k));
  (void)plan;

  return OkStatus();
}

Status CublasLtGemm::LinearF16(const TensorView& x, const TensorView& w,
                               const TensorView& y, cudaStream_t stream) {
  // Validated up front and in full, rather than trusting the caller: a wrong
  // extent here is a silent out-of-bounds read on the device, which surfaces
  // later as a corrupted tensor somewhere unrelated.
  for (const auto& [name, t] : {std::pair{"x", &x}, {"w", &w}, {"y", &y}}) {
    if (!t->IsDefined()) return InvalidArgumentError("LinearF16: ", name,
                                                     " is undefined");
    if (!t->IsCuda()) {
      return InvalidArgumentError("LinearF16: ", name, " is on ",
                                  t->Device().ToString(), ", not a CUDA device");
    }
    if (t->GetDataType() != DataType::kFloat16) {
      return InvalidArgumentError("LinearF16: ", name, " is ",
                                  DataTypeName(t->GetDataType()), ", not f16");
    }
    if (t->Rank() != 2) {
      return InvalidArgumentError("LinearF16: ", name, " has rank ", t->Rank(),
                                  ", expected 2");
    }
  }

  const int64_t m = x.Dim(0);
  const int64_t k = x.Dim(1);
  const int64_t n = w.Dim(0);

  if (w.Dim(1) != k) {
    return InvalidArgumentError("LinearF16: w is [", n, ", ", w.Dim(1),
                                "], expected [", n, ", ", k, "] to match x");
  }

  if (y.Dim(0) != m || y.Dim(1) != n) {
    return InvalidArgumentError("LinearF16: y is [", y.Dim(0), ", ", y.Dim(1),
                                "], expected [", m, ", ", n, "]");
  }

  if (m == 0 || n == 0 || k == 0) return OkStatus();

  INFERX_ASSIGN_OR_RETURN(const Plan* plan, impl_->GetOrBuild(m, n, k));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  INFERX_CUBLAS_RETURN_IF_ERROR(cublasLtMatmul(
      impl_->lt, plan->desc, &alpha, w.Data(), plan->a, x.Data(), plan->b,
      &beta, y.Data(), plan->c, y.Data(), plan->c, &plan->algo,
      impl_->workspace.data(), impl_->workspace.size(), stream));

  return OkStatus();
}

}  // namespace inferx::kernels
