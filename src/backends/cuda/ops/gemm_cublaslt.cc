#include <cublasLt.h>

#include <algorithm>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "inferx/backends/cuda/cuda_utils.h"
#include "inferx/core/shape.h"
#include "inferx/ops/gemm.h"

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

// Mirrors cuda::ErrorToStatus: the status code carries what the caller can do
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

// FP8 and FP16 plans for the same shape are different plans: the layouts carry
// different element types, so an algorithm the heuristic chose for one is not
// valid for the other. Keying on the shape alone would hand an FP16 algorithm
// to an FP8 matmul.
enum class Path { kF16, kBF16, kF8E4M3, kF8E4M3BF16Out };

struct ShapeKey {
  int64_t m, n, k;
  Path path;

  friend bool operator==(const ShapeKey& a, const ShapeKey& b) {
    return a.m == b.m && a.n == b.n && a.k == b.k && a.path == b.path;
  }

  template <typename H>
  friend H AbslHashValue(H h, const ShapeKey& s) {
    return H::combine(std::move(h), s.m, s.n, s.k,
                      static_cast<int>(s.path));
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

  // The heuristic returns a ranked list, and its top rank is not uniformly the
  // fastest by wall-clock: R3b measured gate_up at m=8 at 0.88x on the top-1
  // algo. We keep the first kMaxAlgos candidates and let Tune() time them on
  // the FP8 decode path, keeping whichever the device actually runs fastest.
  static constexpr int kMaxAlgos = 8;
  cublasLtMatmulHeuristicResult_t candidates[kMaxAlgos]{};
  int num_candidates = 0;

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

// RAII for the two CUDA events Tune() times candidates with, so its several
// abandon-on-failure paths cannot leak them. Local to this file because the
// engine owns the only other event pairs and this is the kernel layer's single
// timed path.
struct EventPair {
  cudaEvent_t a = nullptr;
  cudaEvent_t b = nullptr;
  bool created = false;

  Status Init() {
    if (cudaEventCreate(&a) != cudaSuccess) {
      return InternalError("cudaEventCreate failed in GEMM tuning");
    }
    if (cudaEventCreate(&b) != cudaSuccess) {
      cudaEventDestroy(a);
      a = nullptr;
      return InternalError("cudaEventCreate failed in GEMM tuning");
    }
    created = true;
    return OkStatus();
  }

  ~EventPair() {
    if (created) {
      cudaEventDestroy(a);
      cudaEventDestroy(b);
    }
  }

  EventPair() = default;
  EventPair(const EventPair&) = delete;
  EventPair& operator=(const EventPair&) = delete;
};

}  // namespace

struct CublasLtGemm::Impl {
  cublasLtHandle_t lt = nullptr;
  DeviceBuffer workspace;
  // Two floats holding 1.0, used as placeholder scales while the FP8 heuristic
  // is queried. cuBLASLt wants the descriptor complete before it will pick an
  // algorithm, but the scales themselves do not influence the choice -- the
  // real pointers are bound per call, since they belong to the tensors rather
  // than to the plan.
  DeviceBuffer unit_scales;
  // When false, Tune is skipped and plans keep the heuristic's top-1 -- the
  // reference path for differential testing, and a faster cold start.
  bool autotune = true;
  absl::flat_hash_map<ShapeKey, std::unique_ptr<Plan>> plans;

  ~Impl() {
    // Plans hold cuBLASLt descriptors and must be gone before the handle is.
    plans.clear();
    if (lt) cublasLtDestroy(lt);
  }

  StatusOr<const Plan*> GetOrBuild(int64_t m, int64_t n, int64_t k, Path path);

  // Times the heuristic's candidate algorithms for one plan and keeps the
  // fastest, retiring R3b (the top-1 algo is not uniformly the fastest on the
  // FP8 decode path). Scoped to the FP8 path and to memory-bound shapes, where
  // the heuristic is the thing that is wrong and the scratch cost of timing is
  // small. Best-effort: any internal failure abandons tuning and leaves the
  // heuristic's top-1 in place, so it can never make a shape fail.
  Status Tune(Plan& plan, int64_t m, int64_t n, int64_t k, Path path);
};

StatusOr<const Plan*> CublasLtGemm::Impl::GetOrBuild(int64_t m, int64_t n,
                                                     int64_t k, Path path) {
  const ShapeKey key{m, n, k, path};

  if (auto it = plans.find(key); it != plans.end()) return it->second.get();

  auto plan = std::make_unique<Plan>();

  // Both paths accumulate in FP32 with FP32 scales, which is what makes
  // alpha/beta below floats rather than halves.
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

  if (path == Path::kF8E4M3 || path == Path::kF8E4M3BF16Out) {
    // Bound to placeholders here and rebound per call in LinearF8E4M3. A null
    // scale pointer makes the heuristic decline the problem outright, which
    // surfaces as "no algorithm" rather than as a missing-attribute error.
    float* const unit = reinterpret_cast<float*>(unit_scales.data());

    INFERX_CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
        plan->desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &unit, sizeof(unit)));
    INFERX_CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
        plan->desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &unit, sizeof(unit)));
  }

  // The FP8 path is not a variation on the FP16 one, it is the reason the
  // mapping above is shaped the way it is: cuBLASLt only implements FP8 matmul
  // for TN -- A transposed, B not -- and that is exactly what the row-major
  // linear-layer mapping produces. Storing weights [out, in] as checkpoints do
  // is therefore not just convenient, it is what makes FP8 reachable at all.
  cudaDataType_t ab_type = CUDA_R_16F;
  cudaDataType_t out_type = CUDA_R_16F;

  switch (path) {
    case Path::kF16:
      break;
    case Path::kBF16:
      // bf16 in and out. This is the path the model layer actually uses, since
      // it is what checkpoints store; f16 stays as the M1 benchmark baseline.
      ab_type = CUDA_R_16BF;
      out_type = CUDA_R_16BF;
      break;
    case Path::kF8E4M3:
      ab_type = CUDA_R_8F_E4M3;
      // Output stays f16: an f8 output would need a third scale and would throw
      // away precision the FP32 epilogue is already holding.
      break;
    case Path::kF8E4M3BF16Out:
      // The path the model runs on. FP8 operands with a bf16 result, so a
      // quantized GEMM drops into a bf16 stack without a conversion on either
      // side of it.
      ab_type = CUDA_R_8F_E4M3;
      out_type = CUDA_R_16BF;
      break;
  }

  INFERX_CUBLAS_RETURN_IF_ERROR(
      cublasLtMatrixLayoutCreate(&plan->a, ab_type, k, n, k));
  INFERX_CUBLAS_RETURN_IF_ERROR(
      cublasLtMatrixLayoutCreate(&plan->b, ab_type, k, m, k));
  INFERX_CUBLAS_RETURN_IF_ERROR(
      cublasLtMatrixLayoutCreate(&plan->c, out_type, n, m, n));

  cublasLtMatmulPreference_t pref = nullptr;
  INFERX_CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceCreate(&pref));

  const size_t ws_bytes = workspace.size();
  const cublasStatus_t pref_status = cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes,
      sizeof(ws_bytes));

  // Ask for several candidates rather than one. The heuristic ranks them, but
  // its top rank is not uniformly the fastest by wall-clock -- R3b measured
  // gate_up at m=8 at 0.88x on the top-1 algo, an algorithm-selection issue
  // rather than a bandwidth one. Tune() times them and keeps the fastest on
  // the FP8 path. Every candidate is already workspace-filtered by the
  // preference above, so all of them fit the buffer handed to cublasLtMatmul.
  const cublasStatus_t heur_status =
      pref_status == CUBLAS_STATUS_SUCCESS
          ? cublasLtMatmulAlgoGetHeuristic(lt, plan->desc, plan->a, plan->b,
                                           plan->c, plan->c, pref,
                                           Plan::kMaxAlgos, plan->candidates,
                                           &plan->num_candidates)
          : pref_status;

  // Destroyed before the error checks below so that neither early return leaks
  // it; the preference object is not needed once the heuristic has answered.
  cublasLtMatmulPreferenceDestroy(pref);

  if (heur_status != CUBLAS_STATUS_SUCCESS) {
    return CublasErrorToStatus(heur_status, "cublasLtMatmulAlgoGetHeuristic",
                               __FILE__, __LINE__);
  }

  if (plan->num_candidates == 0) {
    return UnimplementedError(
        "cuBLASLt has no algorithm for m=", m, " n=", n, " k=", k, " (",
        (path == Path::kF8E4M3 || path == Path::kF8E4M3BF16Out) ? "f8e4m3"
                                                                 : "f16",
        " in, f32 accumulate, workspace ", ws_bytes, " B)");
  }

  // Default to the heuristic's top rank; Tune may overwrite this for FP8.
  plan->algo = plan->candidates[0].algo;

  // Best-effort and never propagated: the heuristic's top-1 is today's
  // behavior, so abandoning tuning leaves a correct -- if sometimes 0.88x --
  // plan rather than a failed one. Skipped entirely when autotune is off, which
  // is the differential-testing reference and a faster cold start.
  if (autotune) (void)Tune(*plan, m, n, k, path);

  const Plan* raw = plan.get();
  plans.emplace(key, std::move(plan));

  return raw;
}

Status CublasLtGemm::Impl::Tune(Plan& plan, int64_t m, int64_t n, int64_t k,
                                Path path) {
  // R3b is an FP8-path issue; the f16/bf16 heuristic has no known inversion,
  // and timing it would burn warmup for nothing.
  if (path != Path::kF8E4M3 && path != Path::kF8E4M3BF16Out) return OkStatus();

  // Nothing to choose between. Most decode shapes return several candidates;
  // a shape that returns one has already been answered optimally by definition.
  if (plan.num_candidates < 2) return OkStatus();

  // Scope tuning to the memory-bound regime, which is where R3b lives and
  // where the algorithm swings the result. The FP8 roofline on this card
  // crosses around m ~ 80 (2*m FLOP/byte against a ~160 FLOP:s-:Byte peak
  // ratio), so 64 sits safely inside the decode range. Prefill m is
  // compute-bound, the heuristic is reliable there, and timing it would
  // allocate a y buffer of hundreds of MB for no return.
  constexpr int64_t kTuneMaxTokens = 64;
  if (m > kTuneMaxTokens) return OkStatus();

  const DataType out_dt = (path == Path::kF8E4M3BF16Out) ? DataType::kBFloat16
                                                          : DataType::kFloat16;

  // Operand scratch. Values do not affect FP8-GEMM timing -- the tensor cores
  // have no data-dependent path, and the scale folds into a fixed epilogue --
  // so zeroed buffers time identically to real ones and avoid a quantize pass
  // per candidate. The descriptor already carries the unit scales bound in
  // GetOrBuild, which is all timing needs.
  auto xb = DeviceBuffer::Allocate(static_cast<size_t>(m * k), DeviceId::Cuda(0));
  auto wb = DeviceBuffer::Allocate(static_cast<size_t>(n * k), DeviceId::Cuda(0));
  auto yb =
      DeviceBuffer::Allocate(static_cast<size_t>(m * n) * 2, DeviceId::Cuda(0));
  if (!xb.ok() || !wb.ok() || !yb.ok()) return OkStatus();

  if (cudaMemsetAsync(xb->data(), 0, xb->size()) != cudaSuccess ||
      cudaMemsetAsync(wb->data(), 0, wb->size()) != cudaSuccess ||
      cudaMemsetAsync(yb->data(), 0, yb->size()) != cudaSuccess) {
    return OkStatus();
  }

  auto x = TensorView::Create(xb->data(), DataType::kFloat8E4M3FN,
                              Shape({m, k}), DeviceId::Cuda(0));
  auto w = TensorView::Create(wb->data(), DataType::kFloat8E4M3FN,
                              Shape({n, k}), DeviceId::Cuda(0));
  auto y = TensorView::Create(yb->data(), out_dt, Shape({m, n}),
                              DeviceId::Cuda(0));
  if (!x.ok() || !w.ok() || !y.ok()) return OkStatus();

  EventPair events;
  if (!events.Init().ok()) return OkStatus();

  const float alpha = 1.0f;
  const float beta = 0.0f;

  auto launch = [&](const cublasLtMatmulAlgo_t& algo) -> cublasStatus_t {
    return cublasLtMatmul(lt, plan.desc, &alpha, w->Data(), plan.a, x->Data(),
                          plan.b, &beta, y->Data(), plan.c, y->Data(), plan.c,
                          &algo, workspace.data(), workspace.size(), nullptr);
  };

  // Ramp the device before timing. Only the SM clock is locked (-lgc); the
  // memory clock floats, and on a consumer Ada part its response to a workload
  // is sluggish and state-dependent -- a bandwidth-bound decode GEMM measured
  // before it has settled can read at half the achievable bandwidth, ranking
  // candidates wrong. Running candidate 0 back-to-back is the same ramp the
  // bench's own warmup performs, and reaches a steady state more often than a
  // cold start does. It does not fully eliminate cross-run variance: that is a
  // property of the unlocked memory clock (R3b), and locking it is the only
  // complete fix. The tuner is still a strict improvement on the heuristic --
  // its top-1 lands in the slow cluster even when a faster algo is reachable.
  constexpr int kRampLaunches = 128;
  for (int i = 0; i < kRampLaunches; ++i) {
    if (launch(plan.candidates[0].algo) != CUBLAS_STATUS_SUCCESS) {
      return OkStatus();
    }
  }
  if (cudaDeviceSynchronize() != cudaSuccess) return OkStatus();

  // Size the batch from a pilot on candidate 0 -- now that the clock has
  // settled -- so every candidate is measured over the same launch count.
  // Clamped because m=1 shapes would otherwise ask for thousands of launches,
  // and a millisecond of device time is all the resolution the comparison needs.
  constexpr int kWarmup = 5;
  constexpr double kTargetSampleMs = 1.0;
  int batch = 1;

  if (cudaEventRecord(events.a) == cudaSuccess &&
      launch(plan.candidates[0].algo) == CUBLAS_STATUS_SUCCESS &&
      cudaEventRecord(events.b) == cudaSuccess &&
      cudaEventSynchronize(events.b) == cudaSuccess) {
    float pilot_ms = 0.0f;
    if (cudaEventElapsedTime(&pilot_ms, events.a, events.b) == cudaSuccess &&
        pilot_ms > 0.0f) {
      const double want = kTargetSampleMs / static_cast<double>(pilot_ms);
      batch = static_cast<int>(
          want < 1.0 ? 1.0 : (want > 256.0 ? 256.0 : want));
    }
  }

  // Time each candidate over several samples and keep the one with the best
  // minimum. Min, not median, for one reason only: the bench reports min-of-
  // samples too, so Tune and the bench agree on what "fastest algorithm" means
  // and a plan that looks fast to Tune looks fast to the bench. Within a run
  // the two are within ~1% for these shapes anyway -- a bandwidth-bound GEMM is
  // launch-bound and its per-sample spread is small once the device has settled
  // -- so the choice between them does not move the pick. What does move it is
  // the device's DVFS state, which shifts every candidate together across runs
  // and is addressed by the ramp above and by locking the memory clock, not by
  // the summary statistic.
  constexpr int kSamples = 20;
  double sample_ms[kSamples];

  double best_min = -1.0;
  int best_idx = 0;

  for (int ci = 0; ci < plan.num_candidates; ++ci) {
    if (plan.candidates[ci].state != CUBLAS_STATUS_SUCCESS) continue;

    bool good = true;
    for (int i = 0; i < kWarmup && good; ++i) {
      good = (launch(plan.candidates[ci].algo) == CUBLAS_STATUS_SUCCESS);
    }
    if (!good || cudaDeviceSynchronize() != cudaSuccess) continue;

    int got = 0;
    for (int s = 0; s < kSamples; ++s) {
      // Drained between samples, not between launches: samples stay
      // independent while the device never idles long enough mid-sample to
      // drop clocks. The sync sits outside the event pair so it is not timed.
      if (cudaDeviceSynchronize() != cudaSuccess) break;
      if (cudaEventRecord(events.a) != cudaSuccess) break;

      int done = 0;
      for (int i = 0; i < batch; ++i) {
        if (launch(plan.candidates[ci].algo) != CUBLAS_STATUS_SUCCESS) break;
        ++done;
      }
      if (done != batch || cudaEventRecord(events.b) != cudaSuccess ||
          cudaEventSynchronize(events.b) != cudaSuccess) {
        break;
      }

      float ms = 0.0f;
      if (cudaEventElapsedTime(&ms, events.a, events.b) != cudaSuccess) break;
      sample_ms[got++] = static_cast<double>(ms) / batch;
    }

    // A candidate that could not be timed cleanly is out of the running; it is
    // not an error for the plan, which keeps the heuristic's choice otherwise.
    if (got != kSamples) continue;

    double cmin = sample_ms[0];
    for (int s = 1; s < kSamples; ++s) cmin = std::min(cmin, sample_ms[s]);

    if (best_min < 0.0 || cmin < best_min) {
      best_min = cmin;
      best_idx = ci;
    }
  }

  if (best_min >= 0.0) plan.algo = plan.candidates[best_idx].algo;
  return OkStatus();
}

StatusOr<CublasLtGemm> CublasLtGemm::Create(size_t workspace_bytes,
                                            bool autotune) {
  auto impl = std::make_unique<Impl>();
  impl->autotune = autotune;

  INFERX_CUBLAS_RETURN_IF_ERROR(cublasLtCreate(&impl->lt));

  INFERX_ASSIGN_OR_RETURN(
      impl->workspace,
      DeviceBuffer::Allocate(workspace_bytes, DeviceId::Cuda(0)));

  INFERX_ASSIGN_OR_RETURN(
      impl->unit_scales,
      DeviceBuffer::Allocate(2 * sizeof(float), DeviceId::Cuda(0)));

  const float ones[2] = {1.0f, 1.0f};
  INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(impl->unit_scales.data(), ones,
                                         sizeof(ones), cudaMemcpyHostToDevice));

  return CublasLtGemm(std::move(impl));
}

CublasLtGemm::CublasLtGemm(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

CublasLtGemm::~CublasLtGemm() = default;
CublasLtGemm::CublasLtGemm(CublasLtGemm&&) noexcept = default;
CublasLtGemm& CublasLtGemm::operator=(CublasLtGemm&&) noexcept = default;

size_t CublasLtGemm::PlanCacheSize() const { return impl_->plans.size(); }

Status CublasLtGemm::Warm(int64_t m, int64_t n, int64_t k, bool fp8) {
  if (m <= 0 || n <= 0 || k <= 0) {
    return InvalidArgumentError("GEMM shape must be positive, got m=", m,
                                " n=", n, " k=", k);
  }

  INFERX_ASSIGN_OR_RETURN(
      const Plan* plan,
      impl_->GetOrBuild(m, n, k, fp8 ? Path::kF8E4M3 : Path::kF16));
  (void)plan;

  return OkStatus();
}

Status CublasLtGemm::LinearBF16(const TensorView& x, const TensorView& w,
                                const TensorView& y, Stream stream) {
  for (const auto& [name, t] : {std::pair{"x", &x}, {"w", &w}, {"y", &y}}) {
    if (!t->IsDefined()) {
      return InvalidArgumentError("LinearBF16: ", name, " is undefined");
    }
    if (!t->IsCuda()) {
      return InvalidArgumentError("LinearBF16: ", name, " is on ",
                                  t->Device().ToString(),
                                  ", not a CUDA device");
    }
    if (t->GetDataType() != DataType::kBFloat16) {
      return InvalidArgumentError("LinearBF16: ", name, " is ",
                                  DataTypeName(t->GetDataType()), ", not bf16");
    }
    if (t->Rank() != 2) {
      return InvalidArgumentError("LinearBF16: ", name, " has rank ", t->Rank(),
                                  ", expected 2");
    }
  }

  const int64_t m = x.Dim(0);
  const int64_t k = x.Dim(1);
  const int64_t n = w.Dim(0);

  if (w.Dim(1) != k) {
    return InvalidArgumentError("LinearBF16: w is [", n, ", ", w.Dim(1),
                                "], expected [", n, ", ", k, "] to match x");
  }

  if (y.Dim(0) != m || y.Dim(1) != n) {
    return InvalidArgumentError("LinearBF16: y is [", y.Dim(0), ", ", y.Dim(1),
                                "], expected [", m, ", ", n, "]");
  }

  if (m == 0 || n == 0 || k == 0) return OkStatus();

  INFERX_ASSIGN_OR_RETURN(const Plan* plan,
                          impl_->GetOrBuild(m, n, k, Path::kBF16));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  INFERX_CUBLAS_RETURN_IF_ERROR(cublasLtMatmul(
      impl_->lt, plan->desc, &alpha, w.Data(), plan->a, x.Data(), plan->b,
      &beta, y.Data(), plan->c, y.Data(), plan->c, &plan->algo,
      impl_->workspace.data(), impl_->workspace.size(), stream));

  return OkStatus();
}

Status CublasLtGemm::LinearF16(const TensorView& x, const TensorView& w,
                               const TensorView& y, Stream stream) {
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

  INFERX_ASSIGN_OR_RETURN(const Plan* plan,
                          impl_->GetOrBuild(m, n, k, Path::kF16));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  INFERX_CUBLAS_RETURN_IF_ERROR(cublasLtMatmul(
      impl_->lt, plan->desc, &alpha, w.Data(), plan->a, x.Data(), plan->b,
      &beta, y.Data(), plan->c, y.Data(), plan->c, &plan->algo,
      impl_->workspace.data(), impl_->workspace.size(), stream));

  return OkStatus();
}

Status CublasLtGemm::LinearF8E4M3(const TensorView& x, const TensorView& w,
                                  const TensorView& y, const float* x_scale_dev,
                                  const float* w_scale_dev,
                                  Stream stream) {
  // The output dtype selects the path. f16 is M1's benchmark form; bf16 is what
  // the model uses, and having one entry point means the two cannot drift.
  if (y.GetDataType() != DataType::kFloat16 &&
      y.GetDataType() != DataType::kBFloat16) {
    return InvalidArgumentError("LinearF8E4M3: y is ",
                                DataTypeName(y.GetDataType()),
                                ", expected f16 or bf16");
  }

  const Path path = y.GetDataType() == DataType::kBFloat16
                        ? Path::kF8E4M3BF16Out
                        : Path::kF8E4M3;

  for (const auto& [name, t, want] :
       {std::tuple{"x", &x, DataType::kFloat8E4M3FN},
        {"w", &w, DataType::kFloat8E4M3FN},
        {"y", &y, y.GetDataType()}}) {
    if (!t->IsDefined()) {
      return InvalidArgumentError("LinearF8E4M3: ", name, " is undefined");
    }
    if (!t->IsCuda()) {
      return InvalidArgumentError("LinearF8E4M3: ", name, " is on ",
                                  t->Device().ToString(),
                                  ", not a CUDA device");
    }
    if (t->GetDataType() != want) {
      return InvalidArgumentError("LinearF8E4M3: ", name, " is ",
                                  DataTypeName(t->GetDataType()),
                                  ", expected ", DataTypeName(want));
    }
    if (t->Rank() != 2) {
      return InvalidArgumentError("LinearF8E4M3: ", name, " has rank ",
                                  t->Rank(), ", expected 2");
    }
  }

  if (x_scale_dev == nullptr || w_scale_dev == nullptr) {
    return InvalidArgumentError("LinearF8E4M3: scale pointers must be non-null");
  }

  const int64_t m = x.Dim(0);
  const int64_t k = x.Dim(1);
  const int64_t n = w.Dim(0);

  if (w.Dim(1) != k) {
    return InvalidArgumentError("LinearF8E4M3: w is [", n, ", ", w.Dim(1),
                                "], expected [", n, ", ", k, "] to match x");
  }

  if (y.Dim(0) != m || y.Dim(1) != n) {
    return InvalidArgumentError("LinearF8E4M3: y is [", y.Dim(0), ", ",
                                y.Dim(1), "], expected [", m, ", ", n, "]");
  }

  // Checked explicitly because the alternative is worse than an error: the
  // FP8 tensor-core path wants 16-byte-aligned rows, and k is the leading
  // dimension of both operands in this layout. cuBLASLt answers a misaligned
  // problem with "no algorithm", which reads as "FP8 is unsupported here"
  // rather than "your k is odd".
  if (k % 16 != 0) {
    return InvalidArgumentError(
        "LinearF8E4M3: k must be a multiple of 16 for the FP8 path, got ", k);
  }

  if (m == 0 || n == 0 || k == 0) return OkStatus();

  INFERX_ASSIGN_OR_RETURN(const Plan* plan, impl_->GetOrBuild(m, n, k, path));

  // Rebound per call: the plan is shared by every tensor of this shape, but the
  // scales belong to the tensors. Cheap -- this copies a pointer into the
  // descriptor, it does not touch the device.
  INFERX_CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
      plan->desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_scale_dev,
      sizeof(w_scale_dev)));
  INFERX_CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescSetAttribute(
      plan->desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &x_scale_dev,
      sizeof(x_scale_dev)));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  INFERX_CUBLAS_RETURN_IF_ERROR(cublasLtMatmul(
      impl_->lt, plan->desc, &alpha, w.Data(), plan->a, x.Data(), plan->b,
      &beta, y.Data(), plan->c, y.Data(), plan->c, &plan->algo,
      impl_->workspace.data(), impl_->workspace.size(), stream));

  return OkStatus();
}

}  // namespace inferx::kernels
