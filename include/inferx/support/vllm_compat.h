#pragma once

#include <deque>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "inferx/core/status.h"

namespace CLI {
class App;
class Option;
}  // namespace CLI

namespace inferx {

/// One vLLM flag that InferX accepts but cannot act on. The flag always
/// parses; a value other than the vLLM default is a startup error naming the
/// missing feature, so a vLLM launch script either runs with identical
/// semantics or fails loudly -- never silently diverges.
enum class CompatKind {
  kBool,    // registered as --x / --no-x flag pair
  kInt,     // value compared as an integer
  kDouble,  // value compared as a floating-point number
  kString,  // value compared verbatim
};

struct CompatStub {
  std::string_view name;  // long option name, e.g. "--enable-lora"
  CompatKind kind;
  /// The vLLM default as a literal ("false", "1", "0.92", "fcfs"). Empty
  /// means vLLM's None: any explicit value is rejected. An explicitly passed
  /// value equal to this literal is accepted.
  std::string_view default_text;
  /// Names the capability in the error, e.g. "LoRA adapters".
  std::string_view feature;
  /// Accept any value with an info log instead of default-checking. For flags
  /// that ask for less than InferX already does or that only shape logs vLLM
  /// would emit (e.g. --trust-remote-code, --disable-log-requests).
  bool accept_any = false;
};

/// The vLLM 0.26.0 `vllm serve` flags that InferX stubs. Engine + frontend.
std::span<const CompatStub> ServeCompatStubs();
/// The frontend-only subset, for inferx-gateway.
std::span<const CompatStub> GatewayCompatStubs();

/// Parse-time storage for stub values. Deques keep addresses stable across
/// registration, which CLI11's bindings require.
struct CompatState {
  std::deque<std::string> values;
  std::deque<bool> bools;
  std::vector<std::pair<const CompatStub*, CLI::Option*>> bound;
};

/// Registers every stub on `app` under the "vLLM compatibility" help group.
/// CLI11 throws OptionAlreadyAdded if a stub collides with a real flag, so a
/// test that constructs the full app guards the table.
void AddCompatStubs(CLI::App& app, std::span<const CompatStub> stubs,
                    CompatState& state);

/// Post-parse: OK when every stub is unset, at its vLLM default, or marked
/// accept_any; otherwise InvalidArgumentError naming the flag, the value, and
/// the unsupported feature.
Status CheckCompatStubs(std::span<const CompatStub> stubs,
                        const CompatState& state);

/// Lower-case hex SHA-256, for hashing --api-key values into the store of
/// digests that `--api-key-sha256` feeds directly.
std::string Sha256Hex(std::string_view data);

}  // namespace inferx
