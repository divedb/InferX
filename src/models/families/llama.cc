#include "inferx/models/causal/build.h"
#include "inferx/models/families/families.h"

namespace inferx::families {
namespace {

struct LlamaTraits {
  static constexpr std::string_view kModelType = "llama";
  static constexpr std::string_view kArch = "LlamaForCausalLM";
  static constexpr causal::NormPlacement kNorm = causal::NormPlacement::kPre;
  static constexpr models::WeightNames kNames{};
  static constexpr models::WeightLayout kLayout{};
  using Norm = components::RmsNorm;
  using Attn =
      components::GqaAttention<components::QkvBias::kDisabled, components::QkNorm::kNone,
                               components::RopeStyle::kNeox>;
  using Mlp = components::GatedMlp<ops::Activation::kSilu>;
};

static_assert(causal::ModelTraits<LlamaTraits>);

}  // namespace

Family Llama() { return causal::MakeFamily<LlamaTraits>(); }

}  // namespace inferx::families
