#include "inferx/core/shape.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace inferx {

std::string Shape::ToString() const {
  return absl::StrCat("[", absl::StrJoin(dims_, ", "), "]");
}

}  // namespace inferx
