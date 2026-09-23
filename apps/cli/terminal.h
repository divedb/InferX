#pragma once

#include <cstdio>
#include <string>
#include <string_view>

namespace inferx::cli::term {

enum class ColorMode { kAuto, kAlways, kNever };
enum class Style { kError, kWarning, kSuccess, kCommand, kHeading, kDim, kPlaceholder };

bool IsTerminal(FILE* stream);
bool UseColor(ColorMode mode, bool terminal);

// All terminal escape sequences live here. Callers choose semantic styles.
class StyleSheet {
 public:
  explicit StyleSheet(bool color) : color_(color) {}
  std::string Apply(Style style, std::string_view text) const;
  std::string Error(std::string_view text) const { return Apply(Style::kError, text); }
  std::string Warning(std::string_view text) const { return Apply(Style::kWarning, text); }
  std::string Success(std::string_view text) const { return Apply(Style::kSuccess, text); }
  std::string Command(std::string_view text) const { return Apply(Style::kCommand, text); }
  std::string Heading(std::string_view text) const { return Apply(Style::kHeading, text); }
  std::string Dim(std::string_view text) const { return Apply(Style::kDim, text); }
  std::string Placeholder(std::string_view text) const {
    return Apply(Style::kPlaceholder, text);
  }

 private:
  bool color_;
};

}  // namespace inferx::cli::term
