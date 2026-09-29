#pragma once

#include <cstdio>
#include <string>
#include <string_view>

namespace inferx::cli {

/// \brief The semantic meaning of a piece of text, independent of how it is drawn.
enum class TextRole {
  kError,
  kWarning,
  kSuccess,
  kCommand,
  kHeading,
  kDimmed,
  kPlaceholder,
};

/// \brief Returns true if `stream` is an interactive terminal. On Windows this also
///        enables ANSI escape sequence processing for the console.
///
/// \param stream The file stream to check.
/// \return       True if `stream` is an interactive terminal; otherwise false.
bool IsInteractiveTerminal(FILE* stream);

/// \brief Decides whether colored output should be used for a stream. Color is
///        configured only through the environment: a non-empty NO_COLOR disables
///        it, a non-empty CLICOLOR_FORCE enables it even when the stream is
///        redirected, and otherwise color is used only for terminals with
///        TERM != dumb. NO_COLOR wins over CLICOLOR_FORCE.
///
/// \param is_terminal Whether the output is a terminal.
/// \return            True if color should be used; otherwise false.
bool ShouldUseColor(bool is_terminal);

/// \brief Represents a terminal color and/or text style. Applies role-based styling to text, or
///        passes it through unchanged when color is disabled.
class TextStyler {
 public:
  explicit TextStyler(bool color_enabled) : color_enabled_(color_enabled) {}

  std::string Style(TextRole role, std::string_view text) const;

  std::string Error(std::string_view text) const { return Style(TextRole::kError, text); }
  std::string Warning(std::string_view text) const { return Style(TextRole::kWarning, text); }
  std::string Success(std::string_view text) const { return Style(TextRole::kSuccess, text); }
  std::string Command(std::string_view text) const { return Style(TextRole::kCommand, text); }
  std::string Heading(std::string_view text) const { return Style(TextRole::kHeading, text); }
  std::string Dimmed(std::string_view text) const { return Style(TextRole::kDimmed, text); }
  std::string Placeholder(std::string_view text) const {
    return Style(TextRole::kPlaceholder, text);
  }

 private:
  bool color_enabled_;
};

}  // namespace inferx::cli