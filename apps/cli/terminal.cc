#include "cli/terminal.h"

#include <fmt/color.h>
#include <fmt/format.h>

#include <cstdlib>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace inferx::cli {

namespace {

fmt::text_style StyleForRole(TextRole role) {
  switch (role) {
    case TextRole::kError:
      return fmt::emphasis::bold | fmt::fg(fmt::terminal_color::red);
    case TextRole::kWarning:
    case TextRole::kPlaceholder:
      return fmt::fg(fmt::terminal_color::yellow);
    case TextRole::kSuccess:
      return fmt::fg(fmt::terminal_color::green);
    case TextRole::kCommand:
      return fmt::fg(fmt::terminal_color::cyan);
    case TextRole::kHeading:
      return fmt::emphasis::bold;
    case TextRole::kDimmed:
      return fmt::emphasis::faint;
  }
  return {};
}

bool IsEnvVarSet(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && *value != '\0';
}

bool IsDumbTerminalType() {
  const char* term = std::getenv("TERM");
  return term != nullptr && std::string_view(term) == "dumb";
}

}  // namespace

bool IsInteractiveTerminal(FILE* stream) {
#ifdef _WIN32
  if (!_isatty(_fileno(stream))) return false;

  HANDLE console = reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(stream)));
  DWORD console_mode = 0;
  return GetConsoleMode(console, &console_mode) &&
         SetConsoleMode(console, console_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#else
  return isatty(fileno(stream)) != 0;
#endif
}

bool ShouldUseColor(bool is_terminal) {
  if (IsEnvVarSet("NO_COLOR")) return false;
  if (IsEnvVarSet("CLICOLOR_FORCE")) return true;
  return is_terminal && !IsDumbTerminalType();
}

std::string TextStyler::Style(TextRole role, std::string_view text) const {
  if (!color_enabled_ || text.empty()) return std::string(text);

  return fmt::format("{}", fmt::styled(text, StyleForRole(role)));
}

}  // namespace inferx::cli