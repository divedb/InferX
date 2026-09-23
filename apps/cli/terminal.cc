#include "cli/terminal.h"

#include <cstdlib>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace inferx::cli::term {

bool IsTerminal(FILE* stream) {
#ifdef _WIN32
  if (!_isatty(_fileno(stream))) return false;
  HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(stream)));
  DWORD mode = 0;
  return GetConsoleMode(handle, &mode) &&
         SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#else
  return isatty(fileno(stream)) != 0;
#endif
}

bool UseColor(ColorMode mode, bool terminal) {
  if (mode == ColorMode::kAlways) return true;
  if (mode == ColorMode::kNever) return false;
  const char* no_color = std::getenv("NO_COLOR");
  const char* term = std::getenv("TERM");
  return terminal && !(no_color && *no_color) && !(term && std::string_view(term) == "dumb");
}

std::string StyleSheet::Apply(Style style, std::string_view text) const {
  if (!color_ || text.empty()) return std::string(text);
  const char* code = "";
  switch (style) {
    case Style::kError: code = "\033[1;31m"; break;
    case Style::kWarning: code = "\033[33m"; break;
    case Style::kSuccess: code = "\033[32m"; break;
    case Style::kCommand: code = "\033[36m"; break;
    case Style::kHeading: code = "\033[1m"; break;
    case Style::kDim: code = "\033[2m"; break;
    case Style::kPlaceholder: code = "\033[33m"; break;
  }
  return std::string(code) + std::string(text) + "\033[0m";
}

}  // namespace inferx::cli::term
