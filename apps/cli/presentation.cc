// Presentation: shared help formatting, --color handling, parse-error
// translation, and styled status messages. tests/cli_output_test.py and
// tests/cli_presentation_test.cc pin the process-level contract.
#include "cli/presentation.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace inferx::cli {
namespace {

constexpr int kWrapWidth = 79;     // Total line width for help and listings.
constexpr int kMaxLeftWidth = 36;  // Longest left column before wrapping.

bool StartsWith(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

std::string_view StripDashes(std::string_view name) {
  while (!name.empty() && name.front() == '-') name.remove_prefix(1);
  return name;
}

std::vector<std::string> SplitWords(std::string_view text) {
  std::vector<std::string> words;
  std::size_t i = 0;
  while (i < text.size()) {
    while (i < text.size() && text[i] == ' ') ++i;
    std::size_t j = i;
    while (j < text.size() && text[j] != ' ') ++j;
    if (j > i) words.emplace_back(text.substr(i, j - i));
    i = j;
  }
  return words;
}

std::size_t EditDistance(std::string_view a, std::string_view b) {
  std::vector<std::size_t> previous(b.size() + 1), current(b.size() + 1);
  for (std::size_t j = 0; j <= b.size(); ++j) previous[j] = j;
  for (std::size_t i = 1; i <= a.size(); ++i) {
    current[0] = i;
    for (std::size_t j = 1; j <= b.size(); ++j) {
      current[j] = std::min({current[j - 1] + 1, previous[j] + 1,
                             previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
    }
    std::swap(previous, current);
  }
  return previous[b.size()];
}

/// Best "Did you mean" candidate for `typed`, or empty when nothing is
/// close enough.
std::string Suggest(std::string_view typed, const std::vector<std::string>& candidates) {
  const std::size_t threshold = std::max<std::size_t>(2, typed.size() / 3);
  const std::string* best = nullptr;
  std::size_t best_distance = threshold + 1;
  for (const std::string& candidate : candidates) {
    const std::size_t distance = EditDistance(typed, candidate);
    if (distance < best_distance) {
      best_distance = distance;
      best = &candidate;
    }
  }
  return best == nullptr ? std::string() : *best;
}

std::vector<std::string> WrapText(std::string_view text, int width) {
  std::vector<std::string> lines;
  std::string line;
  for (const std::string& word : SplitWords(text)) {
    if (line.empty()) {
      line = word;
    } else if (static_cast<int>(line.size() + 1 + word.size()) <= width) {
      line += " " + word;
    } else {
      lines.push_back(line);
      line = word;
    }
  }
  if (!line.empty()) lines.push_back(line);
  return lines;
}

std::vector<const CLI::App*> Children(const CLI::App& app) {
  // Option groups register as nameless subcommand apps; only named
  // subcommands are real commands.
  std::vector<const CLI::App*> commands;
  for (const CLI::App* sub : app.get_subcommands([](const CLI::App*) { return true; })) {
    if (!sub->get_name().empty()) commands.push_back(sub);
  }
  return commands;
}

/// Option-group subcommand apps (nameless children owning grouped options).
std::vector<const CLI::App*> OptionGroups(const CLI::App& app) {
  std::vector<const CLI::App*> groups;
  for (const CLI::App* sub : app.get_subcommands([](const CLI::App*) { return true; })) {
    if (sub->get_name().empty()) groups.push_back(sub);
  }
  return groups;
}

/// Every option visible on `app`'s help: its own plus option-group-owned.
std::vector<const CLI::Option*> AllOptions(const CLI::App& app) {
  std::vector<const CLI::Option*> options = app.get_options();
  for (const CLI::App* group : OptionGroups(app)) {
    for (const CLI::Option* option : group->get_options()) options.push_back(option);
  }
  return options;
}

std::string UsageSuffix(const CLI::App& app) {
  return Children(app).empty() ? " [options]" : " <command> [options]";
}

/// Wraps every simple <placeholder> token in the placeholder style. A no-op
/// in plain mode, so message text stays byte-identical when color is off.
std::string StylePlaceholders(const std::string& text, const term::StyleSheet& style) {
  std::string out;
  std::size_t i = 0;
  while (i < text.size()) {
    const std::size_t open = text.find('<', i);
    if (open == std::string::npos) break;
    const std::size_t close = text.find('>', open);
    if (close == std::string::npos) break;
    out += text.substr(i, open - i);
    const std::string token = text.substr(open, close - open + 1);
    out += token.find(' ') == std::string::npos ? style.Placeholder(token) : token;
    i = close + 1;
  }
  out += text.substr(i);
  return out;
}

std::string UsageLine(const CLI::App& app, const term::StyleSheet& style) {
  return style.Heading("Usage:") + "\n  " + style.Command(CommandPath(app)) +
         StylePlaceholders(UsageSuffix(app), style);
}

std::string LongName(const CLI::Option& option) {
  const std::vector<std::string>& long_names = option.get_lnames();
  if (!long_names.empty()) return "--" + long_names.front();
  const std::vector<std::string>& short_names = option.get_snames();
  return short_names.empty() ? std::string() : "-" + short_names.front();
}

std::vector<std::string> OptionCandidates(const CLI::App& app) {
  std::vector<std::string> candidates{"--help", "--color"};
  for (const CLI::Option* option : AllOptions(app)) {
    if (option == app.get_help_ptr() || option == app.get_version_ptr()) continue;
    std::string name = LongName(*option);
    if (!name.empty()) candidates.push_back(name);
  }
  return candidates;
}

std::vector<std::string> CommandCandidates(const CLI::App& app) {
  std::vector<std::string> candidates;
  for (const CLI::App* sub : Children(app)) candidates.push_back(sub->get_name());
  return candidates;
}

/// One aligned help row; `left` stays unstyled in the entry so column math
/// uses plain widths.
struct HelpEntry {
  std::string left;
  std::string right;
  bool command = false;  // Style the left column as a command name.
};

std::string RenderEntries(const std::vector<HelpEntry>& entries,
                          const term::StyleSheet& style) {
  std::size_t column = 0;
  for (const HelpEntry& entry : entries) {
    if (entry.left.size() <= static_cast<std::size_t>(kMaxLeftWidth))
      column = std::max(column, entry.left.size() + 2);
  }
  column = std::max<std::size_t>(column, 4);
  std::string out;
  for (const HelpEntry& entry : entries) {
    const std::string left = entry.command
                                 ? style.Command(entry.left)
                                 : StylePlaceholders(entry.left, style);
    std::vector<std::string> lines =
        WrapText(entry.right, static_cast<int>(std::max<std::size_t>(
                                  1, kWrapWidth - column)));
    const std::string first = lines.empty() ? std::string() : lines.front();
    if (entry.left.size() > static_cast<std::size_t>(kMaxLeftWidth)) {
      out += "  " + left + "\n" + std::string(column, ' ') + first + "\n";
    } else {
      out += "  " + left + std::string(column - entry.left.size(), ' ') + first + "\n";
    }
    for (std::size_t i = 1; i < lines.size(); ++i)
      out += std::string(column, ' ') + lines[i] + "\n";
  }
  return out;
}

std::string OptionLeft(const CLI::Option& option) {
  const std::string name = LongName(option);
  // Flags carry no type name; value options show a name-derived placeholder.
  if (option.get_type_name().empty()) return name;
  return name + " <" + std::string(StripDashes(name)) + ">";
}

std::string OptionRight(const CLI::Option& option) {
  std::string description = option.get_description();
  if (!option.get_default_str().empty())
    description += " [default: " + option.get_default_str() + "]";
  return description;
}

std::string MakeHelp(const CLI::App* app, const term::StyleSheet& style) {
  std::string out;
  if (!app->get_description().empty()) out += app->get_description() + "\n\n";
  out += UsageLine(*app, style) + "\n";

  const std::vector<const CLI::App*> children = Children(*app);
  if (!children.empty()) {
    std::vector<HelpEntry> entries;
    for (const CLI::App* sub : children)
      entries.push_back({sub->get_name(), sub->get_description(), true});
    out += "\n" + style.Heading("Commands:") + "\n" + RenderEntries(entries, style);
  }

  // Options: the built-in flags always lead the ungrouped section and never
  // show value placeholders; option groups follow in declaration order.
  std::vector<HelpEntry> options;
  options.push_back({"-h, --help", "Print this help message and exit"});
  options.push_back({"-v, --version", "Display program version information and exit"});
  for (const CLI::Option* option : app->get_options()) {
    if (option == app->get_help_ptr() || option == app->get_version_ptr()) continue;
    std::string group = option->get_group();
    if (!group.empty() && group != "OPTIONS") continue;  // Owned by a group.
    options.push_back({OptionLeft(*option), OptionRight(*option)});
  }
  out += "\n" + style.Heading("Options:") + "\n" + RenderEntries(options, style);
  for (const CLI::App* group_app : OptionGroups(*app)) {
    const std::vector<const CLI::Option*> group_options = AllOptions(*group_app);
    if (group_options.empty()) continue;
    std::vector<HelpEntry> entries;
    for (const CLI::Option* option : group_options)
      entries.push_back({OptionLeft(*option), OptionRight(*option)});
    out += "\n" + style.Heading(group_app->get_group() + ":") + "\n" +
           RenderEntries(entries, style);
  }

  out += "\n" + style.Heading("Global options:") + "\n" +
         RenderEntries({{"--color <color>",
                         "Color output: auto, always, never [default: auto]"}},
                       style);

  if (app->get_parent() == nullptr) {
    out += "\nExamples:\n  $ inferx serve --model models/Qwen3-0.6B\n"
           "  $ inferx bench latency --help\n  $ inferx bench throughput --help\n\n"
           "Run 'inferx <command> --help' for more information.\n";
  } else {
    out += "\nRun '" + CommandPath(*app) + " --help' for more information.\n";
  }
  return out;
}

// --- Parse-error reports ---------------------------------------------------

void WriteTryLine(std::ostream& err, const term::StyleSheet& style,
                  const CLI::App* target) {
  err << "\nFor more information, try '"
      << style.Command(CommandPath(*target) + " --help") << "'.\n";
}

std::string PlaceholderFor(const CLI::App&, const std::string& option_name) {
  return "<" + std::string(StripDashes(option_name)) + ">";
}

void ReportUnknownOption(std::ostream& err, const term::StyleSheet& style,
                         const CLI::App* target, const std::string& name) {
  err << style.Error("error:") << " unknown option '" << style.Command(name) << "'\n";
  const std::string best = Suggest(name, OptionCandidates(*target));
  if (!best.empty()) err << "\n  Did you mean '" << style.Command(best) << "'?\n";
  WriteTryLine(err, style, target);
}

void ReportUnknownCommand(std::ostream& err, const term::StyleSheet& style,
                          const CLI::App* target, const std::string& name) {
  err << style.Error("error:") << " unknown command '" << style.Command(name) << "'\n";
  const std::string best = Suggest(name, CommandCandidates(*target));
  if (!best.empty()) err << "\n  Did you mean '" << style.Command(best) << "'?\n";
  const std::vector<const CLI::App*> children = Children(*target);
  if (!children.empty()) {
    std::vector<HelpEntry> entries;
    for (const CLI::App* sub : children)
      entries.push_back({sub->get_name(), sub->get_description(), true});
    err << "\n" << style.Heading("Available commands:") << "\n"
        << RenderEntries(entries, style);
  }
  WriteTryLine(err, style, target);
}

void ReportMissingSubcommand(std::ostream& err, const term::StyleSheet& style,
                             const CLI::App* target) {
  err << style.Error("error:") << " missing required command "
      << style.Placeholder("<command>") << "\n\n"
      << UsageLine(*target, style) << "\n\n"
      << style.Heading("Available commands:") << "\n";
  std::vector<HelpEntry> entries;
  for (const CLI::App* sub : Children(*target))
    entries.push_back({sub->get_name(), sub->get_description(), true});
  err << RenderEntries(entries, style);
  WriteTryLine(err, style, target);
}

void ReportMissingOptions(std::ostream& err, const term::StyleSheet& style,
                          const CLI::App* target,
                          const std::vector<std::string>& names) {
  err << style.Error("error:");
  bool first = true;
  for (const std::string& name : names) {
    err << (first ? " " : "\n") << "missing required option '" << style.Command(name)
        << "' " << style.Placeholder(PlaceholderFor(*target, name));
    first = false;
  }
  err << "\n\n" << UsageLine(*target, style) << "\n";
  WriteTryLine(err, style, target);
}

void ReportMissingValue(std::ostream& err, const term::StyleSheet& style,
                        const CLI::App* target, const std::string& option_name) {
  err << style.Error("error:") << " missing value "
      << style.Placeholder(PlaceholderFor(*target, option_name)) << " for '"
      << style.Command(option_name) << "'\n";
  WriteTryLine(err, style, target);
}

void ReportInvalidValue(std::ostream& err, const term::StyleSheet& style,
                        const CLI::App* target, const std::string& value,
                        const std::string& option_name,
                        const std::string& detail) {
  err << style.Error("error:") << " invalid value '" << value << "' for '"
      << style.Command(option_name) << "'";
  if (!detail.empty()) err << ": " << detail;
  err << "\n";
  WriteTryLine(err, style, target);
}

void ReportGeneric(std::ostream& err, const term::StyleSheet& style,
                   const CLI::App* target, const std::string& what) {
  err << style.Error("error:") << " " << what << "\n";
  WriteTryLine(err, style, target);
}

/// First "--name" token inside a CLI11 message.
std::string FirstOptionToken(const std::string& what) {
  const std::size_t start = what.find("--");
  if (start == std::string::npos) return {};
  const std::size_t end = what.find_first_of(" \t,:)", start);
  return what.substr(start, end == std::string::npos ? end : end - start);
}

}  // namespace

std::string CommandPath(const CLI::App& app) {
  if (const CLI::App* parent = app.get_parent())
    return CommandPath(*parent) + " " + app.get_name();
  return app.get_name();
}

Presentation::Presentation(CLI::App& app, std::ostream& out, std::ostream& err,
                           bool out_is_terminal, bool err_is_terminal)
    : app_(app),
      out_(out),
      err_(err),
      out_is_terminal_(out_is_terminal),
      err_is_terminal_(err_is_terminal) {
  // CLI11 copies the formatter into subcommands when they are created, and
  // the tree already exists by now, so walk it.
  std::function<void(CLI::App*)> install = [&](CLI::App* app) {
    app->formatter_fn([this](const CLI::App* help_app, std::string,
                            CLI::AppFormatMode) {
      return MakeHelp(help_app, out_style_);
    });
    for (CLI::App* sub : app->get_subcommands([](CLI::App*) { return true; }))
      install(sub);
  };
  install(&app_);
  CollectValueOptions(&app_);
}

void Presentation::CollectValueOptions(CLI::App* app) {
  for (const CLI::Option* option : AllOptions(*app)) {
    const bool takes_value = !option->get_type_name().empty();
    for (const std::string& name : option->get_lnames())
      value_options_["--" + name] = takes_value;
    for (const std::string& name : option->get_snames())
      value_options_["-" + name] = takes_value;
  }
  for (CLI::App* sub : app->get_subcommands([](CLI::App*) { return true; }))
    CollectValueOptions(sub);
}

void Presentation::SetStyles(term::ColorMode mode) {
  out_style_ = term::StyleSheet(term::UseColor(mode, out_is_terminal_));
  err_style_ = term::StyleSheet(term::UseColor(mode, err_is_terminal_));
}

void Presentation::Error(std::string_view message) const {
  err_ << err_style_.Error("error:") << ' ' << message << '\n';
}

void Presentation::Warning(std::string_view message) const {
  err_ << err_style_.Warning("warning:") << ' ' << message << '\n';
}

void Presentation::Success(std::string_view message) const {
  out_ << out_style_.Success("success:") << ' ' << message << '\n';
}

int Presentation::PrintHelp(const CLI::App* app) {
  out_ << MakeHelp(app, out_style_);
  return 0;
}

int Presentation::Parse(int argc, const char* const argv[]) {
  // --color and --version are engine-level: they are recognized anywhere
  // before "--", the last --color wins, and --version short-circuits the
  // parse. Everything else goes to CLI11 unchanged.
  std::vector<std::string> args;
  bool past_dashdash = false;
  bool have_color = false;
  bool color_missing_value = false;
  bool want_version = false;
  std::string color_value;
  for (int i = 1; i < argc; ++i) {
    std::string token = argv[i];
    if (!past_dashdash) {
      if (token == "--") {
        past_dashdash = true;
      } else if (token == "--color") {
        if (i + 1 < argc && argv[i + 1][0] != '-') {
          color_value = argv[++i];
          have_color = true;
          continue;
        }
        color_missing_value = true;
        continue;
      } else if (StartsWith(token, "--color=")) {
        color_value = token.substr(8);
        have_color = true;
        continue;
      } else if (token == "-v" || token == "--version") {
        want_version = true;
        continue;
      }
    }
    args.push_back(std::move(token));
  }
  if (want_version) args.insert(args.begin(), "--version");

  // Deepest subcommand named on the command line: the help target and the
  // context for error hints. Tokens naming no subcommand feed unknown-
  // command reporting.
  const CLI::App* target = &app_;
  std::string unmatched;
  for (const std::string& token : args) {
    if (token == "--") break;
    if (!token.empty() && token[0] == '-') continue;
    const CLI::App* next = nullptr;
    for (const CLI::App* sub : Children(*target)) {
      if (sub->get_name() == token) {
        next = sub;
        break;
      }
    }
    if (next != nullptr) {
      target = next;
    } else if (unmatched.empty()) {
      unmatched = token;
    }
  }

  term::ColorMode mode = term::ColorMode::kAuto;
  if (have_color) {
    if (color_value == "always") {
      mode = term::ColorMode::kAlways;
    } else if (color_value == "never") {
      mode = term::ColorMode::kNever;
    } else if (color_value != "auto") {
      SetStyles(term::ColorMode::kNever);
      ReportInvalidValue(err_, err_style_, target, color_value, "--color",
                         color_value + " not in {auto,always,never}");
      return 105;  // CLI11 ValidationError exit code.
    }
  }
  if (color_missing_value) {
    SetStyles(mode);
    ReportMissingValue(err_, err_style_, target, "--color");
    return 114;  // CLI11 ArgumentMismatch exit code.
  }
  SetStyles(mode);

  // A value option followed by another option (or "--") never takes that
  // token as its value; report the missing value instead of letting CLI11
  // consume it and strand a later token as an extra. Values that merely
  // start with '-' (e.g. negative numbers) still parse as values.
  bool past_positionals = false;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& token = args[i];
    if (!past_positionals && token == "--") past_positionals = true;
    if (past_positionals || token.empty() || token[0] != '-') continue;
    const std::string name = token.substr(0, token.find('='));
    const auto known = value_options_.find(name);
    if (known == value_options_.end() || !known->second) continue;
    if (i + 1 >= args.size()) continue;  // CLI11 reports the dangling option.
    const std::string& next = args[i + 1];
    if (next == "--" || (StartsWith(next, "-") && value_options_.count(next) > 0)) {
      ReportMissingValue(err_, err_style_, target, name);
      return 114;  // CLI11 ArgumentMismatch exit code.
    }
  }

  if (args.empty()) return PrintHelp(&app_);

  std::vector<const char*> raw;
  raw.push_back("inferx");  // argv[0]; the tree already carries its names.
  for (const std::string& token : args) raw.push_back(token.c_str());
  try {
    app_.parse(static_cast<int>(raw.size()), raw.data());
  } catch (const CLI::CallForHelp&) {
    return PrintHelp(target);
  } catch (const CLI::CallForAllHelp&) {
    return PrintHelp(target);
  } catch (const CLI::CallForVersion& e) {
    out_ << e.what() << "\n";
    return 0;
  } catch (CLI::ParseError& e) {
    return ReportParseError(e, target, unmatched, e.get_exit_code());
  }
  return 0;
}

int Presentation::ReportParseError(const CLI::ParseError& error,
                                   const CLI::App* target,
                                   const std::string& unmatched, int code) {
  const std::string name = error.get_name();
  const std::string what = error.what();

  if (name == "ExtrasError") {
    // "inferx: The following argument(s) were not expected: <tokens>"
    std::vector<std::string> tokens;
    const std::size_t marker = what.find("not expected");
    if (marker != std::string::npos) {
      const std::size_t colon = what.find(':', marker);
      if (colon != std::string::npos) tokens = SplitWords(what.substr(colon + 1));
    }
    std::string option, word;
    for (const std::string& token : tokens) {
      if (token.empty()) continue;
      if (token[0] == '-') {
        if (option.empty()) option = token.substr(0, token.find('='));
      } else if (word.empty()) {
        word = token;
      }
    }
    if (!option.empty()) {
      ReportUnknownOption(err_, err_style_, target, option);
    } else if (!word.empty() && !Children(*target).empty()) {
      ReportUnknownCommand(err_, err_style_, target, word);
    } else {
      ReportGeneric(err_, err_style_, target, what);
    }
  } else if (name == "RequiredError") {
    if (what.find("subcommand") != std::string::npos) {
      if (!unmatched.empty()) {
        ReportUnknownCommand(err_, err_style_, target, unmatched);
      } else {
        ReportMissingSubcommand(err_, err_style_, target);
      }
    } else {
      // "--fixture is required" (option names joined by spaces).
      std::string names = what;
      if (names.size() > 12 && names.compare(names.size() - 12, 12, " is required") == 0)
        names = names.substr(0, names.size() - 12);
      ReportMissingOptions(err_, err_style_, target, SplitWords(names));
    }
  } else if (name == "ArgumentMismatch") {
    std::string option = FirstOptionToken(what);
    if (option.empty()) option = unmatched;
    ReportMissingValue(err_, err_style_, target, option);
  } else if (name == "ValidationError") {
    const std::size_t colon = what.find(": ");
    const std::string option = colon == std::string::npos ? what : what.substr(0, colon);
    std::string detail = colon == std::string::npos ? std::string() : what.substr(colon + 2);
    std::string value = detail;
    if (StartsWith(value, "Value ")) value = value.substr(6);
    const std::size_t space = value.find(' ');
    if (space != std::string::npos) value = value.substr(0, space);
    ReportInvalidValue(err_, err_style_, target, value, option, detail);
  } else if (name == "ConversionError") {
    std::string value;
    const std::size_t marker = what.find("The value ");
    if (marker != std::string::npos) {
      value = what.substr(marker + 10);
      const std::size_t space = value.find(' ');
      if (space != std::string::npos) value = value.substr(0, space);
    }
    ReportInvalidValue(err_, err_style_, target, value, FirstOptionToken(what), "");
  } else {
    ReportGeneric(err_, err_style_, target, what);
  }
  return code;
}

}  // namespace inferx::cli
