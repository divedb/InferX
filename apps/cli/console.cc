#include "cli/console.h"

#include <fmt/ostream.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <rapidfuzz/distance/Levenshtein.hpp>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace inferx::cli {
namespace {

constexpr std::size_t kWrapWidth = 79;     // Total line width for help and listings.
constexpr std::size_t kMaxLeftWidth = 24;  // Longest aligned left column; longer signatures hang.
constexpr std::size_t kColumnGap = 2;
constexpr std::size_t kMinColumn = 4;
constexpr std::size_t kHangIndent = 6;     // Description column for hanging signatures.

constexpr int kExtrasErrorExit = static_cast<int>(CLI::ExitCodes::ExtrasError);
constexpr int kArgumentMismatchExit = static_cast<int>(CLI::ExitCodes::ArgumentMismatch);

bool StartsWith(std::string_view text, std::string_view prefix) {
  return text.substr(0, prefix.size()) == prefix;
}

bool EndsWith(std::string_view text, std::string_view suffix) {
  return text.size() >= suffix.size() && text.substr(text.size() - suffix.size()) == suffix;
}

std::string_view StripDashes(std::string_view name) {
  const std::size_t first = name.find_first_not_of('-');
  return first == std::string_view::npos ? std::string_view() : name.substr(first);
}

std::vector<std::string> SplitWords(const std::string& text) {
  std::istringstream stream(text);
  std::vector<std::string> words;
  for (std::string word; stream >> word;) words.push_back(std::move(word));
  return words;
}

std::vector<std::string> WrapText(const std::string& text, std::size_t width) {
  std::vector<std::string> lines;
  std::string line;
  for (const std::string& word : SplitWords(text)) {
    if (line.empty()) {
      line = word;
    } else if (line.size() + 1 + word.size() <= width) {
      line += ' ' + word;
    } else {
      lines.push_back(std::move(line));
      line = word;
    }
  }
  if (!line.empty()) lines.push_back(std::move(line));
  return lines;
}

/// Best "Did you mean" candidate for `typed`, or empty when nothing is close.
std::string Suggest(std::string_view typed, const std::vector<std::string>& candidates) {
  const std::size_t threshold = std::max<std::size_t>(2, typed.size() / 3);
  const std::string* best = nullptr;
  std::size_t best_distance = threshold + 1;
  for (const std::string& candidate : candidates) {
    const std::size_t distance = rapidfuzz::levenshtein_distance(typed, candidate);
    if (distance < best_distance) {
      best_distance = distance;
      best = &candidate;
    }
  }
  return best ? *best : std::string();
}

/// Option groups register as nameless subcommand apps; only named
/// subcommands are real commands.
std::vector<const CLI::App*> Subcommands(const CLI::App& app, bool named) {
  std::vector<const CLI::App*> result;

  for (const CLI::App* sub : app.get_subcommands([](const CLI::App*) { return true; })) {
    if (sub->get_name().empty() != named) result.push_back(sub);
  }

  return result;
}

std::vector<const CLI::App*> Children(const CLI::App& app) { return Subcommands(app, true); }
std::vector<const CLI::App*> OptionGroups(const CLI::App& app) {
  return Subcommands(app, false);
}

const CLI::App* FindChild(const CLI::App& app, const std::string& name) {
  for (const CLI::App* sub : Children(app)) {
    if (sub->get_name() == name) return sub;
  }

  return nullptr;
}

/// Every option visible on `app`'s help: its own plus option-group-owned.
std::vector<const CLI::Option*> AllOptions(const CLI::App& app) {
  std::vector<const CLI::Option*> options = app.get_options();

  for (const CLI::App* group : OptionGroups(app)) {
    const auto grouped = group->get_options();
    options.insert(options.end(), grouped.begin(), grouped.end());
  }

  return options;
}

bool IsHelpFlag(const CLI::App& app, const CLI::Option* option) {
  return option == app.get_help_ptr();
}

/// True for options shown in the main "Options:" section (not owned by a group).
bool IsUngrouped(const CLI::Option& option) {
  const std::string& group = option.get_group();
  return group.empty() || group == "OPTIONS";
}

bool TakesValue(const CLI::Option& option) { return !option.get_type_name().empty(); }

std::string LongName(const CLI::Option& option) {
  if (!option.get_lnames().empty()) return "--" + option.get_lnames().front();
  if (!option.get_snames().empty()) return "-" + option.get_snames().front();
  return {};
}

std::string PlaceholderFor(const std::string& option_name) {
  return fmt::format("<{}>", StripDashes(option_name));
}

std::vector<std::string> OptionCandidates(const CLI::App& app) {
  std::vector<std::string> candidates{"--help"};
  for (const CLI::Option* option : AllOptions(app)) {
    if (IsHelpFlag(app, option)) continue;
    if (std::string name = LongName(*option); !name.empty())
      candidates.push_back(std::move(name));
  }
  return candidates;
}

std::vector<std::string> CommandCandidates(const CLI::App& app) {
  std::vector<std::string> candidates;
  for (const CLI::App* sub : Children(app)) candidates.push_back(sub->get_name());
  return candidates;
}

}  // namespace

std::string CommandPath(const CLI::App& app) {
  if (const CLI::App* parent = app.get_parent())
    return CommandPath(*parent) + " " + app.get_name();
  return app.get_name();
}

namespace {

/// One aligned help row; `left` stays unstyled so column math uses plain widths.
struct HelpEntry {
  std::string left;
  std::string right;
  std::string default_annotation;  // "[default: X]" unit; wrapped whole.
  bool is_command = false;         // Style the left column as a command name.
};

std::vector<HelpEntry> CommandEntries(const CLI::App& app) {
  std::vector<HelpEntry> entries;

  for (const CLI::App* sub : Children(app))
    entries.push_back({sub->get_name(), sub->get_description(), {}, true});

  return entries;
}

std::string OptionLeft(const CLI::Option& option) {
  const std::string name = LongName(option);
  if (!TakesValue(option)) return name;
  // The registered metavariable, e.g. "--batch-size <N>". CLI11 appends
  // validator detail after a ':' ("N:POSITIVE"); the help layout omits it.
  const std::string& full = option.get_type_name();
  return fmt::format("{} <{}>", name, full.substr(0, full.find(':')));
}

/// The "[default: VALUE]" annotation, or empty; wrapped as one unit so it
/// never splits across lines.
std::string DefaultAnnotation(const CLI::Option& option) {
  return option.get_default_str().empty()
             ? std::string()
             : fmt::format("[default: {}]", option.get_default_str());
}

std::vector<HelpEntry> OptionEntries(const std::vector<const CLI::Option*>& options) {
  std::vector<HelpEntry> entries;

  for (const CLI::Option* option : options)
    entries.push_back({OptionLeft(*option), option->get_description(), DefaultAnnotation(*option)});

  return entries;
}

/// An option and the option-group app that owns it (null for the app's own).
struct OwnedOption {
  const CLI::Option* option;
  const CLI::App* owner;
};

std::vector<OwnedOption> OwnedOptions(const CLI::App& app) {
  std::vector<OwnedOption> result;
  for (const CLI::Option* option : app.get_options()) result.push_back({option, nullptr});
  for (const CLI::App* group : OptionGroups(app))
    for (const CLI::Option* option : group->get_options()) {
      // Group apps inherit the CLI's own help flags; they are not content.
      if (option == group->get_help_ptr() || option == group->get_help_all_ptr()) continue;
      result.push_back({option, group});
    }
  return result;
}

/// The section an option renders under: its own group when set (e.g.
/// "Advanced"), else its owning group's name, else the main Options:.
std::string EffectiveGroup(const OwnedOption& owned) {
  const std::string& own = owned.option->get_group();
  if (!own.empty() && own != "OPTIONS") return own;
  return owned.owner != nullptr ? owned.owner->get_group() : std::string();
}

/// True when any option on `app` is filed under the advanced group.
bool HasAdvanced(const CLI::App& app) {
  for (const OwnedOption& owned : OwnedOptions(app))
    if (EffectiveGroup(owned) == "Advanced") return true;
  return false;
}

class HelpRenderer {
 public:
  /// `all` renders the advanced options and omits the "more help" pointer.
  explicit HelpRenderer(const TextStyler& style, bool all = false)
      : style_(style), all_(all) {}

  std::string Render(const CLI::App& app) const {
    std::string out;
    if (!app.get_description().empty()) out += app.get_description() + "\n\n";
    out += Usage(app) + "\n";
    out += CommandsSection(app);
    out += OptionsSection(app);
    out += OptionGroupSections(app);
    out += Footer(app);
    return out;
  }

  std::string Usage(const CLI::App& app) const {
    const char* suffix = Children(app).empty() ? " [options]" : " <command> [options]";
    return fmt::format("{}\n  {}{}", style_.Heading("Usage:"), style_.Command(CommandPath(app)),
                       StylePlaceholders(suffix));
  }

  /// Aligned rows, wrapped to kWrapWidth. A signature longer than
  /// kMaxLeftWidth hangs: it gets its own line with the description at the
  /// fixed kHangIndent, so one long option never pushes every description
  /// into a deep column. A blank line separates aligned from hanging rows.
  std::string Entries(const std::vector<HelpEntry>& entries) const {
    const std::size_t column = LeftColumnWidth(entries);
    std::string out;
    bool hanging = false;
    for (std::size_t i = 0; i < entries.size(); ++i) {
      const bool entry_hangs = entries[i].left.size() > kMaxLeftWidth;
      if (i > 0 && entry_hangs && !hanging) out += "\n";
      out += Entry(entries[i], column);
      hanging = entry_hangs;
    }
    return out;
  }

  std::string Section(const std::string& title, const std::vector<HelpEntry>& entries) const {
    return fmt::format("\n{}\n{}", style_.Heading(title), Entries(entries));
  }

 private:
  /// Wraps every simple <placeholder> token in the placeholder style. A no-op
  /// in plain mode, so message text stays byte-identical when color is off.
  std::string StylePlaceholders(const std::string& text) const {
    std::string out;
    std::size_t pos = 0;
    while (pos < text.size()) {
      const std::size_t open = text.find('<', pos);
      if (open == std::string::npos) break;
      const std::size_t close = text.find('>', open);
      if (close == std::string::npos) break;
      const std::string token = text.substr(open, close - open + 1);
      out += text.substr(pos, open - pos);
      out += token.find(' ') == std::string::npos ? style_.Placeholder(token) : token;
      pos = close + 1;
    }
    return out + text.substr(pos);
  }

  static std::size_t LeftColumnWidth(const std::vector<HelpEntry>& entries) {
    std::size_t column = kMinColumn;
    for (const HelpEntry& entry : entries) {
      if (entry.left.size() <= kMaxLeftWidth)
        column = std::max(column, entry.left.size() + kColumnGap);
    }
    return column;
  }

  std::string Entry(const HelpEntry& entry, std::size_t column) const {
    const std::string left =
        entry.is_command ? style_.Command(entry.left) : StylePlaceholders(entry.left);
    // All description lines share one column -- the row indent plus the left
    // column (or the hanging indent) -- so wrapped continuations align under
    // the first description line.
    const bool hangs = entry.left.size() > kMaxLeftWidth;
    const std::size_t indent_width = hangs ? kHangIndent : column + 2;
    const std::size_t width = std::max<std::size_t>(1, kWrapWidth - indent_width);
    std::vector<std::string> lines = WrapText(entry.right, width);
    if (!entry.default_annotation.empty()) {
      // Keep "[default: VALUE]" whole: append to the last line when it fits.
      const std::size_t used = lines.empty() ? 0 : lines.back().size();
      if (lines.empty() || used + 1 + entry.default_annotation.size() > width)
        lines.push_back(entry.default_annotation);
      else
        lines.back() += " " + entry.default_annotation;
    }
    const std::string indent(indent_width, ' ');
    const std::string first = lines.empty() ? std::string() : lines.front();

    // An over-long left column gets its own line; the description drops below.
    std::string out = hangs
                          ? fmt::format("  {}\n{}{}\n", left, indent, first)
                          : fmt::format("  {}{}{}\n", left,
                                        std::string(column - entry.left.size(), ' '), first);
    for (std::size_t i = 1; i < lines.size(); ++i) out += indent + lines[i] + "\n";
    return out;
  }

  std::string CommandsSection(const CLI::App& app) const {
    const std::vector<HelpEntry> entries = CommandEntries(app);
    return entries.empty() ? std::string() : Section("Commands:", entries);
  }

  /// The built-in help flag always leads the ungrouped section. The help-all
  /// flag stays hidden until there is something advanced to show with it.
  std::string OptionsSection(const CLI::App& app) const {
    std::vector<const CLI::Option*> ungrouped;
    for (const OwnedOption& owned : OwnedOptions(app)) {
      if (owned.owner != nullptr || IsHelpFlag(app, owned.option)) continue;
      if (!IsUngrouped(*owned.option)) continue;
      if (!all_ && !HasAdvanced(app) && owned.option == app.get_help_all_ptr()) continue;
      ungrouped.push_back(owned.option);
    }
    std::vector<HelpEntry> entries = OptionEntries(ungrouped);
    entries.insert(entries.begin(), {"-h, --help", "Print this help message and exit", {}});
    return Section("Options:", entries);
  }

  /// Named option groups in declaration order; the advanced group renders
  /// only in --help=all mode.
  std::string OptionGroupSections(const CLI::App& app) const {
    std::vector<std::string> names;
    std::map<std::string, std::vector<const CLI::Option*>> sections;
    for (const OwnedOption& owned : OwnedOptions(app)) {
      const std::string group = EffectiveGroup(owned);
      if (group.empty() || (group == "Advanced" && !all_)) continue;
      if (sections.find(group) == sections.end()) names.push_back(group);
      sections[group].push_back(owned.option);
    }
    std::string out;
    for (const std::string& group : names)
      out += Section(group + ":", OptionEntries(sections[group]));
    return out;
  }

  /// The pointer to more help: --help=all where advanced options exist, and
  /// nothing in all mode -- the reader is already viewing the detail.
  std::string Footer(const CLI::App& app) const {
    if (all_) return {};
    if (app.get_parent() != nullptr)
      return HasAdvanced(app)
                 ? fmt::format("\nRun '{} --help=all' to show all options.\n", CommandPath(app))
                 : fmt::format("\nRun '{} --help' for more information.\n", CommandPath(app));
    return "\nExamples:\n"
           "  $ inferx serve --model models/Qwen3-0.6B\n"
           "  $ inferx bench latency --help\n"
           "  $ inferx bench throughput --help\n\n"
           "Run 'inferx <command> --help' for more information.\n";
  }

  const TextStyler& style_;
  bool all_;
};

/// First "--name" token inside a CLI11 message.
std::string FirstOptionToken(const std::string& what) {
  const std::size_t start = what.find("--");

  if (start == std::string::npos) return {};

  const std::size_t end = what.find_first_of(" \t,:)", start);

  return what.substr(start, end == std::string::npos ? end : end - start);
}

/// First word of `text` (up to the first space).
std::string FirstWord(std::string text) { return text.substr(0, text.find(' ')); }

/// Writes user-facing error reports for one parse against `target`, the
/// deepest subcommand named on the command line.
class ParseErrorReporter {
 public:
  ParseErrorReporter(std::ostream& err, const TextStyler& style, const CLI::App& target)
      : err_(err), style_(style), help_(style), target_(target) {}

  /// Translates a CLI11 error into a report. `unmatched` is the first command
  /// -line token that named no subcommand.
  void Report(const CLI::ParseError& error, const std::string& unmatched) {
    const std::string name = error.get_name();
    const std::string what = error.what();

    if (name == "ExtrasError") return ReportExtras(what);
    if (name == "RequiredError") return ReportRequired(what, unmatched);
    if (name == "ArgumentMismatch") return ReportArgumentMismatch(what, unmatched);
    if (name == "ValidationError") return ReportValidation(what);
    if (name == "ConversionError") return ReportConversion(what);

    Generic(what);
  }

  void UnknownOption(const std::string& name) {
    Line("unknown option '{}'", style_.Command(name));
    Suggestion(Suggest(name, OptionCandidates(target_)));
    TryHelp();
  }

  void UnknownCommand(const std::string& name) {
    Line("unknown command '{}'", style_.Command(name));
    Suggestion(Suggest(name, CommandCandidates(target_)));
    const std::vector<HelpEntry> commands = CommandEntries(target_);

    if (!commands.empty())
      fmt::print(err_, "{}", help_.Section("Available commands:", commands));

    TryHelp();
  }

  void MissingSubcommand() {
    Line("missing required command {}", style_.Placeholder("<command>"));
    fmt::print(err_, "\n{}\n{}", help_.Usage(target_),
               help_.Section("Available commands:", CommandEntries(target_)));
    TryHelp();
  }

  void MissingOptions(const std::vector<std::string>& names) {
    fmt::print(err_, "{}", style_.Error("error:"));

    for (std::size_t i = 0; i < names.size(); ++i) {
      fmt::print(err_, "{}missing required option '{}' {}", i == 0 ? " " : "\n",
                 style_.Command(names[i]), style_.Placeholder(PlaceholderFor(names[i])));
    }

    fmt::print(err_, "\n\n{}\n", help_.Usage(target_));
    TryHelp();
  }

  void MissingValue(const std::string& option_name) {
    Line("missing value {} for '{}'", style_.Placeholder(PlaceholderFor(option_name)),
         style_.Command(option_name));
    TryHelp();
  }

  void InvalidValue(const std::string& value, const std::string& option_name,
                    const std::string& detail) {
    fmt::print(err_, "{} invalid value '{}' for '{}'", style_.Error("error:"), value,
               style_.Command(option_name));
    if (!detail.empty()) fmt::print(err_, ": {}", detail);
    fmt::print(err_, "\n");
    TryHelp();
  }

  void Generic(const std::string& what) {
    Line("{}", what);
    TryHelp();
  }

 private:
  template <typename... Args>
  void Line(fmt::format_string<Args...> format, Args&&... args) {
    fmt::print(err_, "{} {}\n", style_.Error("error:"),
               fmt::format(format, std::forward<Args>(args)...));
  }

  void Suggestion(const std::string& best) {
    if (!best.empty()) fmt::print(err_, "\n  Did you mean '{}'?\n", style_.Command(best));
  }

  void TryHelp() {
    fmt::print(err_, "\nFor more information, try '{}'.\n",
               style_.Command(CommandPath(target_) + " --help"));
  }

  // "inferx: The following argument(s) were not expected: <tokens>"
  void ReportExtras(const std::string& what) {
    std::vector<std::string> tokens;

    if (const std::size_t marker = what.find("not expected"); marker != std::string::npos) {
      if (const std::size_t colon = what.find(':', marker); colon != std::string::npos)
        tokens = SplitWords(what.substr(colon + 1));
    }

    std::string option, word;

    for (const std::string& token : tokens) {
      if (token[0] == '-') {
        if (option.empty()) option = token.substr(0, token.find('='));
      } else if (word.empty()) {
        word = token;
      }
    }

    if (!option.empty()) return UnknownOption(option);
    if (!word.empty() && !Children(target_).empty()) return UnknownCommand(word);
    Generic(what);
  }

  void ReportRequired(const std::string& what, const std::string& unmatched) {
    if (what.find("subcommand") != std::string::npos) {
      return unmatched.empty() ? MissingSubcommand() : UnknownCommand(unmatched);
    }

    // "--fixture is required" (option names joined by spaces).
    constexpr std::string_view kSuffix = " is required";
    std::string names = what;

    if (EndsWith(names, kSuffix)) names.resize(names.size() - kSuffix.size());

    MissingOptions(SplitWords(names));
  }

  void ReportArgumentMismatch(const std::string& what, const std::string& unmatched) {
    const std::string option = FirstOptionToken(what);
    MissingValue(option.empty() ? unmatched : option);
  }

  // "--port: Value 99999 not in range 1 - 65535"
  void ReportValidation(const std::string& what) {
    const std::size_t colon = what.find(": ");

    if (colon == std::string::npos) return InvalidValue(FirstWord(""), what, "");

    const std::string detail = what.substr(colon + 2);
    std::string value = detail;

    if (StartsWith(value, "Value ")) value.erase(0, 6);

    InvalidValue(FirstWord(value), what.substr(0, colon), detail);
  }

  // "... The value abc ... --port"
  void ReportConversion(const std::string& what) {
    constexpr std::string_view kMarker = "The value ";
    std::string value;

    if (const std::size_t marker = what.find(kMarker); marker != std::string::npos)
      value = FirstWord(what.substr(marker + kMarker.size()));

    InvalidValue(value, FirstOptionToken(what), "");
  }

  std::ostream& err_;
  const TextStyler& style_;
  HelpRenderer help_;
  const CLI::App& target_;
};

/// Where the command line points: the help target and error-hint context.
struct CommandScan {
  const CLI::App* target;  // Deepest subcommand named on the command line.
  std::string unmatched;   // First token naming no subcommand.
  bool in_help_command = false;
};

/// The words after the root `help` command name a command path ("inferx help
/// bench latency" describes `inferx bench latency`), so the scan keeps
/// descending from the root instead of into the help subcommand itself.
CommandScan ScanCommandPath(const CLI::App& root, const std::vector<std::string>& args) {
  CommandScan scan{&root, {}, false};

  for (const std::string& token : args) {
    if (token == "--") break;
    if (StartsWith(token, "-")) continue;

    const CLI::App* next = FindChild(*scan.target, token);

    if (next && scan.target == &root && next->get_name() == "help") {
      scan.in_help_command = true;
    } else if (next) {
      scan.target = next;
    } else if (scan.unmatched.empty()) {
      scan.unmatched = token;
    }
  }
  return scan;
}

/// A value option followed by another option (or "--") never takes that token
/// as its value. Returns the option missing its value, if any. Values that
/// merely start with '-' (e.g. negative numbers) still parse as values.
template <typename ValueOptionMap>
std::optional<std::string> FindOptionMissingValue(const std::vector<std::string>& args,
                                                  const ValueOptionMap& value_options) {
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {  // CLI11 reports a dangling last option.
    const std::string& token = args[i];

    if (token == "--") break;
    if (!StartsWith(token, "-")) continue;

    const std::string name = token.substr(0, token.find('='));
    const auto known = value_options.find(name);

    if (known == value_options.end() || !known->second) continue;

    const std::string& next = args[i + 1];

    if (next == "--" || (StartsWith(next, "-") && value_options.count(next) > 0)) return name;
  }

  return std::nullopt;
}

}  // namespace

Console::Console(CLI::App& app, std::ostream& out, std::ostream& err)
    : app_(app),
      out_(out),
      err_(err),
      out_style_(ShouldUseColor(IsInteractiveTerminal(stdout))),
      err_style_(ShouldUseColor(IsInteractiveTerminal(stderr))) {
  // CLI11 copies the formatter into subcommands when they are created, and
  // the tree already exists by now, so walk it.
  std::function<void(CLI::App*)> install = [&](CLI::App* node) {
    node->formatter_fn([this](const CLI::App* help_app, std::string, CLI::AppFormatMode mode) {
      return HelpRenderer(out_style_, mode == CLI::AppFormatMode::All).Render(*help_app);
    });
    for (CLI::App* sub : node->get_subcommands([](const CLI::App*) { return true; })) install(sub);
  };
  install(&app_);
  CollectValueOptions(&app_);
}

void Console::CollectValueOptions(CLI::App* app) {
  for (const CLI::Option* option : AllOptions(*app)) {
    const bool takes_value = TakesValue(*option);
    for (const std::string& name : option->get_lnames())
      value_options_["--" + name] = takes_value;
    for (const std::string& name : option->get_snames())
      value_options_["-" + name] = takes_value;
  }
  for (CLI::App* sub : app->get_subcommands([](CLI::App*) { return true; }))
    CollectValueOptions(sub);
}

void Console::Error(std::string_view message) const {
  fmt::print(err_, "{} {}\n", err_style_.Error("error:"), message);
}

void Console::Warning(std::string_view message) const {
  fmt::print(err_, "{} {}\n", err_style_.Warning("warning:"), message);
}

void Console::Success(std::string_view message) const {
  fmt::print(out_, "{} {}\n", out_style_.Success("success:"), message);
}

int Console::PrintHelp(const CLI::App* app, bool all) {
  fmt::print(out_, "{}", HelpRenderer(out_style_, all).Render(*app));
  return 0;
}

int Console::Parse(int argc, const char* const argv[]) {
  std::vector<std::string> args(argv + 1, argv + argc);
  // CLI11 rejects "=" in flag names, so the friendlier "--help=all" spelling
  // is normalized onto the registered "--help-all" flag.
  for (std::string& token : args)
    if (token == "--help=all") token = "--help-all";
  const CommandScan scan = ScanCommandPath(app_, args);
  ParseErrorReporter reporter(err_, err_style_, *scan.target);

  // CLI11 accepts any `help` path (the positional swallows the words), so a
  // path naming no command is reported here instead of by the parse.
  if (scan.in_help_command && !scan.unmatched.empty()) {
    reporter.UnknownCommand(scan.unmatched);
    return kExtrasErrorExit;
  }

  // Report the missing value ourselves instead of letting CLI11 consume the
  // next option and strand a later token as an extra.
  if (const auto option = FindOptionMissingValue(args, value_options_)) {
    reporter.MissingValue(*option);
    return kArgumentMismatchExit;
  }

  if (args.empty()) return PrintHelp(&app_);

  std::vector<const char*> raw{"inferx"};  // argv[0]; the tree already carries its names.

  for (const std::string& token : args) raw.push_back(token.c_str());

  try {
    app_.parse(static_cast<int>(raw.size()), raw.data());
  } catch (const CLI::CallForHelp&) {
    return PrintHelp(scan.target);
  } catch (const CLI::CallForAllHelp&) {
    return PrintHelp(scan.target, /*all=*/true);
  } catch (const CLI::ParseError& error) {
    return ReportParseError(error, scan.target, scan.unmatched, error.get_exit_code());
  }

  return 0;
}

int Console::ReportParseError(const CLI::ParseError& error, const CLI::App* target,
                              const std::string& unmatched, int code) {
  ParseErrorReporter(err_, err_style_, *target).Report(error, unmatched);
  return code;
}

}  // namespace inferx::cli