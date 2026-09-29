// In-process help-rendering contracts: metavariables, long-signature
// wrapping, default annotations, and the --help / --help=all / help-command
// footer behavior. Everything renders through the one Console path.
#include <cstdlib>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "cli/app.h"
#include "cli/console.h"

namespace {

using namespace inferx::cli;

std::vector<std::string> Lines(const std::string& text) {
  std::vector<std::string> lines;
  std::istringstream stream(text);
  for (std::string line; std::getline(stream, line);) lines.push_back(line);
  return lines;
}

class ConsoleHelpTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Pin plain rendering regardless of the runner's environment.
    ::setenv("NO_COLOR", "1", 1);
    ::unsetenv("CLICOLOR_FORCE");
  }

  /// Parses `args` (no program name) against a fresh command tree and
  /// returns the exit code plus the captured help output.
  std::pair<int, std::string> Render(const std::vector<std::string>& args) {
    InferxCli cli{"InferX test tree", "inferx"};
    std::ostringstream out, err;
    Console console(cli.app(), out, err);
    std::vector<const char*> argv{"inferx"};
    for (const std::string& arg : args) argv.push_back(arg.c_str());
    const int code = console.Parse(static_cast<int>(argv.size()), argv.data());
    return {code, out.str()};
  }

  /// Every "[default: ...]" annotation must close on the line it opens.
  static void ExpectDefaultsIntact(const std::string& help) {
    for (const std::string& line : Lines(help)) {
      const std::size_t open = line.find("[default:");
      if (open == std::string::npos) continue;
      EXPECT_NE(line.find(']', open), std::string::npos) << "split default: " << line;
    }
  }
};

TEST_F(ConsoleHelpTest, ConciseMetavariables) {
  const auto [code, help] = Render({"bench", "latency", "--help"});
  ASSERT_EQ(code, 0);
  for (const std::string& signature :
       {"--batch-size <N>", "--output-len <N>", "--input-len <N>", "--num-iters <N>",
        "--num-iters-warmup <N>", "--model <PATH>", "--tokenizer <PATH>", "--device <TYPE>",
        "--device-ids <IDS>", "--dtype <TYPE>", "--kv-cache-memory-bytes <BYTES>"}) {
    EXPECT_NE(help.find(signature), std::string::npos) << signature;
  }
  // Name-derived placeholders are gone.
  EXPECT_EQ(help.find("<batch-size>"), std::string::npos);
  EXPECT_EQ(help.find("<model>"), std::string::npos);
}

TEST_F(ConsoleHelpTest, LongSignaturesHangInsteadOfAligning) {
  const auto [code, help] = Render({"bench", "latency", "--help"});
  ASSERT_EQ(code, 0);
  // A long signature gets its own line; the description sits at the fixed
  // hanging indent rather than a deep per-section column.
  EXPECT_NE(help.find("\n  --max-num-batched-tokens <N>\n"
                      "      Maximum number of tokens per batch [default: 4096]\n"),
            std::string::npos);
  // Short signatures keep the aligned two-column layout.
  EXPECT_NE(help.find("  --max-num-seqs <N>  Maximum number of sequences per iteration"),
            std::string::npos);
  // A blank line separates aligned from hanging rows in a section.
  EXPECT_NE(help.find("[default: 32]\n\n  --max-num-batched-tokens <N>"), std::string::npos);
}

TEST_F(ConsoleHelpTest, DefaultAnnotationsStayWhole) {
  for (const std::vector<std::string>& args :
       {std::vector<std::string>{"bench", "latency", "--help"},
        std::vector<std::string>{"bench", "latency", "--help=all"},
        std::vector<std::string>{"serve", "--help=all"},
        std::vector<std::string>{"bench", "throughput", "--help"}}) {
    const auto [code, help] = Render(args);
    ASSERT_EQ(code, 0) << args.front();
    ExpectDefaultsIntact(help);
  }
}

TEST_F(ConsoleHelpTest, NormalHelpHidesAdvancedOptions) {
  const auto [code, help] = Render({"bench", "latency", "--help"});
  ASSERT_EQ(code, 0);
  // Signatures, not substrings: descriptions may cross-reference hidden
  // options ("0 derives it from --num-kv-blocks").
  EXPECT_EQ(help.find("\n  --num-kv-blocks <"), std::string::npos);
  EXPECT_EQ(help.find("\n  --tensor-parallel-size <"), std::string::npos);
  EXPECT_EQ(help.find("Advanced:"), std::string::npos);
}

TEST_F(ConsoleHelpTest, FooterPointsToHelpAllAndHelpAllOmitsIt) {
  const auto [normal_code, normal] = Render({"bench", "latency", "--help"});
  ASSERT_EQ(normal_code, 0);
  EXPECT_NE(normal.find("Run 'inferx bench latency --help=all' to show all options.\n"),
            std::string::npos);

  const auto [all_code, all] = Render({"bench", "latency", "--help=all"});
  ASSERT_EQ(all_code, 0);
  EXPECT_EQ(all.find("for more information"), std::string::npos);
  EXPECT_NE(all.find("\n  --num-kv-blocks <"), std::string::npos);
  EXPECT_NE(all.find("Advanced:"), std::string::npos);
}

TEST_F(ConsoleHelpTest, HelpAllSpellingsRenderIdentically) {
  const auto [eq_sign_code, eq_sign] = Render({"bench", "latency", "--help=all"});
  const auto [dash_code, dash] = Render({"bench", "latency", "--help-all"});
  ASSERT_EQ(eq_sign_code, 0);
  ASSERT_EQ(dash_code, 0);
  EXPECT_EQ(eq_sign, dash);
}

TEST_F(ConsoleHelpTest, HelpCommandMatchesHelpFlag) {
  const auto [cmd_code, cmd] = Render({"help", "bench", "latency"});
  const auto [flag_code, flag] = Render({"bench", "latency", "--help"});
  ASSERT_EQ(cmd_code, 0);
  ASSERT_EQ(flag_code, 0);
  EXPECT_EQ(cmd, flag);

  const auto [root_cmd_code, root_cmd] = Render({"help"});
  const auto [root_flag_code, root_flag] = Render({"--help"});
  ASSERT_EQ(root_cmd_code, 0);
  ASSERT_EQ(root_flag_code, 0);
  EXPECT_EQ(root_cmd, root_flag);
  EXPECT_NE(root_cmd.find("Run 'inferx <command> --help' for more information.\n"),
            std::string::npos);
}

TEST_F(ConsoleHelpTest, ParseErrorsKeepCli11ExitCodes) {
  InferxCli cli{"InferX test tree", "inferx"};
  std::ostringstream out, err;
  Console console(cli.app(), out, err);
  const char* argv[] = {"inferx", "--frob"};
  EXPECT_EQ(console.Parse(2, argv), 109);  // CLI11 ExtrasError
  EXPECT_TRUE(err.str().find("unknown option '--frob'") != std::string::npos);
}

}  // namespace
