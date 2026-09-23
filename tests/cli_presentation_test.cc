#include "cli/presentation.h"

#include <sstream>

#include "gtest/gtest.h"

namespace inferx::cli {
namespace {

TEST(TerminalStyles, SemanticStylesAndPlainText) {
  const term::StyleSheet plain(false), color(true);
  EXPECT_EQ(plain.Error("error:"), "error:");
  EXPECT_EQ(plain.Warning("warning:"), "warning:");
  EXPECT_EQ(plain.Success("success:"), "success:");
  EXPECT_EQ(plain.Command("--model"), "--model");
  EXPECT_EQ(plain.Placeholder("<model>"), "<model>");
  EXPECT_EQ(color.Error("error:"), "\033[1;31merror:\033[0m");
  EXPECT_EQ(color.Warning("<model>"), "\033[33m<model>\033[0m");
  EXPECT_EQ(color.Success("ok"), "\033[32mok\033[0m");
  EXPECT_EQ(color.Command("serve"), "\033[36mserve\033[0m");
  EXPECT_EQ(color.Heading("Usage:"), "\033[1mUsage:\033[0m");
  EXPECT_EQ(color.Dim("hint"), "\033[2mhint\033[0m");
  EXPECT_EQ(color.Placeholder("<model>"), "\033[33m<model>\033[0m");
}

TEST(Presentation, MessagesUseTheCorrectStream) {
  CLI::App app{"test", "custom"};
  app.add_subcommand("run");
  std::ostringstream out, err;
  Presentation presentation(app, out, err, false, false);
  const char* args[] = {"custom", "--color=never", "run"};
  ASSERT_EQ(presentation.Parse(3, args), 0);
  presentation.Error("cannot open model");
  presentation.Warning("using default model");
  presentation.Success("model loaded");
  EXPECT_EQ(out.str(), "success: model loaded\n");
  EXPECT_EQ(err.str(), "error: cannot open model\nwarning: using default model\n");
}

TEST(Presentation, DerivesPathsAndPreservesPositionalParsing) {
  CLI::App app{"test", "custom"};
  auto* group = app.add_subcommand("tools");
  auto* command = group->add_subcommand("inspect");
  std::string input;
  command->add_option("input", input)->required();
  std::ostringstream out, err;
  Presentation presentation(app, out, err, false, false);
  EXPECT_EQ(CommandPath(*command), "custom tools inspect");
  const char* args[] = {"custom", "tools", "inspect", "--", "--color=always"};
  ASSERT_EQ(presentation.Parse(5, args), 0);
  EXPECT_EQ(input, "--color=always");
  EXPECT_EQ(out.str(), "");
  EXPECT_EQ(err.str(), "");
}

}  // namespace
}  // namespace inferx::cli
