#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "inferx/util/config_view.h"

namespace inferx {
namespace {

/// Parses `text` into `doc` and returns a root view over it; the caller must
/// keep `doc` alive for the view's lifetime, mirroring the HfConfig contract.
ConfigView RootView(const char* text, nlohmann::json& doc) {
  doc = nlohmann::json::parse(text);
  return ConfigView(&doc, "");
}

std::string Msg(const Status& s) { return s.ToString(); }

TEST(ConfigViewTest, PathsAreDottedFromRoot) {
  EXPECT_EQ(ConfigView().Path(), "");

  nlohmann::json doc;
  const auto view = RootView(R"({"child": {}})", doc);
  EXPECT_EQ(view.PathOf("hidden_size"), "hidden_size");

  const auto child = view.Sub("child");
  ASSERT_TRUE(child.ok()) << child.status();
  ASSERT_TRUE(child->has_value());
  EXPECT_EQ((*child)->Path(), "child");
  EXPECT_EQ((*child)->PathOf("hidden_size"), "child.hidden_size");
}

TEST(ConfigViewTest, HasTreatsAbsentAndNullAsMissing) {
  nlohmann::json doc;
  const auto view = RootView(R"({"a": 1, "b": null})", doc);
  EXPECT_TRUE(view.Has("a"));
  EXPECT_FALSE(view.Has("b"));
  EXPECT_FALSE(view.Has("c"));
}

TEST(ConfigViewTest, RequireIntReadsIntegerAndIntegralFloat) {
  nlohmann::json doc;
  const auto view = RootView(R"({"a": 128, "b": 128.0, "c": 1.5, "d": "7"})", doc);

  const auto a = view.RequireInt("a");
  ASSERT_TRUE(a.ok()) << a.status();
  EXPECT_EQ(*a, 128);

  // Some checkpoints write integral values as 128.0.
  const auto b = view.RequireInt("b");
  ASSERT_TRUE(b.ok()) << b.status();
  EXPECT_EQ(*b, 128);

  const auto c = view.RequireInt("c");
  ASSERT_FALSE(c.ok());
  EXPECT_EQ(c.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(Msg(c.status()).find("'c': expected integer"), std::string::npos);

  const auto d = view.RequireInt("d");
  ASSERT_FALSE(d.ok());
  EXPECT_NE(Msg(d.status()).find("expected integer, got string"), std::string::npos);
}

TEST(ConfigViewTest, RequireIntReportsMissingKeyPath) {
  nlohmann::json doc;
  const auto view = RootView(R"({"hidden_size": 32})", doc);
  const auto v = view.RequireInt("vocab_size");
  ASSERT_FALSE(v.ok());
  EXPECT_EQ(v.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(Msg(v.status()).find("'vocab_size' is required"), std::string::npos);
}

TEST(ConfigViewTest, IntFallsBackToDefaultOnlyOnAbsence) {
  nlohmann::json doc;
  const auto view = RootView(R"({"a": 8, "b": null})", doc);

  const auto missing = view.Int("vocab_size", 32000);
  ASSERT_TRUE(missing.ok()) << missing.status();
  EXPECT_EQ(*missing, 32000);

  // JSON null counts as absent.
  const auto null_key = view.Int("b", 4);
  ASSERT_TRUE(null_key.ok()) << null_key.status();
  EXPECT_EQ(*null_key, 4);

  // A malformed value must not silently become the default.
  const auto bad = RootView(R"({"a": "many"})", doc).Int("a", 4);
  ASSERT_FALSE(bad.ok());
  EXPECT_NE(Msg(bad.status()).find("expected integer"), std::string::npos);
}

TEST(ConfigViewTest, FloatAcceptsAnyNumber) {
  nlohmann::json doc;
  const auto view = RootView(R"({"a": 2, "b": 2.5, "c": "fast"})", doc);

  const auto a = view.Float("a", 0.5);
  ASSERT_TRUE(a.ok()) << a.status();
  EXPECT_DOUBLE_EQ(*a, 2.0);

  const auto b = view.RequireFloat("b");
  ASSERT_TRUE(b.ok()) << b.status();
  EXPECT_DOUBLE_EQ(*b, 2.5);

  const auto dflt = view.Float("missing", 0.25);
  ASSERT_TRUE(dflt.ok()) << dflt.status();
  EXPECT_DOUBLE_EQ(*dflt, 0.25);

  const auto bad = view.RequireFloat("c");
  ASSERT_FALSE(bad.ok());
  EXPECT_NE(Msg(bad.status()).find("'c': expected number"), std::string::npos);
}

TEST(ConfigViewTest, OptFloatIsNulloptOnlyWhenAbsent) {
  nlohmann::json doc;
  const auto view = RootView(R"({"factor": 2.0, "bad": "two"})", doc);

  const auto absent = view.OptFloat("missing");
  ASSERT_TRUE(absent.ok()) << absent.status();
  EXPECT_FALSE(absent->has_value());

  const auto present = view.OptFloat("factor");
  ASSERT_TRUE(present.ok()) << present.status();
  ASSERT_TRUE(present->has_value());
  EXPECT_DOUBLE_EQ(**present, 2.0);

  const auto bad = view.OptFloat("bad");
  ASSERT_FALSE(bad.ok());
  EXPECT_NE(Msg(bad.status()).find("expected number"), std::string::npos);
}

TEST(ConfigViewTest, BoolChecksType) {
  nlohmann::json doc;
  const auto view = RootView(R"({"a": true, "b": 1})", doc);

  const auto a = view.Bool("a", false);
  ASSERT_TRUE(a.ok()) << a.status();
  EXPECT_TRUE(*a);

  const auto dflt = view.Bool("missing", true);
  ASSERT_TRUE(dflt.ok()) << dflt.status();
  EXPECT_TRUE(*dflt);

  const auto bad = view.Bool("b", false);
  ASSERT_FALSE(bad.ok());
  EXPECT_NE(Msg(bad.status()).find("'b': expected boolean"), std::string::npos);
}

TEST(ConfigViewTest, StrChecksType) {
  nlohmann::json doc;
  const auto view = RootView(R"({"dtype": "bfloat16", "n": 3})", doc);

  const auto req = view.RequireStr("dtype");
  ASSERT_TRUE(req.ok()) << req.status();
  EXPECT_EQ(*req, "bfloat16");

  const auto dflt = view.Str("missing", "float16");
  ASSERT_TRUE(dflt.ok()) << dflt.status();
  EXPECT_EQ(*dflt, "float16");

  const auto req_missing = view.RequireStr("absent");
  ASSERT_FALSE(req_missing.ok());
  EXPECT_NE(Msg(req_missing.status()).find("'absent' is required"), std::string::npos);

  const auto bad = view.Str("n", "float16");
  ASSERT_FALSE(bad.ok());
  EXPECT_NE(Msg(bad.status()).find("expected string"), std::string::npos);
}

TEST(ConfigViewTest, IntListAcceptsScalarArrayAndAbsence) {
  nlohmann::json doc;
  const auto view = RootView(
      R"({"eos_token_id": 7, "stop": [2, 3, 5], "integral": [9.0], "mixed": [2, "x"]})",
      doc);

  const auto absent = view.IntList("missing");
  ASSERT_TRUE(absent.ok()) << absent.status();
  EXPECT_EQ(*absent, (std::vector<int64_t>{}));

  const auto scalar = view.IntList("eos_token_id");
  ASSERT_TRUE(scalar.ok()) << scalar.status();
  EXPECT_EQ(*scalar, (std::vector<int64_t>{7}));

  const auto arr = view.IntList("stop");
  ASSERT_TRUE(arr.ok()) << arr.status();
  EXPECT_EQ(*arr, (std::vector<int64_t>{2, 3, 5}));

  const auto integral = view.IntList("integral");
  ASSERT_TRUE(integral.ok()) << integral.status();
  EXPECT_EQ(*integral, (std::vector<int64_t>{9}));

  const auto mixed = view.IntList("mixed");
  ASSERT_FALSE(mixed.ok());
  EXPECT_NE(Msg(mixed.status()).find("'mixed[1]'"), std::string::npos);
  EXPECT_NE(Msg(mixed.status()).find("expected integer"), std::string::npos);
}

TEST(ConfigViewTest, SubReturnsNestedViewOrNullopt) {
  nlohmann::json doc;
  const auto view = RootView(
      R"({"text_config": {"rope_scaling": {"factor": "big"}}, "flat": 1})", doc);

  const auto absent = view.Sub("missing");
  ASSERT_TRUE(absent.ok()) << absent.status();
  EXPECT_FALSE(absent->has_value());

  const auto not_object = view.Sub("flat");
  ASSERT_FALSE(not_object.ok());
  EXPECT_NE(Msg(not_object.status()).find("'flat': expected object"), std::string::npos);

  // Errors from nested views report the full dotted key path.
  const auto text = view.Sub("text_config");
  ASSERT_TRUE(text.ok()) << text.status();
  ASSERT_TRUE(text->has_value());
  const auto rope = (*text)->Sub("rope_scaling");
  ASSERT_TRUE(rope.ok()) << rope.status();
  ASSERT_TRUE(rope->has_value());
  const auto factor = (*rope)->RequireInt("factor");
  ASSERT_FALSE(factor.ok());
  EXPECT_NE(Msg(factor.status()).find("'text_config.rope_scaling.factor'"),
            std::string::npos);
}

TEST(ConfigViewTest, RejectIfPresentGivesUnimplementedWithReason) {
  nlohmann::json doc;
  const auto view = RootView(R"({"bias": true})", doc);

  const auto absent = view.RejectIfPresent("absent", "bias");
  ASSERT_TRUE(absent.ok()) << absent;

  const auto present = view.RejectIfPresent("bias", "bias");
  ASSERT_FALSE(present.ok());
  EXPECT_EQ(present.code(), absl::StatusCode::kUnimplemented);
  EXPECT_NE(Msg(present).find("'bias' is set: bias is not supported"), std::string::npos);
}

TEST(ConfigViewTest, DefaultViewAndNullNodeBehaveLikeEmpty) {
  const ConfigView empty;
  EXPECT_FALSE(empty.Has("a"));
  EXPECT_FALSE(empty.RequireInt("a").ok());
  const auto sub = empty.Sub("a");
  ASSERT_TRUE(sub.ok()) << sub.status();
  EXPECT_FALSE(sub->has_value());

  nlohmann::json null_doc = nullptr;
  const ConfigView null_view(&null_doc, "");
  EXPECT_FALSE(null_view.Has("a"));
  EXPECT_FALSE(null_view.RequireStr("a").ok());
}

}  // namespace
}  // namespace inferx
