#include "inferx/common/json.h"

#include <gtest/gtest.h>

namespace inferx {
namespace {

TEST(Json, ParsesScalars) {
  EXPECT_TRUE(ParseJson("null")->IsNull());
  EXPECT_TRUE(*ParseJson("true")->AsBool());
  EXPECT_FALSE(*ParseJson("false")->AsBool());
  EXPECT_EQ(*ParseJson("42")->AsInt(), 42);
  EXPECT_EQ(*ParseJson("-7")->AsInt(), -7);
  EXPECT_DOUBLE_EQ(*ParseJson("1e-06")->AsDouble(), 1e-6);
  EXPECT_DOUBLE_EQ(*ParseJson("1000000.0")->AsDouble(), 1e6);
  EXPECT_EQ(*ParseJson("\"hello\"")->AsString(), "hello");
}

TEST(Json, ParsesTheShapeOfASafetensorsEntry) {
  const auto v = ParseJson(
      R"({"dtype":"BF16","shape":[151936,2048],"data_offsets":[0,622329856]})");
  ASSERT_TRUE(v.ok()) << v.status();

  EXPECT_EQ(*v->RequiredString("dtype"), "BF16");

  const auto* shape = *v->Find("shape")->AsArray();
  ASSERT_EQ(shape->size(), 2u);
  EXPECT_EQ(*(*shape)[0].AsInt(), 151936);
  EXPECT_EQ(*(*shape)[1].AsInt(), 2048);

  const auto* offs = *v->Find("data_offsets")->AsArray();
  EXPECT_EQ(*(*offs)[1].AsInt(), 622329856);
}

TEST(Json, HandlesEscapesAndNesting) {
  const auto v = ParseJson(R"({"a":{"b":["x\ty","z\"q","A"]}})");
  ASSERT_TRUE(v.ok()) << v.status();

  const auto* arr = *v->Find("a")->Find("b")->AsArray();
  ASSERT_EQ(arr->size(), 3u);
  EXPECT_EQ(*(*arr)[0].AsString(), "x\ty");
  EXPECT_EQ(*(*arr)[1].AsString(), "z\"q");
  EXPECT_EQ(*(*arr)[2].AsString(), "A");
}

TEST(Json, EmptyContainers) {
  EXPECT_EQ((*ParseJson("{}")->AsObject())->size(), 0u);
  EXPECT_EQ((*ParseJson("[]")->AsArray())->size(), 0u);
  EXPECT_EQ((*ParseJson(" { \"a\" : [ ] } ")->Find("a")->AsArray())->size(), 0u);
}

// Every one of these is a way a checkpoint header could be subtly wrong. The
// parser is strict on purpose: silently accepting them produces a model with
// quietly incorrect weights, which is far worse than failing at startup.
TEST(Json, RejectsMalformedInput) {
  for (const char* bad : {
           "",           "{",          "}",         "[1,",
           "{\"a\"}",    "{\"a\":}",   "{,}",       "[1 2]",
           "\"unterminated", "tru",    "nul",       "01x",
           "{\"a\":1}x",                 // trailing content
           "{\"a\":1,\"a\":2}",          // duplicate key
           "1.5.2",      "--3",        "1e",
       }) {
    EXPECT_FALSE(ParseJson(bad).ok()) << "should have rejected: " << bad;
  }
}

// The escape form is rejected because decoding it correctly means emitting
// UTF-8, and a half-decoded tensor name fails a lookup much later and much less
// legibly. Raw UTF-8 bytes in the source text are a different matter -- they
// pass through untouched, since copying bytes needs no decoding at all.
TEST(Json, RejectsNonAsciiUnicodeEscapeButPassesRawUtf8Through) {
  // The JSON text is  "\u00e9"  -- the escape form, which is rejected.
  EXPECT_FALSE(ParseJson("\"\\u00e9\"").ok());
  // and  "\u0041"  -- 'A', inside ASCII, which decodes.
  const auto ascii = ParseJson("\"\\u0041\"");
  ASSERT_TRUE(ascii.ok()) << ascii.status();
  EXPECT_EQ(*ascii->AsString(), "A");

  const auto raw = ParseJson("\"\xc3\xa9\"");
  ASSERT_TRUE(raw.ok()) << raw.status();
  EXPECT_EQ(*raw->AsString(), "\xc3\xa9");
}

TEST(Json, AccessorsDoNotCoerce) {
  const auto v = ParseJson(R"({"n":"4","f":2.5,"b":1})");
  ASSERT_TRUE(v.ok()) << v.status();

  EXPECT_FALSE(v->Find("n")->AsInt().ok());     // string "4" is not a number
  EXPECT_FALSE(v->Find("f")->AsInt().ok());     // 2.5 is not an integer
  EXPECT_FALSE(v->Find("b")->AsBool().ok());    // 1 is not a bool
}

TEST(Json, RequiredAndOptional) {
  const auto v = ParseJson(R"({"present":5})");
  ASSERT_TRUE(v.ok()) << v.status();

  EXPECT_EQ(*v->RequiredInt("present"), 5);
  EXPECT_FALSE(v->RequiredInt("absent").ok());
  EXPECT_EQ(*v->OptionalInt("absent", 99), 99);
  EXPECT_EQ(*v->OptionalInt("present", 99), 5);

  // Present but wrong type is still an error, even for an optional field --
  // a config that spells a number as a string is not one we understand.
  const auto w = ParseJson(R"({"x":"nope"})");
  EXPECT_FALSE(w->OptionalInt("x", 0).ok());
}

TEST(Json, ErrorsNameTheOffendingKey) {
  const auto v = ParseJson(R"({"hidden_size":"2048"})");
  ASSERT_TRUE(v.ok()) << v.status();

  const auto r = v->RequiredInt("hidden_size");
  ASSERT_FALSE(r.ok());
  EXPECT_NE(r.status().message().find("hidden_size"), std::string_view::npos)
      << r.status();
}

TEST(Json, DeepNestingIsRejectedNotOverflowed) {
  std::string deep;
  for (int i = 0; i < 500; ++i) deep += "[";
  for (int i = 0; i < 500; ++i) deep += "]";

  EXPECT_FALSE(ParseJson(deep).ok());
}

}  // namespace
}  // namespace inferx
