#include "inferx/support/log.h"

#include <fstream>
#include <sstream>
#include <string>

#include "gtest/gtest.h"

namespace inferx {
namespace {

TEST(LogTest, FormatsRequestId) {
  std::ostringstream out;
  out << Rid("req_0198abc");
  EXPECT_EQ(out.str(), "[req_0198abc] ");
}

TEST(LogTest, JsonFileHasExpectedShape) {
  const std::string path = testing::TempDir() + "/inferx_log_test.jsonl";
  LogOptions options;
  options.json = true;
  options.file = path;
  InitLogging(options);
  LOG(INFO) << "logging-test-message";

  // Reinitializing unregisters and flushes the file sink before it is read.
  InitLogging();
  std::ifstream input(path);
  const std::string line((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  EXPECT_NE(line.find("\"ts\":"), std::string::npos);
  EXPECT_NE(line.find("\"severity\":\"INFO\""), std::string::npos);
  EXPECT_NE(line.find("\"file\":\"log_test.cc\""), std::string::npos);
  EXPECT_NE(line.find("\"line\":"), std::string::npos);
  EXPECT_NE(line.find("\"tid\":"), std::string::npos);
  EXPECT_NE(line.find("\"msg\":\"logging-test-message\""),
            std::string::npos);
}

}  // namespace
}  // namespace inferx
