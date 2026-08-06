#include "inferx/support/file_util.h"

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <string>

namespace inferx {
namespace {

class TemporaryFile {
 public:
  TemporaryFile() {
    std::array<char, 64> pattern{};
    const std::string value = "/tmp/inferx-read-file-test-XXXXXX";
    std::copy(value.begin(), value.end(), pattern.begin());
    fd_ = ::mkstemp(pattern.data());
    path_ = pattern.data();
  }

  ~TemporaryFile() {
    if (fd_ >= 0) ::close(fd_);
    if (!path_.empty()) ::unlink(path_.c_str());
  }

  int fd() const { return fd_; }
  const std::string& path() const { return path_; }

 private:
  int fd_ = -1;
  std::string path_;
};

TEST(ReadFileTest, ReadsBinaryData) {
  TemporaryFile file;
  ASSERT_GE(file.fd(), 0);
  std::string expected(10'000, '\0');
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<char>(i % 251);
  }
  ASSERT_EQ(::write(file.fd(), expected.data(), expected.size()),
            static_cast<ssize_t>(expected.size()));

  const StatusOr<std::string> contents = ReadFile(file.path());
  ASSERT_TRUE(contents.ok()) << contents.status();
  EXPECT_EQ(*contents, expected);
}

TEST(ReadFileTest, ReadsProcFileThatReportsAZeroSize) {
  struct stat file_info {};
  ASSERT_EQ(::stat("/proc/self/cmdline", &file_info), 0);
  ASSERT_EQ(file_info.st_size, 0);

  const StatusOr<std::string> contents = ReadFile("/proc/self/cmdline");
  ASSERT_TRUE(contents.ok()) << contents.status();
  EXPECT_FALSE(contents->empty());
}

TEST(ReadFileTest, MapsMissingFileToNotFound) {
  const StatusOr<std::string> contents =
      ReadFile("/tmp/inferx-file-that-does-not-exist-7e1176bb");
  ASSERT_FALSE(contents.ok());
  EXPECT_TRUE(absl::IsNotFound(contents.status())) << contents.status();
}

}  // namespace
}  // namespace inferx
