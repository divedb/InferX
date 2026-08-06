#include "inferx/support/file_util.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "inferx/support/scope_exit.h"

namespace inferx {

StatusOr<std::string> ReadFile(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);

  if (fd < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("cannot open ", path));
  }
  const auto close_fd = MakeScopeExit([fd] { ::close(fd); });

  struct stat file_info {};

  if (::fstat(fd, &file_info) != 0) {
    const int stat_error = errno;
    return absl::ErrnoToStatus(stat_error, absl::StrCat("cannot stat ", path));
  }

  std::string contents;

  static_assert(sizeof(contents[0]) == 1,
                "ReadFile requires byte-sized string elements");

  // procfs, sysfs, and files changing concurrently may report an inaccurate
  // size. Treat a positive size as a reservation hint plus one probe byte; for
  // zero-sized files, start small and attempt a read anyway.
  constexpr size_t kInitialAllocation = 4 * 1024;
  const size_t maximum_size = contents.max_size();
  size_t initial_size = kInitialAllocation;

  if (file_info.st_size > 0) {
    const uintmax_t advertised = static_cast<uintmax_t>(file_info.st_size);
    initial_size = advertised < maximum_size
                       ? static_cast<size_t>(advertised) + 1
                       : maximum_size;
  }

  contents.resize(std::min(initial_size, maximum_size));
  size_t bytes_read = 0;

  for (;;) {
    const ssize_t size =
        ::read(fd, contents.data() + bytes_read, contents.size() - bytes_read);

    if (size > 0) {
      bytes_read += static_cast<size_t>(size);
    } else if (size == 0) {
      break;
    } else if (errno == EINTR) {
      continue;
    } else {
      const int read_error = errno;
      return absl::ErrnoToStatus(read_error,
                                 absl::StrCat("cannot read ", path));
    }

    if (bytes_read < contents.size()) continue;

    if (contents.size() == maximum_size) {
      return ResourceExhaustedError("file is too large to read: ", path);
    }

    const size_t growth = std::max(contents.size() / 2, size_t{1});
    contents.resize(std::min(maximum_size - contents.size(), growth) +
                    contents.size());
  }

  contents.resize(bytes_read);

  return contents;
}

}  // namespace inferx
