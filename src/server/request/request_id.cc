#include "inferx/server/request/request_id.h"

#include <array>
#include <cstdint>
#include <iomanip>
#include <random>
#include <sstream>

namespace inferx::server::request {
namespace {

RequestId Generate(std::chrono::system_clock::time_point now) {
  const uint64_t millis = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          now.time_since_epoch())
          .count());
  std::array<uint8_t, 16> bytes{};
  for (size_t i = 0; i < 6; ++i) {
    bytes[5 - i] = static_cast<uint8_t>(millis >> (i * 8));
  }

  thread_local std::mt19937_64 random(std::random_device{}());
  const uint64_t high_random = random();
  const uint64_t low_random = random();
  bytes[6] = static_cast<uint8_t>(0x70 | ((high_random >> 56) & 0x0f));
  bytes[7] = static_cast<uint8_t>(high_random >> 48);
  bytes[8] = static_cast<uint8_t>(0x80 | ((high_random >> 40) & 0x3f));
  for (size_t i = 9; i < bytes.size(); ++i) {
    bytes[i] = static_cast<uint8_t>(low_random >> ((15 - i) * 8));
  }

  std::ostringstream result;
  result << "req_" << std::hex << std::setfill('0');
  for (size_t i = 0; i < bytes.size(); ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) result << '-';
    result << std::setw(2) << static_cast<unsigned int>(bytes[i]);
  }
  return result.str();
}

}  // namespace

RequestId GenerateRequestId() { return Generate(std::chrono::system_clock::now()); }

RequestId GenerateRequestId(std::chrono::system_clock::time_point now) {
  return Generate(now);
}

}  // namespace inferx::server::request
