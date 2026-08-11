#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>

#include "inference/scheduler/v1/scheduler.pb.h"

namespace {

namespace wire = ::inference::scheduler::v1;

// Captured from the original inference.scheduler.v1 schema. Keeping this as
// bytes makes accidental field renumbering, wire-type changes, and enum-value
// changes visible even though the test is compiled with current generated code.
constexpr std::array<uint8_t, 109> kLegacySubmitRequest = {
    0x0a, 0x0a, 0x72, 0x65, 0x71, 0x2d, 0x6c, 0x65, 0x67, 0x61, 0x63,
    0x79, 0x10, 0x07, 0x1a, 0x08, 0x74, 0x65, 0x6e, 0x61, 0x6e, 0x74,
    0x2d, 0x61, 0x22, 0x08, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x40, 0x76,
    0x31, 0x28, 0x01, 0x30, 0x02, 0x3a, 0x02, 0x0b, 0x16, 0x42, 0x13,
    0x08, 0x20, 0x15, 0x00, 0x00, 0x00, 0x3f, 0x1d, 0x66, 0x66, 0x66,
    0x3f, 0x20, 0x2a, 0x2a, 0x03, 0x45, 0x4e, 0x44, 0x4a, 0x06, 0x74,
    0x6f, 0x6b, 0x2d, 0x76, 0x31, 0x50, 0xfb, 0xd0, 0x95, 0xff, 0xbc,
    0x31, 0x5a, 0x1d, 0x0a, 0x0d, 0x30, 0x30, 0x2d, 0x61, 0x62, 0x63,
    0x2d, 0x64, 0x65, 0x66, 0x2d, 0x30, 0x31, 0x12, 0x0c, 0x76, 0x65,
    0x6e, 0x64, 0x6f, 0x72, 0x3d, 0x76, 0x61, 0x6c, 0x75, 0x65};

TEST(SchedulerContractCompatibilityTest, ReadsLegacyV1SubmitRequest) {
  wire::SubmitRequest request;
  ASSERT_TRUE(request.ParseFromArray(kLegacySubmitRequest.data(),
                                     kLegacySubmitRequest.size()));

  EXPECT_EQ(request.request_id(), "req-legacy");
  EXPECT_EQ(request.attempt(), 7);
  EXPECT_EQ(request.tenant_id(), "tenant-a");
  EXPECT_EQ(request.model_version(), "model@v1");
  EXPECT_EQ(request.workload(), wire::WORKLOAD_CLASS_GENERATION);
  EXPECT_EQ(request.priority(), wire::PRIORITY_CLASS_BATCH);
  ASSERT_EQ(request.prompt_tokens_size(), 2);
  EXPECT_EQ(request.prompt_tokens(0), 11);
  EXPECT_EQ(request.prompt_tokens(1), 22);
  EXPECT_EQ(request.sampling().max_tokens(), 32);
  EXPECT_FLOAT_EQ(request.sampling().temperature(), 0.5F);
  EXPECT_FLOAT_EQ(request.sampling().top_p(), 0.9F);
  EXPECT_EQ(request.sampling().seed(), 42);
  ASSERT_EQ(request.sampling().stop_size(), 1);
  EXPECT_EQ(request.sampling().stop(0), "END");
  EXPECT_EQ(request.tokenizer_revision(), "tok-v1");
  EXPECT_EQ(request.deadline_unix_millis(), 1700000000123);
  EXPECT_EQ(request.trace().traceparent(), "00-abc-def-01");
  EXPECT_EQ(request.trace().tracestate(), "vendor=value");
}

TEST(SchedulerContractCompatibilityTest, PreservesLegacyV1WireEncoding) {
  wire::SubmitRequest request;
  ASSERT_TRUE(request.ParseFromArray(kLegacySubmitRequest.data(),
                                     kLegacySubmitRequest.size()));
  const std::string expected(
      reinterpret_cast<const char*>(kLegacySubmitRequest.data()),
      kLegacySubmitRequest.size());
  EXPECT_EQ(request.SerializeAsString(), expected);
}

// The vLLM-compat sampling fields were appended (6+) with defaults that all
// mean "feature off". A request that does not use them must serialize to the
// exact legacy bytes, which is what makes new clients readable by old
// servers; and the new fields must survive a round trip, which is what makes
// old clients' silence indistinguishable from "off" on new servers.
TEST(SchedulerContractCompatibilityTest, AppendedSamplingFieldsStayAppended) {
  wire::SubmitRequest legacy;
  ASSERT_TRUE(legacy.ParseFromArray(kLegacySubmitRequest.data(),
                                    kLegacySubmitRequest.size()));
  const std::string expected(
      reinterpret_cast<const char*>(kLegacySubmitRequest.data()),
      kLegacySubmitRequest.size());
  EXPECT_EQ(legacy.SerializeAsString(), expected);

  wire::SamplingParams sampling;
  sampling.set_top_k(40);
  sampling.set_min_p(0.05F);
  sampling.set_presence_penalty(0.5F);
  sampling.set_frequency_penalty(-0.5F);
  sampling.set_repetition_penalty(1.1F);
  sampling.add_stop_token_ids(7);
  sampling.set_ignore_eos(true);
  sampling.set_min_tokens(4);
  sampling.set_want_logprobs(true);
  sampling.set_top_logprobs(5);
  sampling.set_keep_special_tokens(true);
  sampling.set_include_stop_str_in_output(true);

  wire::SamplingParams round_trip;
  ASSERT_TRUE(round_trip.ParseFromString(sampling.SerializeAsString()));
  EXPECT_EQ(round_trip.top_k(), 40);
  EXPECT_FLOAT_EQ(round_trip.min_p(), 0.05F);
  EXPECT_FLOAT_EQ(round_trip.presence_penalty(), 0.5F);
  EXPECT_FLOAT_EQ(round_trip.frequency_penalty(), -0.5F);
  EXPECT_FLOAT_EQ(round_trip.repetition_penalty(), 1.1F);
  ASSERT_EQ(round_trip.stop_token_ids_size(), 1);
  EXPECT_EQ(round_trip.stop_token_ids(0), 7);
  EXPECT_TRUE(round_trip.ignore_eos());
  EXPECT_EQ(round_trip.min_tokens(), 4);
  EXPECT_TRUE(round_trip.want_logprobs());
  EXPECT_EQ(round_trip.top_logprobs(), 5);
  EXPECT_TRUE(round_trip.keep_special_tokens());
  EXPECT_TRUE(round_trip.include_stop_str_in_output());
}

TEST(SchedulerContractCompatibilityTest, EventLogprobsRoundTrip) {
  wire::GenerationEvent event;
  event.set_request_id("req");
  event.add_token_ids(3);
  auto* token = event.add_logprobs();
  token->set_token_id(3);
  token->set_logprob(-0.25F);
  auto* top = token->add_top();
  top->set_token_id(3);
  top->set_logprob(-0.25F);

  wire::GenerationEvent round_trip;
  ASSERT_TRUE(round_trip.ParseFromString(event.SerializeAsString()));
  ASSERT_EQ(round_trip.logprobs_size(), 1);
  EXPECT_EQ(round_trip.logprobs(0).token_id(), 3);
  EXPECT_FLOAT_EQ(round_trip.logprobs(0).logprob(), -0.25F);
  ASSERT_EQ(round_trip.logprobs(0).top_size(), 1);
}

}  // namespace
