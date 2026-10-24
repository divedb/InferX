// Router unit tests: exact and parameterized matching, method masks,
// 404/405 with Allow, dispatch through handlers, and OPTIONS discovery.
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <boost/asio.hpp>

#include "inferx/server/http_types.h"
#include "inferx/server/request_context.h"
#include "inferx/server/router.h"
#include "inferx/server/sse_writer.h"

namespace inferx::server {
namespace {

class FakeWriter final : public IStreamWriter {
 public:
  boost::asio::awaitable<Result<void>> Begin(int, HeaderMap) override {
    co_return Result<void>{};
  }
  boost::asio::awaitable<Result<void>> WriteEvent(std::string_view) override {
    co_return Result<void>{};
  }
  boost::asio::awaitable<Result<void>> WriteComment(std::string_view) override {
    co_return Result<void>{};
  }
  boost::asio::awaitable<Result<void>> WriteDone() override {
    co_return Result<void>{};
  }
  void Abort() noexcept override {}
  bool committed() const noexcept override { return false; }
  std::size_t bytes_written() const noexcept override { return 0; }
};

std::unique_ptr<RequestContext> MakeContext(HttpMethod method,
                                            const std::string& path) {
  HttpRequest request;
  request.method = method;
  request.path = path;
  static FakeWriter writer;
  return std::make_unique<RequestContext>(
      ClientRequestId("test"), ConnectionId("conn-1"), TenantId{},
      std::move(request), boost::asio::system_executor{}, Clock::now(),
      Clock::now() + std::chrono::seconds(10), &writer);
}

RouteHandler OkHandler(std::string* seen_path, RouteParams* seen_params) {
  return [seen_path, seen_params](RequestContext& ctx)
             -> boost::asio::awaitable<RouteResult> {
    if (seen_path) *seen_path = ctx.request().path;
    if (seen_params) *seen_params = ctx.route_params();
    HttpResponse response;
    response.status = 200;
    co_return response;
  };
}

class RouterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    router_ = std::make_unique<Router>(std::vector<RouteSpec>{
        RouteSpec{"/v1/models", HttpMethodMask::kGet, "list_models", false,
                  false},
        RouteSpec{"/v1/models/:id", HttpMethodMask::kGet | HttpMethodMask::kDelete,
                  "model_detail", false, false},
    });
    router_->Add(RouteSpec{"/v1/models", HttpMethodMask::kGet, "list_models",
                           false, false},
                 OkHandler(nullptr, nullptr));
    router_->Add(RouteSpec{"/v1/models/:id",
                           HttpMethodMask::kGet | HttpMethodMask::kDelete,
                           "model_detail", false, false},
                 OkHandler(nullptr, nullptr));
  }

  RouteResult Dispatch(HttpMethod method, const std::string& path) {
    auto ctx = MakeContext(method, path);
    auto outcome = boost::asio::co_spawn(
        boost::asio::system_executor{}, router_->Dispatch(*ctx),
        boost::asio::use_future);
    return outcome.get();
  }

  std::unique_ptr<Router> router_;
};

TEST_F(RouterTest, ExactMatchDispatches) {
  auto result = Dispatch(HttpMethod::get, "/v1/models");
  ASSERT_TRUE(std::holds_alternative<HttpResponse>(result));
  EXPECT_EQ(std::get<HttpResponse>(result).status, 200);
}

TEST_F(RouterTest, ParamsAreCapturedAndExactSegmentsDiffer) {
  RouteParams params;
  RouteSpec spec{"/v1/models/:id", HttpMethodMask::kGet, "model_detail", false,
                 false};
  Router single({spec});
  single.Add(spec, OkHandler(nullptr, &params));
  auto ctx = MakeContext(HttpMethod::get, "/v1/models/gpt-x");
  auto future = boost::asio::co_spawn(boost::asio::system_executor{},
                                      single.Dispatch(*ctx),
                                      boost::asio::use_future);
  const RouteResult result = future.get();
  ASSERT_TRUE(std::holds_alternative<HttpResponse>(result));
  EXPECT_EQ(params.get("id"), "gpt-x");

  // Different segment count must not match the parameterized route.
  auto miss = MakeContext(HttpMethod::get, "/v1/models/a/b");
  auto miss_future = boost::asio::co_spawn(boost::asio::system_executor{},
                                           single.Dispatch(*miss),
                                           boost::asio::use_future);
  EXPECT_TRUE(std::holds_alternative<ApiError>(
      miss_future.get()));
}

TEST_F(RouterTest, WrongMethodYields405WithAllow) {
  auto result = Dispatch(HttpMethod::post, "/v1/models");
  ASSERT_TRUE(std::holds_alternative<ApiError>(result));
  const ApiError& error = std::get<ApiError>(result);
  EXPECT_EQ(error.http_status, 405);
  ASSERT_FALSE(error.headers.empty());
  EXPECT_EQ(error.headers.front().first, "Allow");
  EXPECT_EQ(error.headers.front().second, "GET");
}

TEST_F(RouterTest, MultiMethodAllowListsBoth) {
  const std::string allow = router_->AllowedMethods("/v1/models/x");
  EXPECT_EQ(allow, "GET, DELETE");
}

TEST_F(RouterTest, UnknownPathYields404) {
  auto result = Dispatch(HttpMethod::get, "/nope");
  ASSERT_TRUE(std::holds_alternative<ApiError>(result));
  EXPECT_EQ(std::get<ApiError>(result).http_status, 404);
}

TEST_F(RouterTest, HasRouteAndAllowedMethods) {
  EXPECT_TRUE(router_->HasRoute(HttpMethod::get, "/v1/models"));
  EXPECT_TRUE(router_->HasRoute(HttpMethod::delete_, "/v1/models/abc"));
  EXPECT_FALSE(router_->HasRoute(HttpMethod::post, "/v1/models"));
  EXPECT_FALSE(router_->HasRoute(HttpMethod::get, "/unknown"));
  EXPECT_TRUE(router_->AllowedMethods("/v1/models").find("GET") !=
              std::string::npos);
  EXPECT_TRUE(router_->AllowedMethods("/missing").empty());
}

}  // namespace
}  // namespace inferx::server
