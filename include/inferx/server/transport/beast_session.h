#pragma once

#include <boost/asio/ip/tcp.hpp>
#include <folly/Executor.h>

#include <memory>

#include "inferx/server/transport/beast_listener.h"

namespace inferx::server::transport {

class BeastSession;

std::shared_ptr<BeastSession> MakeBeastSession(
    boost::asio::ip::tcp::socket socket, BeastListenerConfig config,
    std::shared_ptr<RequestHandler> handler,
    folly::Executor::KeepAlive<> coroutine_executor);

void StartBeastSession(const std::shared_ptr<BeastSession>& session);
void StopBeastSession(const std::shared_ptr<BeastSession>& session);

}  // namespace inferx::server::transport
