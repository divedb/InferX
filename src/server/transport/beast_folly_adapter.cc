#include "inferx/server/transport/beast_folly_adapter.h"

// The adapters are templates. This translation unit gives the transport
// target a stable implementation boundary and a home for future non-template
// error/category translation without exposing it to application targets.
