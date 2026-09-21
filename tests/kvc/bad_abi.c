// Fixture provider whose entry point rejects every ABI major: the loader
// must fail with Unsupported.

#include <stddef.h>

#include "kvc/abi/provider.h"

static KvcStatus KVC_CALL Entry(uint32_t major, uint32_t minor, const KvcPlugin** out) {
  (void)major;
  (void)minor;
  if (!out) return (KvcStatus){.code = KVC_INVALID_ARGUMENT};
  *out = NULL;
  return (KvcStatus){.code = KVC_UNSUPPORTED};
}

KVC_EXPORT KvcStatus KVC_CALL KvcGetProvider(uint32_t major, uint32_t minor,
                                             const KvcPlugin** out) {
  return Entry(major, minor, out);
}
