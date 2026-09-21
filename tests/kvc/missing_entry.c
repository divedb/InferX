// Fixture library that exports no provider entry symbol at all.

#include "kvc/abi/basics.h"

KVC_EXPORT int kvc_missing_entry_helper(void) { return 1; }
