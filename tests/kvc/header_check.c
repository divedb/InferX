// Compiles every provider-SDK header as strict C11 with no provider
// defined: a pure ABI compatibility check.

#include <assert.h>

#include "kvc/abi/basics.h"
#include "kvc/abi/config.h"
#include "kvc/abi/control.h"
#include "kvc/abi/data.h"
#include "kvc/abi/provider.h"
#include "kvc/abi/types.h"

/* The records must stay fixed-width and self-contained in C. */
static void Sizes(void) {
  KvcStatus status;
  status.code = KVC_OK;
  KvcDigest digest;
  (void)digest;
  KvcConfig config;
  config.struct_size = (uint32_t)sizeof(KvcConfig);
  KvcLookupRequest lookup;
  lookup.struct_size = (uint32_t)sizeof(KvcLookupRequest);
  KvcTransferRequest transfer;
  transfer.struct_size = (uint32_t)sizeof(KvcTransferRequest);
  KvcMemoryImport import;
  import.struct_size = (uint32_t)sizeof(KvcMemoryImport);
  (void)status;
  (void)config;
  (void)lookup;
  (void)transfer;
  (void)import;
}

int main(void) {
  Sizes();
  return KVC_ABI_MAJOR == 1 && KVC_ABI_MINOR >= 0 ? 0 : 1;
}
