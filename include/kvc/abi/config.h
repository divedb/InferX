#ifndef KVC_ABI_CONFIG_H_
#define KVC_ABI_CONFIG_H_

#include "kvc/abi/basics.h"

#ifdef __cplusplus
extern "C" {
#endif

/// \brief Component memory layout. Bit N of KvcCapabilities::layouts
/// advertises layout N.
enum KvcLayout {
  KVC_OPAQUE_BYTES = 1,   ///< Fixed byte count per block; provider never
                          ///< interprets the contents.
  KVC_STRIDED_TENSOR = 2  ///< Typed tensor with shape and byte strides.
};

/// \brief Optional provider abilities, combined with bitwise OR in
/// KvcConfig::required_features and KvcCapabilities::features.
enum KvcFeature {
  KVC_FEATURE_MAPPING = 1u,       ///< Immutable reads can be memory-mapped.
  KVC_FEATURE_CANCELLATION = 2u,  ///< Transfers can be cancelled.
  KVC_FEATURE_LAYERWISE = 4u      ///< Per-layer selection in spans.
};

/// \brief One cached state component of a group: attention K, V, an MLA
/// latent, or custom state, identified by meaning rather than position.
///
/// EXAMPLE: A GQA layer stores K and V in separate GPU buffers. With 16
/// tokens per block, 8 KV heads, head dimension 128, and FP16 elements,
/// each buffer contributes 16 * 8 * 128 * 2 = 32768 bytes per block.
/// \code{.c}
/// KvcComponent components[] = {
///     {0, KVC_OPAQUE_BYTES, {{{"kvc.bytes", 9}, 1, 0}, {0, 0}}, 32768},
///     {1, KVC_OPAQUE_BYTES, {{{"kvc.bytes", 9}, 1, 0}, {0, 0}}, 32768},
/// };
/// \endcode
/// Attach this array to a KvcGroup with tokens_per_block = 16,
/// component_count = 2, and semantics "kvc.opaque" version 1.0 with empty
/// parameters. The group's layers select which model layers use this pair.
/// Here the caller assigns ID 0 to K and ID 1 to V; these are not reserved
/// ABI IDs, and "kvc.bytes" does not itself distinguish K from V.
///
/// For each selected block and layer, supply a KvcBinding for component 0
/// pointing to the K memory region and another for component 1 pointing
/// to the V region, each covering 32768 bytes for a full block. This lets
/// the provider transfer separate buffers without requiring a packed copy
/// and check that both components are present before publication.
/// An engine that already packs K and V together can instead declare one
/// opaque component of 65536 bytes per block.
typedef struct KvcComponent {
  uint32_t id;               ///< Component identifier, unique within its group.
  uint32_t layout;           ///< A KvcLayout value.
  KvcDescriptor meaning;     ///< What the bytes represent.
  uint64_t bytes_per_block;  ///< OpaqueBytes; tensor geometry is schema-defined.
} KvcComponent;

/// \brief Layers sharing logical allocation and block semantics (block
/// size, component set, schema). A layer may appear in multiple groups.
typedef struct KvcGroup {
  uint32_t id;                ///< Group identifier, unique within the configuration.
  uint32_t tokens_per_block;  ///< Block size; zero is invalid.
  KvcDescriptor semantics;    ///< Block semantics (e.g. "kvc.opaque").
  const uint32_t* layers;     ///< Ascending, unique layer indices.
  uint64_t layer_count;
  const KvcComponent* components;
  uint64_t component_count;
} KvcGroup;

/// \brief Session configuration, supplied once to configure(). The provider
/// deep-copies everything borrowed here; the caller may release its copy on
/// return.
typedef struct KvcConfig {
  uint32_t struct_size;
  KvcDigest cache_namespace;  ///< Isolates unrelated cache namespaces.
  /// Digest of the complete model fingerprint, including execution
  /// semantics; mismatched state is IncompatibleModel.
  KvcDigest model_identity;
  KvcDescriptor logical_partition;  ///< Canonical tensor slices, never
                                    ///< runtime rank numbers.
  const KvcGroup* groups;
  uint64_t group_count;
  uint64_t required_features;  ///< OR of KvcFeature values.
  const KvcSchema* required_extensions;
  uint64_t required_extension_count;
} KvcConfig;

/// \brief What a provider can do, reported by capabilities(). Describes
/// possibilities; configure() still validates each concrete combination.
typedef struct KvcCapabilities {
  uint32_t struct_size;
  uint64_t features;         ///< OR of supported KvcFeature values.
  uint64_t memory_types;     ///< Bit N advertises memory type N.
  uint64_t layouts;          ///< Bit N advertises layout N.
  const KvcSchema* schemas;  ///< Supported semantic schemas.
  uint64_t schema_count;
} KvcCapabilities;

/// \brief Vendor extension: a versioned function table plus the owner
/// keeping it alive. Extension schemas define their own contracts.
typedef struct KvcExtension {
  uint32_t struct_size;
  const void* table;
  // One owned reference, keeping the table and its context alive.
  KvcObject* owner;
} KvcExtension;

#ifdef __cplusplus
}
#endif
#endif  // KVC_ABI_CONFIG_H_
