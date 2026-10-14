add_subdirectory(core)       # element types, tensors, status, memory, runtimes
add_subdirectory(config)     # device flag parsing, validation, resolution
add_subdirectory(util)       # typed views over HuggingFace JSON configs
add_subdirectory(cache)      # paged KV block pool + prefix cache
add_subdirectory(ops)        # portable compute ops + cpu/cuda backends
add_subdirectory(sampling)   # sampling pipeline
add_subdirectory(models)     # families, reusable layers, registry, runner
add_subdirectory(engine)     # scheduling and orchestration
add_subdirectory(dist)       # collectives + tensor-parallel topology
add_subdirectory(bench)      # benchmark entry points
add_subdirectory(diagnostic) # numerical debugging
add_subdirectory(server)     # HTTP serving layer
