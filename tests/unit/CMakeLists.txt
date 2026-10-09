# Unit tests: one CMakeLists per component area, registered next to the
# sources it covers. Labels follow the source layout (everything here is
# "unit"); integration suites live under tests/integration.
add_subdirectory(cache)
add_subdirectory(cli)
add_subdirectory(config)
add_subdirectory(core)
add_subdirectory(dist)
add_subdirectory(engine)
add_subdirectory(models)
add_subdirectory(ops)
add_subdirectory(sampling)
add_subdirectory(server)
add_subdirectory(util)
