# inferx_prompt_service -- the tokenizer subsystem's serving components
# (docs/tokenizer_design.md): the tokenizers-cpp facade, the bounded
# prompt-processor pool, and the incremental detokenizer. Deliberately free
# of engine dependencies so front ends and tests can link it standalone.
#
# inferx_http -- the vLLM-aligned HTTP surface (docs/inference_server_design.md):
# framing/lifecycle (http_server), the response writer with SSE semantics,
# the OpenAI handlers, the wire schemas, the stable error model, and
# metrics. Depends on the engine only through the IEngineClient interface,
# so the whole surface is testable with a fake engine and no engine link.
#
# inferx_server -- `inferx serve` composition: the in-process engine client
# adapter over EngineGateway plus RunServe lifecycle wiring.
inferx_add_library(
  NAME prompt_service
  SRCS
    chat_template.cc
    ../tokenizer/tokenizer.cc
    prompt_processor_pool.cc
    detokenizer.cc
  DEPS inferx::core inferx::boost nlohmann_json::nlohmann_json
  PRIVATE_DEPS absl::status absl::statusor absl::strings
)

if(INFERX_ENABLE_TOKENIZERS)
  target_link_libraries(inferx_prompt_service PUBLIC inferx::tokenizers)
endif()

inferx_add_library(
  NAME http
  SRCS
    api.cc
    http_server.cc
    sse_writer.cc
    metrics.cc
    openai_handlers.cc
  DEPS inferx::core inferx::boost inferx::beast inferx::prompt_service
    inferx::sampling nlohmann_json::nlohmann_json
  PRIVATE_DEPS
    absl::status
    absl::statusor
    absl::strings
)

inferx_add_library(
  NAME server
  SRCS
    serve.cc
    engine_gateway.cc
    generation_config.cc
  DEPS inferx::engine inferx::http inferx::prompt_service inferx::boost
  PRIVATE_DEPS
    inferx::beast
    nlohmann_json::nlohmann_json
    absl::flat_hash_map
    absl::status
    absl::statusor
)
