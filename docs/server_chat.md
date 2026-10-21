# Qwen3 chat completions

`POST /v1/chat/completions` supports text conversations using the recognized
Qwen3 checkpoint template. Rendering happens in the prompt worker pool before
tokenization. It is a native implementation of that template's text-only path,
not a general Jinja interpreter. A missing or different `chat_template` in
`tokenizer_config.json` returns 501 for chat; text completions remain available.
The metadata file is included in the tokenizer fingerprint.

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"models/Qwen3-0.6B","messages":[{"role":"user","content":"What is 2 + 2?"}],"chat_template_kwargs":{"enable_thinking":false},"temperature":0,"max_tokens":64}'
```

Rebuild and restart the server to use the new endpoint.

Supported messages have `system`, `user`, or `assistant` roles and string
content. The renderer handles Qwen3's historical `<think>` removal and default
assistant generation prefix. `chat_template_kwargs.enable_thinking` is a boolean
and defaults to true, as in the checkpoint. Thinking text is returned in
`message.content`/`delta.content`; there is no separate reasoning parser.

Sampling and stop controls share the text-completion implementation.
`max_completion_tokens` is an alias for `max_tokens`; specifying both is an
error. JSON responses use `chat.completion` with an assistant message. SSE
responses use `chat.completion.chunk`, an initial assistant role delta, content
deltas, a finish reason, optional usage, and one `[DONE]` marker. Usage counts
the templated prompt and includes the generated terminating EOS token, while
that EOS token is excluded from text.

Tool calls, named messages, explicit `reasoning_content`, content-part arrays,
multimodal input, custom templates, and continuation options remain unsupported
and are rejected. Other special tokens still follow the existing literal
tokenizer decoding behavior.

Validation includes 20 Hugging Face reference renderings in
`tests/testdata/qwen3_chat_reference.json`, parser validation, fake-engine HTTP
tests, and a real Qwen3 smoke test comparing chat with a manually HF-templated
completion (text, token usage, streaming, stops, and EOS suppression).

```sh
ctest --test-dir build-dbg --output-on-failure \
  -R '^(chat_template_test|server_api_test|http_api_test|prompt_processor_pool_test)$'
```
