# Completion stopping

`POST /v1/completions` accepts:

- `stop`: a string or an array of at most four non-empty strings.
- `stop_token_ids`: an array of non-negative int32 token IDs. The engine
  rejects IDs outside the model vocabulary.

Both fields also accept `null` or an empty array. Matching applies to generated
output only, never the prompt. Explicit token stops still apply when
`ignore_eos` is true.

Stop strings are excluded from returned text. Streaming holds a possible stop
prefix until it either matches or becomes ordinary output. A partial match is
flushed when generation ends without a full match. Matching consumes settled
UTF-8 text, so an incomplete character's temporary replacement does not leak
into the stream.

Ordinary stop-token text remains in the response. Completion usage includes
the token that triggers either kind of stop. A matched stop at the generation
limit returns `finish_reason: "stop"`; an unmatched limit returns `"length"`.
The scheduler releases KV blocks through its normal completion path.

Current limits: `include_stop_str_in_output` and `min_tokens` remain unsupported
API fields. A terminating model EOS token is suppressed from text. Other
special tokens are decoded literally; configurable special-token skipping
remains a separate compatibility gap with vLLM.

Focused checks:

```sh
cmake --build build-dbg --target inferx inferx_server_api_test \
  inferx_http_api_test inferx_text_delta_test inferx_scheduler_test -j 4
ctest --test-dir build-dbg --output-on-failure \
  -R '^(server_api_test|http_api_test|text_delta_test|scheduler_test)$'
```

The HTTP tests use a fake engine and verify parameter propagation. The text
tests exhaustively partition a stop-containing string into chunks and exercise
overlapping matches, unfinished prefixes, and UTF-8 settling. Scheduler tests
check finish precedence, usage token retention, and cache release.
