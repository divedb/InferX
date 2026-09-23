# CLI presentation

CLI11 owns command registration, parsing, validation, and dispatch. Construct
`Presentation` after registering the command tree and call `Parse` at the entry
point. It installs the shared help formatter and translates parse errors while
preserving CLI11's exit codes. Business errors still use `CommandError` and are
rendered through `Presentation::Error` in `main`.

Help and success messages go to stdout; errors and warnings go to stderr. Use
`Presentation::Warning` and `Presentation::Success` for human-facing status
messages. Keep machine-readable command output unchanged. `term::StyleSheet`
provides semantic styles; ANSI codes belong only in `terminal.cc`. Styling is
restrained: red+bold `error:`, yellow warnings and `<placeholder>` tokens,
green success, cyan commands/option names, bold headings, dim secondary text.

`--color=auto|always|never` works before or after subcommands. The last occurrence
wins. Auto mode checks each output stream independently and disables styling for
non-terminals, `TERM=dumb`, and a non-empty `NO_COLOR`. An explicit `always`
overrides these checks, including `NO_COLOR`; `never` always produces plain text.

Command paths, option groups, required markers, and defaults come from CLI11
metadata. Register descriptions and validators with the command; do not add a
hardcoded full usage path. Unknown options and commands get suggestions only
when there is a close, unambiguous match. Custom validation messages remain
available as a fallback when CLI11 cannot identify a particular option.

Run `ctest --test-dir build-cuda13 -R 'cli_' --output-on-failure` after building.
The process tests check stdout/stderr, pipes, files, pseudo-terminals, color
precedence, and common parsing failures without starting an inference engine.

## Attention backend

CUDA models use FlashInfer by default. `--attention-backend flashinfer` selects it
explicitly; `default` and `flash` are compatibility aliases. Unknown backends fail
validation. `INFERX_EXPERIMENTAL_FLASH_ATTENTION` no longer selects execution.

The BF16 integration supports full causal attention, head dimensions 64/128/256,
and query/KV-head ratios 1 through 32. It handles prefill, cached-prefix chunks,
mixed batches and decode. Unsupported geometry returns an error before model
weights are uploaded. CUTLASS/CuTe is reserved for demonstrated requirements that
FlashInfer cannot meet; there is no scalar attention fallback.
