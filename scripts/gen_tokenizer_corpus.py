#!/usr/bin/env python3
"""Generates the tokenizer conformance corpus from HuggingFace's tokenizer.

ARCHITECTURE.md R7 argues that tokenizer exactness should hold by construction,
via FFI to the real Rust implementation.  We reimplemented instead, which moves
exactness from a property of the design to a claim that has to be checked -- and
this script produces the evidence for that claim.

It runs HuggingFace's own tokenizer over a corpus chosen to hit the places a
reimplementation goes wrong, and writes the expected token ids.  The C++ test
replays it and requires exact equality.  A disagreement on any string is a bug
in our tokenizer, not a tolerance to be widened: token ids are discrete, and
"close" is meaningless.

    .venv-ref/bin/python scripts/gen_tokenizer_corpus.py \
        --model <checkpoint-dir> --out testdata/tokenizer_corpus.txt

Output format, one record per line, tab-separated:

    <base64 of the input text>\t<space-separated ids>

The text is base64-encoded so that a record is a single line no matter what
control characters, newlines or invalid-looking sequences it contains -- which
is exactly the input the tokenizer most needs to be tested on.
"""

import argparse
import base64
import sys

from tokenizers import Tokenizer


def corpus():
    """The strings to check, grouped by what each group is trying to break."""

    cases = []

    def add(group, *items):
        cases.extend((group, item) for item in items)

    add("empty and trivial", "", " ", "a", "\n", "\t")

    add("plain english",
        "Hello, world!",
        "The capital of France is Paris.",
        "The quick brown fox jumps over the lazy dog.")

    # The pre-tokenizer's leading-space rule is the single most common place a
    # reimplementation diverges: " hello" is one token, "hello" is another.
    add("leading space",
        "hello", " hello", "  hello", "   hello",
        "hello world", "hello  world", "hello   world")

    # `\s+(?!\S)` gives back its last character; `\s*[\r\n]+` does not.  These
    # separate the two.
    add("trailing and interior whitespace",
        "hello ", "hello  ", "hello\n", "hello\n\n", "hello \n",
        "hello \n world", "a  \n  b", "   \n   ", "\n   ", "   \n",
        "a\r\nb", "\r\n\r\n", "a\tb", "  \t  \n  ")

    # The contraction alternative is tried first and is case-insensitive.
    add("contractions",
        "don't", "DON'T", "Don'T", "it's", "IT'S", "we've", "they'll",
        "I'd", "I'm", "you're", "can't've", "'s", "'S", "s'", "o'clock",
        "rock 'n' roll")

    # `\p{N}` matches one digit at a time, so numbers never merge across the
    # pre-tokenizer -- a rule that surprises people writing this by hand.
    add("digits",
        "0", "7", "42", "1234567890", "3.14159", "-273.15",
        "1,000,000", "v1.2.3", "0x1F", "2026-08-01", "12:34:56")

    # Punctuation runs, with and without the leading space the pattern allows.
    add("punctuation",
        "!!!", "...", " ...", "?!", "--", " -- ", "( )", "[]", "{}",
        "a.b.c", "e.g.", "C++", "a->b", "=>", "<|", "|>", "&&", "||")

    add("code",
        "def main():\n    return 0\n",
        "int x = 1;  // comment\n",
        "#include <cstdint>",
        'printf("%s\\n", s);',
        "for (int i = 0; i < n; ++i) {\n  sum += a[i];\n}",
        "SELECT * FROM t WHERE id = 1;",
        "{\"key\": [1, 2, 3], \"nested\": {\"a\": null}}",
        "  indented\n    more\n      deeper\n")

    # Long whitespace runs exercise the BPE merge queue, which is where a
    # quadratic implementation would be visible.
    add("whitespace runs",
        " " * 8, " " * 16, " " * 64, " " * 256,
        "\n" * 16, "\t" * 32, "a" + " " * 128 + "b")

    # Multi-byte characters, where a token boundary can fall inside a character.
    add("cjk",
        "你好", "你好，世界！", "日本語のテキストです。", "한국어 텍스트",
        "中文和English混排", "これはテストです")

    add("other scripts",
        "Привет, мир!", "مرحبا بالعالم", "שלום עולם",
        "Ελληνικά", "हिन्दी", "ไทย")

    # Emoji are four-byte characters, and skin-tone and ZWJ sequences are
    # several codepoints that must not be merged across.
    add("emoji",
        "🙂", "🙂🙂", "hello 🙂 world", "👍🏽", "👨‍👩‍👧‍👦", "🇬🇧",
        "🔥" * 8)

    # NFC is the normalizer, so decomposed input must tokenize as its composed
    # form.  The pairs below are byte-different and NFC-identical.
    add("normalization",
        "café",            # composed U+00E9
        "café",      # decomposed e + combining acute
        "Ångström",
        "Ångström",
        "한국",            # composed Hangul
        "한국",  # conjoining jamo
        "ཱི",    # Tibetan, non-trivial combining classes
        "q̣̇",   # two marks needing canonical reordering
        "q̣̇")

    # Non-breaking and exotic spaces are \p{White_Space} but not ASCII space.
    add("unicode whitespace",
        "a b", "a b", "a　b", "a b", "a b",
        "  ", "ab")

    # Special tokens must come back as single ids from Encode.
    add("special tokens",
        "<|im_start|>", "<|im_end|>", "<|endoftext|>",
        "<|im_start|>user\nhi<|im_end|>\n",
        "<|im_start|>system\nYou are Qwen.<|im_end|>\n"
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n",
        "text<|im_end|>text",
        "<|im_start|", "|im_end|>", "<|not_a_real_token|>")

    add("adversarial",
        "​", "﻿", "�", "\x00" if False else "",
        "a" * 200,
        "".join(chr(c) for c in range(0x20, 0x7f)),
        "mixed 中文 with 🙂 and café and 123 and <|im_end|>")

    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="checkpoint directory containing tokenizer.json")
    parser.add_argument("--out", required=True, help="corpus file to write")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(f"{args.model}/tokenizer.json")

    records = 0
    groups = {}

    with open(args.out, "w", encoding="utf-8") as out:
        out.write("# Generated by scripts/gen_tokenizer_corpus.py\n")
        out.write(f"# tokenizer: {args.model}/tokenizer.json\n")
        out.write("# format: base64(text) <TAB> space-separated expected ids\n")

        for group, text in corpus():
            ids = tokenizer.encode(text, add_special_tokens=False).ids

            # A round-trip check on the reference side, so that a corpus entry
            # can never encode an expectation the reference itself fails.
            assert tokenizer.decode(ids, skip_special_tokens=False) is not None

            encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
            out.write(f"{encoded}\t{' '.join(str(i) for i in ids)}\n")

            records += 1
            groups[group] = groups.get(group, 0) + 1

    print(f"wrote {records} records to {args.out}", file=sys.stderr)

    for group, count in groups.items():
        print(f"  {count:4d}  {group}", file=sys.stderr)


if __name__ == "__main__":
    main()
