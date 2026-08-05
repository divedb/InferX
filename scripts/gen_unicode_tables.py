#!/usr/bin/env python3
"""Generates the Unicode tables the tokenizer needs, from Python's unicodedata.

The tokenizer has to reproduce two pieces of Unicode behaviour exactly, because
HF's tokenizer does and any divergence is a divergence in token ids:

  * the pre-tokenizer regex uses \\p{L} and \\p{N}, so we need the set of
    codepoints in the general categories L* and N*;
  * the normalizer is NFC, so we need canonical combining classes, full
    canonical decompositions, and the canonical composition pairs.

Hangul is left out of the tables deliberately -- its decomposition and
composition are arithmetic, and the tables would otherwise gain ~11k entries
for no information.  The C++ side implements the arithmetic directly.

Emitting this as a checked-in header rather than generating it at build time
keeps Python off the build path.  Re-run it only when moving to a new Unicode
version; the header records which one it came from.

    python3 scripts/gen_unicode_tables.py > include/inferx/tokenizer/unicode_data.h
"""

import sys
import unicodedata

MAX_CP = 0x110000

# Hangul syllable arithmetic, from UAX #15.  These bounds are also used to skip
# the region when building the tables.
HANGUL_S_BASE = 0xAC00
HANGUL_S_COUNT = 11172


def ranges(predicate):
    """Collapses the codepoints satisfying `predicate` into inclusive ranges."""
    out = []
    start = None

    for cp in range(MAX_CP):
        if predicate(cp):
            if start is None:
                start = cp
        elif start is not None:
            out.append((start, cp - 1))
            start = None

    if start is not None:
        out.append((start, MAX_CP - 1))

    return out


def category_ranges(prefix):
    return ranges(lambda cp: unicodedata.category(chr(cp)).startswith(prefix))


def combining_ranges():
    """Ranges of equal, non-zero canonical combining class."""
    out = []
    start = None
    current = 0

    for cp in range(MAX_CP):
        ccc = unicodedata.combining(chr(cp))

        if ccc != current:
            if start is not None and current != 0:
                out.append((start, cp - 1, current))
            start = cp if ccc != 0 else None
            current = ccc

    if start is not None and current != 0:
        out.append((start, MAX_CP - 1, current))

    return out


def canonical_decomposition(cp):
    """The one-step canonical decomposition of `cp`, or None.

    `unicodedata.decomposition` returns compatibility decompositions too, marked
    with a `<tag>` prefix.  Those must not be applied under NFC, so they are
    filtered out here.
    """
    spec = unicodedata.decomposition(chr(cp))

    if not spec or spec.startswith("<"):
        return None

    return tuple(int(part, 16) for part in spec.split())


def full_decomposition(cp):
    """Recursively expands a canonical decomposition to starters and marks."""
    step = canonical_decomposition(cp)

    if step is None:
        return [cp]

    out = []
    for part in step:
        out.extend(full_decomposition(part))

    return out


def composition_exclusions():
    """Codepoints that must not be recomposed by NFC.

    Rather than parse CompositionExclusions.txt, this derives the set the way
    UAX #15 defines it operationally: a canonical decomposable whose NFC form is
    not itself is excluded (script exclusions and post-composition versions), as
    are singletons and non-starter decompositions, which are handled separately
    by the caller.
    """
    out = set()

    for cp in range(MAX_CP):
        if canonical_decomposition(cp) is None:
            continue

        if unicodedata.normalize("NFC", chr(cp)) != chr(cp):
            out.add(cp)

    return out


def specific_category_ranges(categories):
    """Ranges of codepoints whose general category is in `categories`.

    `category_ranges` is startswith-based and therefore emits unions of the L*,
    N* etc. subclasses; the o200k pre-tokenizer needs the *specific* subclasses
    Lu, Lt, Ll and M, because it splits on case rather than on "is a letter".
    """
    return ranges(lambda cp: unicodedata.category(chr(cp)) in categories)


def main():
    letters = category_ranges("L")
    numbers = category_ranges("N")
    combining = combining_ranges()

    # The o200k pre-tokenizer's CamelCase alternatives are written with
    # \p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M} and \p{Ll}\p{Lm}\p{Lo}\p{M}. \p{Lm} and
    # \p{Lo} appear in both sets, so a single membership test for "is an
    # uncased letter" is enough for them -- and that is exactly `IsLetter &&
    # !IsUpperLetter && !IsLowerLetter`. Only the case-bearing letters and the
    # marks need their own tables.
    upper_letters = specific_category_ranges({"Lu", "Lt"})
    lower_letters = specific_category_ranges({"Ll"})
    marks = category_ranges("M")

    excluded = composition_exclusions()

    decompositions = []  # (cp, [cp, ...])
    compositions = []    # (first, second, composed)

    for cp in range(MAX_CP):
        if HANGUL_S_BASE <= cp < HANGUL_S_BASE + HANGUL_S_COUNT:
            continue

        step = canonical_decomposition(cp)

        if step is None:
            continue

        decompositions.append((cp, full_decomposition(cp)))

        # A pair recomposes only if it is a true pair, is not excluded, and its
        # first element is a starter.  Singletons never recompose.
        if len(step) == 2 and cp not in excluded:
            if unicodedata.combining(chr(step[0])) == 0:
                compositions.append((step[0], step[1], cp))

    # The composition table is searched by (first, second), so sort it that way
    # and let the C++ side binary-search a packed 64-bit key.
    compositions.sort()
    decompositions.sort()

    out = sys.stdout
    w = out.write

    w("// Generated by scripts/gen_unicode_tables.py -- do not edit.\n")
    w(f"// Unicode {unicodedata.unidata_version}, via Python's unicodedata.\n")
    w("//\n")
    w("// Tables backing the tokenizer's pre-tokenizer character classes and its\n")
    w("// NFC normalizer.  Hangul is absent by design: it is arithmetic, and\n")
    w("// unicode.cc implements the arithmetic from UAX #15 directly.\n")
    w("#pragma once\n\n")
    w("#include <cstdint>\n\n")
    w("namespace inferx::tokenizer::unicode_data {\n\n")

    w(f'inline constexpr const char* kUnicodeVersion = "{unicodedata.unidata_version}";\n\n')

    w("/// Inclusive codepoint ranges, sorted, non-overlapping.\n")
    w("struct Range {\n")
    w("  uint32_t lo;\n")
    w("  uint32_t hi;\n")
    w("};\n\n")

    def emit_ranges(name, rs):
        w(f"inline constexpr Range k{name}[] = {{\n")
        for i in range(0, len(rs), 4):
            chunk = rs[i:i + 4]
            w("    " + " ".join(f"{{0x{lo:04X}, 0x{hi:04X}}}," for lo, hi in chunk) + "\n")
        w("};\n")
        w(f"inline constexpr int k{name}Count = {len(rs)};\n\n")

    emit_ranges("Letter", letters)
    emit_ranges("Number", numbers)
    emit_ranges("UpperLetter", upper_letters)
    emit_ranges("LowerLetter", lower_letters)
    emit_ranges("Mark", marks)

    w("/// A run of codepoints sharing one non-zero canonical combining class.\n")
    w("struct CombiningRange {\n")
    w("  uint32_t lo;\n")
    w("  uint32_t hi;\n")
    w("  uint8_t ccc;\n")
    w("};\n\n")
    w("inline constexpr CombiningRange kCombining[] = {\n")
    for i in range(0, len(combining), 3):
        chunk = combining[i:i + 3]
        w("    " + " ".join(f"{{0x{lo:04X}, 0x{hi:04X}, {ccc}}}," for lo, hi, ccc in chunk) + "\n")
    w("};\n")
    w(f"inline constexpr int kCombiningCount = {len(combining)};\n\n")

    # Decompositions are variable-length, so they are stored as an index into a
    # flat pool rather than as a ragged array.
    pool = []
    entries = []

    for cp, seq in decompositions:
        entries.append((cp, len(pool), len(seq)))
        pool.extend(seq)

    w("/// Full canonical decomposition, as an (offset, length) into kDecompPool.\n")
    w("struct Decomposition {\n")
    w("  uint32_t cp;\n")
    w("  uint32_t offset;\n")
    w("  uint32_t length;\n")
    w("};\n\n")
    w("inline constexpr Decomposition kDecomp[] = {\n")
    for i in range(0, len(entries), 4):
        chunk = entries[i:i + 4]
        w("    " + " ".join(f"{{0x{cp:04X}, {off}, {ln}}}," for cp, off, ln in chunk) + "\n")
    w("};\n")
    w(f"inline constexpr int kDecompCount = {len(entries)};\n\n")

    w("inline constexpr uint32_t kDecompPool[] = {\n")
    for i in range(0, len(pool), 8):
        chunk = pool[i:i + 8]
        w("    " + " ".join(f"0x{cp:04X}," for cp in chunk) + "\n")
    w("};\n")
    w(f"inline constexpr int kDecompPoolCount = {len(pool)};\n\n")

    w("/// Canonical composition, keyed on (first << 32) | second so that the\n")
    w("/// lookup is a binary search on a single integer.\n")
    w("struct Composition {\n")
    w("  uint64_t pair;\n")
    w("  uint32_t composed;\n")
    w("};\n\n")
    w("inline constexpr Composition kComposition[] = {\n")
    for i in range(0, len(compositions), 3):
        chunk = compositions[i:i + 3]
        w("    " + " ".join(
            f"{{0x{(a << 32) | b:012X}ULL, 0x{c:04X}}}," for a, b, c in chunk) + "\n")
    w("};\n")
    w(f"inline constexpr int kCompositionCount = {len(compositions)};\n\n")

    w("}  // namespace inferx::tokenizer::unicode_data\n")


if __name__ == "__main__":
    main()
