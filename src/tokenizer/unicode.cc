#include "inferx/tokenizer/unicode.h"

#include <algorithm>

#include "inferx/tokenizer/unicode_data.h"

namespace inferx::tokenizer::unicode {
namespace {

using namespace unicode_data;  // NOLINT(build/namespaces) -- generated tables

// Hangul composition and decomposition are arithmetic, per UAX #15.  The
// generated tables omit the 11k syllables for that reason.
constexpr uint32_t kHangulSBase = 0xAC00;
constexpr uint32_t kHangulLBase = 0x1100;
constexpr uint32_t kHangulVBase = 0x1161;
constexpr uint32_t kHangulTBase = 0x11A7;
constexpr uint32_t kHangulLCount = 19;
constexpr uint32_t kHangulVCount = 21;
constexpr uint32_t kHangulTCount = 28;
constexpr uint32_t kHangulNCount = kHangulVCount * kHangulTCount;  // 588
constexpr uint32_t kHangulSCount = kHangulLCount * kHangulNCount;  // 11172

bool InRanges(const Range* table, int count, uint32_t cp) {
  const Range* end = table + count;

  // Ranges are sorted and disjoint, so the last range starting at or below `cp`
  // is the only one that can contain it.
  const Range* it = std::upper_bound(
      table, end, cp, [](uint32_t v, const Range& r) { return v < r.lo; });

  if (it == table) return false;

  --it;
  return cp <= it->hi;
}

int CanonicalCombiningClass(uint32_t cp) {
  const CombiningRange* end = kCombining + kCombiningCount;

  const CombiningRange* it =
      std::upper_bound(kCombining, end, cp, [](uint32_t v,
                                               const CombiningRange& r) {
        return v < r.lo;
      });

  if (it == kCombining) return 0;

  --it;
  return cp <= it->hi ? it->ccc : 0;
}

const Decomposition* FindDecomposition(uint32_t cp) {
  const Decomposition* end = kDecomp + kDecompCount;

  const Decomposition* it = std::lower_bound(
      kDecomp, end, cp,
      [](const Decomposition& d, uint32_t v) { return d.cp < v; });

  if (it == end || it->cp != cp) return nullptr;

  return it;
}

// Returns the canonical composition of `a` and `b`, or 0 if they do not
// compose.  Hangul is arithmetic; everything else is a table lookup.
uint32_t Compose(uint32_t a, uint32_t b) {
  if (a >= kHangulLBase && a < kHangulLBase + kHangulLCount &&
      b >= kHangulVBase && b < kHangulVBase + kHangulVCount) {
    const uint32_t l = a - kHangulLBase;
    const uint32_t v = b - kHangulVBase;
    return kHangulSBase + (l * kHangulVCount + v) * kHangulTCount;
  }

  if (a >= kHangulSBase && a < kHangulSBase + kHangulSCount &&
      (a - kHangulSBase) % kHangulTCount == 0 && b > kHangulTBase &&
      b < kHangulTBase + kHangulTCount) {
    return a + (b - kHangulTBase);
  }

  const uint64_t key = (static_cast<uint64_t>(a) << 32) | b;
  const Composition* end = kComposition + kCompositionCount;

  const Composition* it = std::lower_bound(
      kComposition, end, key,
      [](const Composition& c, uint64_t v) { return c.pair < v; });

  if (it == end || it->pair != key) return 0;

  return it->composed;
}

void DecomposeInto(uint32_t cp, std::vector<uint32_t>* out) {
  if (cp >= kHangulSBase && cp < kHangulSBase + kHangulSCount) {
    const uint32_t index = cp - kHangulSBase;

    out->push_back(kHangulLBase + index / kHangulNCount);
    out->push_back(kHangulVBase + (index % kHangulNCount) / kHangulTCount);

    if (const uint32_t t = index % kHangulTCount; t != 0) {
      out->push_back(kHangulTBase + t);
    }

    return;
  }

  if (const Decomposition* d = FindDecomposition(cp); d != nullptr) {
    // The table already holds the *full* decomposition, so this does not
    // recurse.
    out->insert(out->end(), kDecompPool + d->offset,
                kDecompPool + d->offset + d->length);
    return;
  }

  out->push_back(cp);
}

// The set of codepoints that can appear as the second element of a composition
// pair.  Built once, so the NFC quick check below is a binary search rather
// than a scan of the composition table.
const std::vector<uint32_t>& CompositionSeconds() {
  static const std::vector<uint32_t>* const seconds = [] {
    auto* v = new std::vector<uint32_t>;
    v->reserve(kCompositionCount);

    for (int i = 0; i < kCompositionCount; ++i) {
      v->push_back(static_cast<uint32_t>(kComposition[i].pair & 0xFFFFFFFFULL));
    }

    std::sort(v->begin(), v->end());
    v->erase(std::unique(v->begin(), v->end()), v->end());
    return v;
  }();

  return *seconds;
}

// Whether `cp` can be left alone by NFC no matter what surrounds it.
//
// It is safe when it does not decompose, carries no combining class, and can
// never be absorbed into a preceding character.  A string made entirely of such
// codepoints is already in NFC, which lets the common case skip the three-pass
// algorithm entirely.
bool IsNfcInert(uint32_t cp) {
  if (cp < 0x80) return true;

  if (cp >= kHangulLBase && cp < kHangulLBase + kHangulLCount) return false;
  if (cp >= kHangulVBase && cp < kHangulVBase + kHangulVCount) return false;
  if (cp > kHangulTBase && cp < kHangulTBase + kHangulTCount) return false;
  if (cp >= kHangulSBase && cp < kHangulSBase + kHangulSCount) return false;

  if (CanonicalCombiningClass(cp) != 0) return false;
  if (FindDecomposition(cp) != nullptr) return false;

  const std::vector<uint32_t>& seconds = CompositionSeconds();
  return !std::binary_search(seconds.begin(), seconds.end(), cp);
}

}  // namespace

std::vector<uint32_t> Utf8Decode(std::string_view text) {
  std::vector<uint32_t> out;
  out.reserve(text.size());

  size_t i = 0;

  while (i < text.size()) {
    const auto lead = static_cast<unsigned char>(text[i]);
    const int length = Utf8SequenceLength(lead);

    if (length == 0 || i + length > text.size()) {
      out.push_back(kReplacementChar);
      ++i;
      continue;
    }

    uint32_t cp = 0;

    switch (length) {
      case 1:
        cp = lead;
        break;
      case 2:
        cp = lead & 0x1Fu;
        break;
      case 3:
        cp = lead & 0x0Fu;
        break;
      default:
        cp = lead & 0x07u;
        break;
    }

    bool valid = true;

    for (int k = 1; k < length; ++k) {
      const auto byte = static_cast<unsigned char>(text[i + k]);

      if ((byte & 0xC0u) != 0x80u) {
        valid = false;
        break;
      }

      cp = (cp << 6) | (byte & 0x3Fu);
    }

    // An overlong encoding is a different byte sequence for a codepoint that
    // has a shorter one, and treating the two as equal is a classic filter
    // bypass.  Surrogates are not scalar values and must not survive decoding.
    if (!valid || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF) ||
        (length == 2 && cp < 0x80) || (length == 3 && cp < 0x800) ||
        (length == 4 && cp < 0x10000)) {
      out.push_back(kReplacementChar);
      ++i;
      continue;
    }

    out.push_back(cp);
    i += length;
  }

  return out;
}

void Utf8Encode(uint32_t cp, std::string* out) {
  if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) cp = kReplacementChar;

  if (cp < 0x80) {
    out->push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out->push_back(static_cast<char>(0xC0u | (cp >> 6)));
    out->push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
  } else if (cp < 0x10000) {
    out->push_back(static_cast<char>(0xE0u | (cp >> 12)));
    out->push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
    out->push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
  } else {
    out->push_back(static_cast<char>(0xF0u | (cp >> 18)));
    out->push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
    out->push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
    out->push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
  }
}

std::string Utf8Encode(const std::vector<uint32_t>& codepoints) {
  std::string out;
  out.reserve(codepoints.size());

  for (const uint32_t cp : codepoints) Utf8Encode(cp, &out);

  return out;
}

int Utf8SequenceLength(unsigned char lead) {
  if (lead < 0x80) return 1;
  if ((lead & 0xE0u) == 0xC0u) return 2;
  if ((lead & 0xF0u) == 0xE0u) return 3;
  if ((lead & 0xF8u) == 0xF0u) return 4;
  return 0;  // continuation byte or an invalid lead
}

bool EndsWithPartialUtf8(std::string_view text) {
  // A sequence is at most four bytes, so at most the last three can be an
  // incomplete one.
  const size_t limit = std::min<size_t>(text.size(), 4);

  for (size_t back = 1; back <= limit; ++back) {
    const auto byte = static_cast<unsigned char>(text[text.size() - back]);

    if ((byte & 0xC0u) == 0x80u) continue;  // continuation, keep walking back

    const int length = Utf8SequenceLength(byte);

    // Zero means an invalid lead byte, which is not a partial sequence -- it is
    // simply broken, and holding it back would stall the stream forever.
    return length != 0 && static_cast<size_t>(length) > back;
  }

  return false;
}

bool IsLetter(uint32_t cp) { return InRanges(kLetter, kLetterCount, cp); }

bool IsNumber(uint32_t cp) { return InRanges(kNumber, kNumberCount, cp); }

bool IsUpperLetter(uint32_t cp) {
  return InRanges(kUpperLetter, kUpperLetterCount, cp);
}

bool IsLowerLetter(uint32_t cp) {
  return InRanges(kLowerLetter, kLowerLetterCount, cp);
}

bool IsMark(uint32_t cp) { return InRanges(kMark, kMarkCount, cp); }

bool IsWhitespace(uint32_t cp) {
  // \p{White_Space}.  Small and stable enough to inline rather than generate.
  switch (cp) {
    case 0x09:
    case 0x0A:
    case 0x0B:
    case 0x0C:
    case 0x0D:
    case 0x20:
    case 0x85:
    case 0xA0:
    case 0x1680:
    case 0x2028:
    case 0x2029:
    case 0x202F:
    case 0x205F:
    case 0x3000:
      return true;
    default:
      return cp >= 0x2000 && cp <= 0x200A;
  }
}

std::string NormalizeNfc(std::string_view text) {
  const std::vector<uint32_t> input = Utf8Decode(text);

  bool inert = true;

  for (const uint32_t cp : input) {
    if (!IsNfcInert(cp)) {
      inert = false;
      break;
    }
  }

  // Already NFC, and -- because decoding replaced nothing -- byte-identical to
  // the input, so hand it straight back.
  if (inert && Utf8Encode(input).size() == text.size()) {
    return std::string(text);
  }

  std::vector<uint32_t> decomposed;
  decomposed.reserve(input.size() + input.size() / 4);

  for (const uint32_t cp : input) DecomposeInto(cp, &decomposed);

  // Canonical ordering: a stable sort of each run of non-starters by combining
  // class.  An insertion sort over adjacent pairs is the standard formulation
  // and is stable by construction; runs are a handful of characters at most.
  for (size_t i = 1; i < decomposed.size(); ++i) {
    const int ccc = CanonicalCombiningClass(decomposed[i]);

    if (ccc == 0) continue;

    size_t j = i;

    while (j > 0) {
      const int prev = CanonicalCombiningClass(decomposed[j - 1]);

      if (prev <= ccc) break;

      std::swap(decomposed[j - 1], decomposed[j]);
      --j;
    }
  }

  std::vector<uint32_t> out;
  out.reserve(decomposed.size());

  // Composition, per UAX #15: fold each character into the most recent starter
  // unless something between them blocks it.  `prev_ccc` is the class of the
  // character immediately preceding the candidate, or -1 when the candidate
  // directly follows the starter; a candidate is blocked exactly when that
  // class is greater than or equal to its own.
  ptrdiff_t last_starter = -1;
  int prev_ccc = -1;

  for (const uint32_t cp : decomposed) {
    const int ccc = CanonicalCombiningClass(cp);

    if (last_starter >= 0 && !(prev_ccc >= 0 && prev_ccc >= ccc)) {
      if (const uint32_t composed = Compose(out[last_starter], cp);
          composed != 0) {
        out[last_starter] = composed;
        continue;  // absorbed, so `prev_ccc` is unchanged
      }
    }

    out.push_back(cp);

    if (ccc == 0) {
      last_starter = static_cast<ptrdiff_t>(out.size()) - 1;
      prev_ccc = -1;
    } else {
      prev_ccc = ccc;
    }
  }

  return Utf8Encode(out);
}

}  // namespace inferx::tokenizer::unicode
