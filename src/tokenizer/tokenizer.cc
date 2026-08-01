#include "inferx/tokenizer/tokenizer.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <queue>
#include <sstream>
#include <utility>

#include "inferx/common/json.h"
#include "inferx/tokenizer/unicode.h"
#include "inferx/tokenizer/unicode_data.h"

namespace inferx::tokenizer {
namespace {

// The pre-tokenizer pattern this class implements, verbatim from Qwen2's
// tokenizer.json.  Loading refuses any file declaring a different one: the
// split is hand-written, so a pattern we have not read is a pattern we would
// silently get wrong.
constexpr std::string_view kExpectedSplitPattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

// GPT-2's byte-to-unicode alphabet: every byte gets a distinct printable
// codepoint, so that arbitrary binary survives a text-only BPE unchanged.  The
// 188 bytes that are already printable map to themselves; the other 68 are
// pushed into U+0100 and up, in byte order.
//
// This is what makes `Ġ` mean "space" and `Ċ` mean "newline" in the vocabulary.
std::array<uint32_t, 256> BuildByteToCodepoint() {
  std::array<uint32_t, 256> table{};
  std::array<bool, 256> direct{};

  const auto mark = [&](int lo, int hi) {
    for (int b = lo; b <= hi; ++b) direct[b] = true;
  };

  mark('!', '~');
  mark(0xA1, 0xAC);
  mark(0xAE, 0xFF);

  uint32_t next = 256;

  for (int b = 0; b < 256; ++b) {
    table[b] = direct[b] ? static_cast<uint32_t>(b) : next++;
  }

  return table;
}

const std::array<std::string, 256>& ByteToChars() {
  static const auto* const table = [] {
    auto* out = new std::array<std::string, 256>();
    const auto codepoints = BuildByteToCodepoint();

    for (int b = 0; b < 256; ++b) unicode::Utf8Encode(codepoints[b], &(*out)[b]);

    return out;
  }();

  return *table;
}

// Reverse of the above.  Sparse over a small range, so a flat array indexed by
// codepoint beats a hash map.
const std::array<int16_t, 324>& CodepointToByte() {
  static const auto* const table = [] {
    auto* out = new std::array<int16_t, 324>();
    out->fill(-1);

    const auto codepoints = BuildByteToCodepoint();

    for (int b = 0; b < 256; ++b) {
      (*out)[codepoints[b]] = static_cast<int16_t>(b);
    }

    return out;
  }();

  return *table;
}

bool IsCrLf(uint32_t cp) { return cp == '\r' || cp == '\n'; }

char AsciiLower(uint32_t cp) {
  return static_cast<char>(cp >= 'A' && cp <= 'Z' ? cp + 32 : cp);
}

// Length of the run of codepoints from `p` satisfying `pred`.
template <typename Pred>
size_t RunLength(const std::vector<uint32_t>& cps, size_t p, Pred pred) {
  size_t n = 0;
  while (p + n < cps.size() && pred(cps[p + n])) ++n;
  return n;
}

/// Matches one alternative of the pre-tokenizer pattern at `p`, returning the
/// number of codepoints consumed, or 0 for no match.
///
/// HuggingFace runs this pattern through Oniguruma, which is a backtracking
/// engine: alternatives are tried in written order and the first that matches
/// wins, and quantifiers are greedy but give ground when the rest of the
/// alternative cannot proceed.  Both behaviours are load-bearing here and are
/// reproduced explicitly below -- a leftmost-*longest* engine would produce
/// different, and wrong, token ids.

// (?i:'s|'t|'re|'ve|'m|'ll|'d)
//
// Case folding is ASCII-only.  Oniguruma would also fold U+017F (long s) into
// `s`, so `'ſ` divides here; it is not worth a case-folding table.
size_t MatchContraction(const std::vector<uint32_t>& cps, size_t p) {
  if (cps[p] != '\'') return 0;

  static constexpr std::string_view kSuffixes[] = {"s", "t", "re", "ve",
                                                   "m", "ll", "d"};

  for (const std::string_view suffix : kSuffixes) {
    if (p + 1 + suffix.size() > cps.size()) continue;

    bool ok = true;

    for (size_t k = 0; k < suffix.size(); ++k) {
      if (AsciiLower(cps[p + 1 + k]) != suffix[k]) {
        ok = false;
        break;
      }
    }

    if (ok) return 1 + suffix.size();
  }

  return 0;
}

// [^\r\n\p{L}\p{N}]?\p{L}+
size_t MatchWord(const std::vector<uint32_t>& cps, size_t p) {
  // The optional prefix is greedy, so try consuming one first and fall back to
  // consuming none.  This is what attaches the leading space to a word.
  for (int prefix = 1; prefix >= 0; --prefix) {
    if (prefix == 1) {
      const uint32_t c = cps[p];
      if (IsCrLf(c) || unicode::IsLetter(c) || unicode::IsNumber(c)) continue;
      if (p + 1 >= cps.size()) continue;
    }

    const size_t letters =
        RunLength(cps, p + prefix, [](uint32_t c) { return unicode::IsLetter(c); });

    if (letters >= 1) return prefix + letters;
  }

  return 0;
}

//  ?[^\s\p{L}\p{N}]+[\r\n]*
size_t MatchPunctuation(const std::vector<uint32_t>& cps, size_t p) {
  for (int space = 1; space >= 0; --space) {
    if (space == 1 && cps[p] != ' ') continue;

    const size_t q = p + space;

    const size_t run = RunLength(cps, q, [](uint32_t c) {
      return !unicode::IsWhitespace(c) && !unicode::IsLetter(c) &&
             !unicode::IsNumber(c);
    });

    if (run == 0) continue;

    const size_t newlines = RunLength(cps, q + run, IsCrLf);
    return space + run + newlines;
  }

  return 0;
}

// \s*[\r\n]+
size_t MatchNewlineRun(const std::vector<uint32_t>& cps, size_t p) {
  const size_t whitespace =
      RunLength(cps, p, [](uint32_t c) { return unicode::IsWhitespace(c); });

  // `\s*` is greedy, so give back one codepoint at a time until `[\r\n]+` can
  // match at the split point.
  for (size_t k = whitespace + 1; k-- > 0;) {
    if (p + k < cps.size() && IsCrLf(cps[p + k])) {
      return k + RunLength(cps, p + k, IsCrLf);
    }
  }

  return 0;
}

// \s+(?!\S)
size_t MatchTrailingWhitespace(const std::vector<uint32_t>& cps, size_t p) {
  const size_t run =
      RunLength(cps, p, [](uint32_t c) { return unicode::IsWhitespace(c); });

  if (run == 0) return 0;

  // The full run satisfies the lookahead only at end of input.  Otherwise the
  // engine backs off by one, which leaves the last space for the word that
  // follows -- the reason " hello" is one token and not two.
  if (p + run == cps.size()) return run;

  return run >= 2 ? run - 1 : 0;
}

// \s+
size_t MatchWhitespace(const std::vector<uint32_t>& cps, size_t p) {
  return RunLength(cps, p, [](uint32_t c) { return unicode::IsWhitespace(c); });
}

// Splits into the pieces BPE is applied to, as [begin, end) codepoint ranges.
std::vector<std::pair<size_t, size_t>> PreTokenize(
    const std::vector<uint32_t>& cps) {
  std::vector<std::pair<size_t, size_t>> pieces;
  size_t p = 0;

  while (p < cps.size()) {
    size_t len = MatchContraction(cps, p);

    if (len == 0) len = MatchWord(cps, p);
    if (len == 0) len = unicode::IsNumber(cps[p]) ? 1 : 0;
    if (len == 0) len = MatchPunctuation(cps, p);
    if (len == 0) len = MatchNewlineRun(cps, p);
    if (len == 0) len = MatchTrailingWhitespace(cps, p);
    if (len == 0) len = MatchWhitespace(cps, p);

    // Every codepoint matches one of the alternatives -- a character is a
    // letter, a digit, whitespace, or falls to the punctuation branch -- so
    // this is unreachable.  It consumes a codepoint anyway rather than looping
    // forever, because "unreachable" and "cannot happen on a socket" are
    // different claims.
    if (len == 0) len = 1;

    pieces.emplace_back(p, p + len);
    p += len;
  }

  return pieces;
}

StatusOr<std::string> ReadWholeFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);

  if (!in) return NotFoundError("cannot open ", path);

  std::ostringstream buffer;
  buffer << in.rdbuf();

  if (!in.good() && !in.eof()) return InternalError("error reading ", path);

  return buffer.str();
}

// Confirms a pipeline component is the one we implement, by type name.
Status ExpectType(const JsonValue* node, std::string_view expected,
                  std::string_view what) {
  if (node == nullptr || node->IsNull()) {
    return InvalidArgumentError("tokenizer.json has no ", what, ", expected ",
                                expected);
  }

  INFERX_ASSIGN_OR_RETURN(const std::string_view type,
                          node->RequiredString("type"));

  if (type != expected) {
    return UnimplementedError("tokenizer.json declares ", what, " \"", type,
                              "\", but this tokenizer implements only \"",
                              expected, "\"");
  }

  return OkStatus();
}

}  // namespace

const char* Tokenizer::UnicodeVersion() {
  return unicode_data::kUnicodeVersion;
}

StatusOr<std::unique_ptr<Tokenizer>> Tokenizer::LoadFromFile(
    const std::string& path) {
  INFERX_ASSIGN_OR_RETURN(const std::string text, ReadWholeFile(path));
  INFERX_ASSIGN_OR_RETURN(const JsonValue root, ParseJson(text));

  auto tokenizer = std::unique_ptr<Tokenizer>(new Tokenizer());

  // --- Pipeline conformance -------------------------------------------------
  //
  // Each of these is a component whose behaviour is hard-coded below.  A file
  // declaring something else gets a clear error at load, which is enormously
  // cheaper than tokenizing subtly wrongly and debugging it from bad output.

  if (const JsonValue* normalizer = root.Find("normalizer");
      normalizer != nullptr && !normalizer->IsNull()) {
    INFERX_RETURN_IF_ERROR(ExpectType(normalizer, "NFC", "normalizer"));
  }

  INFERX_RETURN_IF_ERROR(
      ExpectType(root.Find("pre_tokenizer"), "Sequence", "pre_tokenizer"));
  INFERX_RETURN_IF_ERROR(ExpectType(root.Find("decoder"), "ByteLevel",
                                    "decoder"));

  {
    INFERX_ASSIGN_OR_RETURN(
        const std::vector<JsonValue>* stages,
        root.Find("pre_tokenizer")->Find("pretokenizers") == nullptr
            ? StatusOr<const std::vector<JsonValue>*>(InvalidArgumentError(
                  "pre_tokenizer Sequence has no \"pretokenizers\""))
            : root.Find("pre_tokenizer")->Find("pretokenizers")->AsArray());

    if (stages->size() != 2) {
      return UnimplementedError(
          "pre_tokenizer has ", stages->size(),
          " stages, expected 2 (Split then ByteLevel)");
    }

    INFERX_RETURN_IF_ERROR(ExpectType(&(*stages)[0], "Split", "pre-tokenizer"));
    INFERX_RETURN_IF_ERROR(
        ExpectType(&(*stages)[1], "ByteLevel", "pre-tokenizer"));

    const JsonValue* pattern = (*stages)[0].Find("pattern");

    if (pattern == nullptr) {
      return InvalidArgumentError("Split pre-tokenizer has no pattern");
    }

    INFERX_ASSIGN_OR_RETURN(const std::string_view regex,
                            pattern->RequiredString("Regex"));

    if (regex != kExpectedSplitPattern) {
      return UnimplementedError(
          "the Split pre-tokenizer's pattern is not the one this tokenizer "
          "implements; it splits text differently and would produce different "
          "token ids");
    }

    INFERX_ASSIGN_OR_RETURN(const std::string_view behavior,
                           (*stages)[0].RequiredString("behavior"));

    if (behavior != "Isolated") {
      return UnimplementedError("Split behavior is \"", behavior,
                                "\", expected \"Isolated\"");
    }
  }

  // --- Model ----------------------------------------------------------------

  const JsonValue* model = root.Find("model");
  INFERX_RETURN_IF_ERROR(ExpectType(model, "BPE", "model"));

  // Each of these changes what BPE does, and none is implemented.
  if (const JsonValue* v = model->Find("byte_fallback");
      v != nullptr && !v->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const bool byte_fallback, v->AsBool());
    if (byte_fallback) return UnimplementedError("BPE byte_fallback is on");
  }

  if (const JsonValue* v = model->Find("dropout");
      v != nullptr && !v->IsNull()) {
    return UnimplementedError("BPE dropout is set");
  }

  for (const std::string_view affix :
       {"continuing_subword_prefix", "end_of_word_suffix"}) {
    if (const JsonValue* v = model->Find(affix);
        v != nullptr && !v->IsNull()) {
      INFERX_ASSIGN_OR_RETURN(const std::string_view value, v->AsString());
      if (!value.empty()) {
        return UnimplementedError("BPE ", affix, " is \"", value, "\"");
      }
    }
  }

  const JsonValue* vocab = model->Find("vocab");

  if (vocab == nullptr) return InvalidArgumentError("model has no vocab");

  INFERX_ASSIGN_OR_RETURN(const auto* entries, vocab->AsObject());

  tokenizer->token_to_id_.reserve(entries->size() * 2);

  int64_t max_id = -1;

  for (const auto& [token, id_value] : *entries) {
    INFERX_ASSIGN_OR_RETURN(const int64_t id, id_value.AsInt());

    if (id < 0) return InvalidArgumentError("vocab entry has negative id ", id);

    max_id = std::max(max_id, id);
    tokenizer->token_to_id_.emplace(token, static_cast<int32_t>(id));
  }

  // --- Added tokens ---------------------------------------------------------

  if (const JsonValue* added = root.Find("added_tokens"); added != nullptr) {
    INFERX_ASSIGN_OR_RETURN(const auto* list, added->AsArray());

    for (const JsonValue& entry : *list) {
      AddedToken token;

      INFERX_ASSIGN_OR_RETURN(const std::string_view content,
                              entry.RequiredString("content"));
      INFERX_ASSIGN_OR_RETURN(const int64_t id, entry.RequiredInt("id"));
      INFERX_ASSIGN_OR_RETURN(token.special,
                              entry.OptionalBool("special", false));

      token.content = std::string(content);
      token.id = static_cast<int32_t>(id);

      max_id = std::max(max_id, id);

      tokenizer->token_to_id_[token.content] = token.id;
      tokenizer->added_special_[token.id] = token.special;
      tokenizer->added_.push_back(std::move(token));
    }
  }

  // Longest first, so that the scan in EncodeWithAddedTokens is
  // leftmost-longest without a second pass.
  std::sort(tokenizer->added_.begin(), tokenizer->added_.end(),
            [](const AddedToken& a, const AddedToken& b) {
              return a.content.size() > b.content.size();
            });

  tokenizer->id_to_token_.assign(static_cast<size_t>(max_id) + 1, std::string());

  for (const auto& [token, id] : tokenizer->token_to_id_) {
    tokenizer->id_to_token_[static_cast<size_t>(id)] = token;
  }

  // Every byte must be representable, or some inputs would have no encoding at
  // all.  Checking it here turns a corrupt vocabulary into a load failure
  // rather than tokens silently going missing from the middle of a prompt.
  for (const std::string& byte_char : ByteToChars()) {
    if (tokenizer->token_to_id_.count(byte_char) == 0) {
      return InvalidArgumentError(
          "vocabulary is missing a byte-level character, so not every byte can "
          "be encoded");
    }
  }

  // --- Merges ---------------------------------------------------------------

  const JsonValue* merges = model->Find("merges");

  if (merges == nullptr) return InvalidArgumentError("model has no merges");

  INFERX_ASSIGN_OR_RETURN(const auto* merge_list, merges->AsArray());

  tokenizer->merge_rank_.reserve(merge_list->size() * 2);

  int32_t rank = 0;

  for (const JsonValue& entry : *merge_list) {
    std::string key;

    // Older files write a merge as "a b"; newer ones as ["a", "b"].
    if (entry.IsArray()) {
      INFERX_ASSIGN_OR_RETURN(const auto* pair, entry.AsArray());

      if (pair->size() != 2) {
        return InvalidArgumentError("merge ", rank, " has ", pair->size(),
                                    " parts, expected 2");
      }

      INFERX_ASSIGN_OR_RETURN(const std::string_view left,
                              (*pair)[0].AsString());
      INFERX_ASSIGN_OR_RETURN(const std::string_view right,
                              (*pair)[1].AsString());

      key.reserve(left.size() + right.size() + 1);
      key.append(left).push_back(' ');
      key.append(right);
    } else {
      INFERX_ASSIGN_OR_RETURN(const std::string_view pair, entry.AsString());

      if (pair.find(' ') == std::string_view::npos) {
        return InvalidArgumentError("merge ", rank, " is not a pair: \"", pair,
                                    "\"");
      }

      key = std::string(pair);
    }

    // A merge whose result is not in the vocabulary can never be applied --
    // there would be no id to emit for it -- so dropping it is what makes the
    // rest of the table usable.  Well-formed files have none.
    std::string joined = key;
    joined.erase(joined.find(' '), 1);

    if (tokenizer->token_to_id_.count(joined) != 0) {
      tokenizer->merge_rank_.emplace(std::move(key), rank);
    }

    ++rank;
  }

  // --- End of sequence ------------------------------------------------------
  //
  // Overridden from tokenizer_config.json by LoadFromDirectory when present;
  // this is the fallback for a bare tokenizer.json.
  for (const std::string_view candidate : {"<|im_end|>", "<|endoftext|>"}) {
    if (const auto it = tokenizer->token_to_id_.find(std::string(candidate));
        it != tokenizer->token_to_id_.end()) {
      tokenizer->eos_id_ = it->second;
      break;
    }
  }

  return tokenizer;
}

StatusOr<std::unique_ptr<Tokenizer>> Tokenizer::LoadFromDirectory(
    const std::string& dir) {
  INFERX_ASSIGN_OR_RETURN(std::unique_ptr<Tokenizer> tokenizer,
                          LoadFromFile(dir + "/tokenizer.json"));

  // The end-of-sequence token is a property of the *checkpoint*, not of the
  // tokenizer: base and instruct Qwen2 models share a vocabulary but stop on
  // different tokens.  Absent config is not an error -- the fallback above
  // already picked a reasonable one.
  if (StatusOr<std::string> config = ReadWholeFile(dir + "/tokenizer_config.json");
      config.ok()) {
    INFERX_ASSIGN_OR_RETURN(const JsonValue root, ParseJson(*config));

    if (const JsonValue* eos = root.Find("eos_token");
        eos != nullptr && !eos->IsNull()) {
      // Either a bare string or an AddedToken object with a "content" field.
      std::string_view content;

      if (eos->kind() == JsonValue::Kind::kString) {
        INFERX_ASSIGN_OR_RETURN(content, eos->AsString());
      } else {
        INFERX_ASSIGN_OR_RETURN(content, eos->RequiredString("content"));
      }

      if (const std::optional<int32_t> id = tokenizer->TokenToId(content)) {
        tokenizer->eos_id_ = *id;
      } else {
        return InvalidArgumentError("tokenizer_config.json names eos_token \"",
                                    content,
                                    "\", which is not in the vocabulary");
      }
    }
  }

  return tokenizer;
}

std::optional<int32_t> Tokenizer::TokenToId(std::string_view token) const {
  const auto it = token_to_id_.find(std::string(token));

  if (it == token_to_id_.end()) return std::nullopt;

  return it->second;
}

std::string_view Tokenizer::IdToToken(int32_t id) const {
  if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return {};

  return id_to_token_[static_cast<size_t>(id)];
}

bool Tokenizer::IsSpecial(int32_t id) const {
  const auto it = added_special_.find(id);

  return it != added_special_.end() && it->second;
}

std::vector<int32_t> Tokenizer::Encode(std::string_view text) const {
  std::vector<int32_t> out;
  EncodeWithAddedTokens(text, &out);

  return out;
}

std::vector<int32_t> Tokenizer::EncodeOrdinary(std::string_view text) const {
  std::vector<int32_t> out;
  EncodeSpan(text, &out);

  return out;
}

void Tokenizer::EncodeWithAddedTokens(std::string_view text,
                                      std::vector<int32_t>* out) const {
  size_t span_start = 0;
  size_t p = 0;

  while (p < text.size()) {
    const AddedToken* hit = nullptr;

    // `added_` is sorted longest-first, so the first content that matches here
    // is the longest one that matches at `p`.
    for (const AddedToken& token : added_) {
      if (text.compare(p, token.content.size(), token.content) == 0) {
        hit = &token;
        break;
      }
    }

    if (hit == nullptr) {
      ++p;
      continue;
    }

    if (p > span_start) {
      EncodeSpan(text.substr(span_start, p - span_start), out);
    }

    out->push_back(hit->id);

    p += hit->content.size();
    span_start = p;
  }

  if (span_start < text.size()) EncodeSpan(text.substr(span_start), out);
}

void Tokenizer::EncodeSpan(std::string_view text,
                           std::vector<int32_t>* out) const {
  if (text.empty()) return;

  const std::string normalized = unicode::NormalizeNfc(text);
  const std::vector<uint32_t> cps = unicode::Utf8Decode(normalized);

  const std::array<std::string, 256>& byte_chars = ByteToChars();

  std::string word;
  std::string utf8;

  for (const auto& [begin, end] : PreTokenize(cps)) {
    word.clear();

    // Byte level: encode the piece to UTF-8, then replace each *byte* with its
    // alphabet character.  Going through bytes rather than codepoints is the
    // whole point -- it is what lets a token boundary fall inside a multi-byte
    // character, which Qwen's vocabulary makes heavy use of for CJK.
    for (size_t i = begin; i < end; ++i) {
      utf8.clear();
      unicode::Utf8Encode(cps[i], &utf8);

      for (const char byte : utf8) {
        word += byte_chars[static_cast<unsigned char>(byte)];
      }
    }

    ApplyBpe(word, out);
  }
}

void Tokenizer::ApplyBpe(const std::string& word,
                         std::vector<int32_t>* out) const {
  if (word.empty()) return;

  // Symbols are always contiguous substrings of `word` -- merging concatenates
  // neighbours -- so each is just an offset and a length.
  struct Symbol {
    uint32_t start;
    uint32_t length;
    int32_t prev;
    int32_t next;
    bool alive;
  };

  std::vector<Symbol> symbols;
  symbols.reserve(word.size());

  for (size_t i = 0; i < word.size();) {
    const int length =
        unicode::Utf8SequenceLength(static_cast<unsigned char>(word[i]));
    const size_t step = length == 0 ? 1 : static_cast<size_t>(length);

    symbols.push_back({static_cast<uint32_t>(i), static_cast<uint32_t>(step),
                       static_cast<int32_t>(symbols.size()) - 1,
                       static_cast<int32_t>(symbols.size()) + 1, true});

    i += step;
  }

  symbols.back().next = -1;

  const auto emit = [&](const Symbol& symbol) {
    const std::string piece = word.substr(symbol.start, symbol.length);

    if (const auto it = token_to_id_.find(piece); it != token_to_id_.end()) {
      out->push_back(it->second);
      return;
    }

    // Unreachable for a well-formed vocabulary: every single byte character is
    // present (checked at load) and every merge that survived loading has its
    // result in the vocabulary.  Falling back to the constituent bytes keeps a
    // corrupt file from dropping text silently.
    for (size_t i = 0; i < symbol.length;) {
      const int length = unicode::Utf8SequenceLength(
          static_cast<unsigned char>(word[symbol.start + i]));
      const size_t step = length == 0 ? 1 : static_cast<size_t>(length);

      const auto it =
          token_to_id_.find(word.substr(symbol.start + i, step));

      if (it != token_to_id_.end()) out->push_back(it->second);

      i += step;
    }
  };

  if (symbols.size() == 1) {
    emit(symbols[0]);
    return;
  }

  // A candidate merge of symbol `left` with its successor.  `combined` records
  // the total length at the time it was queued, which is how a stale entry is
  // recognised after either side has since grown.
  struct Candidate {
    int32_t rank;
    int32_t left;
    uint32_t combined;

    // Lowest rank first; ties broken leftmost, matching the reference
    // implementation's scan order.
    bool operator>(const Candidate& other) const {
      return rank != other.rank ? rank > other.rank : left > other.left;
    }
  };

  std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> queue;

  std::string key;

  const auto consider = [&](int32_t left) {
    if (left < 0) return;

    const int32_t right = symbols[left].next;

    if (right < 0) return;

    key.assign(word, symbols[left].start, symbols[left].length);
    key.push_back(' ');
    key.append(word, symbols[right].start, symbols[right].length);

    if (const auto it = merge_rank_.find(key); it != merge_rank_.end()) {
      queue.push({it->second, left,
                  symbols[left].length + symbols[right].length});
    }
  };

  for (int32_t i = 0; i + 1 < static_cast<int32_t>(symbols.size()); ++i) {
    consider(i);
  }

  // Lazy deletion rather than a decrease-key: entries invalidated by a merge
  // stay in the queue and are recognised on pop.  This keeps the whole thing
  // O(n log n), which matters because a pre-tokenized piece is not bounded --
  // a long run of spaces arrives as one word, and the naive rescan-for-minimum
  // formulation is quadratic in it.
  while (!queue.empty()) {
    const Candidate candidate = queue.top();
    queue.pop();

    const int32_t left = candidate.left;

    if (!symbols[left].alive) continue;

    const int32_t right = symbols[left].next;

    if (right < 0 || !symbols[right].alive) continue;

    if (symbols[left].length + symbols[right].length != candidate.combined) {
      continue;  // one side grew since this was queued
    }

    symbols[left].length += symbols[right].length;
    symbols[left].next = symbols[right].next;
    symbols[right].alive = false;

    if (symbols[left].next >= 0) symbols[symbols[left].next].prev = left;

    consider(left);
    consider(symbols[left].prev);
  }

  for (int32_t i = 0; i >= 0; i = symbols[i].next) emit(symbols[i]);
}

std::string Tokenizer::Decode(const std::vector<int32_t>& ids,
                              bool skip_special) const {
  const std::array<int16_t, 324>& to_byte = CodepointToByte();

  std::string out;

  for (const int32_t id : ids) {
    const std::string_view token = IdToToken(id);

    if (token.empty()) continue;

    if (added_special_.count(id) != 0) {
      // Added tokens carry their literal content; the byte-level map is not
      // applied to them.
      if (!(skip_special && IsSpecial(id))) out.append(token);
      continue;
    }

    for (const uint32_t cp : unicode::Utf8Decode(token)) {
      if (cp < to_byte.size() && to_byte[cp] >= 0) {
        out.push_back(static_cast<char>(to_byte[cp]));
      } else {
        // Not an alphabet character, so there is no byte it stands for. Pass it
        // through rather than dropping it.
        unicode::Utf8Encode(cp, &out);
      }
    }
  }

  return out;
}

IncrementalDecoder::IncrementalDecoder(const Tokenizer* tokenizer,
                                       bool skip_special)
    : tokenizer_(tokenizer), skip_special_(skip_special) {}

std::string IncrementalDecoder::Push(int32_t id) {
  pending_ += tokenizer_->Decode({id}, skip_special_);

  // Emit everything up to the start of a trailing incomplete sequence.
  size_t emit = pending_.size();

  const size_t limit = std::min<size_t>(pending_.size(), 4);

  for (size_t back = 1; back <= limit; ++back) {
    const auto byte = static_cast<unsigned char>(pending_[pending_.size() - back]);

    if ((byte & 0xC0u) == 0x80u) continue;

    const int length = unicode::Utf8SequenceLength(byte);

    // An invalid lead byte is broken input, not a partial character: holding it
    // back would stall the stream waiting for bytes that will never complete
    // it.
    if (length != 0 && static_cast<size_t>(length) > back) {
      emit = pending_.size() - back;
    }

    break;
  }

  std::string out = pending_.substr(0, emit);
  pending_.erase(0, emit);

  return out;
}

std::string IncrementalDecoder::Flush() {
  std::string out = std::move(pending_);
  pending_.clear();

  return out;
}

}  // namespace inferx::tokenizer
