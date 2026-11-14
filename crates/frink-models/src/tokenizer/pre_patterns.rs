//! The regex lists of `llm_tokenizer_bpe` (`llama-vocab.cpp` at the
//! `604cc430` pin), one constant per `LLAMA_VOCAB_PRE_TYPE_*` list.
//!
//! Data, transcribed, and reviewed against that switch; which
//! `tokenizer.ggml.pre` selects which list is [`super::pretokenize`].
//! `tools/oracle/regex_split.py` reads the same lists out of
//! `llama-vocab.cpp` itself and checks frink's cut against libllama's.
//!
//! Each list entry is the C++ literal copied verbatim: C and Rust spell
//! `\\`, `\"`, `\r` and `\n` alike, so these are the strings upstream
//! executes, which matters because upstream picks an engine per pattern
//! by comparing the string (see [`super::regex_split`]). The
//! single-pattern constants are raw strings with the same value as
//! upstream's literal. Do not tidy the lists into raw strings.

// The strings upstream has a hand-written splitter for
// (`unicode_regex_split_custom`). Frink stands in for each with the
// pattern itself, run with Rust's Unicode classes (`Step::Native`).

/// The llama3 family (`LLAMA_VOCAB_PRE_TYPE_LLAMA3`, shared with
/// `CHATGLM4`, `DBRX` and `SMAUG`). Digits group in threes; contractions fold case; a
/// punctuation run may swallow the newlines after it.
pub(super) const LLAMA3: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Qwen2 and the arms that share its regex (`QWEN2`, `STABLELM2`,
/// `HUNYUAN`, `SOLAR_OPEN`, `GROK_2`, and the second step of `LAGUNA`). Identical to [`LLAMA3`] except that digits
/// are emitted ONE at a time: `\p{N}` rather than `\p{N}{1,3}`.
///
/// The contractions fold case here too. Frink used to spell this arm
/// with lowercase-only contractions on the theory that qwen2 differed
/// from llama3 in that respect; upstream says otherwise, and the
/// difference is invisible on the obvious input (`DON'T` splits the same
/// either way, because `[^\r\n\p{L}\p{N}]?\p{L}+` also matches `'T`).
/// It shows on input like `'Sa`, where the contraction alternative wins
/// only if it folds case.
pub(super) const QWEN2: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// `LLAMA_VOCAB_PRE_TYPE_QWEN35` (`llama-vocab.cpp:400-406`): Qwen2's
/// pattern with combining marks (`\p{M}`) admitted into a letter run
/// and excluded from a punctuation run. Not an alias: on the parity
/// corpus the two split `(x` apart (qwen2 arm, `(` then `x`) and
/// together (this arm) on Bonsai 2 27B, found by `frink parity`.
pub(super) const QWEN35: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Plain GPT-2 (`LLAMA_VOCAB_PRE_TYPE_GPT2`, shared verbatim with
/// `MPT`, `OLMO`, `JAIS`, `TRILLION` and `GRANITE_DOCLING`, and a step
/// of several sequences).
///
/// Note there is no trailing `|\s+`. That is not an omission: upstream
/// ends this pattern at `\s+(?!\S)`, and the whitespace it therefore
/// leaves unmatched comes out between the matches instead. Frink
/// used to carry two arms here — one with `|\s+` for "gpt2" and one
/// without for "olmo" — which is the same code path written twice to
/// vary it, and only the second one was ever exercised.
pub(super) const GPT2: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";

/// Gemma-4 / SARVAM-style pre-split (`LLAMA_VOCAB_PRE_TYPE_GEMMA4`):
/// carve on newlines only, so BPE can merge across ordinary word
/// boundaries after `" "` → `▁` escaping.
pub(super) const NEWLINES: &str = r"[^\n]+|[\n]+";

// The lists, as `llm_tokenizer_bpe`'s `regex_exprs`.

/// `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM` (`llama-vocab.cpp:317-326`).
///
/// Six codepoints of the letter class are spelled `\u{..}` (U+1F7D,
/// U+1FBE, U+1FD3, U+1FDB, U+2126, U+212A): each has a canonical
/// equivalent, and an editor that normalises to NFC rewrites it into a
/// different codepoint and the range around it into an invalid one. The
/// string's value is still upstream's.
pub(super) const DEEPSEEK_LLM: &[&str] = &[
    "[\r\n]",
    "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-\u{1F7D}ᾀ-ᾴᾶ-ᾼ\u{1FBE}ῂ-ῄῆ-ῌῐ-\u{1FD3}ῖ-\u{1FDB}ῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤ\u{2126}ℨ\u{212A}-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
    "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
    "\\s+$",
    "[一-龥ࠀ-一가-퟿]+",
    "\\p{N}+",
];

/// `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM`, shared with `HUNYUAN_DENSE`,
/// `JOYAI_LLM` and `HY_V4` (`llama-vocab.cpp:327-337`).
pub(super) const DEEPSEEK3_LLM: &[&str] = &[
    "\\p{N}{1,3}",
    "[一-龥぀-ゟ゠-ヿ]+",
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
];

/// `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER` (`llama-vocab.cpp:351-359`).
pub(super) const DEEPSEEK_CODER: &[&str] = &[
    "[\r\n]",
    "\\s?\\p{L}+",
    "\\s?\\p{P}+",
    "[一-龥ࠀ-一가-퟿]+",
    "\\p{N}",
];

/// The arm `STARCODER`, `REFACT`, `COMMAND_R`, `SMOLLM`, `CODESHELL`,
/// `EXAONE`, `MINERVA` and `MELLUM2` share (`llama-vocab.cpp:367-380`):
/// digits one at a time, then the GPT-2 pattern, which upstream runs
/// with its hand-written GPT-2 splitter and frink with [`GPT2`].
pub(super) const STARCODER: &[&str] = &[
    "\\p{N}",
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
];

/// `LLAMA_VOCAB_PRE_TYPE_PORO`, shared with `BLOOM` and `GPT3_FINNISH`
/// (`llama-vocab.cpp:407-412`). One regex, but not one of the patterns
/// upstream has a hand-written splitter for, so it runs on codepoints
/// with the whitespace substitution like any sequence step. The
/// parentheses and `|` are literal: they are inside the class.
pub(super) const PORO: &[&str] = &[" ?[^(\\s|.,!?…。，、।۔،)]+"];

/// `LLAMA_VOCAB_PRE_TYPE_VIKING` (`llama-vocab.cpp:419-424`).
pub(super) const VIKING: &[&str] = &[" ?[^(\\s|.,!?…。，、।۔،)]+", "\\p{N}"];

/// tiny_aya's digit pattern, which upstream does not run as a regex: it
/// dispatches it to `unicode_regex_split_custom_afmoe`
/// (`unicode.cpp`, `unicode_regex_split_custom`).
pub(super) const TINY_AYA_DIGITS: &str = "\\d{1,3}(?=(?:\\d{3})*\\b)";

/// `LLAMA_VOCAB_PRE_TYPE_TINY_AYA` (`llama-vocab.cpp:464-471`).
pub(super) const TINY_AYA: &[&str] = &[
    TINY_AYA_DIGITS,
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
];

/// `LLAMA_VOCAB_PRE_TYPE_SUPERBPE` (`llama-vocab.cpp:479-483`). The
/// second pattern only ever matches empty: it cuts a digit run into
/// threes from the right.
pub(super) const SUPERBPE: &[&str] = &["\\p{N}+", "(?=(\\d{3})+(?!\\d))"];

/// `LLAMA_VOCAB_PRE_TYPE_FLINT` (`llama-vocab.cpp:285-290`).
pub(super) const FLINT: &[&str] = &[
    "\\p{N}{1,3}",
    "[一-龥\u{3040}-ゟ゠-ヿ]+",
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
];

/// `LLAMA_VOCAB_PRE_TYPE_JAIS2` (`llama-vocab.cpp:301-308`): llama3's
/// pattern with a cascade of whitespace-run lengths. Not one of the
/// strings upstream has a splitter for, so it runs on the collapsed text.
pub(super) const JAIS2: &[&str] = &[
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s{512}(?!\\S)|\\s{256}(?!\\S)|\\s{128}(?!\\S)|\\s{64}(?!\\S)|\\s{32}(?!\\S)|\\s{16}(?!\\S)|\\s{8}(?!\\S)|\\s{4}(?!\\S)|\\s{1,2}(?!\\S)|\\s{1}",
];

/// `LLAMA_VOCAB_PRE_TYPE_SPARK2_5` (`llama-vocab.cpp:337-344`).
pub(super) const SPARK2_5: &[&str] = &[
    "\\p{N}{1,3}",
    "[一-龥\u{3040}-ゟ゠-ヿ]+",
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+|[\r\n]|\\s+(?!\\S)|\\s+",
    "\\p{N}",
];

/// `LLAMA_VOCAB_PRE_TYPE_YOUTU` (`llama-vocab.cpp:345-350`).
pub(super) const YOUTU: &[&str] = &[
    "[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥\u{3040}-ゟ゠-ヿ]+",
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
];

/// `LLAMA_VOCAB_PRE_TYPE_FALCON` (`llama-vocab.cpp:360-366`).
pub(super) const FALCON: &[&str] = &[
    "[\\p{P}\\$\\+<=>\\^~\\|`]+",
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
    "[0-9][0-9][0-9]",
];

/// `LLAMA_VOCAB_PRE_TYPE_TEKKEN` (`llama-vocab.cpp:425-432`), which
/// `pixtral` also selects.
pub(super) const TEKKEN: &[&str] = &[
    "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
];

/// `LLAMA_VOCAB_PRE_TYPE_CHAMELEON` (`llama-vocab.cpp:433-446`).
pub(super) const CHAMELEON: &[&str] = &[
    "<sentinel:[0-9]+>",
    "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",
    "([\\t\\n]|    |  )",
    "\\p{N}",
    "[\\p{P}!-/:-@\\[-`{-~]",
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
];

/// `LLAMA_VOCAB_PRE_TYPE_GPT4O`, shared with `MINIMAX_M2` and, as the
/// same string, `GRANITE_EMB_MULTI` (`llama-vocab.cpp:447-463`). Not a
/// pattern upstream has a splitter for: it runs on the collapsed text,
/// where every non-ASCII letter and mark is one marker, so a non-ASCII
/// letter takes the `[^a-z]` branch of the case split.
pub(super) const GPT4O: &[&str] = &[
    "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
];

/// `LLAMA_VOCAB_PRE_TYPE_KIMI_K2` (`llama-vocab.cpp:472-478`): a
/// trigger string for `unicode_regex_split_custom_kimi_k2`, never run as
/// a regex.
pub(super) const KIMI_K2: &[&str] = &["\\p{Han}+"];

/// `LLAMA_VOCAB_PRE_TYPE_BAILINGMOE` (`llama-vocab.cpp:485-492`).
pub(super) const BAILINGMOE: &[&str] = &[
    "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
];

/// `LLAMA_VOCAB_PRE_TYPE_SEED_CODER` (`llama-vocab.cpp:493-499`).
pub(super) const SEED_CODER: &[&str] = &[
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
];

/// `LLAMA_VOCAB_PRE_TYPE_UFAKZEKA` (`llama-vocab.cpp:500-505`).
pub(super) const UFAKZEKA: &[&str] = &[
    "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
];

/// `LLAMA_VOCAB_PRE_TYPE_AFMOE` (`llama-vocab.cpp:513-523`). The first
/// string is a trigger for `unicode_regex_split_custom_afmoe`. Some
/// codepoints of the second are spelled `\u{..}` because they are
/// unassigned or NFC-unstable (U+F900 normalises to U+8C48).
pub(super) const AFMOE: &[&str] = &[
    "\\p{AFMoE_digits}",
    "[一-\u{9FFF}㐀-䶿豈-\u{FAFF}\u{3040}-ゟ゠-ヿ･-ﾟ⼀-\u{2FDF}เ-\u{E7F}\u{E80}-\u{EFF}ក-\u{17FF}က-႟ꩠ-ꩿꧠ-\u{A9FF}가-\u{D7AF}ᄀ-ᇿ]+",
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
];

/// `LLAMA_VOCAB_PRE_TYPE_LAGUNA` (`llama-vocab.cpp:524-529`): the
/// newline splitter, then the qwen2 splitter.
pub(super) const LAGUNA: &[&str] = &[
    "[^\\n]+|[\\n]+",
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
];

/// `LLAMA_VOCAB_PRE_TYPE_EXAONE_MOE` (`llama-vocab.cpp:530-536`).
pub(super) const EXAONE_MOE: &[&str] = &[
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
];

/// `LLAMA_VOCAB_PRE_TYPE_MINICPM5` (`llama-vocab.cpp:563-570`).
pub(super) const MINICPM5: &[&str] = &[
    "\\p{N}{1,3}",
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
];

/// `LLAMA_VOCAB_PRE_TYPE_DEFAULT` (`llama-vocab.cpp:587-595`): a file
/// with no `tokenizer.ggml.pre`, or `"default"`. Not the GPT-2 pattern
/// alone: punctuation runs are carved out first, and digit runs are cut
/// again after it.
pub(super) const DEFAULT: &[&str] = &[
    "[\\p{P}\\$\\+<=>\\^~\\|]+",
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
    "\\p{N}+",
    "[0-9][0-9][0-9]",
];
