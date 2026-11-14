//! Tokenizer parity against llama.cpp, on the same GGUF file.
//!
//! # Why this exists
//!
//! The logit half of `frink parity` hands llama.cpp explicit token ids.
//! That is the right call for measuring the *graph* — see the module
//! docs next door — but it means the tokenizer sat outside the only
//! cross-engine oracle this repo has, and a real defect lived there for
//! the life of the project: one hardcoded pre-tokenizer regex served
//! every BPE checkpoint, so every run of four or more digits and every
//! run of two or more whitespace characters tokenized differently from
//! llama.cpp on Llama-3.x, Qwen, DeepSeek and SmolLM. The BPE tests were
//! round-trip only (encode then decode, which is invariant to *where*
//! the splits fall), so nothing failed.
//!
//! So the tokenizer gets its own oracle, with the same shape as the
//! logit one: same file, same library, frink's answer against
//! llama.cpp's answer, and a first-divergence report detailed enough to
//! debug from.
//!
//! # What is compared, and what is deliberately not
//!
//! * **Ids for raw text**, `add_special = false` on both sides. This is
//!   the pre-tokenizer + merge behaviour, which is where the defects are.
//! * **The add-BOS decision**, as a flag, compared separately. Baking it
//!   into every sequence would make one policy disagreement misreport
//!   every case in the corpus as a tokenization divergence, hiding
//!   whatever else was wrong.
//! * **Vocab size**, because two different id spaces make every
//!   comparison below it meaningless rather than merely failing.
//!
//! * **Both `parse_special` settings**, every case. They are different
//!   questions -- is `<|im_end|>` in the input the end-of-turn token or
//!   six characters of prose -- and llama.cpp answers both, so frink
//!   has to. The sweep used to dump only `parse_special = true`, on the
//!   grounds that frink always parsed specials and no corpus case
//!   contained one. Both halves of that were the hole: frink's
//!   unconditional parsing WAS the divergence from llama.cpp's default,
//!   and a corpus with no marker in it could not see it. The
//!   `special-markers-as-prose` case and the second run per case exist
//!   so that this class turns the sweep red.

mod corpus;

use anyhow::Context;
use corpus::{Case, CORPUS};
use frink_gguf::ShardedGguf;
use frink_models::tokenizer::{
    should_add_bos_token, GgufBpeTokenizer, GgufPlamo2Tokenizer, GgufSpmTokenizer,
    GgufUnigramTokenizer, GgufWordPieceTokenizer, SpecialTokens,
};
use std::path::Path;
use std::process::Command;

/// Exit code `tools/llama_logits.c` uses for "llama.cpp cannot load this
/// checkpoint", as distinct from "the run failed". Kept in step with
/// `EXIT_MODEL_UNSUPPORTED` there.
const EXIT_MODEL_UNSUPPORTED: i32 = 3;

/// llama.cpp's answer for the whole corpus, from one dumper invocation.
#[derive(Debug)]
struct Reference {
    add_bos: bool,
    add_eos: bool,
    n_vocab: usize,
    /// One entry per corpus case, in corpus order; each holds the ids
    /// for every [`SpecialTokens`] setting, in [`SETTINGS`] order.
    cases: Vec<[Vec<u32>; SETTINGS.len()]>,
}

/// The `parse_special` settings every case is compared under. The
/// dumper writes them in THIS order (`tools/llama_logits.c`, FXTK v2:
/// `false` first), and the reader below indexes by it.
const SETTINGS: [SpecialTokens; 2] = [SpecialTokens::AsText, SpecialTokens::Parse];

/// How a setting is named in the report, as llama.cpp spells it.
fn setting_name(mode: SpecialTokens) -> &'static str {
    match mode {
        SpecialTokens::AsText => "parse_special=false",
        SpecialTokens::Parse => "parse_special=true",
    }
}

/// Where two id sequences first stop agreeing, with enough around it to
/// debug from.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Divergence {
    /// Index of the first token that differs, or of the first token past
    /// the end of the shorter sequence.
    pub index: usize,
    /// Byte offset into the case text, summed over the decoded pieces of
    /// the agreed prefix. Approximate for SentencePiece vocabularies,
    /// whose first piece carries a dummy `▁` that is not in the input.
    pub byte_offset: usize,
    /// `(id, piece)` for a window of tokens ending just past `index`.
    pub llama: Vec<(u32, String)>,
    pub frink: Vec<(u32, String)>,
    pub llama_len: usize,
    pub frink_len: usize,
}

/// One corpus case under one `parse_special` setting.
pub(super) struct CaseOutcome {
    pub name: &'static str,
    pub why: &'static str,
    pub text: &'static str,
    pub mode: SpecialTokens,
    pub n_tokens: usize,
    pub divergence: Option<Divergence>,
}

pub(super) struct Report {
    pub cases: Vec<CaseOutcome>,
    pub bos_llama: bool,
    pub bos_frink: bool,
    pub eos_llama: bool,
    pub vocab_llama: usize,
    pub vocab_frink: usize,
    pub pre: String,
    pub model: String,
}

impl Report {
    pub fn diverged(&self) -> bool {
        self.bos_llama != self.bos_frink
            || self.vocab_llama != self.vocab_frink
            || self.cases.iter().any(|c| c.divergence.is_some())
    }

    fn n_bad(&self) -> usize {
        self.cases.iter().filter(|c| c.divergence.is_some()).count()
    }

    /// Total tokens llama.cpp produced for the corpus. Printed so a
    /// MATCH cannot be read without seeing that work was actually done:
    /// a corpus that tokenized to nothing would also "agree".
    fn n_tokens(&self) -> usize {
        self.cases.iter().map(|c| c.n_tokens).sum()
    }
}

/// Frink's tokenizer for a checkpoint, without its decoder.
///
/// `verify_engine::load_and_tokenize` builds the same three variants but
/// returns a fully loaded `Decoder` with them, which is the wrong trade
/// here: this runs one encode per corpus case and needs no weights at
/// all. It also needs two things that path does not expose — the vocab
/// size, and the decoded piece for a single id — so the two are not the
/// same function with a flag. If a third caller ever wants this shape,
/// this is the one to lift into `frink-models` and have both use.
enum Encoder {
    Bpe(Box<GgufBpeTokenizer>),
    Spm(GgufSpmTokenizer),
    Unigram(GgufUnigramTokenizer),
    /// `tokenizer.ggml.model == "bert"`, i.e. WordPiece — every BGE,
    /// E5, nomic-embed, jina-embed and GTE checkpoint.
    WordPiece(GgufWordPieceTokenizer),
    /// `tokenizer.ggml.model == "plamo2"`, PLaMo-2's suffix-table segmenter.
    Plamo2(Box<GgufPlamo2Tokenizer>),
}

impl Encoder {
    fn from_gguf(file: &ShardedGguf) -> anyhow::Result<Self> {
        Ok(match file.metadata_str("tokenizer.ggml.model") {
            Some("gpt2" | "gemma4") => Encoder::Bpe(Box::new(GgufBpeTokenizer::from_gguf(file)?)),
            Some("llama") => Encoder::Spm(GgufSpmTokenizer::from_gguf(file)?),
            Some("t5") => Encoder::Unigram(GgufUnigramTokenizer::from_gguf(file)?),
            Some("bert") => Encoder::WordPiece(GgufWordPieceTokenizer::from_gguf(file)?),
            Some("plamo2") => Encoder::Plamo2(Box::new(GgufPlamo2Tokenizer::from_gguf(file)?)),
            other => anyhow::bail!(
                "parity does not cover tokenizer {other:?} — it builds gpt2/gemma4 (BPE), \
                 llama (SPM), t5 (unigram), bert (WordPiece) and plamo2 vocabularies only"
            ),
        })
    }

    fn encode(&self, text: &str, mode: SpecialTokens) -> Vec<u32> {
        match self {
            Encoder::Bpe(t) => t.encode(text, mode),
            Encoder::Spm(t) => t.encode(text, mode),
            Encoder::Unigram(t) => t.encode(text, mode),
            Encoder::WordPiece(t) => t.encode(text, mode),
            Encoder::Plamo2(t) => t.encode(text, mode),
        }
    }

    /// The text a single id stands for. Both sides' ids are decoded with
    /// THIS table on purpose: the id space comes from the same GGUF, so
    /// a piece is a property of the file rather than of either engine,
    /// and a disagreement about the table itself would already have
    /// shown up as a vocab-size mismatch above.
    fn piece(&self, id: u32) -> String {
        match self {
            Encoder::Bpe(t) => t.decode(&[id]),
            Encoder::Spm(t) => t.decode(&[id]),
            Encoder::Unigram(t) => t.decode(&[id]),
            Encoder::WordPiece(t) => t.decode(&[id]),
            Encoder::Plamo2(t) => t.decode(&[id]),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Encoder::Bpe(t) => t.vocab_size(),
            Encoder::Spm(t) => t.vocab_size(),
            Encoder::Unigram(t) => t.vocab_size(),
            Encoder::WordPiece(t) => t.vocab_size(),
            Encoder::Plamo2(t) => t.vocab_size(),
        }
    }
}

/// Runs the corpus through both tokenizers and returns the comparison,
/// or `None` when llama.cpp itself cannot load the checkpoint.
///
/// `None` is not a pass and not a failure. A reference that has no
/// answer produces no evidence about frink, and folding that into
/// either verdict would make the oracle lie in one direction or the
/// other — which is the failure mode this whole module exists to fix.
pub(super) fn run(dumper: &Path, model: &Path) -> anyhow::Result<Option<Report>> {
    let file = ShardedGguf::open(model)?;
    let encoder = Encoder::from_gguf(&file)?;
    let pre = file
        .metadata_str("tokenizer.ggml.pre")
        .unwrap_or("(unset)")
        .to_string();

    let Some(reference) = reference_tokenization(dumper, model, CORPUS)? else {
        return Ok(None);
    };
    if reference.cases.len() != CORPUS.len() {
        anyhow::bail!(
            "reference tokenizer returned {} cases for {} sent",
            reference.cases.len(),
            CORPUS.len()
        );
    }

    let encoder = &encoder;
    let cases = CORPUS
        .iter()
        .zip(&reference.cases)
        .flat_map(|(case, llama_runs)| {
            SETTINGS
                .iter()
                .zip(llama_runs)
                .map(move |(&mode, llama_ids)| {
                    let frink_ids = encoder.encode(case.text, mode);
                    CaseOutcome {
                        name: case.name,
                        why: case.why,
                        text: case.text,
                        mode,
                        n_tokens: llama_ids.len(),
                        divergence: first_divergence(llama_ids, &frink_ids, &|id| {
                            encoder.piece(id)
                        }),
                    }
                })
        })
        .collect();

    Ok(Some(Report {
        cases,
        bos_llama: reference.add_bos,
        bos_frink: should_add_bos_token(&file),
        eos_llama: reference.add_eos,
        vocab_llama: reference.n_vocab,
        vocab_frink: encoder.vocab_size(),
        pre,
        model: model.display().to_string(),
    }))
}

/// How many tokens of context to show on either side of a divergence.
const WINDOW: usize = 3;

/// The first index at which the two id sequences differ, or `None` when
/// they are equal. A length difference with an equal prefix diverges at
/// the end of the shorter sequence — silently passing that would be the
/// same class of hole this whole module exists to close.
pub(super) fn first_divergence(
    llama: &[u32],
    frink: &[u32],
    piece: &dyn Fn(u32) -> String,
) -> Option<Divergence> {
    let common = llama
        .iter()
        .zip(frink)
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| llama.len().min(frink.len()));
    if common == llama.len() && common == frink.len() {
        return None;
    }

    let byte_offset = llama[..common].iter().map(|&id| piece(id).len()).sum();
    let lo = common.saturating_sub(WINDOW);
    let hi = common + WINDOW + 1;
    let window = |ids: &[u32]| -> Vec<(u32, String)> {
        ids[lo.min(ids.len())..hi.min(ids.len())]
            .iter()
            .map(|&id| (id, piece(id)))
            .collect()
    };

    Some(Divergence {
        index: common,
        byte_offset,
        llama: window(llama),
        frink: window(frink),
        llama_len: llama.len(),
        frink_len: frink.len(),
    })
}

pub(super) fn print_report(r: &Report) {
    let name = Path::new(&r.model)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| r.model.clone());
    let verdict = if r.diverged() { "DIVERGES" } else { "MATCH" };
    println!(
        "tokenizer {name}: {verdict} ({} cases x {} parse_special settings / {} tokens, pre={}, \
         frink vs llama.cpp)",
        r.cases.len() / SETTINGS.len(),
        SETTINGS.len(),
        r.n_tokens(),
        r.pre
    );
    println!(
        "  vocab  llama {} / frink {}     add_bos  llama {} / frink {}  (llama add_eos {})",
        r.vocab_llama, r.vocab_frink, r.bos_llama, r.bos_frink, r.eos_llama
    );

    if r.vocab_llama != r.vocab_frink {
        println!(
            "  vocab sizes differ: the two id spaces are not the same, so every case below \
             compares numbers that do not mean the same thing."
        );
    }
    if r.bos_llama != r.bos_frink {
        println!(
            "  add-BOS policy differs. Frink's rule is `should_add_bos_token`; llama's is \
             `llama_vocab_get_add_bos`. Every prompt on this checkpoint is off by one token."
        );
    }

    if r.n_bad() == 0 {
        println!(
            "  all {} cases tokenize identically under both parse_special settings (digit \
             runs, multi-space, indents, blank lines, CJK, emoji, contractions, special-token \
             markers as prose).",
            r.cases.len() / SETTINGS.len()
        );
        return;
    }

    println!("  {}/{} case runs diverge:", r.n_bad(), r.cases.len());
    for c in r.cases.iter().filter(|c| c.divergence.is_some()) {
        let d = c.divergence.as_ref().expect("filtered on is_some");
        println!(
            "\n  [{} @ {}] token {} of {} (llama) / {} (frink), byte ~{} of {}",
            c.name,
            setting_name(c.mode),
            d.index,
            d.llama_len,
            d.frink_len,
            d.byte_offset,
            c.text.len()
        );
        println!("      why this case: {}", c.why);
        println!("      input around it: {}", around(c.text, d.byte_offset));
        println!("      llama  {}", render_window(&d.llama, d.index));
        println!("      frink {}", render_window(&d.frink, d.index));
    }
    println!(
        "\n  A divergence here means frink and llama.cpp disagree about the PROMPT, so a \
         logit comparison below is measuring two different inputs. Fix this first."
    );
}

/// A readable slice of the case text centred on `offset`, with the split
/// point marked. Byte offsets are clamped to char boundaries so this can
/// never panic on the CJK and emoji cases.
fn around(text: &str, offset: usize) -> String {
    let mid = floor_boundary(text, offset.min(text.len()));
    let lo = floor_boundary(text, mid.saturating_sub(24));
    let hi = ceil_boundary(text, (mid + 24).min(text.len()));
    format!("{:?} >|< {:?}", &text[lo..mid], &text[mid..hi])
}

fn floor_boundary(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_boundary(s: &str, mut i: usize) -> usize {
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// `id:"piece"` for each token in the window, with the diverging one
/// starred so the eye lands on it without counting.
fn render_window(window: &[(u32, String)], index: usize) -> String {
    let lo = index.saturating_sub(WINDOW);
    window
        .iter()
        .enumerate()
        .map(|(k, (id, piece))| {
            let mark = if lo + k == index { "*" } else { "" };
            format!("{mark}{id}:{piece:?}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Serialises the corpus into the FXTK case file the dumper reads.
///
/// Length-prefixed rather than delimited because the corpus contains
/// newlines, CRLFs and control bytes on purpose — any delimiter it could
/// use is also a thing it has to test.
fn encode_cases(cases: &[Case]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"FXTK");
    out.extend_from_slice(&(cases.len() as u32).to_le_bytes());
    for c in cases {
        out.extend_from_slice(&(c.text.len() as u32).to_le_bytes());
        out.extend_from_slice(c.text.as_bytes());
    }
    out
}

/// Reads the FXTK result file the dumper writes.
fn parse_result(bytes: &[u8]) -> anyhow::Result<Reference> {
    let u32_at = |at: &mut usize| -> anyhow::Result<u32> {
        let end = *at + 4;
        let slice = bytes
            .get(*at..end)
            .context("reference tokenization is truncated")?;
        *at = end;
        Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
    };

    if bytes.len() < 4 || &bytes[..4] != b"FXTK" {
        anyhow::bail!(
            "reference tokenization is not an FXTK file. The dumper at hand is probably older \
             than the --tokenize mode; rebuild it with ./tools/build_llama_logits.sh"
        );
    }
    let mut at = 4usize;
    let version = u32_at(&mut at)?;
    if version != 2 {
        anyhow::bail!(
            "reference tokenization is FXTK v{version}, this build reads v2 (one run per \
             parse_special setting). Rebuild the dumper with ./tools/build_llama_logits.sh"
        );
    }
    let flags = u32_at(&mut at)?;
    let n_vocab = u32_at(&mut at)? as usize;
    let n_cases = u32_at(&mut at)? as usize;

    let mut cases = Vec::with_capacity(n_cases);
    for i in 0..n_cases {
        let mut runs: [Vec<u32>; SETTINGS.len()] = Default::default();
        for run in runs.iter_mut() {
            let n = u32_at(&mut at)? as usize;
            // Every id is four bytes on the wire, so a declared count
            // cannot exceed the bytes left; a torn file says so here
            // rather than in a giant allocation.
            let remaining = bytes.len().saturating_sub(at) / 4;
            if n > remaining {
                anyhow::bail!(
                    "reference tokenization case {i} declares {n} ids with {remaining} left"
                );
            }
            let mut ids = Vec::with_capacity(n);
            for _ in 0..n {
                let raw = u32_at(&mut at)? as i32;
                if raw < 0 {
                    anyhow::bail!(
                        "reference tokenization case {i} holds a negative token id {raw}"
                    );
                }
                ids.push(raw as u32);
            }
            *run = ids;
        }
        cases.push(runs);
    }
    if at != bytes.len() {
        anyhow::bail!(
            "reference tokenization has {} trailing bytes after {n_cases} cases",
            bytes.len() - at
        );
    }

    Ok(Reference {
        add_bos: flags & 1 != 0,
        add_eos: flags & 2 != 0,
        n_vocab,
        cases,
    })
}

/// Runs the dumper's `--tokenize` mode over the whole corpus in ONE
/// invocation. Batched because the per-case cost is microseconds and the
/// per-invocation cost is a model open; a case-per-process loop would
/// make the corpus expensive enough that someone would trim it.
fn reference_tokenization(
    dumper: &Path,
    model: &Path,
    cases: &[Case],
) -> anyhow::Result<Option<Reference>> {
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let in_path = dir.join(format!("frink-parity-cases-{pid}.bin"));
    let out_path = dir.join(format!("frink-parity-toks-{pid}.bin"));

    std::fs::write(&in_path, encode_cases(cases))
        .with_context(|| format!("writing corpus to {}", in_path.display()))?;

    let out = Command::new(dumper)
        .arg("--tokenize")
        .arg(model)
        .arg(&in_path)
        .arg(&out_path)
        .output()
        .context("running the reference tokenizer")?;
    let _ = std::fs::remove_file(&in_path);

    if !out.status.success() {
        let _ = std::fs::remove_file(&out_path);
        if out.status.code() == Some(EXIT_MODEL_UNSUPPORTED) {
            // The installed libllama is older than this checkpoint's
            // architecture, or does not build with it. Not a frink
            // result either way.
            return Ok(None);
        }
        anyhow::bail!(
            "reference tokenizer failed: {}\n(if it says \"failed to load --tokenize\", the \
             dumper predates this mode — rebuild it with ./tools/build_llama_logits.sh)",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .last()
                .unwrap_or("(no stderr)")
        );
    }

    let bytes = std::fs::read(&out_path)
        .with_context(|| format!("reading reference tokenization from {}", out_path.display()))?;
    let _ = std::fs::remove_file(&out_path);
    parse_result(&bytes).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pieces good enough to exercise the report: one character per id.
    fn fake_piece(id: u32) -> String {
        char::from_u32(id).unwrap_or('?').to_string()
    }

    #[test]
    fn identical_sequences_do_not_diverge() {
        let ids = vec![97u32, 98, 99, 100];
        assert_eq!(first_divergence(&ids, &ids, &fake_piece), None);
    }

    /// The test the whole module exists for: feed it a mismatched pair
    /// and confirm it SAYS SO. Sabotaging `first_divergence` to return
    /// `None` must turn this red — a comparison that cannot fail is the
    /// exact hole being closed here, not a new one.
    #[test]
    fn a_mismatched_pair_is_reported_as_a_divergence() {
        // The shape of the original defect: llama.cpp splits "1234567"
        // into 123|456|7 and frink (before the fix) took it whole.
        let llama = vec![9000u32, 123, 456, 7, 46];
        let frink = vec![9000u32, 1234567, 46];
        let d = first_divergence(&llama, &frink, &fake_piece)
            .expect("a digit-run split difference must be reported");
        assert_eq!(d.index, 1, "they agree only on the leading token");
        assert_eq!(d.llama_len, 5);
        assert_eq!(d.frink_len, 3);
        assert_eq!(d.llama.first().map(|(id, _)| *id), Some(9000));
        assert!(d.llama.iter().any(|(id, _)| *id == 123));
        assert!(d.frink.iter().any(|(id, _)| *id == 1234567));
    }

    #[test]
    fn an_equal_prefix_with_a_longer_tail_still_diverges() {
        // The subtle half: everything matched until one side ran out.
        // Comparing only the overlap would call this a pass.
        let llama = vec![97u32, 98, 99];
        let frink = vec![97u32, 98, 99, 100];
        let d = first_divergence(&llama, &frink, &fake_piece)
            .expect("a trailing extra token is a divergence");
        assert_eq!(d.index, 3);
        assert_eq!((d.llama_len, d.frink_len), (3, 4));
    }

    #[test]
    fn the_byte_offset_counts_the_agreed_prefix() {
        // 'a','b','c' are one byte each, so the fourth token starts at 3.
        let llama = vec![97u32, 98, 99, 100];
        let frink = vec![97u32, 98, 99, 101];
        let d = first_divergence(&llama, &frink, &fake_piece).expect("differs at index 3");
        assert_eq!(d.index, 3);
        assert_eq!(d.byte_offset, 3);
    }

    #[test]
    fn an_empty_pair_agrees_and_a_half_empty_pair_does_not() {
        assert_eq!(first_divergence(&[], &[], &fake_piece), None);
        let d = first_divergence(&[], &[97], &fake_piece).expect("empty vs one token differs");
        assert_eq!(d.index, 0);
        assert_eq!(d.byte_offset, 0);
    }

    #[test]
    fn the_window_is_clamped_at_both_ends() {
        // Divergence at index 0 with a long tail: no underflow below the
        // start, and no read past the end of the shorter side.
        let llama: Vec<u32> = (97..107).collect();
        let frink = vec![200u32];
        let d = first_divergence(&llama, &frink, &fake_piece).expect("differs at 0");
        assert_eq!(d.index, 0);
        assert_eq!(d.llama.len(), WINDOW + 1);
        assert_eq!(d.frink.len(), 1);
    }

    #[test]
    fn the_case_file_is_length_prefixed_so_newlines_survive() {
        let cases = [
            Case {
                name: "a",
                why: "",
                text: "hi\n\n",
            },
            Case {
                name: "b",
                why: "",
                text: "",
            },
        ];
        let blob = encode_cases(&cases);
        assert_eq!(&blob[..4], b"FXTK");
        assert_eq!(u32::from_le_bytes(blob[4..8].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(blob[8..12].try_into().unwrap()), 4);
        assert_eq!(&blob[12..16], b"hi\n\n");
        assert_eq!(u32::from_le_bytes(blob[16..20].try_into().unwrap()), 0);
        assert_eq!(blob.len(), 20);
    }

    /// A v2 result: every case is two runs, `parse_special` false then
    /// true.
    fn result_blob(flags: u32, n_vocab: u32, cases: &[[&[i32]; 2]]) -> Vec<u8> {
        let mut b = Vec::from(*b"FXTK");
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&flags.to_le_bytes());
        b.extend_from_slice(&n_vocab.to_le_bytes());
        b.extend_from_slice(&(cases.len() as u32).to_le_bytes());
        for runs in cases {
            for c in runs {
                b.extend_from_slice(&(c.len() as u32).to_le_bytes());
                for id in *c {
                    b.extend_from_slice(&id.to_le_bytes());
                }
            }
        }
        b
    }

    #[test]
    fn a_result_file_round_trips_flags_vocab_and_both_runs_of_every_case() {
        // The prose case's shape: `<s>` is three ids as text and one
        // when parsed, so the two runs of one case differ in length.
        let blob = result_blob(0b11, 128_256, &[[&[1, 2, 3], &[9]], [&[], &[]]]);
        let r = parse_result(&blob).expect("well-formed result must parse");
        assert!(r.add_bos && r.add_eos);
        assert_eq!(r.n_vocab, 128_256);
        assert_eq!(r.cases, vec![[vec![1u32, 2, 3], vec![9]], [vec![], vec![]]]);

        let none = parse_result(&result_blob(0, 32, &[[&[7], &[7]]])).unwrap();
        assert!(!none.add_bos && !none.add_eos);
    }

    /// A dumper built from the v1 source writes one run per case. Read
    /// as v2 that would pair case N's ids with case N+1's and report
    /// nonsense divergences, so the version is refused by name.
    #[test]
    fn a_version_one_result_is_refused_and_names_the_rebuild() {
        let mut blob = result_blob(0, 32, &[[&[1], &[1]]]);
        blob[4..8].copy_from_slice(&1u32.to_le_bytes());
        let err = parse_result(&blob).unwrap_err().to_string();
        assert!(
            err.contains("FXTK v1") && err.contains("build_llama_logits.sh"),
            "got {err}"
        );
    }

    #[test]
    fn a_malformed_result_is_refused_rather_than_half_read() {
        // An old dumper handed "--tokenize" as a model path exits before
        // writing anything FXTK-shaped; the error has to name the fix.
        let err = parse_result(b"not fxtk at all").unwrap_err().to_string();
        assert!(err.contains("build_llama_logits.sh"), "got {err}");

        // Truncated mid-case: a short read must not become a short
        // sequence that then reports a divergence at the wrong place.
        let mut blob = result_blob(0, 32, &[[&[1, 2, 3], &[1, 2, 3]]]);
        blob.truncate(blob.len() - 5);
        assert!(parse_result(&blob).is_err());

        // Trailing bytes mean the two sides disagree about the layout.
        let mut blob = result_blob(0, 32, &[[&[1], &[1]]]);
        blob.push(0);
        assert!(parse_result(&blob).is_err());

        // A negative id would silently become a huge u32 index.
        assert!(parse_result(&result_blob(0, 32, &[[&[-3], &[3]]])).is_err());

        // A count larger than the bytes left is a torn file, refused
        // before any allocation sized by it.
        let mut blob = result_blob(0, 32, &[[&[1], &[1]]]);
        let at = blob.len() - 8;
        blob[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        let err = parse_result(&blob).unwrap_err().to_string();
        assert!(err.contains("declares"), "got {err}");
    }

    #[test]
    fn the_context_slice_never_splits_a_character() {
        // Offsets landing inside a multi-byte character must clamp, not
        // panic: the CJK and emoji cases are the ones most likely to
        // diverge, so this path runs exactly when it matters.
        for case in CORPUS {
            for off in 0..=case.text.len() {
                let _ = around(case.text, off);
            }
        }
        assert!(around("日本", 1).contains(">|<"));
    }

    /// Paths are anchored on the manifest dir, not on the cwd: cargo
    /// runs a test binary from its own package directory, so
    /// `target/llama_logits` from here would look inside
    /// `crates/frink-cli/`.
    const REPO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

    /// The two sides of the exit-code contract live in different
    /// languages and no compiler links them, so the C source is read and
    /// checked. If they drift, "llama.cpp cannot load this checkpoint"
    /// starts being reported as a frink failure, or worse, a real
    /// failure starts being skipped.
    #[test]
    fn the_unsupported_exit_code_matches_the_c_side() {
        let c = std::fs::read_to_string(std::path::Path::new(REPO).join("tools/llama_logits.c"))
            .expect("the dumper source is tracked next to this crate");
        assert!(
            c.contains(&format!(
                "#define EXIT_MODEL_UNSUPPORTED {EXIT_MODEL_UNSUPPORTED}"
            )),
            "tools/llama_logits.c no longer defines EXIT_MODEL_UNSUPPORTED as \
             {EXIT_MODEL_UNSUPPORTED}"
        );
        // And it is actually returned, rather than only defined.
        assert!(
            c.matches("return EXIT_MODEL_UNSUPPORTED;").count() >= 2,
            "both dumper modes must report an unloadable checkpoint with that code"
        );
    }

    /// One checkpoint per pre-tokenizer arm that `pretokenize::pre_split`
    /// distinguishes, because the defect this closes was *per arm*: the
    /// engine looked correct on whichever family happened to be tested.
    /// Missing files are skipped, so a partial `models/` still checks
    /// what it has.
    const SWEEP: &[&str] = &[
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf",         // llama-bpe
        "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",         // qwen2
        "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf", // deepseek / qwen2
        "olmoe-1b-7b-0924-q4_0.gguf",                // olmo (no trailing \s+ clause)
        "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",        // SPM
        "gemma-2-2b-it-Q4_K_M.gguf",                 // SPM, gemma vocab
        "gemma-4-E2B-it-Q4_K_M.gguf",                // gemma4 BPE, ▁-escaped merges
        "Phi-4-mini-instruct-Q4_K_M.gguf",           // gpt-4o / cl100k-shaped
        "Yi-1.5-6B-Chat-Q4_K_M.gguf",
        "Mistral-7B-Instruct-v0.2-Q4_K_M.gguf",
    ];

    /// End-to-end against the real reference. Ignored because it needs
    /// two things the workspace cannot build: `target/llama_logits`
    /// (`./tools/build_llama_logits.sh`, links libllama) and real
    /// checkpoints under `models/`.
    ///
    /// # This is RED as written, on purpose
    ///
    /// The first run of this oracle, 2026-08-31, found four open defects
    /// in `frink-models`. They are named here rather than allow-listed,
    /// because an oracle with an exception list is the hole it replaced:
    ///
    /// 1. `pre = "deepseek-r1-qwen"` has no arm in
    ///    `pretokenize_regex_for`, so it falls through to the GPT-2
    ///    default. llama.cpp maps it to `LLAMA_VOCAB_PRE_TYPE_QWEN2`.
    ///    7 of 19 cases diverge on DeepSeek-R1-Distill-Qwen-1.5B.
    /// 2. `pre = "gpt-4o"` is mapped onto the qwen2 arm, but upstream
    ///    `LLAMA_VOCAB_PRE_TYPE_GPT4O` is its own pattern with
    ///    `\p{N}{1,3}` grouping, case-split letter runs and `[\r\n/]*`.
    ///    7 of 19 diverge on Phi-4-mini; every multi-digit number is
    ///    tokenized one digit at a time.
    /// 3. The `olmo` arm deliberately has no trailing `\s+` clause, and
    ///    `encode_text_run` uses `find_iter(..).flat_map(..)`, which
    ///    DROPS the text between matches. llama.cpp's
    ///    `unicode_regex_split_stl` emits each unmatched gap as its own
    ///    chunk instead. Frink loses input bytes on OLMoE: tabs, NBSP,
    ///    form feeds and interior newlines vanish from the prompt.
    /// 4. `should_add_bos_token` misses the `llama-bpe` group, so every
    ///    raw Llama-3.x completion prompt is short one
    ///    `<|begin_of_text|>` relative to llama.cpp.
    ///
    /// All four live in `crates/frink-models/src/tokenizer.rs`. This
    /// test goes green when they are fixed, and must not be weakened to
    /// get there.
    #[test]
    #[ignore = "needs ./tools/build_llama_logits.sh and checkpoints under models/"]
    fn frink_and_llama_cpp_tokenize_the_corpus_identically() {
        let dumper = std::path::PathBuf::from(REPO).join("target/llama_logits");
        assert!(
            dumper.exists(),
            "run ./tools/build_llama_logits.sh first ({} is missing)",
            dumper.display()
        );

        let mut checked = 0usize;
        let mut bad_ids: Vec<String> = Vec::new();
        let mut bad_policy: Vec<String> = Vec::new();
        for name in SWEEP {
            let model = std::path::PathBuf::from(REPO).join("models").join(name);
            if !model.exists() {
                println!("skip {name}: not downloaded");
                continue;
            }
            let Some(report) = run(&dumper, &model).expect("tokenizer parity must run") else {
                println!("skip {name}: the installed libllama cannot load it");
                continue;
            };
            checked += 1;
            print_report(&report);
            println!();
            if report.n_bad() > 0 || report.vocab_llama != report.vocab_frink {
                bad_ids.push(format!("{name} ({} cases)", report.n_bad()));
            }
            if report.bos_llama != report.bos_frink {
                bad_policy.push(format!(
                    "{name} (pre={}, llama {} / frink {})",
                    report.pre, report.bos_llama, report.bos_frink
                ));
            }
        }
        assert!(
            checked > 0,
            "no checkpoint in SWEEP is present under models/"
        );

        assert!(
            bad_ids.is_empty(),
            "these checkpoints tokenize the corpus differently from llama.cpp: {bad_ids:?}"
        );
        // Reported separately from the ids because the cause is
        // elsewhere: `frink_models::tokenizer::should_add_bos_token`
        // implements llama.cpp's BPE add_bos defaults, and upstream
        // (`src/llama-vocab.cpp`, the `tokenizer_pre == "llama-bpe"`
        // arm) sets add_bos for llama-bpe/falcon3/falcon-h1/pixtral/
        // midm-2.0/lfm2/jina-v5-nano as well as tekken and chameleon.
        // That file is not this module's to change.
        assert!(
            bad_policy.is_empty(),
            "these checkpoints disagree about add-BOS, so every raw completion prompt is off \
             by one token — see `should_add_bos_token` against llama.cpp's per-pre arms: \
             {bad_policy:?}"
        );
    }
}
