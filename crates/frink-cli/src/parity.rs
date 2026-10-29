//! Answer parity against llama.cpp, measured on the distribution rather
//! than on sampled text.
//!
//! Why this exists: `frink verify` compares frink-CPU against
//! frink-Metal, so it can prove the two frink backends agree with each
//! other while both disagree with the reference implementation. The
//! quality gate in `docs/plans/llama-cpp-parity-push.md` says a row is
//! only closed when "answers match llama", and nothing measured that.
//!
//! Why not greedy text: it was tried. With tokenization matched by hand
//! (6 tokens on both sides) frink and llama.cpp still diverged after
//! about three tokens on TinyLlama Q8_0, a model with no exotic graph
//! features. Greedy text is a chain of argmaxes, so ordinary
//! last-bit numeric drift flips one token and every token after it is
//! then conditioned on different input. A text diff cannot separate "the
//! graph is wrong" from "two near-tied logits swapped", and those are
//! the only two hypotheses worth telling apart.
//!
//! So this compares the FIRST-TOKEN distribution at the last prompt
//! position: one forward pass per engine, no sampling, no feedback. The
//! same token ids are fed to both sides, which removes the tokenizer
//! from the experiment — frink tokenizes, and llama.cpp is handed the
//! resulting ids.
//!
//! Removing the tokenizer from THIS experiment is right, and for years
//! it also meant nothing tested the tokenizer at all: a single hardcoded
//! pre-tokenizer regex mistokenized every digit run of four or more and
//! every whitespace run of two or more, on Llama-3.x, Qwen, DeepSeek and
//! SmolLM, and the repo's only cross-engine oracle was structurally
//! unable to see it. So `parity` now runs a SECOND comparison first, in
//! [`tokenize`]: same file, same library, frink's token ids against
//! llama.cpp's for a corpus built out of the inputs the pre-tokenizers
//! actually disagree on. It runs before the logit comparison because a
//! tokenizer divergence means the distribution comparison below is
//! measuring two different prompts.
//!
//! The reference side is a small C program (`tools/llama_logits.c`)
//! linked against the installed `libllama`, in the same spirit as the
//! IQ-tier goldens, which link ggml's own `ggml-quants.c` rather than
//! re-reading the format spec. A reference you re-implemented is not a
//! reference. Build it with `tools/build_llama_logits.sh`.
//!
//! `--dumper` may be given MORE THAN ONCE, each pointing at a dumper
//! built against a different libllama. That is not a convenience: it is
//! what turns the WRONG verdict from a fitted constant into a
//! measurement this run makes for itself. See [`calibration`].

mod calibration;
mod decode;
mod dump;
mod metrics;
mod quant;
mod tokenize;

use anyhow::Context;
use calibration::{Band, WrongLine, KL_WRONG};
use quant::DominantQuant;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Where the reference dumper is looked for, in order. `target/` is
/// where `tools/build_llama_logits.sh` puts it; the second entry keeps
/// working for anyone who built it by hand before it moved into the
/// tracked tree.
const DUMPER_CANDIDATES: &[&str] = &["target/llama_logits", ".local-scripts/llama_logits"];

/// Environment override, for a dumper built somewhere else entirely.
const DUMPER_ENV: &str = "FRINK_LLAMA_LOGITS";

/// Prompt held fixed so runs are comparable across models and sessions.
const PROMPT: &str = "The capital of France is";

/// Below this KL the two engines are doing the same arithmetic to within
/// f32 accumulation-order noise. Chosen as the scale at which a
/// reordered reduction over a few thousand terms lands: it is a noise
/// floor, not a quality target.
///
/// A statement about the PRIMARY reference, unlike the WRONG line: the
/// report's numbers are that reference's, and "indistinguishable from
/// the reference you named" is what MATCH has always meant.
const KL_NOISE: f64 = 1e-3;

/// The verdict ladder, checked at COMPILE time.
///
/// One rung is a constant now — [`KL_WRONG`], the floor under every
/// calibrated line — and the other is measured per checkpoint, so the
/// only ordering left to enforce is that MATCH is tighter than the
/// tightest possible WRONG. The three constants this assert used to
/// hold in order (`KL_REFERENCE_BUILD_SPREAD_KQUANT`,
/// `KL_WRONG_Q8K_DOTTED` and its 1.1 margin) are DELETED: a rule that
/// measures its own line has no thresholds left to keep in step. See
/// [`calibration`] for why keeping them was not an option.
const _: () = {
    assert!(
        KL_NOISE < KL_WRONG,
        "MATCH must be tighter than the tightest WRONG line"
    );
};

pub struct ParityArgs {
    pub model: String,
    pub prompt: Option<String>,
    pub prompt_tokens: Option<usize>,
    pub top_k: usize,
    /// Paths to compiled reference dumpers. The first is the PRIMARY —
    /// the one every printed number is about. Any further ones exist to
    /// measure how much llama.cpp disagrees with itself on this
    /// checkpoint, which is what the WRONG line is made of.
    pub dumper: Vec<String>,
    /// Prefix to write every compared logit vector under. See [`dump`].
    pub dump_logits: Option<String>,
    /// Token ids to compare on instead of tokenizing a prompt. The
    /// tokenizer half is skipped, said so, and not counted as a pass.
    pub tokens: Option<Vec<u32>>,
    /// Greedy decode steps to compare after the first token; 0 skips
    /// the decode half. See [`decode`].
    pub decode_steps: usize,
}

pub fn run(args: ParityArgs) -> anyhow::Result<()> {
    let path = crate::pull::resolve_model_path(&args.model)?;

    // Resolved before anything expensive runs: a missing dumper is the
    // most common way this command fails, and finding that out after a
    // multi-second model load helps nobody.
    let dumpers = dumper_paths(&args.dumper)?;

    // The tokenizer comparison goes first. It is vocab-only on both
    // sides, so it costs a fraction of the prefill below, and a
    // divergence here means the logit numbers underneath it were
    // computed from a prompt the two engines do not even agree on.
    // Only the primary reference tokenizes: the tokenizer half is not
    // the half that needs calibrating.
    let tokens_report = match args.tokens {
        Some(_) => None,
        None => tokenize::run(&dumpers[0], Path::new(&path))?,
    };
    match &tokens_report {
        Some(r) => tokenize::print_report(r),
        None if args.tokens.is_some() => {
            println!("tokenizer: SKIPPED — --tokens gave the ids, so there is no text to tokenize.")
        }
        None => println!(
            "tokenizer: SKIPPED — the installed libllama cannot load this checkpoint, so it has \
             no tokenization to compare against. The logit comparison below will fail for the \
             same reason."
        ),
    }
    println!();

    let prompt = args.prompt.clone().unwrap_or_else(|| PROMPT.to_string());

    let (tokens, frink_logits) = match &args.tokens {
        Some(ids) if ids.is_empty() => anyhow::bail!("--tokens is empty"),
        Some(ids) => {
            let ids_usize: Vec<usize> = ids.iter().map(|&t| t as usize).collect();
            let mut rows =
                crate::verify_engine::teacher_forced_logits(Path::new(&path), &ids_usize, &[])
                    .context("frink prefill")?;
            (ids.clone(), rows.swap_remove(0))
        }
        None => crate::verify_engine::prefill_logits(
            Path::new(&path),
            &prompt,
            frink_models::tokenizer::SpecialTokens::Parse,
            args.prompt_tokens,
        )
        .context("frink prefill")?,
    };

    // The reference is pinned to llama.cpp's CPU path (n_gpu_layers = 0),
    // because frink's CPU path is the one cross-validated against NumPy.
    // Say which side frink ran on rather than leaving it to the
    // environment: a Metal-vs-llama-CPU verdict has two possible causes
    // and must not be read as one.
    let backend = frink_core::weight_matrix::active_backend();
    if backend != frink_core::kernel_registry::Backend::Cpu {
        eprintln!(
            "parity: frink ran on {}, the reference on llama.cpp CPU — a non-MATCH verdict \
             here could be either engine or the backend difference. Run with FRINK_METAL=0 \
             FRINK_CUDA=0 to isolate the engines.",
            backend.as_str()
        );
    }

    let references = collect_references(&dumpers, &path, &tokens, frink_logits.len())?;

    // Dumped BEFORE the verdict is computed, so a run that ends in the
    // `bail!` below still leaves the evidence behind. The whole reason
    // to dump is to investigate a bad verdict; losing the vectors
    // exactly when the verdict is bad would defeat it.
    let ref_logits: Vec<&[f32]> = references.iter().map(|r| r.logits.as_slice()).collect();
    if let Some(prefix) = args.dump_logits.as_deref() {
        for p in dump::write(prefix, &tokens, &ref_logits, &frink_logits)? {
            println!("wrote {}", p.display());
        }
        println!();
    }

    // ONE value, read by the DRIFT message and by the uncalibrated
    // fallback of the WRONG line. Deriving it twice was #109.
    let gguf = frink_gguf::GgufFile::open(&path).ok();
    let quant = DominantQuant::of(gguf.as_ref().map_or(&[][..], |f| &f.tensors));

    let band = Band::measure(&ref_logits, &frink_logits, &quant);
    let report = compare(ref_logits[0], &frink_logits, args.top_k, &band);
    print_report(
        &Run {
            model: &args.model,
            n_tokens: tokens.len(),
            top_k: args.top_k,
            backend: backend.as_str(),
        },
        &quant,
        &references,
        &band,
        &report,
    );

    // Both halves are reported before either is allowed to abort the
    // command: a run that printed the tokenizer divergence and then
    // exited would hide the logit numbers that say whether the graph is
    // also wrong, and those are two separate pieces of work.
    let mut failures: Vec<String> = Vec::new();
    if tokens_report.as_ref().is_some_and(|r| r.diverged()) {
        failures.push("frink and llama.cpp tokenize the same text differently".to_string());
    }
    if report.verdict == Verdict::Wrong {
        failures.push(format!(
            "frink disagrees with llama.cpp beyond numeric noise (KL {:.3e} nats to its \
             nearest reference)",
            band.nearest()
        ));
    }
    if args.decode_steps > 0 {
        println!();
        let worst = decode::run(&dumpers[0], Path::new(&path), &tokens, args.decode_steps)?;
        if worst == decode::StepVerdict::Wrong {
            failures.push("frink's greedy decode disagrees with llama.cpp's".to_string());
        }
    }
    if !failures.is_empty() {
        anyhow::bail!("{}", failures.join("; "));
    }
    Ok(())
}

/// Runs every dumper on the same token ids and keeps the ones that
/// answered with a comparable vector.
///
/// The PRIMARY reference is load-bearing and its failures are fatal. A
/// SECONDARY one is evidence, and evidence that did not arrive leaves a
/// narrower experiment rather than no experiment — but it is said out
/// loud, because a run that silently lost its calibration would print a
/// WRONG line derived from one reference while looking like a
/// two-reference run.
fn collect_references(
    dumpers: &[PathBuf],
    model: &str,
    tokens: &[u32],
    n_vocab: usize,
) -> anyhow::Result<Vec<Reference>> {
    let mut out: Vec<Reference> = Vec::new();
    for (i, dumper) in dumpers.iter().enumerate() {
        let primary = i == 0;
        let reference = match reference_logits(dumper, model, tokens, i) {
            Ok(r) => r,
            Err(e) if primary => return Err(e),
            Err(e) => {
                eprintln!(
                    "parity: reference [{i}] {} produced nothing ({e:#}); the WRONG line will be \
                     measured without it",
                    dumper.display()
                );
                continue;
            }
        };
        if reference.logits.len() != n_vocab {
            if primary {
                // Not a tolerance question: the two engines disagree
                // about how many tokens the model can emit, which is a
                // loader bug on one side and makes every other number
                // here meaningless.
                anyhow::bail!(
                    "vocab size disagrees: llama.cpp {} vs frink {n_vocab} — the logit vectors \
                     are not comparable, fix the loader before reading any metric",
                    reference.logits.len()
                );
            }
            eprintln!(
                "parity: reference [{i}] reports {} vocabulary entries against frink's \
                 {n_vocab}; dropped from the WRONG line rather than compared to a different \
                 distribution",
                reference.logits.len()
            );
            continue;
        }
        out.push(reference);
    }
    Ok(out)
}

#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    /// Same distribution to within accumulation-order noise.
    Match,
    /// Same top-1, but the distributions drift more than noise explains.
    Drift,
    /// Top-1 differs, and the reference's own top-2 margin is so small
    /// that noise of the observed size is enough to swap them. This is
    /// NOT evidence of a wrong graph and must not be reported as one.
    TieFlip,
    /// Different distributions.
    Wrong,
}

struct Report {
    verdict: Verdict,
    kl_ref_frink: f64,
    kl_frink_ref: f64,
    total_variation: f64,
    max_prob_delta: f64,
    top1_ref: usize,
    top1_frink: usize,
    /// Where the reference's top-1 sits in frink's ordering (0 = top).
    ref_top1_rank_in_frink: usize,
    /// p1 - p2 in the reference distribution: how close the call was.
    ref_top2_margin: f64,
    /// The same call in LOGIT space, on both sides, and how many
    /// representable f32 values separate the reference's two
    /// candidates.
    ///
    /// A TIE-FLIP verdict is a claim about magnitude — "the two
    /// candidates are closer than the arithmetic can resolve" — and
    /// probability margin does not settle it, because softmax over a
    /// 128k vocabulary turns a wide logit gap into a small probability
    /// gap whenever the distribution is flat. The gap in ulps is the
    /// claim's own units: single digits is a tie, millions is a
    /// difference that happens to sit near the argmax
    /// ([#103](https://github.com/antonellof/frink/issues/103)).
    ref_top2_logit_gap: f32,
    ref_top2_logit_gap_ulps: i64,
    /// The same gap as frink sees it, signed by the reference's
    /// ordering: negative means frink ranks them the other way round.
    frink_gap_on_ref_pair: f32,
    topk_overlap: usize,
}

/// Scores frink against the PRIMARY reference, and takes the WRONG
/// rung from `band`.
///
/// The split is deliberate. MATCH and DRIFT are statements about the
/// reference the report names — "indistinguishable from it", "differs
/// from it by more than summation order" — and they move with it, which
/// is correct: they describe a comparison, not a defect. WRONG is a
/// claim about frink itself, so it is the one rung that must not
/// change with which libllama happens to be installed (#102), and
/// `band` is what makes it not.
fn compare(ref_logits: &[f32], frink_logits: &[f32], k: usize, band: &Band) -> Report {
    let p = metrics::softmax(ref_logits);
    let q = metrics::softmax(frink_logits);

    // The primary's KL comes from the band rather than being recomputed
    // here: one number, one owner. Two evaluations would be two places
    // for the guarded-zero convention to drift apart.
    let kl_pq = band.kl_to_frink(0);
    let kl_qp = metrics::kl(&q, &p);
    let (tv, max_delta) = metrics::total_variation(&p, &q);

    let ref_order = metrics::order_desc(&p);
    let fx_order = metrics::order_desc(&q);
    let top1_ref = ref_order[0];
    let top1_frink = fx_order[0];
    let ref_top1_rank_in_frink = fx_order.iter().position(|&i| i == top1_ref).unwrap_or(0);
    let (ref_top2_margin, ref_top2_logit_gap, ref_top2_logit_gap_ulps, frink_gap_on_ref_pair) =
        if ref_order.len() > 1 {
            let (i1, i2) = (ref_order[0], ref_order[1]);
            (
                p[i1] - p[i2],
                ref_logits[i1] - ref_logits[i2],
                metrics::ulps_between(ref_logits[i1], ref_logits[i2]),
                // Same pair, frink's numbers, keeping the reference's
                // order: the sign is the flip itself.
                frink_logits[i1] - frink_logits[i2],
            )
        } else {
            (1.0, 0.0, 0, 0.0)
        };

    let k = k.min(ref_order.len());
    let ref_top: std::collections::HashSet<usize> = ref_order[..k].iter().copied().collect();
    let topk_overlap = fx_order[..k].iter().filter(|i| ref_top.contains(i)).count();

    // ONE predicate, read by both branches below. They were two
    // spellings of `kl_pq < kl_wrong` before, which is the shape this
    // repo keeps paying for.
    let outside = band.frink_is_outside();
    let verdict = if top1_ref == top1_frink {
        if kl_pq < KL_NOISE {
            Verdict::Match
        } else if !outside {
            Verdict::Drift
        } else {
            Verdict::Wrong
        }
    } else if !outside && ref_top2_margin <= max_delta {
        // The two engines put nearly the same mass on both candidates and
        // the reference itself could barely separate them. Calling this a
        // failure would make the instrument cry wolf on exactly the case
        // that motivated building it.
        Verdict::TieFlip
    } else {
        Verdict::Wrong
    };

    Report {
        verdict,
        kl_ref_frink: kl_pq,
        kl_frink_ref: kl_qp,
        total_variation: tv,
        max_prob_delta: max_delta,
        top1_ref,
        top1_frink,
        ref_top1_rank_in_frink,
        ref_top2_margin,
        ref_top2_logit_gap,
        ref_top2_logit_gap_ulps,
        frink_gap_on_ref_pair,
        topk_overlap,
    }
}

/// What the run WAS, as opposed to what it found. Grouped because a
/// print function taking eight positional arguments is one careless
/// edit away from two of them being swapped, and `&str` next to `&str`
/// swaps silently.
struct Run<'a> {
    model: &'a str,
    n_tokens: usize,
    top_k: usize,
    backend: &'a str,
}

fn print_report(
    run: &Run<'_>,
    quant: &DominantQuant,
    references: &[Reference],
    band: &Band,
    r: &Report,
) {
    let Run {
        model,
        n_tokens,
        top_k: k,
        backend,
    } = *run;
    let name = Path::new(model)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| model.to_string());
    let verdict = match r.verdict {
        Verdict::Match => "MATCH",
        Verdict::Drift => "DRIFT",
        Verdict::TieFlip => "TIE-FLIP",
        Verdict::Wrong => "WRONG",
    };
    println!(
        "parity {name}: {verdict} ({n_tokens}-token prompt, first-token distribution, \
         frink {backend} vs llama.cpp cpu)"
    );
    // Named on every row, not only when it surprises. Two libllama
    // builds gave the same checkpoint DRIFT and WRONG (#102), and the
    // difference between the runs was invisible in this report.
    for (i, reference) in references.iter().enumerate() {
        let who = match reference.libllama.as_deref() {
            Some(p) => p.to_string(),
            None => "(this dumper predates the `libllama` line — rebuild with \
                     tools/build_llama_logits.sh)"
                .to_string(),
        };
        println!(
            "  reference [{i}]     {who}   KL(llama||frink) {:.3e}",
            band.kl_to_frink(i)
        );
    }
    print_wrong_line(quant, references, band);
    println!(
        "  KL(llama||frink) {:.3e} nats   KL(frink||llama) {:.3e} nats   [reference 0]",
        r.kl_ref_frink, r.kl_frink_ref
    );
    println!(
        "  total variation   {:.3e}        max |delta p|     {:.3e}",
        r.total_variation, r.max_prob_delta
    );
    println!(
        "  top-1  llama {} / frink {}   (llama's top-1 is rank {} for frink)",
        r.top1_ref, r.top1_frink, r.ref_top1_rank_in_frink
    );
    println!(
        "  top-{k} overlap    {}/{k}          llama top-2 margin {:.3e}",
        r.topk_overlap, r.ref_top2_margin
    );
    // Printed for every verdict, not only TIE-FLIP: how close the
    // argmax call was is what says whether the NEXT prompt would flip
    // too, and a row that did not flip is the evidence that a row which
    // did was one prompt's luck.
    println!(
        "  top-1/top-2 gap   {:+.3e} logits ({} ulps)   frink on the same pair {:+.3e}",
        r.ref_top2_logit_gap, r.ref_top2_logit_gap_ulps, r.frink_gap_on_ref_pair
    );
    match r.verdict {
        Verdict::Match => println!("  same distribution to within f32 accumulation-order noise."),
        // `q8k_dotted` is the SAME predicate on the SAME value the
        // uncalibrated WRONG line reads, so the explanation and the
        // line are never about different tensors (#109).
        Verdict::Drift => match quant.label().filter(|_| quant.q8k_dotted()) {
            // Expected, with a named cause. Saying "go run a per-layer
            // divergence" here would send the reader after a bug that
            // is not there.
            Some(q) => println!(
                "  same token. The distributions differ by more than summation order, and for \
                 {q} that is EXPECTED: llama.cpp declares `vec_dot_type = Q8_K` for it, so it \
                 quantizes the ACTIVATION to 8 bits and accumulates in integers, while frink \
                 keeps activations in f32. Different arithmetic, with frink on the more \
                 precise side. See docs/plans/llama-cpp-gap-inventory.md §10."
            ),
            None => println!(
                "  same token, but the distributions differ by more than accumulation order \
                 explains — worth a per-layer divergence run before trusting this row."
            ),
        },
        Verdict::TieFlip => {
            println!(
                "  top-1 differs, but llama's own top-2 margin ({:.3e}) is under the observed \
                 per-token noise ({:.3e}): a tie swapped, not a wrong graph.",
                r.ref_top2_margin, r.max_prob_delta
            );
            // The magnitude, in the units the claim is made in. Without
            // it "tie" is an assertion; with it a reader can check
            // whether the two candidates really are inseparable or
            // whether a real difference merely landed near the argmax.
            println!(
                "  the two candidates are {} ulps apart in llama's logits ({:+.3e} absolute); \
                 frink puts the same pair {:+.3e} apart, so the flip is a sign change of a gap \
                 this size — not a redistribution.",
                r.ref_top2_logit_gap_ulps, r.ref_top2_logit_gap, r.frink_gap_on_ref_pair
            );
        }
        Verdict::Wrong => println!(
            "  the graphs disagree. This is not sampling noise and not a tie: something in \
             the forward pass differs."
        ),
    }
}

/// The WRONG line, and where it came from.
///
/// Printed on EVERY row, including the ones that are nowhere near it.
/// A verdict whose line is invisible cannot be audited, and the line is
/// no longer a constant a reader could look up.
fn print_wrong_line(quant: &DominantQuant, references: &[Reference], band: &Band) {
    match band.line() {
        WrongLine::Calibrated { spread, line } => {
            let (a, b) = spread.between;
            println!(
                "  WRONG line        {line:.3e}  measured: references [{a}] and [{b}] disagree \
                 with EACH OTHER by {:.3e} on this checkpoint",
                spread.kl
            );
            println!(
                "                    frink's nearest reference is {:.3e}, {:.0}% of the line",
                band.nearest(),
                100.0 * band.nearest() / line
            );
        }
        WrongLine::Absolute(line) => println!(
            "  WRONG line        {line:.3e}  absolute: llama.cpp's own build-to-build spread on \
             {} is 0 to 4.6e-4, two orders under it",
            quant.label().unwrap_or("this checkpoint")
        ),
        WrongLine::Uncalibrated => {
            println!(
                "  WRONG line        NONE — {} is dotted against Q8_K activations, and two \
                 builds of llama.cpp disagree with EACH OTHER by up to 3.5e-2 there (#111), so \
                 no constant can mean `the graphs disagree`.",
                quant.label().unwrap_or("this checkpoint")
            );
            println!(
                "                    Pass --dumper a second time, built against another \
                 libllama, to measure this checkpoint's own line. ({} reference in this run.)",
                references.len()
            );
        }
    }
}

/// Resolves every `--dumper`, or falls back to the single one the
/// environment and the tree offer.
fn dumper_paths(explicit: &[String]) -> anyhow::Result<Vec<PathBuf>> {
    if !explicit.is_empty() {
        return explicit
            .iter()
            .map(|e| {
                let p = PathBuf::from(e);
                // An explicit path that does not exist is a typo, not a
                // reason to silently fall back to some other binary and
                // report its answer as the reference.
                if p.exists() {
                    Ok(p)
                } else {
                    anyhow::bail!("--dumper {} does not exist", p.display())
                }
            })
            .collect();
    }
    if let Some(e) = std::env::var_os(DUMPER_ENV) {
        let p = PathBuf::from(e);
        if p.exists() {
            return Ok(vec![p]);
        }
        anyhow::bail!(
            "{DUMPER_ENV} points at {}, which does not exist",
            p.display()
        );
    }
    for c in DUMPER_CANDIDATES {
        let p = PathBuf::from(c);
        if p.exists() {
            return Ok(vec![p]);
        }
    }
    anyhow::bail!(
        "reference dumper not built. It links llama.cpp's own library, so it is built \
         separately from the cargo workspace:\n\n  ./tools/build_llama_logits.sh\n\n\
         (set LLAMA_CPP_PREFIX if llama.cpp is not a Homebrew install, or --dumper / \
         {DUMPER_ENV} to point at a binary elsewhere)"
    )
}

/// What the reference said, and which library said it.
struct Reference {
    logits: Vec<f32>,
    /// The libllama the dumper actually loaded, as it reported itself.
    ///
    /// `None` when the dumper predates the line — an older binary is
    /// still usable, it just cannot be attributed, and that is worth
    /// saying rather than guessing a path from the dumper's own.
    libllama: Option<String>,
}

/// Prefix the dumper uses to report the library it loaded. Spelled once
/// here and once in `tools/llama_logits.c`; the parse is tolerant of
/// its absence precisely because those two are the only things that
/// have to agree and nothing links them.
const LIBLLAMA_LINE: &str = "libllama ";

/// Runs the reference dumper on the SAME token ids and reads back its
/// logit vector.
///
/// `slot` distinguishes the temporary file per reference: two dumpers
/// run in one process would otherwise write the same path, and the
/// second answer would be read as the first.
fn reference_logits(
    dumper: &Path,
    model: &str,
    tokens: &[u32],
    slot: usize,
) -> anyhow::Result<Reference> {
    let out_path =
        std::env::temp_dir().join(format!("frink-parity-{}-{slot}.bin", std::process::id()));
    let mut cmd = Command::new(dumper);
    cmd.arg(model).arg(&out_path);
    for t in tokens {
        cmd.arg(t.to_string());
    }
    let out = cmd.output().context("running the reference dumper")?;
    if !out.status.success() {
        anyhow::bail!(
            "reference dumper failed: {}",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .last()
                .unwrap_or("(no stderr)")
        );
    }
    let bytes = std::fs::read(&out_path)
        .with_context(|| format!("reading reference logits from {}", out_path.display()))?;
    let _ = std::fs::remove_file(&out_path);
    if bytes.len() % 4 != 0 {
        anyhow::bail!("reference logits file is not a whole number of f32");
    }
    Ok(Reference {
        logits: bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        libllama: parse_libllama(&String::from_utf8_lossy(&out.stdout)),
    })
}

/// Picks the `libllama <path>` line out of the dumper's stdout.
fn parse_libllama(stdout: &str) -> Option<String> {
    stdout
        .lines()
        .find_map(|l| l.trim().strip_prefix(LIBLLAMA_LINE))
        .map(str::trim)
        .filter(|p| !p.is_empty() && *p != "unknown")
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A band from ONE reference, which is what every pre-existing
    /// verdict test is about: the KL to that reference IS the nearest,
    /// so these exercise the ladder and not the calibration.
    fn one_reference(reference: &[f32], frink: &[f32]) -> Band {
        Band::measure(
            &[reference],
            frink,
            &DominantQuant::weigh(Some("Q8_0"), Some("Q8_0")),
        )
    }

    #[test]
    fn identical_logits_are_a_match() {
        let l = vec![0.1f32, 5.0, -2.0, 3.3];
        let r = compare(&l, &l, 3, &one_reference(&l, &l));
        assert_eq!(r.verdict, Verdict::Match);
        assert!(r.kl_ref_frink < 1e-12, "KL was {}", r.kl_ref_frink);
        assert_eq!(r.top1_ref, r.top1_frink);
        assert_eq!(r.topk_overlap, 3);
        assert_eq!(r.ref_top1_rank_in_frink, 0);
    }

    #[test]
    fn a_different_graph_is_reported_wrong() {
        // Mass moved to a different token entirely.
        let a = vec![0.0f32, 8.0, 0.0, 0.0];
        let b = vec![0.0f32, 0.0, 8.0, 0.0];
        let r = compare(&a, &b, 2, &one_reference(&a, &b));
        assert_eq!(r.verdict, Verdict::Wrong);
        assert_ne!(r.top1_ref, r.top1_frink);
    }

    #[test]
    fn a_near_tie_that_swaps_is_a_tie_flip_not_a_failure() {
        // Two candidates the reference itself can barely separate: the
        // whole point of measuring the distribution instead of the draw.
        let a = vec![-10.0f32, 2.000_01, 2.0];
        let b = vec![-10.0f32, 2.0, 2.000_01];
        let r = compare(&a, &b, 2, &one_reference(&a, &b));
        assert_eq!(r.verdict, Verdict::TieFlip);
        assert_ne!(r.top1_ref, r.top1_frink);
        // Both engines still agree on WHICH two tokens are in play.
        assert_eq!(r.topk_overlap, 2);
    }

    #[test]
    fn same_top1_with_a_shifted_tail_is_drift_not_a_match() {
        // Top-1 survives, but the rest of the distribution has moved
        // further than accumulation order explains.
        let a = vec![6.0f32, 1.0, 1.0, 1.0];
        let b = vec![6.0f32, 1.0, 1.0, 2.0];
        let r = compare(&a, &b, 4, &one_reference(&a, &b));
        assert_eq!(r.verdict, Verdict::Drift);
        assert_eq!(r.top1_ref, r.top1_frink);
        assert!(r.kl_ref_frink >= KL_NOISE && r.kl_ref_frink < KL_WRONG);
    }

    #[test]
    fn a_uniform_logit_shift_is_invisible_because_softmax_is_shift_invariant() {
        // Guards against ever "fixing" a false alarm by comparing raw
        // logits: an additive constant is not a disagreement.
        let a = vec![0.5f32, 1.5, -3.0];
        let b: Vec<f32> = a.iter().map(|v| v + 7.25).collect();
        let r = compare(&a, &b, 3, &one_reference(&a, &b));
        assert_eq!(r.verdict, Verdict::Match);
    }

    /// The reported top-1/top-2 gap describes the pair the REFERENCE
    /// picked, on both sides, so the flip shows up as a sign change.
    ///
    /// Reporting frink's own top-2 pair instead would compare two
    /// different questions and could print two positive gaps for a run
    /// whose whole finding is that the order reversed.
    #[test]
    fn the_reported_gap_is_the_same_pair_on_both_sides_and_flips_sign() {
        let a = vec![-10.0f32, 2.000_01, 2.0];
        let b = vec![-10.0f32, 2.0, 2.000_01];
        let r = compare(&a, &b, 2, &one_reference(&a, &b));
        assert_eq!(r.verdict, Verdict::TieFlip);
        assert!(
            r.ref_top2_logit_gap > 0.0,
            "the reference ranks its own pair the right way round"
        );
        assert!(
            r.frink_gap_on_ref_pair < 0.0,
            "frink ranks the SAME pair the other way: gap {}",
            r.frink_gap_on_ref_pair
        );
        assert!(
            r.ref_top2_logit_gap_ulps > 0 && r.ref_top2_logit_gap_ulps < 1_000,
            "a tie must be a handful of ulps, got {}",
            r.ref_top2_logit_gap_ulps
        );
    }

    /// THE VERDICT THAT #102 AND #111 ARE ABOUT: one frink, one file,
    /// two libllama, and the answer must not depend on which one the
    /// report happens to name first.
    ///
    /// The fixture is the shape Qwen3-0.6B `--pure` Q4_K_S has: the two
    /// references sit further from each other than frink sits from the
    /// nearer of them. Under the old absolute constant this read DRIFT
    /// against one and WRONG against the other. It must now read the
    /// same either way round, and it must still be a DRIFT rather than
    /// a MATCH, because the distributions really have moved.
    #[test]
    fn the_verdict_is_the_same_whichever_reference_the_report_names() {
        // frink reproduces reference A closely; B is off on its own,
        // and A and B are further apart than frink is from A.
        let a = vec![0.0f32, 4.0, 1.0, 0.5];
        let b = vec![0.0f32, 2.5, 1.0, 0.5];
        let frink = vec![0.0f32, 3.6, 1.0, 0.5];
        let quant = DominantQuant::weigh(Some("Q4K"), Some("Q4K"));

        let ab = Band::measure(&[&a, &b], &frink, &quant);
        let ba = Band::measure(&[&b, &a], &frink, &quant);

        let first = compare(&a, &frink, 4, &ab);
        let second = compare(&b, &frink, 4, &ba);
        assert_eq!(first.verdict, Verdict::Drift);
        assert_eq!(second.verdict, Verdict::Drift);
        // Both really did drift — neither is a MATCH that would make
        // the agreement trivial.
        assert!(first.kl_ref_frink > KL_NOISE && second.kl_ref_frink > KL_NOISE);
        // The printed KL is still the named reference's, which is the
        // whole reason both numbers appear in the report.
        assert_ne!(first.kl_ref_frink, second.kl_ref_frink);

        // With ONE reference and a K-quant body there is no line at
        // all, so the same B-vs-frink comparison cannot be a WRONG
        // either — even though its KL is an order of magnitude over
        // the 3.008e-2 constant that used to be the line.
        let alone = Band::measure(&[&b], &frink, &quant);
        let solo = compare(&b, &frink, 4, &alone);
        assert!(solo.kl_ref_frink > 3.008e-2, "{}", solo.kl_ref_frink);
        assert_eq!(solo.verdict, Verdict::Drift);
    }

    /// THE WRONG RUNG READS THE NEAREST REFERENCE, NOT THE ONE THE
    /// REPORT NAMES.
    ///
    /// This is the whole change, in the one geometry that separates it
    /// from the old rule: the KL printed against the named reference is
    /// OVER that checkpoint's line, while frink is nowhere near being
    /// the outlier — the two references disagree with each other, and
    /// frink reproduces the other one to within a fiftieth of their
    /// disagreement. Qwen3-0.6B `--pure` Q4_K_S is this shape at
    /// smaller scale (3.889e-2 printed, 3.514e-2 spread, 1.975e-2 to
    /// the other build), and deciding it on the printed number is what
    /// made one file DRIFT on one bottle and WRONG on another (#102,
    /// #111).
    ///
    /// The report still prints the large number. It is a true statement
    /// about the named reference; it is just not evidence about frink.
    #[test]
    fn the_wrong_rung_is_decided_by_the_nearest_reference_not_the_one_being_reported() {
        let a = vec![0.0f32, 4.0, 1.0, 0.5];
        let b = vec![0.0f32, 2.0, 1.0, 0.5];
        let frink = vec![0.0f32, 3.8, 0.2, 0.5];
        let quant = DominantQuant::weigh(Some("Q4K"), Some("Q4K"));

        // B is the primary, so B's KL is what the report carries.
        let band = Band::measure(&[&b, &a], &frink, &quant);
        let line = band.line().value().expect("two references give a line");
        assert!(
            band.kl_to_frink(0) > line,
            "the fixture must put the PRIMARY over the line, else this proves nothing: {:.4e} \
             against {line:.4e}",
            band.kl_to_frink(0)
        );
        assert!(band.nearest() < line);

        let r = compare(&b, &frink, 4, &band);
        assert_eq!(r.verdict, Verdict::Drift);
        assert!(
            r.kl_ref_frink > line,
            "the printed KL is still the named reference's"
        );
    }

    /// A WRONG still happens, and it happens to a calibrated band.
    ///
    /// The rule is a relaxation on the checkpoints measured here, so a
    /// test suite that only proved things are no longer WRONG would be
    /// consistent with a verdict that can never fire.
    #[test]
    fn a_frink_further_from_every_reference_than_they_are_from_each_other_is_wrong() {
        // Two references a hair apart, frink a long way from both.
        let a = vec![0.0f32, 4.00, 1.0, 0.5];
        let b = vec![0.0f32, 3.99, 1.0, 0.5];
        let frink = vec![0.0f32, 1.00, 1.0, 0.5];
        let quant = DominantQuant::weigh(Some("Q4K"), Some("Q4K"));
        let band = Band::measure(&[&a, &b], &frink, &quant);
        assert!(band.frink_is_outside());
        assert_eq!(compare(&a, &frink, 4, &band).verdict, Verdict::Wrong);
    }

    /// The dumper's self-report is read, and its absence is not
    /// mistaken for a library called "unknown".
    #[test]
    fn the_reference_identity_is_parsed_and_its_absence_is_not_invented() {
        assert_eq!(
            parse_libllama("libllama /opt/homebrew/lib/libllama.dylib\nn_vocab 32000\n").as_deref(),
            Some("/opt/homebrew/lib/libllama.dylib")
        );
        // An older dumper prints no such line: report that, do not guess.
        assert_eq!(parse_libllama("n_vocab 32000\n"), None);
        // The dumper says "unknown" when dladdr fails; that is an
        // absence too, and passing it through would print a reference
        // path of "unknown" as though it were one.
        assert_eq!(parse_libllama("libllama unknown\nn_vocab 32000\n"), None);
        assert_eq!(parse_libllama("libllama   \n"), None);
    }

    /// Every `--dumper` is resolved, in order, and a typo in ANY of them
    /// fails the run.
    ///
    /// A missing secondary that fell back to the tree's default would
    /// silently calibrate against the primary itself — a zero spread
    /// wearing the appearance of a two-reference experiment.
    #[test]
    fn every_dumper_is_resolved_and_a_missing_one_is_never_substituted() {
        let me = std::env::current_exe().unwrap();
        let me = me.to_string_lossy().into_owned();
        let resolved = dumper_paths(&[me.clone(), me.clone()]).unwrap();
        assert_eq!(resolved.len(), 2, "both references must survive");

        let err = dumper_paths(&[me, "/nonexistent/llama_logits".into()])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("/nonexistent/llama_logits"),
            "a bad second dumper must name itself, got: {err}"
        );
    }
}
