use super::GpuLayers;
use std::str::FromStr;

use super::{banner_line, kv_elem_for, InferArgs, OffloadDevice};
use clap::Parser;

/// `InferArgs` is a `clap::Args` group, not a `Parser`, so the test
/// gives it the top-level command it is normally flattened into.
#[derive(Parser, Debug)]
struct Cli {
    #[command(flatten)]
    infer: InferArgs,
}

fn args(argv: &[&str]) -> InferArgs {
    let mut full = vec!["frink"];
    full.extend_from_slice(argv);
    Cli::parse_from(full).infer
}

/// GitHub issue #170, at the flag rather than at the predicate: the
/// DEFAULT `frink run -m … --temp 0 --ngl 99` must not let Metal
/// fold `lm_head + argmax` onto the device.
///
/// `--repeat-penalty` defaults to **1.1** here, deliberately unlike
/// llama.cpp's 1.0 (`docs/FEATURES.md` records the difference), and
/// a device argmax over raw logits never applies it. The fold gate
/// read a predicate that tested XTC, typical-p and DRY and not the
/// penalties, so a plain greedy Metal run returned a token the host
/// sampler would not have chosen -- and agreed instead, byte for
/// byte, with the same run at `--repeat-penalty 1.0`.
///
/// This test is here and not only in `frink-models` because the
/// DEFAULT is the thing that made it live: the predicate and the
/// flag are two structures that have to agree, and `frink-models`
/// cannot see this crate's `default_value_t`.
///
/// Sabotage: set `default_value_t = 1.0` on `--repeat-penalty`; the
/// first assertion goes red.
#[test]
fn the_default_flags_forbid_the_metal_greedy_argmax_fold() {
    let step = super::TokenStep::new(frink_models::sampling::Sampler::new(1), None);
    let sampling = |argv: &[&str]| {
        args(argv)
            .sampling(None, 4096)
            .expect("no --dry-multiplier, so no vocabulary is needed")
    };

    let defaults = sampling(&["-m", "m.gguf", "--temp", "0"]);
    assert_eq!(defaults.repetition_penalty, 1.1, "llama.cpp's is 1.0");
    assert!(
        step.needs_vocab_logits(&defaults),
        "the default repetition penalty is applied on the host, so the \
         device must hand back a vocabulary and not one token id"
    );

    // Both of llama.cpp's off switches restore the fold, which is
    // what makes the assertion above about the penalty and not about
    // `--top-k 40` or `--min-p 0.05`, which default on too.
    for off in [
        ["-m", "m.gguf", "--temp", "0", "--repeat-penalty", "1.0"],
        ["-m", "m.gguf", "--temp", "0", "--repeat-last-n", "0"],
    ] {
        let s = sampling(&off);
        assert!(
            !step.needs_vocab_logits(&s),
            "{off:?} switches the penalties off, so the fold is exact again"
        );
    }
}

/// The banner may not promise a KV dtype the run will not use.
///
/// `frink -m m.gguf -dev cpu -ngl all` once printed `ctk=f16` while
/// the host `KvCache` was `Vec<f32>` and the budget priced it at f32.
/// The host store honours `--ctk f16` / `f32` now
/// (`frink_models::host_kv_dtype`), and the banner, the budget and the
/// caches all read that one function, so this pins the three answers
/// the host can give: the flag honoured, the flag honoured the other
/// way, and a device-only wire falling back to f16 by name.
#[test]
fn the_banner_reports_the_kv_dtype_the_run_will_actually_keep() {
    let a = args(&["-m", "m.gguf", "--device", "cpu", "--ctk", "f16"]);
    assert_eq!(kv_elem_for(&a).as_str(), "f16", "llama.cpp's default width");
    assert_eq!(
        frink_models::KvElem::from(super::host_kv_dtype_for(&a)),
        kv_elem_for(&a),
        "the caches and the budget read one function"
    );
    let line = banner_line(&a, OffloadDevice::Cpu);
    assert!(
        line.contains("ctk=f16") && !line.contains("--ctk f16"),
        "{line}"
    );

    let a = args(&["-m", "m.gguf", "--device", "cpu", "--ctk", "f32"]);
    assert_eq!(kv_elem_for(&a).as_str(), "f32");
    assert!(banner_line(&a, OffloadDevice::Cpu).contains("ctk=f32"));

    let a = args(&["-m", "m.gguf", "--device", "cpu", "--ctk", "q8_0"]);
    assert_eq!(kv_elem_for(&a).as_str(), "f16");
    let line = banner_line(&a, OffloadDevice::Cpu);
    assert!(line.contains("ctk=f16"), "{line}");
    assert!(
        line.contains("--ctk q8_0 has no cpu host store; using f16"),
        "a flag the host cannot honour must say so: {line}"
    );
}

/// GitHub issue #297: `FRINK_CTK` is documented as "same as
/// `--ctk`" and could not be.
///
/// The resolution writes `args.ctk` into the variable
/// unconditionally, so before clap read it as the default an
/// environment that said `q4` was overwritten with the flag's
/// `f16` before `frink_metal::attn::metal_kv_dtype` ever looked.
/// Nothing checked it, which is why a documented spelling shipped
/// dead.
///
/// Serialised, because it mutates process-wide state and the other
/// tests in this file parse the same argument.
#[test]
fn the_environment_supplies_the_kv_dtype_when_the_flag_does_not() {
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let restore = std::env::var("FRINK_CTK").ok();

    // SAFETY: single-threaded test body, holding ENV_LOCK.
    unsafe { std::env::set_var("FRINK_CTK", "q4_0") };
    let a = args(&["-m", "m.gguf"]);
    assert_eq!(a.ctk, "q4_0", "the environment was ignored");

    // An explicit flag still wins over it.
    let a = args(&["-m", "m.gguf", "--ctk", "q8_0"]);
    assert_eq!(a.ctk, "q8_0", "the flag lost to the environment");

    // SAFETY: same.
    unsafe {
        match restore {
            Some(v) => std::env::set_var("FRINK_CTK", v),
            None => std::env::remove_var("FRINK_CTK"),
        }
    }
}

/// `-dev cpu` is not `-dev none`, and `-ngl all` under either does
/// nothing. Both were printed as if honoured.
#[test]
fn the_banner_does_not_promise_gpu_layers_on_a_cpu_device() {
    let a = args(&["-m", "m.gguf", "--device", "cpu", "--ngl", "all"]);
    let line = banner_line(&a, OffloadDevice::Cpu);
    assert!(line.contains("device=cpu"), "{line}");
    assert!(line.contains("ignored, no GPU offload"), "{line}");
}

/// And a run that really does select the dtype keeps a clean line.
#[test]
fn a_metal_run_reports_the_requested_dtype_with_no_caveat() {
    let a = args(&[
        "-m", "m.gguf", "--device", "metal", "--ngl", "all", "--ctk", "f16",
    ]);
    let line = banner_line(&a, OffloadDevice::Metal);
    assert!(line.contains("ctk=f16"), "{line}");
    assert!(!line.contains("ignored"), "{line}");
}

/// `-ngl N` must not silently mean "all layers".
///
/// llama.cpp's `-ngl N` puts exactly N layers in VRAM and runs the
/// rest on the CPU, which is how people fit a model that otherwise
/// does not fit. frink parsed the count and read only
/// `offload_enabled()`, a bool, so `--ngl 10` on a 32-layer model
/// offloaded all 32 -- same flag, same value, no error, and the
/// failure arrives as an OOM on the machine the flag existed to
/// accommodate.
///
/// Partial offload is not implemented, so it refuses. The two exact
/// cases still work.
#[test]
fn a_partial_gpu_layer_count_is_refused_rather_than_rounded_up() {
    let err = GpuLayers::Count(10)
        .check_supported(32)
        .expect_err("10 of 32 is partial");
    let msg = err.to_string();
    assert!(msg.contains("PARTIAL"), "{msg}");
    assert!(
        msg.contains("--ngl 0"),
        "the message must say what works: {msg}"
    );
    assert!(msg.contains("--ngl 32"), "{msg}");

    // The exact cases are not partial and must stay accepted.
    GpuLayers::Count(0)
        .check_supported(32)
        .expect("0 = CPU only");
    GpuLayers::Count(32).check_supported(32).expect("32 = all");
    GpuLayers::Count(99)
        .check_supported(32)
        .expect("clamps to all");
    GpuLayers::All.check_supported(32).expect("all");
    GpuLayers::Auto.check_supported(32).expect("auto");
}

/// `--escape` defaults TRUE, as llama.cpp does.
///
/// frink defaulted false, so `-p "a\\nb"` reached the model as a
/// literal backslash-n on frink and as a newline on llama.cpp: the
/// same command, a different prompt, and no error on either side.
#[test]
fn escapes_are_processed_by_default_like_llama_cpp() {
    use clap::Parser;
    #[derive(Parser)]
    struct Probe {
        #[command(flatten)]
        args: super::InferArgs,
    }
    let parsed = Probe::try_parse_from(["frink", "-m", "x.gguf"]).expect("defaults parse");
    assert!(parsed.args.escape, "llama.cpp common/common.h:563 is true");
    assert!(!parsed.args.no_escape);

    let off = Probe::try_parse_from(["frink", "-m", "x.gguf", "--no-escape"])
        .expect("--no-escape parses");
    assert!(off.args.no_escape, "llama.cpp spells the negation this way");
}

/// `--samplers` is llama.cpp's, and the DEFAULT is the chain frink
/// already ran: a command line that does not mention the flag must
/// sample exactly what it did before the flag existed.
///
/// The distribution-level proof of that is
/// `frink_models::sampling::tests::the_default_order_is_the_chain_frink_already_ran`;
/// this is the CLI half -- that the flag's default resolves to the
/// same chain, so the two cannot drift apart.
#[test]
fn samplers_defaults_to_the_chain_frink_already_ran() {
    let default = args(&["-m", "x.gguf"])
        .sampling(None, 4096)
        .expect("no dry, so no vocabulary is needed")
        .sampler_order;
    assert_eq!(default, frink_models::SamplerOrder::default());
    assert_eq!(
        default.to_string(),
        "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature"
    );
}

/// A caller-supplied order reaches the sampler, in the order typed.
#[test]
fn a_caller_supplied_order_reaches_the_sampler() {
    let order = args(&["-m", "x.gguf", "--samplers", "penalties;temperature;top_k"])
        .sampling(None, 4096)
        .expect("no dry")
        .sampler_order;
    assert_eq!(order.to_string(), "penalties;temperature;top_k");
    // llama.cpp's own aliases, so an upstream command line works.
    assert_eq!(
        args(&["-m", "x.gguf", "--samplers", "top-k;min-p;temp"])
            .sampling(None, 4096)
            .expect("no dry")
            .sampler_order
            .to_string(),
        "top_k;min_p;temperature"
    );
}

/// A sampler frink does not implement is refused BY NAME at the
/// command line, rather than dropped out of the chain.
///
/// Upstream's own default string names three samplers this engine
/// lacks, so pasting it must fail loudly. A caller who asked for
/// `xtc` and was given a chain without it was silently handed a
/// different sampler.
#[test]
fn a_sampler_frink_lacks_is_refused_by_name_on_the_command_line() {
    let err = Cli::try_parse_from([
        "frink",
        "-m",
        "x.gguf",
        "--samplers",
        "penalties;mirostat;temperature",
    ])
    .expect_err("mirostat is not a chain member here")
    .to_string();
    assert!(err.contains("mirostat"), "{err}");
    assert!(err.contains("not implemented"), "{err}");

    // And llama.cpp's OWN default string now parses, which is the
    // point of this change: pasting an upstream command line works.
    Cli::try_parse_from([
        "frink",
        "-m",
        "x.gguf",
        "--samplers",
        "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature",
    ])
    .expect("llama.cpp's default chain is frink's default chain");

    let unknown = Cli::try_parse_from(["frink", "-m", "x.gguf", "--samplers", "top_kk"])
        .expect_err("no such sampler")
        .to_string();
    assert!(unknown.contains("top_kk"), "{unknown}");
    assert!(unknown.contains("unknown sampler"), "{unknown}");
}

#[test]
fn parses_llama_gpu_layer_values() {
    assert_eq!(GpuLayers::from_str("0"), Ok(GpuLayers::Count(0)));
    assert_eq!(GpuLayers::from_str("42"), Ok(GpuLayers::Count(42)));
    assert_eq!(GpuLayers::from_str("auto"), Ok(GpuLayers::Auto));
    assert_eq!(GpuLayers::from_str("all"), Ok(GpuLayers::All));
    assert!(GpuLayers::from_str("-1").is_err());
    assert!(GpuLayers::from_str("some").is_err());
}

/// A file under the system temp directory, unique to this test process
/// and `tag`, holding `text`.
/// [`args`] with `-m x.gguf` in front, which every parse needs.
fn with_model(argv: &[&str]) -> InferArgs {
    let mut full = vec!["-m", "x.gguf"];
    full.extend_from_slice(argv);
    args(&full)
}

fn temp_file(tag: &str, text: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("frink-run-{}-{tag}", std::process::id()));
    std::fs::write(&path, text).unwrap();
    path
}

/// `-jf FILE` is `-j "$(cat FILE)"`: the same grammar, from the same
/// converter, and exclusive with the other three grammar flags.
///
/// Sabotage: drop the `json_schema_file` branch from `grammar_source`;
/// the first assertion goes red (no grammar at all).
#[test]
fn json_schema_file_is_the_json_schema_flag_read_from_a_file() {
    let schema = r#"{"type": "object", "properties": {"n": {"type": "integer"}}}"#;
    let path = temp_file("schema.json", schema);
    let p = path.to_str().unwrap();
    let from_file = with_model(&["--json-schema-file", p]);
    let inline = with_model(&["--json-schema", schema]);
    let got = from_file.grammar_source().unwrap();
    assert!(got.is_some(), "-jf produced no grammar");
    assert_eq!(got, inline.grammar_source().unwrap());
    let both = with_model(&["--json-schema-file", p, "--grammar", "root ::= \"a\""]);
    let err = both.grammar_source().unwrap_err().to_string();
    assert!(err.contains("mutually exclusive"), "{err}");
    std::fs::remove_file(path).ok();
}

/// `-sysf` drops exactly one trailing newline, as llama.cpp's
/// `read_file` + `pop_back` does, and is exclusive with `--system`.
///
/// Sabotage: replace the `pop` with nothing; the first assertion goes
/// red with a trailing newline.
#[test]
fn system_prompt_file_drops_one_trailing_newline_like_llama_cpp() {
    for (text, want) in [
        ("be terse\n", "be terse"),
        ("two\n\n", "two\n"),
        ("none", "none"),
    ] {
        let path = temp_file("sys.txt", text);
        let mut a = with_model(&["--system-prompt-file", path.to_str().unwrap()]);
        a.read_system_prompt_file().unwrap();
        assert_eq!(a.system.as_deref(), Some(want), "{text:?}");
        std::fs::remove_file(path).ok();
    }
    let clash = Cli::try_parse_from([
        "frink",
        "-m",
        "x.gguf",
        "--system",
        "a",
        "--system-prompt-file",
        "b",
    ]);
    assert!(
        clash.is_err(),
        "--system and --system-prompt-file both accepted"
    );
    let mut missing = with_model(&["--system-prompt-file", "/nonexistent/frink-sysf"]);
    let err = missing.read_system_prompt_file().unwrap_err().to_string();
    assert!(err.contains("--system-prompt-file"), "{err}");
}

/// `--no-jinja` asks for a template mode frink does not have, so it is
/// refused; `--jinja` is what already happens, so it is accepted.
/// `-fa off` is refused, and `-fa` with no value is `auto`.
///
/// Sabotage: delete the `no_jinja` bail; the first assertion goes red.
#[test]
fn jinja_and_flash_attn_refuse_what_frink_cannot_do() {
    let err = with_model(&["--no-jinja"])
        .apply_llama_flags()
        .unwrap_err()
        .to_string();
    assert!(err.contains("--no-jinja"), "{err}");
    with_model(&["--jinja"]).apply_llama_flags().unwrap();
    for off in ["off", "0", "disabled", "false"] {
        let err = with_model(&["--flash-attn", off])
            .apply_llama_flags()
            .unwrap_err()
            .to_string();
        assert!(err.contains("backend property"), "{off}: {err}");
    }
    let err = with_model(&["--flash-attn", "maybe"])
        .apply_llama_flags()
        .unwrap_err()
        .to_string();
    assert!(err.contains("expected on, off or auto"), "{err}");
    assert_eq!(
        with_model(&["--flash-attn"]).flash_attn.as_deref(),
        Some("auto")
    );
    with_model(&["--flash-attn", "on"])
        .apply_llama_flags()
        .unwrap();
}

/// The paired switches: the default is on, and the later spelling wins,
/// as llama.cpp's `{"--warmup"}, {"--no-warmup"}` pair is last-wins.
///
/// Sabotage: make `warmup_enabled` return `true`; the `--no-warmup`
/// assertion goes red.
#[test]
fn warmup_and_show_timings_default_on_and_the_last_spelling_wins() {
    assert!(with_model(&[]).warmup_enabled());
    assert!(!with_model(&["--no-warmup"]).warmup_enabled());
    assert!(with_model(&["--no-warmup", "--warmup"]).warmup_enabled());
    assert!(!with_model(&["--warmup", "--no-warmup"]).warmup_enabled());
    assert!(with_model(&[]).show_timings_enabled());
    assert!(!with_model(&["--no-show-timings"]).show_timings_enabled());
    assert!(with_model(&["--no-show-timings", "--show-timings"]).show_timings_enabled());
}

/// `-co` takes llama.cpp's truthy, falsey and auto spellings, a bare
/// `-co` is on, and anything else is refused as upstream throws.
///
/// Sabotage: map `Switch::Off` to `Some(true)`; the `off` row goes red.
#[test]
fn color_takes_llama_cpps_values() {
    for (argv, want) in [
        (&["--color", "on"][..], Some(true)),
        (&["--color", "1"], Some(true)),
        (&["--color", "off"], Some(false)),
        (&["--color", "false"], Some(false)),
        (&["--color", "auto"], None),
        (&["--color"], Some(true)),
        (&[], None),
    ] {
        assert_eq!(with_model(argv).colour_choice().unwrap(), want, "{argv:?}");
    }
    let err = with_model(&["--color", "rainbow"])
        .colour_choice()
        .unwrap_err()
        .to_string();
    assert!(err.contains("expected on, off or auto"), "{err}");
}
