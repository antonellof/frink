use super::*;

/// The production rewrite with an EMPTY environment, so a
/// `LLAMA_ARG_*` variable on the test host cannot change a test.
fn rewrite(argv: Vec<String>) -> Vec<String> {
    let mut cmd = ServerArgs::command();
    cmd.build();
    llama_style_argv(&cmd, argv, |_| None).unwrap()
}

/// The server and `frink run` set the same `FRINK_CTK` variable, so
/// they have to agree about which values exist. The server used to
/// accept anything and pass it through, which meant
/// `frink-server --ctk nonsense` served f16 without a word while
/// `frink run --ctk nonsense` refused.
#[test]
fn the_server_accepts_exactly_the_cache_types_the_cli_does() {
    for good in ["f16", "q8_0", "q4_0", "fp8", "q5_1"] {
        let a = ServerArgs::try_parse_from(["frink-server", "--ctk", good]);
        assert!(a.is_ok(), "{good} was refused");
    }
    for bad in ["nonsense", "turbo4", "q3_k"] {
        let e = ServerArgs::try_parse_from(["frink-server", "--ctk", bad])
            .expect_err(&format!("{bad} was accepted"));
        let msg = e.to_string();
        assert!(msg.contains("unsupported cache type"), "{msg}");
    }
}

#[test]
fn parses_llama_server_style_options() {
    let argv = [
        "frink-server",
        "-m",
        "model.gguf",
        "--host",
        "::1",
        "--port",
        "9000",
        "-t",
        "4",
        "-dev",
        "Metal",
        "-ngl",
        "all",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    let args = ServerArgs::try_parse_from(rewrite(argv)).unwrap();

    assert_eq!(args.model.as_deref(), Some("model.gguf"));
    assert_eq!(args.host, Some(IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)));
    assert_eq!(args.port, Some(9000));
    assert_eq!(args.threads, Some(4));
    assert_eq!(args.device, Some(OffloadDevice::Metal));
    assert_eq!(args.n_gpu_layers, Some(GpuLayers::All));
    assert_eq!(
        cli_bind_addr(&args, Some("127.0.0.1:8383")).as_deref(),
        Some("[::1]:9000")
    );
}

#[test]
fn port_zero_survives_argument_parsing_as_a_real_request() {
    // `--port 0` must reach the bind call intact: it is a request
    // for a kernel-assigned port, not a missing value to default to
    // 8383. The address it produces is deliberately provisional --
    // the ready line reports what was actually bound.
    let argv = ["frink-server", "--port", "0"]
        .into_iter()
        .map(String::from)
        .collect();
    let args = ServerArgs::try_parse_from(rewrite(argv)).unwrap();
    assert_eq!(args.port, Some(0));
    assert_eq!(
        cli_bind_addr(&args, Some("127.0.0.1:8383")).as_deref(),
        Some("127.0.0.1:0")
    );
}

#[test]
fn parallel_flag_parses_and_rewrites_np() {
    let argv = ["frink-server", "-np", "4"]
        .into_iter()
        .map(String::from)
        .collect();
    let args = ServerArgs::try_parse_from(rewrite(argv)).unwrap();
    assert_eq!(args.parallel, Some(4));
}

/// `-b` and `-ub` are two tokens in llama.cpp's hand-written
/// parser and one token to clap, which sees `-b` as a short option
/// it has never heard of. The rewrite is what makes a copied
/// `llama-server ... -b 2048 -ub 512` command run here at all.
#[test]
fn batch_flags_parse_and_rewrite_their_llama_cpp_short_forms() {
    let argv = ["frink-server", "-b", "2048", "-ub", "512"]
        .into_iter()
        .map(String::from)
        .collect();
    let args = ServerArgs::try_parse_from(rewrite(argv)).unwrap();
    assert_eq!(args.batch_size, Some(2048));
    assert_eq!(args.ubatch_size, Some(512));
}

/// Zero is not a batch size, and accepting it would make
/// `env_positive` panic the server later with a message naming an
/// environment variable the operator never set.
#[test]
fn a_zero_batch_size_is_refused_by_name_rather_than_lowered_to_the_environment() {
    for flag in ["--batch-size", "--ubatch-size"] {
        let args =
            ServerArgs::try_parse_from(["frink-server", flag, "0"].into_iter().map(String::from))
                .unwrap();
        let err = apply_cli_overrides(&args).unwrap_err().to_string();
        assert!(err.contains(flag), "{flag}: {err}");
    }
}

/// A path that is not a directory is refused at startup rather
/// than on the first save. The repo's rule about gates applies to
/// flags too: a `--slot-save-path` pointing at nothing looks
/// configured until somebody tries to use it, hours later.
#[test]
fn a_slot_save_path_that_is_not_a_directory_is_refused_at_startup() {
    let args = ServerArgs::try_parse_from(
        ["frink-server", "--slot-save-path", "/definitely/not/here"]
            .into_iter()
            .map(String::from),
    )
    .unwrap();
    let err = apply_cli_overrides(&args).unwrap_err().to_string();
    assert!(err.contains("--slot-save-path"), "{err}");
    assert!(
        std::env::var("FRINK_SLOT_SAVE_PATH").is_err(),
        "a refused path must not have been lowered to the environment first"
    );
}

/// llama.cpp's spelling and range (`common/arg.cpp:3608-3614`):
/// `-1`, `0` and `N` parse -- `-1` needs `allow_hyphen_values`, or
/// clap reads it as a flag -- and anything below `-1` is refused by
/// name before it is lowered to the environment.
#[test]
fn reasoning_budget_parses_llama_cpps_range_and_refuses_the_rest() {
    for (value, expect) in [("-1", -1), ("0", 0), ("2000", 2000)] {
        let args = ServerArgs::try_parse_from(
            ["frink-server", "--reasoning-budget", value]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert_eq!(args.reasoning_budget, Some(expect), "{value}");
    }
    let args = ServerArgs::try_parse_from(
        ["frink-server", "--reasoning-budget", "-2"]
            .into_iter()
            .map(String::from),
    )
    .unwrap();
    let err = apply_cli_overrides(&args).unwrap_err().to_string();
    assert!(err.contains("--reasoning-budget"), "{err}");
}

/// Both spellings of llama.cpp's prefill switch parse, and they
/// conflict rather than letting the last one win silently.
#[test]
fn prefill_assistant_has_both_of_llama_cpps_spellings() {
    let on = ServerArgs::try_parse_from(
        ["frink-server", "--prefill-assistant"]
            .into_iter()
            .map(String::from),
    )
    .unwrap();
    assert!(on.prefill_assistant && !on.no_prefill_assistant);
    let off = ServerArgs::try_parse_from(
        ["frink-server", "--no-prefill-assistant"]
            .into_iter()
            .map(String::from),
    )
    .unwrap();
    assert!(off.no_prefill_assistant && !off.prefill_assistant);
    assert!(ServerArgs::try_parse_from(
        [
            "frink-server",
            "--prefill-assistant",
            "--no-prefill-assistant"
        ]
        .into_iter()
        .map(String::from),
    )
    .is_err());
}

/// `-ngl 10` used to offload every layer here, the divergence
/// `frink run` had already been fixed for. A count that cannot be
/// checked against a layer count is refused as well, rather than
/// trusted.
#[test]
fn a_partial_or_uncheckable_gpu_layer_count_is_refused() {
    let args = ServerArgs::try_parse_from(
        // No `-m`: setting `FRINK_MODEL_PATH` here would leak into
        // every other test in this process.
        ["frink-server", "--n-gpu-layers", "10"]
            .into_iter()
            .map(String::from),
    )
    .unwrap();
    let err = apply_cli_overrides(&args).unwrap_err().to_string();
    assert!(err.contains("--n-gpu-layers 10"), "{err}");
}

/// llama.cpp's `--alias` is a comma-separated list; frink serves one
/// name and must not serve `a,b` as a name.
#[test]
fn a_list_of_aliases_is_refused() {
    let args = ServerArgs::try_parse_from(rewrite(
        ["frink-server", "-a", "one,two"]
            .into_iter()
            .map(String::from)
            .collect(),
    ))
    .unwrap();
    let err = apply_cli_overrides(&args).unwrap_err().to_string();
    assert!(err.contains("--alias one,two"), "{err}");
}

#[test]
fn stdin_close_exit_is_opt_in() {
    // Default off: a server whose stdin is /dev/null (systemd, cron,
    // nohup) would otherwise exit the instant it started.
    let args = ServerArgs::try_parse_from(["frink-server"].into_iter().map(String::from)).unwrap();
    assert!(!args.exit_on_stdin_close);
    let args = ServerArgs::try_parse_from(
        ["frink-server", "--exit-on-stdin-close"]
            .into_iter()
            .map(String::from),
    )
    .unwrap();
    assert!(args.exit_on_stdin_close);
}

fn server_args(argv: &[&str]) -> ServerArgs {
    let mut full = vec!["frink-server"];
    full.extend_from_slice(argv);
    ServerArgs::try_parse_from(full).unwrap_or_else(|e| panic!("{argv:?}: {e}"))
}

/// HTTPS needs the key and the certificate; one of them alone is
/// refused instead of serving plain HTTP with the other ignored, and
/// before anything reaches the environment.
///
/// Sabotage: turn the `_ => bail!` arm into `_ => {}`; this goes red.
#[test]
fn ssl_key_and_cert_files_are_given_together_or_not_at_all() {
    for argv in [
        &["--ssl-key-file", "k.pem"][..],
        &["--ssl-cert-file", "c.pem"],
    ] {
        let err = apply_cli_overrides(&server_args(argv))
            .unwrap_err()
            .to_string();
        assert!(err.contains("must be given together"), "{argv:?}: {err}");
    }
}

/// llama.cpp's `*` and `localhost` mean things frink's exact-origin
/// list does not, so both are refused by name; exact origins pass.
///
/// Sabotage: delete the `"localhost"` arm; that row goes red.
#[test]
fn cors_origins_refuse_the_values_llama_cpp_gives_a_special_meaning() {
    for (spec, says) in [
        ("*", "exact origins only"),
        ("http://a.test, localhost", "special value"),
    ] {
        let err = check_cors_origins(spec).unwrap_err().to_string();
        assert!(err.contains(says), "{spec}: {err}");
    }
    check_cors_origins("http://localhost:5173,https://a.test").unwrap();
}

/// `--threads-http -1` (llama.cpp's default) parses despite the dash,
/// and the new switches and the version exist under their llama.cpp
/// names.
///
/// Sabotage: drop `allow_hyphen_values` from `--threads-http`; the
/// first line fails to parse.
#[test]
fn threads_http_metrics_and_version_parse_like_llama_server() {
    assert_eq!(
        server_args(&["--threads-http", "-1"]).threads_http,
        Some(-1)
    );
    assert_eq!(server_args(&["--threads-http", "8"]).threads_http, Some(8));
    assert!(server_args(&["--metrics"]).metrics);
    let hft = rewrite(["frink-server", "-hft", "tok"].map(String::from).to_vec());
    assert_eq!(
        ServerArgs::try_parse_from(hft).unwrap().hf_token.as_deref(),
        Some("tok")
    );
    let err = ServerArgs::built_command()
        .try_get_matches_from(["frink-server", "--version"])
        .unwrap_err();
    assert_eq!(err.kind(), clap::error::ErrorKind::DisplayVersion);
    assert!(err.to_string().starts_with("frink-server "), "{err}");
    let err = ServerArgs::built_command()
        .try_get_matches_from(["frink-server", "--usage"])
        .unwrap_err();
    assert_eq!(err.kind(), clap::error::ErrorKind::DisplayHelp);
}
