use crate::argv::rewrite_llama_style_argv;

fn rewrite(argv: &[&str]) -> Vec<String> {
    rewrite_llama_style_argv(argv.iter().copied().map(String::from).collect())
        .unwrap_or_else(|e| panic!("{argv:?}: {e}"))
}

/// `try_parse_from`, not `parse_from`: clap's non-`try` parser exits
/// the *process* on a parse error, which takes the whole test binary
/// down with it and reports as "test exited abnormally" instead of
/// naming the test that broke.
fn parse_cli(argv: Vec<String>) -> super::Cli {
    use clap::Parser;
    super::Cli::try_parse_from(&argv)
        .unwrap_or_else(|e| panic!("`{}` did not parse: {e}", argv.join(" ")))
}

/// Every subcommand clap knows about has to reach clap as itself, or
/// the rewriter turns it into an implicit `run` and the user gets a
/// completion instead of the command they typed. This is a silent
/// failure -- no error, no clue. The rewriter reads the subcommand
/// names from clap now rather than from a list beside it, and this
/// holds it to that for every one, aliases included.
#[test]
fn every_clap_subcommand_survives_the_argv_rewriter() {
    use clap::CommandFactory;
    for sub in super::Cli::command().get_subcommands() {
        for name in std::iter::once(sub.get_name()).chain(sub.get_all_aliases()) {
            let rewritten = rewrite(&["frink", name, "--help"]);
            assert_eq!(
                rewritten[1], name,
                "`frink {name}` must reach clap as `{name}`, not as `{}`",
                rewritten[1]
            );
        }
    }
}

/// `frink quantize --help` and `Target::ALL` are two structures that
/// must agree about which targets write, and for three PRs they did
/// not: the help said "Q8_0, plus Q4_K_S / Q4_K_M with --pure" while
/// the code wrote six targets, mixes included. So the help is checked
/// against the table: every target the policy accepts is named in
/// the subcommand's long help, and `--imatrix` is mentioned because
/// the encoders take one.
#[test]
fn the_quantize_help_names_every_target_the_policy_accepts() {
    use clap::CommandFactory;
    let cmd = super::Cli::command();
    let quantize = cmd
        .get_subcommands()
        .find(|c| c.get_name() == "quantize")
        .expect("quantize subcommand");
    let help = quantize
        .get_long_about()
        .map(|s| s.to_string())
        .expect("quantize has a long help");
    for target in crate::quantize::policy::Target::ALL {
        assert!(
            help.contains(target.name()),
            "quantize's help does not mention {}, which `--type` accepts",
            target.name()
        );
    }
    assert!(
        help.contains("--imatrix"),
        "quantize's help does not mention --imatrix"
    );
    assert!(
        !help.contains("with `--pure`"),
        "quantize's help still ties the K-quants to --pure"
    );
}

/// The named case of the above, kept explicit because `serve` is the
/// one that would start a *completion* while the user waited for an
/// HTTP server: it prints tokens and exits 0, so a supervisor
/// watching for the `frink.server.ready` line just hangs.
#[test]
fn serve_is_not_rewritten_into_an_implicit_run() {
    assert_eq!(
        rewrite(&["frink", "serve", "-m", "model.gguf", "--port", "0"]),
        ["frink", "serve", "-m", "model.gguf", "--port", "0"]
    );
}

/// The rewriter still has to translate llama.cpp's multi-character
/// short options *inside* a `serve` invocation -- `frink-server`
/// accepted `-ngl`/`-dev`, so `frink serve` has to as well. Without the
/// feature the stub takes its arguments verbatim and only explains.
#[cfg(feature = "serve")]
#[test]
fn serve_keeps_the_llama_style_short_options() {
    assert_eq!(
        rewrite(&["frink", "serve", "-ngl", "all", "-dev", "metal"]),
        [
            "frink",
            "serve",
            "--n-gpu-layers",
            "all",
            "--device",
            "metal"
        ]
    );
}

/// `serve` must not claim a registry slot from here: the server
/// claims its own as `"server"`, and a second guard on the same pid
/// would deregister the live server when it dropped.
#[test]
fn serve_does_not_claim_the_instance_registry_from_the_cli_side() {
    let cli = parse_cli(rewrite(&["frink", "serve", "-m", "model.gguf"]));
    assert!(
        super::instance_target(&cli.command).is_none(),
        "`frink serve` must leave the registry to frink_server::run_server"
    );

    // The other half of the invariant: a completion run still
    // registers, so this test fails if `instance_target` were
    // "fixed" by making it return None for everything.
    let cli = parse_cli(rewrite(&["frink", "-m", "model.gguf"]));
    assert_eq!(
        super::instance_target(&cli.command).map(|(c, _)| c),
        Some("run")
    );
}

/// Flag parity with the standalone binary, asserted against
/// `ServerArgs` itself rather than against the documented subset:
/// `docs/CLI.md` lists eight of the eleven flags and reads as
/// exhaustive, so a `serve` written from the docs would ship a
/// downgrade nobody noticed.
#[cfg(feature = "serve")]
#[test]
fn serve_exposes_every_flag_frink_server_has() {
    use clap::CommandFactory;
    let ids = |cmd: &clap::Command| -> std::collections::BTreeSet<String> {
        cmd.get_arguments()
            .map(|a| a.get_id().to_string())
            .filter(|id| id != "help" && id != "version")
            .collect()
    };
    let standalone = ids(&frink_server::ServerArgs::command());
    let root = super::Cli::command();
    let serve = ids(root.find_subcommand("serve").expect("serve subcommand"));
    assert!(
        standalone.is_subset(&serve),
        "`frink serve` is missing flags `frink-server` accepts: {:?}",
        standalone.difference(&serve).collect::<Vec<_>>()
    );
}

/// Parity of *values*, not just of flag names: every flag has to
/// land in the same field with the same value it would have reached
/// through `frink-server`'s own argv path. `--port 0` is in here on
/// purpose -- it is a request for a kernel-assigned port, and a
/// front end that dropped it would silently serve on 8383.
#[cfg(feature = "serve")]
#[test]
fn serve_parses_the_same_command_line_as_frink_server() {
    use clap::Parser;
    let flags = [
        "-m",
        "model.gguf",
        "--host",
        "0.0.0.0",
        "--port",
        "0",
        "-t",
        "4",
        "--device",
        "metal",
        "--n-gpu-layers",
        "all",
        "--mcp-config",
        "mcp.json",
        "--exit-on-stdin-close",
        "--allow-multiple-instances",
    ];

    let standalone =
        frink_server::ServerArgs::try_parse_from(std::iter::once("frink-server").chain(flags))
            .expect("frink-server accepts this command line");

    let mut argv = vec!["frink".to_string(), "serve".to_string()];
    argv.extend(flags.iter().map(|f| f.to_string()));
    let cli = parse_cli(rewrite_llama_style_argv(argv).unwrap());
    let super::Commands::Serve(via_cli) = cli.command else {
        panic!("`frink serve …` did not parse as the serve subcommand");
    };

    assert_eq!(via_cli, standalone);
}

/// `--allow-multiple-instances` is declared `global = true`, so both
/// sides of the subcommand are legal places to type it and both have
/// to reach the server -- if the leading form silently did nothing,
/// the server would refuse to start on a host where the operator
/// had already said a second instance was fine.
///
/// Two mechanisms make it work, one of them not ours: the argv
/// rewriter must not mistake the leading flag for llama.cpp-style
/// argv (it used to, see the test below), and clap propagates the
/// root global's *value* into the subcommand's own flag because both
/// carry the id `allow_multiple_instances`. That second half is
/// clap's behaviour, not a guarantee we wrote, which is exactly why
/// it is pinned here: rename either side and this test fails while
/// nothing else would.
#[cfg(feature = "serve")]
#[test]
fn allow_multiple_instances_reaches_serve_from_either_side_of_the_subcommand() {
    use clap::Parser;
    let expected = frink_server::ServerArgs::try_parse_from([
        "frink-server",
        "-m",
        "model.gguf",
        "--allow-multiple-instances",
    ])
    .unwrap();

    for argv in [
        &[
            "frink",
            "serve",
            "-m",
            "model.gguf",
            "--allow-multiple-instances",
        ][..],
        &[
            "frink",
            "--allow-multiple-instances",
            "serve",
            "-m",
            "model.gguf",
        ][..],
    ] {
        let cli = parse_cli(rewrite(argv));
        assert!(
            cli.allow_multiple_instances,
            "`{}` did not set the root flag",
            argv.join(" ")
        );
        let super::Commands::Serve(args) = cli.command else {
            panic!("`{}` did not parse as the serve subcommand", argv.join(" "));
        };
        assert_eq!(
            args,
            expected,
            "`{}` did not reach the server as an opt-in",
            argv.join(" ")
        );
    }
}

/// The rewriter half of the above, stated on its own: a global flag
/// before the subcommand must not turn the subcommand into an
/// argument of an implicit `run`.
#[test]
fn a_global_flag_before_a_subcommand_does_not_trigger_the_run_rewrite() {
    assert_eq!(
        rewrite(&[
            "frink",
            "--allow-multiple-instances",
            "serve",
            "-m",
            "x.gguf"
        ]),
        [
            "frink",
            "--allow-multiple-instances",
            "serve",
            "-m",
            "x.gguf"
        ]
    );
    // …and llama.cpp-style argv behind the same flag still gets it.
    assert_eq!(
        rewrite(&["frink", "--allow-multiple-instances", "-m", "x.gguf"]),
        ["frink", "run", "--allow-multiple-instances", "-m", "x.gguf"]
    );
}

/// `--list-devices` is the flag that is easiest to lose, because it
/// exits before anything is served and so no smoke test covers it.
#[cfg(feature = "serve")]
#[test]
fn serve_accepts_list_devices_on_its_own() {
    use clap::Parser;
    let cli = parse_cli(rewrite(&["frink", "serve", "--list-devices"]));
    let super::Commands::Serve(args) = cli.command else {
        panic!("not the serve subcommand");
    };
    assert_eq!(
        args,
        frink_server::ServerArgs::try_parse_from(["frink-server", "--list-devices"]).unwrap()
    );
}

/// Without the feature the subcommand still exists and still
/// swallows the server's flags, so the user gets the sentence that
/// names the fix instead of clap's "unrecognized subcommand".
#[cfg(not(feature = "serve"))]
#[test]
fn serve_without_the_feature_explains_itself_instead_of_erroring_out_of_clap() {
    let cli = parse_cli(rewrite(&["frink", "serve", "-m", "model.gguf"]));
    let super::Commands::Serve { args } = cli.command else {
        panic!("`frink serve …` did not parse as the serve subcommand");
    };
    assert_eq!(args, ["-m", "model.gguf"]);
    assert!(super::SERVE_FEATURE_MISSING.contains("serve is on by default"));
}

#[test]
fn rewrites_llama_multi_character_short_options() {
    let args = ["frink", "-m", "model.gguf", "-dev", "none", "-ngl", "0"]
        .into_iter()
        .map(String::from)
        .collect();
    assert_eq!(
        rewrite_llama_style_argv(args).unwrap(),
        [
            "frink",
            "run",
            "-m",
            "model.gguf",
            "--device",
            "none",
            "--n-gpu-layers",
            "0"
        ]
    );
}

/// **`-st` is not `--no-cnv`.**
///
/// llama.cpp 0.4 dropped `--no-cnv` and offers `-st` /
/// `--single-turn`, and the two are easy to read as synonyms.
/// They are not: `-st` runs the CONVERSATION for one turn, with
/// the chat template applied, where `--no-cnv` skips the template
/// entirely.
///
/// An early version of that rewriter row mapped them together,
/// and it changed the answer -- measured on the same prompt and
/// the same file, the templated reply is "The capital of France
/// is Paris." and the raw one continues " Paris\nThe capital city
/// of France is...". Nothing would have caught that but reading
/// two outputs side by side, which is why it is a test now.
#[test]
fn single_turn_keeps_the_chat_template() {
    let out = rewrite(&["frink", "-m", "model.gguf", "-p", "hi", "-st"]);
    assert!(
        out.contains(&"--single-turn".to_string()),
        "-st must reach the single-turn flag: {out:?}"
    );
    assert!(
        !out.contains(&"--no-cnv".to_string()),
        "-st must NOT become --no-cnv: it keeps the chat template, \
         and mapping the two together changes the generated text: {out:?}"
    );
}

/// And the two flags stay distinct at the clap layer, so neither
/// is an alias of the other.
#[test]
fn single_turn_and_no_cnv_are_separate_flags() {
    use clap::CommandFactory;
    let run = crate::Cli::command();
    let run = run
        .get_subcommands()
        .find(|c| c.get_name() == "run")
        .expect("run subcommand");
    let names: Vec<&str> = run
        .get_arguments()
        .filter_map(|a| a.get_long())
        .filter(|l| *l == "no-cnv" || *l == "single-turn")
        .collect();
    assert_eq!(
        names.len(),
        2,
        "both flags must exist independently, found {names:?}"
    );
}

/// `-md` is llama.cpp's draft-model flag. Before the spellings were
/// derived from clap, clap read it as the cluster `-m d`: the MODEL
/// path silently became the letter `d`.
#[test]
fn md_is_the_draft_model_and_not_a_model_named_d() {
    let cli = parse_cli(rewrite(&["frink", "-m", "t.gguf", "-md", "d.gguf"]));
    let super::Commands::Run(args) = cli.command else {
        panic!("not run");
    };
    assert_eq!(args.model.as_deref(), Some("t.gguf"));
    assert_eq!(args.model_draft.as_deref(), Some("d.gguf"));
}

/// An unknown single-dash name is an error naming the token, not a
/// cluster of short options clap guesses at.
#[test]
fn an_unknown_single_dash_name_is_refused_by_the_rewriter() {
    let err = rewrite_llama_style_argv(
        ["frink", "-m", "t.gguf", "-zzz"]
            .into_iter()
            .map(String::from)
            .collect(),
    )
    .unwrap_err();
    assert!(err.contains("`-zzz`"), "{err}");
}

/// `frink-server -np 4` worked and `frink serve -np 4` did not: the two
/// front ends had separate rewrite tables and only the server's had
/// `-np`, `-cb` and `-b`. They read one set of spellings now.
#[cfg(feature = "serve")]
#[test]
fn serve_reads_the_same_single_dash_names_as_frink_server() {
    use clap::Parser;
    let cli = parse_cli(rewrite(&[
        "frink", "serve", "-np", "4", "-cb", "-b", "256", "-ub", "128", "-hf", "r", "-hff",
        "x.gguf",
    ]));
    let super::Commands::Serve(via_cli) = cli.command else {
        panic!("not serve");
    };
    let standalone = frink_server::ServerArgs::try_parse_from([
        "frink-server",
        "--parallel",
        "4",
        "--cont-batching",
        "--batch-size",
        "256",
        "--ubatch-size",
        "128",
        "--hf-repo",
        "r",
        "--hf-file",
        "x.gguf",
    ])
    .unwrap();
    assert_eq!(via_cli, standalone);
}

/// `--version` on a subcommand names the binary that exists (`frink`,
/// or `frink-server` for `serve`), not clap's `frink-run`; `--usage` is
/// help, as it is in llama.cpp, before and after the run rewrite.
///
/// Sabotage: drop the `display_name("frink")` loop in `built_cli`; the
/// `run` row prints `frink-run` and goes red.
#[test]
fn version_and_usage_work_on_every_front_end_spelling() {
    let mut cases = vec![
        (vec!["frink", "--version"], "frink "),
        (vec!["frink", "run", "--version"], "frink "),
    ];
    if cfg!(feature = "serve") {
        cases.push((vec!["frink", "serve", "--version"], "frink-server "));
    }
    for (argv, want) in cases {
        let argv: Vec<String> = argv.into_iter().map(String::from).collect();
        let err = crate::argv::parse(argv.clone())
            .err()
            .expect("--version must exit");
        assert_eq!(
            err.kind(),
            clap::error::ErrorKind::DisplayVersion,
            "{argv:?}"
        );
        assert!(err.to_string().starts_with(want), "{argv:?}: {err}");
    }
    for argv in [
        &["frink", "--usage"][..],
        &["frink", "run", "--usage"],
        &["frink", "-m", "x.gguf", "--usage"],
    ] {
        let err = crate::argv::parse(rewrite(argv))
            .err()
            .expect("--usage must exit");
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayHelp, "{argv:?}");
    }
}
