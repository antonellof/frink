//! llama.cpp's command-line grammar on top of a clap `Command`.
//!
//! llama.cpp's parser is hand-written and reads every token whole:
//! `-ngl`, `-hff`, `-ctk` and `-np` are each ONE flag. clap reads a
//! single dash as a cluster of short options, so `-ngl 99` became `-n`
//! with the value `gl`, and `-md draft.gguf` became `-m d`: the model
//! path silently replaced by the letter `d`. ADR 0005 says frink takes
//! llama.cpp's command lines as they are, so the token has to be
//! translated before clap sees it.
//!
//! Both front ends used to do that with a hand-written table of
//! spellings each, and the tables disagreed: `frink-server` rewrote
//! `-np`, `-cb` and `-b`, `frink serve` (the same server, through
//! frink-cli's table) did not, and both missed `-ctk`. Here the
//! spellings come from the clap `Command` itself: a single-dash token
//! whose name is a long name or alias of one of the command's
//! arguments is that argument. Adding `visible_alias = "ngl"` to an
//! argument is the whole change, and there is no second table to
//! forget.
//!
//! A single-dash token of two or more letters that names nothing is an
//! ERROR rather than being handed to clap as a cluster, because clap's
//! reading of it is never what a llama.cpp user meant.

use clap::{Arg, Command};

/// A llama.cpp tri-state value (`-fa`, `-co`, a switch's environment
/// variable).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Switch {
    On,
    Off,
    Auto,
}

/// llama.cpp's `common_arg_utils::is_truthy` / `is_falsey` / `is_autoy`
/// (`common/arg.cpp:1345-1355`), exactly: `on enabled true 1`,
/// `off disabled false 0`, `auto -1`. Anything else is `None`, which
/// every caller refuses, as upstream throws on it.
pub fn llama_switch(value: &str) -> Option<Switch> {
    match value {
        "on" | "enabled" | "true" | "1" => Some(Switch::On),
        "off" | "disabled" | "false" | "0" => Some(Switch::Off),
        "auto" | "-1" => Some(Switch::Auto),
        _ => None,
    }
}

/// `-fa` / `--flash-attn`, on every front end that takes it.
///
/// `on` and `auto` are accepted: fused attention is a backend property
/// here, on wherever the kernels support the shape. `off` is refused
/// rather than ignored, and so is a value llama.cpp would not parse.
/// One function, because `frink run`, `frink-server` and `frink bench`
/// all take the flag and must agree about what it accepts.
pub fn check_flash_attn(value: &str) -> Result<(), String> {
    match llama_switch(value.trim()) {
        Some(Switch::On | Switch::Auto) => Ok(()),
        Some(Switch::Off) => Err(format!(
            "--flash-attn {value}: fused attention is a backend property here, not a per-run \
             switch. Set FRINK_METAL_ATTN=0 to take the unfused Metal path, or --device cpu"
        )),
        None => Err(format!(
            "--flash-attn {value}: expected on, off or auto, as llama.cpp's --flash-attn"
        )),
    }
}

/// The argument a flag token names, if `cmd` has one.
///
/// `--name` and `--name=value` match a long name or any alias, `-x`
/// matches a short (or short alias) first and then a one-letter long
/// name, and `-name` matches a long name or alias.
pub fn arg_for_token<'a>(cmd: &'a Command, token: &str) -> Option<&'a Arg> {
    let named = |name: &str| {
        cmd.get_arguments().find(|a| {
            a.get_long() == Some(name)
                || a.get_all_aliases()
                    .is_some_and(|aliases| aliases.contains(&name))
        })
    };
    if let Some(rest) = token.strip_prefix("--") {
        return named(rest.split_once('=').map_or(rest, |(n, _)| n));
    }
    let name = token.strip_prefix('-')?;
    let mut chars = name.chars();
    let first = chars.next()?;
    if chars.next().is_none() {
        let short = cmd.get_arguments().find(|a| {
            a.get_short() == Some(first)
                || a.get_all_short_aliases()
                    .is_some_and(|shorts| shorts.contains(&first))
        });
        if short.is_some() {
            return short;
        }
    }
    named(name)
}

/// llama.cpp's third spelling of help, `--usage` (`common/arg.cpp`:
/// `{"-h", "--help", "--usage"}`), on `cmd` and every subcommand.
///
/// Call it BEFORE `Command::build`. clap creates its own help argument
/// at build time and indexes every spelling then, so an alias added
/// afterwards is never looked up; this replaces that argument with one
/// that has the alias, on every command that has help at all. Done
/// over the tree rather than with a help argument written per command,
/// so a command added later gets the spelling without anyone
/// remembering to give it one.
pub fn with_usage_alias(mut cmd: Command) -> Command {
    if !cmd.is_disable_help_flag_set() {
        cmd = cmd.disable_help_flag(true).arg(
            Arg::new("help")
                .short('h')
                .long("help")
                .alias("usage")
                .action(clap::ArgAction::Help)
                .help("Print help"),
        );
    }
    let names: Vec<String> = cmd
        .get_subcommands()
        .map(|s| s.get_name().to_string())
        .collect();
    for name in names {
        cmd = cmd.mut_subcommand(name, with_usage_alias);
    }
    cmd
}

/// Whether `arg` always consumes the token after it.
///
/// An optional value (`num_args(0..=1)`, `--flash-attn [MODE]`) does
/// not: the next token is then classified like any other.
fn consumes_next(arg: &Arg) -> bool {
    arg.get_action().takes_values() && arg.get_num_args().is_some_and(|r| r.min_values() > 0)
}

/// Rewrites llama.cpp-style single-dash names in `tail` (the tokens
/// after the program or subcommand name) to the `--long` spelling clap
/// parses.
///
/// `cmd` must be BUILT (`Command::build`), or it has no help/version
/// arguments, no propagated globals and no value ranges.
///
/// Tokens that are values of the previous flag, tokens after `--`, and
/// anything that is not a flag pass through untouched, so a prompt that
/// starts with a dash is still a prompt.
pub fn expand_single_dash_names(cmd: &Command, tail: Vec<String>) -> Result<Vec<String>, String> {
    let mut out = Vec::with_capacity(tail.len());
    let mut value_next = false;
    let mut verbatim = false;
    for token in tail {
        if verbatim || value_next {
            value_next = false;
            out.push(token);
            continue;
        }
        if token == "--" {
            verbatim = true;
            out.push(token);
            continue;
        }
        let Some(name) = token.strip_prefix('-').filter(|n| !n.starts_with('-')) else {
            if let Some(arg) = arg_for_token(cmd, &token) {
                value_next = !token.contains('=') && consumes_next(arg);
            }
            out.push(token);
            continue;
        };
        // A value such as `-1` or `-0.5`, or a bare `-` (stdin).
        if !name.starts_with(|c: char| c.is_ascii_alphabetic()) {
            out.push(token);
            continue;
        }
        match arg_for_token(cmd, &token) {
            Some(arg) => {
                value_next = consumes_next(arg);
                let one = name.chars().next().filter(|_| name.chars().count() == 1);
                let by_short = one.is_some_and(|c| {
                    arg.get_short() == Some(c)
                        || arg.get_all_short_aliases().is_some_and(|s| s.contains(&c))
                });
                match arg.get_long() {
                    Some(long) if !by_short => out.push(format!("--{long}")),
                    _ => out.push(token),
                }
            }
            // One letter that names nothing: clap's own error is right.
            None if name.chars().count() == 1 => out.push(token),
            None => {
                return Err(format!(
                    "unknown flag `{token}` for `{}`. A single-dash flag is one name here, as in \
                     llama.cpp, not a cluster of short options",
                    cmd.get_name()
                ))
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::ArgAction;

    fn cmd() -> Command {
        let mut c = Command::new("t")
            .arg(Arg::new("model").short('m').long("model"))
            .arg(Arg::new("ngl").long("n-gpu-layers").visible_alias("ngl"))
            .arg(Arg::new("batch").long("batch-size").visible_alias("b"))
            .arg(
                Arg::new("fa")
                    .long("flash-attn")
                    .alias("fa")
                    .num_args(0..=1)
                    .default_missing_value("auto"),
            )
            .arg(Arg::new("prompt").short('p').long("prompt"))
            .arg(
                Arg::new("quiet")
                    .long("quiet")
                    .short('q')
                    .action(ArgAction::SetTrue),
            );
        c.build();
        c
    }

    fn run(argv: &[&str]) -> Result<Vec<String>, String> {
        expand_single_dash_names(&cmd(), argv.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn a_multi_letter_single_dash_name_is_the_long_flag_it_names() {
        assert_eq!(
            run(&["-ngl", "99", "-fa", "-m", "x.gguf"]).unwrap(),
            ["--n-gpu-layers", "99", "--flash-attn", "-m", "x.gguf"]
        );
    }

    /// `-b` has no short form, only the one-letter long alias, so clap
    /// alone would refuse it; a real short is left for clap.
    #[test]
    fn a_one_letter_name_prefers_a_real_short_and_falls_back_to_a_long_alias() {
        assert_eq!(
            run(&["-b", "512", "-q"]).unwrap(),
            ["--batch-size", "512", "-q"]
        );
    }

    /// `-md draft.gguf` is what turned the model path into `d`.
    #[test]
    fn an_unknown_multi_letter_name_is_refused_instead_of_read_as_a_cluster() {
        let err = run(&["-md", "draft.gguf"]).unwrap_err();
        assert!(err.contains("`-md`"), "{err}");
    }

    /// Upstream's `is_truthy` / `is_falsey` / `is_autoy`, exactly, and
    /// nothing else: case matters there, so it matters here.
    ///
    /// Sabotage: add `"yes"` to the `On` arm; the last loop goes red.
    #[test]
    fn llama_switch_is_upstreams_three_lists_and_nothing_else() {
        for v in ["on", "enabled", "true", "1"] {
            assert_eq!(llama_switch(v), Some(Switch::On), "{v}");
        }
        for v in ["off", "disabled", "false", "0"] {
            assert_eq!(llama_switch(v), Some(Switch::Off), "{v}");
        }
        for v in ["auto", "-1"] {
            assert_eq!(llama_switch(v), Some(Switch::Auto), "{v}");
        }
        for v in ["yes", "no", "ON", "2", ""] {
            assert_eq!(llama_switch(v), None, "{v}");
        }
        assert!(check_flash_attn("auto").is_ok() && check_flash_attn(" on ").is_ok());
        assert!(check_flash_attn("off")
            .unwrap_err()
            .contains("backend property"));
        assert!(check_flash_attn("yes")
            .unwrap_err()
            .contains("expected on, off or auto"));
    }

    #[test]
    fn usage_is_help_on_the_command_and_its_subcommands() {
        let c = with_usage_alias(
            Command::new("t").subcommand(Command::new("sub").arg(Arg::new("x").long("x"))),
        );
        for argv in [&["t", "--usage"][..], &["t", "sub", "--usage"]] {
            let err = c.clone().try_get_matches_from(argv).unwrap_err();
            assert_eq!(err.kind(), clap::error::ErrorKind::DisplayHelp, "{argv:?}");
        }
    }

    #[test]
    fn values_and_numbers_pass_through_untouched() {
        assert_eq!(
            run(&["-p", "-ngl is a prompt", "-m", "-1", "--", "-zz"]).unwrap(),
            ["-p", "-ngl is a prompt", "-m", "-1", "--", "-zz"]
        );
        assert_eq!(
            run(&["--prompt", "-weird", "--prompt=x", "-xx"]).unwrap_err(),
            run(&["-xx"]).unwrap_err()
        );
    }
}
