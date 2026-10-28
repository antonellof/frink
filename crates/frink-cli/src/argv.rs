//! `frink`'s argv before clap sees it: the implicit `run`, and
//! llama.cpp's single-dash names.
//!
//! Both halves are DERIVED from the clap `Command`, not from a list
//! beside it. There used to be two lists here, `SUBCOMMANDS` (which
//! words are subcommands, so `frink serve` is not an implicit `run`)
//! and a table of fourteen short spellings (`-ngl`, `-hf`, `-npp`, ...),
//! and a third table of the same kind in frink-server. The server's
//! rewrote `-np`/`-cb`/`-b` and this one did not, so `frink serve -np 4`
//! failed where `frink-server -np 4` worked. Now an argument's
//! `visible_alias = "np"` is the spelling, on every front end at once
//! ([`frink_core::argv`]).

use clap::{Command, CommandFactory};

use crate::Cli;

/// The root command, built, so globals are propagated into every
/// subcommand and help/version exist as arguments.
pub(crate) fn built_cli() -> Command {
    let mut cmd = Cli::command();
    cmd.build();
    cmd
}

/// Rewrite `frink -m …` into `frink run -m …`, and every llama.cpp
/// single-dash name into the long spelling clap parses.
///
/// Errors on a single-dash token of several letters that names no flag
/// of the subcommand, rather than letting clap read `-md x` as `-m d x`.
pub(crate) fn rewrite_llama_style_argv(args: Vec<String>) -> Result<Vec<String>, String> {
    if args.len() < 2 {
        return Ok(args);
    }
    let root = built_cli();
    // A `global = true` flag that takes no value is allowed to precede
    // the subcommand it applies to, so `frink --allow-multiple-instances
    // serve` has to find `serve` at position 2. One that took a value
    // would make position 2 its value, not a subcommand, so only
    // value-less globals are skipped.
    let skippable = |token: &str| {
        frink_core::argv::arg_for_token(&root, token)
            .is_some_and(|a| a.is_global_set() && !a.get_action().takes_values())
    };
    let first_word = args
        .iter()
        .skip(1)
        .position(|a| !skippable(a))
        .map(|offset| offset + 1)
        .unwrap_or(1);
    let first = args[first_word].as_str();
    if matches!(first, "-h" | "--help" | "-V" | "--version") {
        return Ok(args);
    }
    let (head, sub_name, tail) = match root.find_subcommand(first) {
        Some(sub) => (
            args[..=first_word].to_vec(),
            sub.get_name().to_string(),
            args[first_word + 1..].to_vec(),
        ),
        None => {
            let mut head = vec![args[0].clone(), "run".to_string()];
            head.extend(args[1..first_word].iter().cloned());
            (head, "run".to_string(), args[first_word..].to_vec())
        }
    };
    let sub = root
        .find_subcommand(&sub_name)
        .expect("the subcommand was just found by this name");
    // Without the feature, `serve` is a stub that swallows its
    // arguments to explain itself; they are nobody's flags to rewrite.
    let tail = if cfg!(feature = "serve") || sub_name != "serve" {
        frink_core::argv::expand_single_dash_names(sub, tail)?
    } else {
        tail
    };
    #[cfg(feature = "serve")]
    let tail = if sub_name == "serve" {
        frink_server::llama_env::append_llama_arg_env(sub, tail, |k| std::env::var(k).ok())?
    } else {
        tail
    };
    let mut out = head;
    out.extend(tail);
    Ok(out)
}
