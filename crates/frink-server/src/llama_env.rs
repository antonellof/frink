//! llama-server's environment variables (`LLAMA_ARG_*`), as aliases of
//! the flags they stand for.
//!
//! ADR 0005: `llama-server`'s flags AND its environment variables are
//! accepted. llama.cpp reads each variable as the flag it is declared
//! on (`common/arg.cpp`, `set_env`), applies it first, and lets the
//! command line override it. This module does exactly that, from ONE
//! table: each variable becomes its flag, appended to the argv only when
//! the command line did not already give that flag. Every effect a
//! variable has is then the flag's effect, through the one code path the
//! flag already has (`apply_cli_overrides`), so a variable cannot be
//! honoured differently from its flag or validated less strictly.
//!
//! frink's own `FRINK_*` variables stay as they were. A `LLAMA_ARG_*`
//! variable is a flag, so it wins over the `FRINK_*` variable that flag
//! lowers to, exactly as the flag itself would.
//!
//! The table is held to llama.cpp by
//! `crates/frink-cli/src/llama_flags.rs`: every entry must be the
//! variable llama.cpp declares on that flag, and every llama-server
//! flag frink accepts that HAS a variable must have its entry here.

use clap::Command;

/// `(llama.cpp env var, frink-server long flag)`.
pub const LLAMA_ENV_FLAGS: &[(&str, &str)] = &[
    ("LLAMA_ARG_MODEL", "model"),
    ("LLAMA_ARG_HF_REPO", "hf-repo"),
    ("LLAMA_ARG_HF_FILE", "hf-file"),
    ("LLAMA_ARG_CTX_SIZE", "ctx-size"),
    ("LLAMA_API_KEY", "api-key"),
    ("LLAMA_ARG_API_KEY_FILE", "api-key-file"),
    ("LLAMA_ARG_ALIAS", "alias"),
    ("LLAMA_ARG_CACHE_TYPE_K", "ctk"),
    ("LLAMA_ARG_JINJA", "jinja"),
    ("LLAMA_ARG_FLASH_ATTN", "flash-attn"),
    ("LLAMA_ARG_HOST", "host"),
    ("LLAMA_ARG_PORT", "port"),
    ("LLAMA_ARG_THREADS", "threads"),
    ("LLAMA_ARG_DEVICE", "device"),
    ("LLAMA_ARG_N_GPU_LAYERS", "n-gpu-layers"),
    ("LLAMA_ARG_CONT_BATCHING", "cont-batching"),
    ("LLAMA_ARG_N_PARALLEL", "parallel"),
    ("LLAMA_ARG_BATCH", "batch-size"),
    ("LLAMA_ARG_UBATCH", "ubatch-size"),
    ("LLAMA_ARG_SLOT_SAVE_PATH", "slot-save-path"),
    ("LLAMA_ARG_THINK_BUDGET", "reasoning-budget"),
    ("LLAMA_ARG_PREFILL_ASSISTANT", "prefill-assistant"),
];

/// llama.cpp's `common_arg_utils::is_truthy` / `is_falsey`.
fn switch_value(value: &str) -> Option<bool> {
    match value {
        "on" | "enabled" | "true" | "1" => Some(true),
        "off" | "disabled" | "false" | "0" => Some(false),
        _ => None,
    }
}

/// Appends the flag of every set variable in [`LLAMA_ENV_FLAGS`] to
/// `tail` (the arguments after the program or subcommand name), unless
/// `tail` already gives that flag.
///
/// `cmd` is the server's BUILT command. A switch (a flag with no value)
/// takes llama.cpp's truthy/falsy spellings; falsy selects its `--no-`
/// twin when there is one and does nothing otherwise, which is what
/// upstream does for a flag with no negation. The `LLAMA_ARG_NO_*`
/// spelling upstream also reads for a negatable flag means falsy.
pub fn append_llama_arg_env(
    cmd: &Command,
    mut tail: Vec<String>,
    env: impl Fn(&str) -> Option<String>,
) -> Result<Vec<String>, String> {
    let given: Vec<String> = tail
        .iter()
        .filter_map(|t| frink_core::argv::arg_for_token(cmd, t))
        .map(|a| a.get_id().to_string())
        .collect();
    for &(var, long) in LLAMA_ENV_FLAGS {
        let flag = format!("--{long}");
        let arg = frink_core::argv::arg_for_token(cmd, &flag)
            .ok_or_else(|| format!("{var} names --{long}, which this command does not have"))?;
        if given.iter().any(|id| id == arg.get_id().as_str()) {
            continue;
        }
        if arg.get_action().takes_values() {
            if let Some(value) = env(var) {
                tail.push(format!("{flag}={value}"));
            }
            continue;
        }
        let negated = var.replacen("LLAMA_ARG_", "LLAMA_ARG_NO_", 1);
        let (value, source) = match env(var) {
            Some(v) => (v, var.to_string()),
            None if negated != var && env(&negated).is_some() => ("0".to_string(), negated),
            None => continue,
        };
        match switch_value(value.trim()) {
            Some(true) => tail.push(flag),
            Some(false) => {
                let off = format!("--no-{long}");
                if frink_core::argv::arg_for_token(cmd, &off).is_some() {
                    tail.push(off);
                }
            }
            None => {
                return Err(format!(
                    "{source}={value}: expected one of on/off, true/false, 1/0, enabled/disabled"
                ))
            }
        }
    }
    Ok(tail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, Parser};

    fn built() -> Command {
        let mut cmd = crate::ServerArgs::command();
        cmd.build();
        cmd
    }

    fn with_env(tail: &[&str], vars: &[(&str, &str)]) -> Result<Vec<String>, String> {
        append_llama_arg_env(
            &built(),
            tail.iter().map(|s| s.to_string()).collect(),
            |k| {
                vars.iter()
                    .find(|(name, _)| *name == k)
                    .map(|(_, v)| v.to_string())
            },
        )
    }

    /// Every entry names a flag the server has. The function refuses at
    /// startup otherwise, and that refusal must never reach a user.
    #[test]
    fn every_entry_names_a_flag_the_server_has() {
        let cmd = built();
        for (var, long) in LLAMA_ENV_FLAGS {
            assert!(
                frink_core::argv::arg_for_token(&cmd, &format!("--{long}")).is_some(),
                "{var} -> --{long}"
            );
        }
    }

    /// llama.cpp applies the environment first and the command line
    /// over it, so a flag on the command line wins.
    #[test]
    fn a_variable_is_its_flag_unless_the_command_line_gave_that_flag() {
        assert_eq!(
            with_env(&[], &[("LLAMA_ARG_CTX_SIZE", "8192")]).unwrap(),
            ["--ctx-size=8192"]
        );
        assert_eq!(
            with_env(&["-c", "4096"], &[("LLAMA_ARG_CTX_SIZE", "8192")]).unwrap(),
            ["-c", "4096"]
        );
        assert_eq!(
            with_env(&["--np", "2"], &[("LLAMA_ARG_N_PARALLEL", "8")]).unwrap(),
            ["--np", "2"]
        );
    }

    /// A negative value must reach the flag as a value, not as a flag.
    #[test]
    fn a_negative_value_stays_a_value() {
        let tail = with_env(&[], &[("LLAMA_ARG_THINK_BUDGET", "-1")]).unwrap();
        let args =
            crate::ServerArgs::try_parse_from(std::iter::once("frink-server".into()).chain(tail));
        assert!(args.is_ok(), "{args:?}");
    }

    #[test]
    fn switches_take_llama_cpps_truthy_and_falsy_spellings() {
        assert_eq!(
            with_env(&[], &[("LLAMA_ARG_CONT_BATCHING", "on")]).unwrap(),
            ["--cont-batching"]
        );
        assert_eq!(
            with_env(&[], &[("LLAMA_ARG_CONT_BATCHING", "false")]).unwrap(),
            ["--no-cont-batching"]
        );
        assert_eq!(
            with_env(&[], &[("LLAMA_ARG_NO_CONT_BATCHING", "1")]).unwrap(),
            ["--no-cont-batching"]
        );
        let err = with_env(&[], &[("LLAMA_ARG_JINJA", "maybe")]).unwrap_err();
        assert!(err.contains("LLAMA_ARG_JINJA=maybe"), "{err}");
    }
}
