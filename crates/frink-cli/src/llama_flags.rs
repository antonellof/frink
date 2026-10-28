//! The gap between llama.cpp's flags and frink's, as a number.
//!
//! ADR 0005: each frink entry point accepts its llama.cpp tool's flags
//! with the same names and meanings. This classifies every flag in
//! `tests/llama_cpp_flags.txt` (extracted from llama.cpp at the pin by
//! `scripts/llama_cpp_flags.py`) against frink's own clap commands, as
//! one of:
//!
//! - **accepted**: every spelling llama.cpp has for it parses here;
//! - **different**: listed, with its reason, in the "Deliberate
//!   differences" table of `docs/CLI.md`;
//! - **not accepted**: everything else.
//!
//! The accepted side is read from `Command::get_arguments` through the
//! same `frink_core::argv` lookup the binaries parse with, so a flag
//! cannot be counted as accepted here and refused on the command line.
//! The not-accepted count per tool is PINNED: closing a gap fails this
//! test until the pin is lowered to match, so the number only moves
//! down, and the list is printed so the next gap is one read away.
//!
//! What it cannot check is MEANING. "Accepted" means the spelling
//! parses; that it does what llama.cpp's does is a claim each alias
//! makes in its own doc comment, verified against `common/arg.cpp`
//! when it was added. A spelling that parses with a different meaning
//! belongs in the differences table, where this test then counts it.

use std::collections::BTreeMap;

use clap::Command;
use frink_core::argv::arg_for_token;

const TABLE: &str = include_str!("../tests/llama_cpp_flags.txt");

/// Not-accepted flags per llama.cpp tool. Lower these when a gap
/// closes; the test fails until you do, so they cannot creep back up.
const PINNED_NOT_ACCEPTED: &[(&str, usize)] = &[("cli", 163), ("server", 235), ("bench", 40)];

/// `LLAMA_ARG_*` variables on llama-server flags that frink-server does
/// not read. Same ratchet.
#[cfg(feature = "serve")]
const PINNED_SERVER_ENV_NOT_ACCEPTED: usize = 133;

#[derive(Debug)]
struct Row {
    tools: Vec<&'static str>,
    spellings: Vec<&'static str>,
    neg: Vec<&'static str>,
    // Read only by the server-env test, which needs the `serve` feature.
    #[cfg_attr(not(feature = "serve"), allow(dead_code))]
    env: Option<&'static str>,
}

impl Row {
    fn all_spellings(&self) -> impl Iterator<Item = &&'static str> {
        self.spellings.iter().chain(&self.neg)
    }
    fn label(&self) -> String {
        self.all_spellings().copied().collect::<Vec<_>>().join(",")
    }
}

fn rows() -> Vec<Row> {
    let list = |s: &'static str| -> Vec<&'static str> {
        if s == "-" {
            Vec::new()
        } else {
            s.split(',').collect()
        }
    };
    TABLE
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let cols: Vec<&'static str> = l.split('\t').collect();
            assert_eq!(cols.len(), 4, "malformed row: {l}");
            Row {
                tools: list(cols[0]),
                spellings: list(cols[1]),
                neg: list(cols[2]),
                env: (cols[3] != "-").then_some(cols[3]),
            }
        })
        .collect()
}

/// The frink command each llama.cpp tool maps to (ADR 0005's table),
/// built so globals and help are present.
fn tool_command(tool: &str) -> Option<Command> {
    let root = crate::argv::built_cli();
    let sub = match tool {
        "cli" => "run",
        "server" => "serve",
        "bench" => "bench",
        other => panic!("unknown tool column {other}"),
    };
    // `frink serve` is frink-server's own `ServerArgs`; without the
    // feature it is a stub that swallows everything and must not count.
    if sub == "serve" && !cfg!(feature = "serve") {
        return None;
    }
    root.find_subcommand(sub).cloned()
}

/// `(tool, first spelling)` from the "Deliberate differences" table in
/// `docs/CLI.md`: a row whose first cell is a tool name and whose
/// second is a backticked flag.
fn deliberate_differences() -> Vec<(String, String)> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/CLI.md");
    let doc = std::fs::read_to_string(&path).expect("docs/CLI.md");
    let start = doc
        .find("## Deliberate differences from llama.cpp")
        .expect("docs/CLI.md has a `## Deliberate differences from llama.cpp` section");
    let section = &doc[start..];
    let end = section[3..].find("\n## ").map_or(section.len(), |i| i + 3);
    let mut out = Vec::new();
    for line in section[..end].lines() {
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        if cells.len() < 4 {
            continue;
        }
        let flag = cells[2].trim_matches('`');
        let is_tool = |t: &str| PINNED_NOT_ACCEPTED.iter().any(|(name, _)| *name == t);
        if !flag.starts_with('-') || !cells[1].split(',').all(|t| is_tool(t.trim())) {
            continue;
        }
        for tool in cells[1].split(',').map(str::trim) {
            out.push((tool.to_string(), flag.to_string()));
        }
    }
    out
}

enum Class {
    Accepted,
    Different,
    /// The spellings that do not parse.
    NotAccepted(Vec<&'static str>),
}

fn classify(cmd: &Command, row: &Row, tool: &str, different: &[(String, String)]) -> Class {
    if different
        .iter()
        .any(|(t, f)| t == tool && row.all_spellings().any(|s| s == f))
    {
        return Class::Different;
    }
    let missing: Vec<&'static str> = row
        .all_spellings()
        .copied()
        .filter(|s| arg_for_token(cmd, s).is_none())
        .collect();
    if missing.is_empty() {
        Class::Accepted
    } else {
        Class::NotAccepted(missing)
    }
}

/// **The gap, per tool, pinned.** See the module doc.
#[test]
fn every_llama_cpp_flag_is_accepted_listed_as_different_or_counted() {
    let rows = rows();
    let different = deliberate_differences();
    let mut report = String::new();
    let mut counts = BTreeMap::new();
    for &(tool, pinned) in PINNED_NOT_ACCEPTED {
        let Some(cmd) = tool_command(tool) else {
            continue;
        };
        let mut not_accepted = Vec::new();
        let (mut accepted, mut differs, mut total) = (0, 0, 0);
        for row in rows.iter().filter(|r| r.tools.contains(&tool)) {
            total += 1;
            match classify(&cmd, row, tool, &different) {
                Class::Accepted => accepted += 1,
                Class::Different => differs += 1,
                Class::NotAccepted(missing) => {
                    let partial = missing.len() < row.all_spellings().count();
                    not_accepted.push(if partial {
                        format!("{} (missing {})", row.label(), missing.join(","))
                    } else {
                        row.label()
                    });
                }
            }
        }
        report.push_str(&format!(
            "\n{tool}: {total} llama.cpp flags, {accepted} accepted, {differs} deliberately \
             different, {} not accepted (pinned {pinned}):\n",
            not_accepted.len()
        ));
        for f in &not_accepted {
            report.push_str(&format!("  {f}\n"));
        }
        counts.insert(tool, (not_accepted.len(), pinned));
    }
    println!("{report}");
    for (tool, (now, pinned)) in counts {
        assert!(
            now <= pinned,
            "{tool}: {now} llama.cpp flags are not accepted, above the pin of {pinned}. A flag \
             stopped parsing, or a difference row was dropped from docs/CLI.md.{report}"
        );
        assert_eq!(
            now, pinned,
            "{tool}: the gap closed to {now}; lower PINNED_NOT_ACCEPTED in \
             crates/frink-cli/src/llama_flags.rs to {now} so it cannot reopen.{report}"
        );
    }
}

/// A difference row naming no llama.cpp flag of its tool is a gate that
/// can never fire: it reads as an explained difference and explains
/// nothing. And one naming a flag frink ALSO accepts with llama.cpp's
/// meaning would hide that it has none, so each row must match a real
/// row of its own tool.
#[test]
fn every_deliberate_difference_names_a_real_llama_cpp_flag_of_its_tool() {
    let rows = rows();
    for (tool, flag) in deliberate_differences() {
        assert!(
            rows.iter()
                .any(|r| r.tools.contains(&tool.as_str()) && r.all_spellings().any(|s| *s == flag)),
            "docs/CLI.md lists `{flag}` as a deliberate difference for `{tool}`, and llama.cpp's \
             {tool} has no such flag at the pin"
        );
    }
}

/// llama-server's environment variables, against the one table that
/// maps them (`frink_server::llama_env::LLAMA_ENV_FLAGS`).
///
/// Both directions. Every entry must be the variable llama.cpp declares
/// on the flag the entry names, or `LLAMA_ARG_X=…` would set a flag
/// upstream does not attach `X` to. And every accepted llama-server flag
/// that HAS a variable must have its entry, or the variable is the one
/// spelling of that flag frink ignores.
#[cfg(feature = "serve")]
#[test]
fn llama_server_env_vars_map_to_the_flag_llama_cpp_declares_them_on() {
    let rows = rows();
    let cmd = tool_command("server").expect("serve is built");
    let different = deliberate_differences();
    let table = frink_server::llama_env::LLAMA_ENV_FLAGS;
    for (var, long) in table {
        let arg = arg_for_token(&cmd, &format!("--{long}"))
            .unwrap_or_else(|| panic!("{var} -> --{long}: no such frink-server flag"));
        let row = rows
            .iter()
            .filter(|r| r.tools.contains(&"server"))
            .find(|r| {
                r.all_spellings()
                    .any(|s| arg_for_token(&cmd, s).is_some_and(|a| a.get_id() == arg.get_id()))
            })
            .unwrap_or_else(|| panic!("--{long} is no llama-server flag, so {var} maps nothing"));
        assert_eq!(
            row.env,
            Some(*var),
            "{var} -> --{long}, but llama.cpp declares {:?} on {}",
            row.env,
            row.label()
        );
    }
    let mut missing = Vec::new();
    for row in rows.iter().filter(|r| r.tools.contains(&"server")) {
        let Some(var) = row.env else { continue };
        if table.iter().any(|(v, _)| *v == var) {
            continue;
        }
        if matches!(classify(&cmd, row, "server", &different), Class::Accepted) {
            panic!(
                "{} is accepted by frink-server and its variable {var} is not in LLAMA_ENV_FLAGS",
                row.label()
            );
        }
        missing.push(var);
    }
    println!(
        "server env: {} of {} llama-server variables not read:\n  {}",
        missing.len(),
        rows.iter()
            .filter(|r| r.tools.contains(&"server") && r.env.is_some())
            .count(),
        missing.join("\n  ")
    );
    assert!(
        missing.len() <= PINNED_SERVER_ENV_NOT_ACCEPTED,
        "{} llama-server variables are not read, above the pin of \
         {PINNED_SERVER_ENV_NOT_ACCEPTED}",
        missing.len()
    );
    assert_eq!(
        missing.len(),
        PINNED_SERVER_ENV_NOT_ACCEPTED,
        "the env gap closed; lower PINNED_SERVER_ENV_NOT_ACCEPTED to {}",
        missing.len()
    );
}
