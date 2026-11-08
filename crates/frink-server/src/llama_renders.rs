//! frink-server's prompt for a request, held to llama-server's.
//!
//! `tools/oracle/chat_renders.py` renders each request in
//! `tests/llama_renders/shapes.json` through llama.cpp's own chat code at
//! the pin (`tools/oracle/chat_render.cpp`): llama-server's flag parser
//! and defaults, and the body handled as its `/v1/chat/completions`
//! handler handles it. Here the same request takes this server's path:
//! the flags parsed and resolved as `frink-server` resolves its own, the
//! body read as a [`ChatCompletionRequest`], then
//! `resolve_template_kwargs` and `render_prompt`. A byte that differs is
//! a token that differs.
//!
//! [`checked_in_renders_match_libllama`] runs on every PR over
//! `tests/llama_renders`, whose renders CI's oracle job regenerates from
//! the pin and fails on if one moved. [`census`], ignored, runs every
//! template llama.cpp bundles, from the renders `scripts/oracle.sh`
//! writes to `FRINK_LLAMA_RENDERS`. Both hold frink to
//! `known_differences.txt`, the pairs that differ today and the issue
//! tracking each: a new difference fails, and so does a listed one that
//! has stopped differing.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use clap::Parser as _;
use serde_json::Value;

use crate::chat_template::PromptTemplate;
use crate::ChatCompletionRequest;

#[derive(serde::Deserialize)]
struct Spec {
    bos_token: String,
    eos_token: String,
    shapes: Vec<Shape>,
}

#[derive(serde::Deserialize)]
struct Shape {
    name: String,
    flags: Vec<String>,
    body: Value,
}

fn spec() -> Spec {
    serde_json::from_str(include_str!("../tests/llama_renders/shapes.json"))
        .expect("tests/llama_renders/shapes.json")
}

/// What this server renders for one request, or why it refuses.
fn frink_prompt(source: &str, spec: &Spec, shape: &Shape) -> Result<String, String> {
    let argv = std::iter::once("frink-server".to_string()).chain(shape.flags.iter().cloned());
    let args = crate::ServerArgs::try_parse_from(argv).map_err(|e| format!("flags: {e}"))?;
    let flags = crate::cli::template_flags(&args).map_err(|e| format!("flags: {e}"))?;
    let template =
        PromptTemplate::as_served(source, &spec.bos_token, &spec.eos_token, flags.kwargs);
    let mut body = shape.body.clone();
    body["model"] = Value::from("oracle");
    let request: ChatCompletionRequest =
        serde_json::from_value(body).map_err(|e| format!("body: {e}"))?;
    let kwargs = request.resolve_template_kwargs(&template);
    request
        .render_prompt(
            &request.messages,
            &template,
            &request.tools,
            kwargs,
            "oracle",
        )
        .map_err(|e| format!("{e:?}"))
}

/// libllama's answer, as `chat_renders.py` wrote it.
fn llama_prompt(dir: &Path, template: &str, shape: &str) -> Result<String, String> {
    let rendered = dir.join(template);
    match std::fs::read(rendered.join(format!("{shape}.prompt"))) {
        Ok(bytes) => Ok(String::from_utf8(bytes).expect("a render is UTF-8")),
        Err(_) => Err(
            std::fs::read_to_string(rendered.join(format!("{shape}.error")))
                .unwrap_or_else(|_| panic!("{template}/{shape}: neither .prompt nor .error")),
        ),
    }
}

/// Where two renders part, with a little of each side.
fn first_difference(frink: &str, llama: &str) -> String {
    let at = frink
        .bytes()
        .zip(llama.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or(frink.len().min(llama.len()));
    let around = |s: &str| {
        let start = s.floor_char_boundary(at.saturating_sub(40));
        let end = s.ceil_char_boundary((at + 60).min(s.len()));
        format!("{:?}", &s[start..end])
    };
    format!(
        "byte {at}: frink {} | llama.cpp {}",
        around(frink),
        around(llama)
    )
}

/// The renders in `dir`: how many (template, request) pairs were
/// compared, the templates, and each pair where frink and libllama
/// disagree, keyed `template/request`.
struct Census {
    checked: usize,
    templates: BTreeSet<String>,
    differ: BTreeMap<String, String>,
}

fn census_of(dir: &Path) -> Census {
    let spec = spec();
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "jinja"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no templates in {}", dir.display());
    let mut census = Census {
        checked: 0,
        templates: BTreeSet::new(),
        differ: BTreeMap::new(),
    };
    for path in &paths {
        let name = path.file_stem().unwrap().to_str().unwrap().to_string();
        let source = std::fs::read_to_string(path).unwrap();
        for shape in &spec.shapes {
            census.checked += 1;
            let line = match (
                frink_prompt(&source, &spec, shape),
                llama_prompt(dir, &name, &shape.name),
            ) {
                (Ok(a), Ok(b)) if a == b => continue,
                (Err(_), Err(_)) => continue,
                (Ok(a), Ok(b)) => first_difference(&a, &b),
                (Err(e), Ok(_)) => format!("frink refuses what llama.cpp renders: {e}"),
                (Ok(_), Err(e)) => format!("frink renders what llama.cpp refuses: {}", e.trim()),
            };
            census.differ.insert(format!("{name}/{}", shape.name), line);
        }
        census.templates.insert(name);
    }
    census
}

/// `tests/llama_renders/known_differences.txt`: `template/request` to
/// the issue that tracks it.
fn known_differences() -> BTreeMap<String, String> {
    include_str!("../tests/llama_renders/known_differences.txt")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let (key, issue) = l.split_once(' ').expect("`template/request #issue`");
            (key.to_string(), issue.trim().to_string())
        })
        .collect()
}

/// Every difference must be a known one, and every known one for these
/// templates must still differ. `whole` is the census over everything
/// the pin bundles, where a known line naming a template it no longer
/// bundles is stale too.
fn hold_to_libllama(dir: &Path, whole: bool) {
    let census = census_of(dir);
    let known = known_differences();
    let unknown: Vec<String> = census
        .differ
        .iter()
        .filter(|(key, _)| !known.contains_key(*key))
        .map(|(key, line)| format!("{key}: {line}"))
        .collect();
    let stale: Vec<&String> = known
        .keys()
        .filter(|key| {
            let template = key.split('/').next().unwrap();
            (whole || census.templates.contains(template)) && !census.differ.contains_key(*key)
        })
        .collect();
    let mut by_issue: BTreeMap<&str, usize> = BTreeMap::new();
    for key in census.differ.keys() {
        let issue = known.get(key).map_or("not listed", String::as_str);
        *by_issue.entry(issue).or_default() += 1;
    }
    println!(
        "{} of {} prompts differ from libllama's: {by_issue:?}",
        census.differ.len(),
        census.checked
    );
    let mut report = String::new();
    if !unknown.is_empty() {
        let _ = writeln!(report, "{} prompts differ from libllama's:", unknown.len());
        for line in &unknown {
            let _ = writeln!(report, "  {line}");
        }
    }
    if !stale.is_empty() {
        let _ = writeln!(
            report,
            "{} lines of known_differences.txt now match libllama, delete them:",
            stale.len()
        );
        for key in &stale {
            let _ = writeln!(report, "  {key}");
        }
    }
    assert!(report.is_empty(), "{report}");
}

/// FRINK_TEST_CHAT_TEMPLATE_NOW pins `strftime_now`, to the time the
/// renders were made.
fn pin_clock(dir: &Path) {
    let at = std::fs::read_to_string(dir.join("rendered_at"))
        .expect("rendered_at, which chat_renders.py --all writes");
    // SAFETY: the census runs alone (ignored, run by name), and nothing
    // else in it reads the environment concurrently.
    unsafe { std::env::set_var("FRINK_TEST_CHAT_TEMPLATE_NOW", at.trim()) };
}

/// The templates #508 is about, and the requests that tell keeping a
/// thought from dropping it.
#[test]
fn checked_in_renders_match_libllama() {
    // No checked-in template prints the date (chat_renders.py refuses
    // one), so the clock is left alone.
    hold_to_libllama(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/llama_renders"),
        false,
    );
}

/// Every template llama.cpp bundles, from `scripts/oracle.sh`'s renders.
#[test]
#[ignore = "needs FRINK_LLAMA_RENDERS, the renders scripts/oracle.sh makes from the pin"]
fn census() {
    let dir = PathBuf::from(
        std::env::var("FRINK_LLAMA_RENDERS")
            .expect("FRINK_LLAMA_RENDERS: a chat_renders.py --all output"),
    );
    pin_clock(&dir);
    hold_to_libllama(&dir, true);
}
