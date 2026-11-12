//! Every graph fixture, with each integer metadata value replaced by a
//! hostile one (0, 1, the largest the type holds, ...), must load and
//! decode or refuse by name (ADR 0002). A panic, a slice out of range or
//! a division by zero is a bug: the file is untrusted input.
//!
//! The values are patched in place in a copy of the fixture, at the byte
//! offset this file's own header walk finds, so the rest of the file is
//! the fixture's own bytes.

mod common;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

/// One integer metadata value: its key, its GGUF type tag and where its
/// bytes start in the file. For an array of integers, the first element.
struct IntSlot {
    key: String,
    tag: u32,
    offset: usize,
}

fn width(tag: u32) -> Option<usize> {
    match tag {
        2 | 3 => Some(2),
        4 | 5 => Some(4),
        10 | 11 => Some(8),
        _ => None,
    }
}

fn scalar_len(tag: u32) -> Option<usize> {
    match tag {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], at: usize) -> usize {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap()) as usize
}

/// Skips one value of type `tag` at `at`, recording integer slots.
fn walk_value(b: &[u8], mut at: usize, tag: u32, key: &str, out: &mut Vec<IntSlot>) -> usize {
    if let Some(n) = scalar_len(tag) {
        if width(tag).is_some() {
            out.push(IntSlot {
                key: key.to_string(),
                tag,
                offset: at,
            });
        }
        return at + n;
    }
    match tag {
        8 => at + 8 + u64_at(b, at),
        9 => {
            let elem = u32_at(b, at);
            let len = u64_at(b, at + 4);
            at += 12;
            for i in 0..len {
                let key = if i == 0 { key } else { "" };
                let before = out.len();
                at = walk_value(b, at, elem, key, out);
                if i > 0 {
                    out.truncate(before);
                }
            }
            at
        }
        other => panic!("{key}: unknown value type {other}"),
    }
}

fn int_slots(b: &[u8]) -> Vec<IntSlot> {
    let n_kv = u64_at(b, 16);
    let mut at = 24;
    let mut out = Vec::new();
    for _ in 0..n_kv {
        let klen = u64_at(b, at);
        let key = String::from_utf8_lossy(&b[at + 8..at + 8 + klen]).into_owned();
        at += 8 + klen;
        let tag = u32_at(b, at);
        at = walk_value(b, at + 4, tag, &key, &mut out);
    }
    out
}

/// The hostile values for a slot of `tag`, as little-endian bytes.
fn hostile(tag: u32) -> Vec<Vec<u8>> {
    match tag {
        2 => [0u16, 1, u16::MAX]
            .iter()
            .map(|v| v.to_le_bytes().to_vec())
            .collect(),
        3 => [0i16, -1, i16::MAX]
            .iter()
            .map(|v| v.to_le_bytes().to_vec())
            .collect(),
        4 => [0u32, 1, i32::MAX as u32, u32::MAX]
            .iter()
            .map(|v| v.to_le_bytes().to_vec())
            .collect(),
        5 => [0i32, -1, i32::MAX]
            .iter()
            .map(|v| v.to_le_bytes().to_vec())
            .collect(),
        10 => [0u64, 1, 1 << 40, u64::MAX]
            .iter()
            .map(|v| v.to_le_bytes().to_vec())
            .collect(),
        11 => [0i64, -1, i64::MAX]
            .iter()
            .map(|v| v.to_le_bytes().to_vec())
            .collect(),
        _ => Vec::new(),
    }
}

fn describe(v: &[u8]) -> String {
    let mut w = [0u8; 8];
    w[..v.len()].copy_from_slice(v);
    format!("{}", u64::from_le_bytes(w))
}

/// Loads `path` and decodes two tokens. `Err` is a refusal; a panic is
/// returned as its message.
fn load_and_decode(path: &Path) -> Result<Result<(), String>, String> {
    catch_unwind(AssertUnwindSafe(|| {
        let file = frink_gguf::GgufFile::open(path).map_err(|e| e.to_string())?;
        let config = frink_models::ModelConfig::from_gguf(&file).map_err(|e| e.to_string())?;
        let decoder = frink_models::Decoder::from_gguf(path, config).map_err(|e| e.to_string())?;
        let mut kv = decoder.config.new_kv_caches();
        let tok = 1.min(decoder.config.vocab_size.saturating_sub(1));
        let _ = decoder.forward_token(tok, 0, &mut kv);
        let _ = decoder.forward_token(tok, 1, &mut kv);
        Ok(())
    }))
    .map_err(|p| {
        p.downcast_ref::<String>()
            .cloned()
            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "<non-string panic>".into())
    })
}

fn scratch() -> PathBuf {
    let d = std::env::temp_dir().join(format!("frink-hostile-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The full sweep: every fixture, every integer key, every hostile value.
#[test]
#[ignore = "sweep of every fixture: run with --release -- --ignored hostile"]
fn hostile_integer_metadata_refuses_or_runs_but_never_panics() {
    common::backend();
    let dir = scratch();
    let only = std::env::var("FRINK_HOSTILE_ONLY").ok();
    let mut panics = Vec::new();
    let mut cases = 0usize;
    for name in common::graph_fixture_names() {
        if only.as_deref().is_some_and(|o| o != name) {
            continue;
        }
        let src = common::graph_fixture_path(&name);
        let bytes = std::fs::read(&src).unwrap();
        for slot in int_slots(&bytes) {
            for v in hostile(slot.tag) {
                let mut b = bytes.clone();
                b[slot.offset..slot.offset + v.len()].copy_from_slice(&v);
                let path = dir.join(format!("{name}.gguf"));
                std::fs::write(&path, &b).unwrap();
                cases += 1;
                if std::env::var_os("FRINK_HOSTILE_TRACE").is_some() {
                    use std::io::Write as _;
                    let mut e = std::io::stderr();
                    let _ = writeln!(e, "case {name} {}={}", slot.key, describe(&v));
                    let _ = e.flush();
                }
                if let Err(msg) = load_and_decode(&path) {
                    let first = msg.lines().next().unwrap_or("").to_string();
                    panics.push(format!("{name} {}={}: {first}", slot.key, describe(&v)));
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(cases > 0, "no cases ran");
    assert!(
        panics.is_empty(),
        "{} of {cases} hostile cases panicked:\n{}",
        panics.len(),
        panics.join("\n")
    );
}
