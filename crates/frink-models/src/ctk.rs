//! The `-ctk` / `--cache-type-k` value vocabulary, which is
//! llama.cpp's.
//!
//! Frink's command shapes are llama.cpp's, so the set of KV cache types
//! a user may name is llama.cpp's set and nothing else:
//! `common/arg.cpp:304-313` lists nine `ggml_type`s and
//! `kv_cache_type_from_str` THROWS on anything outside them
//! (`Unsupported cache type: ...`). A value that works there has to
//! work here, and a value that errors there must not silently become
//! f16 here.
//!
//! What frink's device store actually keeps is a smaller set, and the
//! two facts are separate on purpose:
//!
//! * [`is_llama_cpp_type`] answers "may a user name this", and the CLI
//!   refuses what it rejects, as llama.cpp does.
//! * [`crate::kv_budget::KvElem::from_ctk`] answers "what will the
//!   store actually keep", which is where a valid-but-unserved type
//!   lands on the nearest one that exists.
//!
//! Keeping them apart is what lets `--ctk q5_1` be ACCEPTED (llama.cpp
//! accepts it) and REPORTED as falling back, rather than either
//! erroring on a valid flag or pretending to honour it.

/// llama.cpp's `kv_cache_types`, in its order (`common/arg.cpp:304`).
///
/// `bf16` is in the list upstream and is not a store frink writes; it
/// resolves like the other unserved ones.
pub const LLAMA_CPP_KV_CACHE_TYPES: [&str; 9] = [
    "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
];

/// Frink's own additions, which llama.cpp has no spelling for.
///
/// `fp8` is the 8-bit store under the name a client may ask for it by;
/// its wire is Q8_0's, and the codes are absmax-scaled int8 rather than
/// real E4M3 (`frink_metal::kv_wire::MetalKvDtype::Fp8` says so).
pub const FRINK_EXTRA_KV_CACHE_TYPES: [&str; 2] = ["fp8", "e4m3"];

/// Whether llama.cpp would accept this `-ctk` value.
pub fn is_llama_cpp_type(value: &str) -> bool {
    let v = value.trim().to_ascii_lowercase();
    LLAMA_CPP_KV_CACHE_TYPES.contains(&v.as_str())
}

/// Whether frink accepts it: llama.cpp's set plus frink's own.
pub fn is_accepted(value: &str) -> bool {
    let v = value.trim().to_ascii_lowercase();
    is_llama_cpp_type(&v) || FRINK_EXTRA_KV_CACHE_TYPES.contains(&v.as_str())
}

/// The spellings frink has a store for, as opposed to merely accepts.
///
/// The rest of llama.cpp's list (`bf16`, `q4_1`, `iq4_nl`, `q5_0`,
/// `q5_1`) is accepted so a command line carries over, and resolves to
/// the nearest store that exists. A caller that reports to the user
/// has to say so, which is what this predicate is for: without it
/// `--ctk q5_1` printed `ctk=f16` and no word about the substitution,
/// which is the flag-accepted-and-ignored shape.
pub fn is_served(value: &str) -> bool {
    let v = value.trim().to_ascii_lowercase();
    matches!(v.as_str(), "f32" | "f16" | "q8_0" | "q4_0" | "fp8" | "e4m3")
}

/// The message for a value outside both sets, naming what is allowed
/// rather than falling back to f16 behind the user's back.
pub fn unsupported_message(value: &str) -> String {
    format!(
        "unsupported cache type: {}. Supported: {} (llama.cpp's set), plus {}",
        value.trim(),
        LLAMA_CPP_KV_CACHE_TYPES.join(", "),
        FRINK_EXTRA_KV_CACHE_TYPES.join(", "),
    )
}

/// The clap `value_parser` both front ends use.
///
/// It lives here rather than beside either flag because `frink run`
/// and `frink-server` set the SAME `FRINK_CTK` variable: two copies of
/// this function is two vocabularies waiting to drift, which is the
/// shape that let the server accept `nonsense` while the CLI refused
/// it. Returns the value normalised, so what reaches the environment
/// is what the parser validated.
pub fn parse_value(raw: &str) -> Result<String, String> {
    if is_accepted(raw) {
        Ok(raw.trim().to_ascii_lowercase())
    } else {
        Err(unsupported_message(raw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_budget::KvElem;

    /// The nine llama.cpp spells, verbatim. A user carrying a command
    /// line over must not meet an error frink invented.
    #[test]
    fn every_llama_cpp_cache_type_is_accepted() {
        for t in LLAMA_CPP_KV_CACHE_TYPES {
            assert!(is_accepted(t), "{t} is llama.cpp's and frink refused it");
            assert!(is_llama_cpp_type(t), "{t}");
        }
    }

    #[test]
    fn frinks_own_names_are_accepted_and_are_not_claimed_as_llama_cpps() {
        for t in FRINK_EXTRA_KV_CACHE_TYPES {
            assert!(is_accepted(t), "{t}");
            assert!(!is_llama_cpp_type(t), "{t} is not in llama.cpp's list");
        }
    }

    /// The point of refusing rather than defaulting: a typo used to
    /// become f16 silently, so a user asking for a smaller cache got
    /// the largest one and no word about it.
    #[test]
    fn an_unknown_value_is_refused_and_the_message_names_the_set() {
        assert!(!is_accepted("turbo4"));
        assert!(!is_accepted("q3_k"));
        let m = unsupported_message(" turbo4 ");
        assert!(m.contains("turbo4"), "{m}");
        assert!(m.contains("q4_0") && m.contains("iq4_nl"), "{m}");
    }

    /// Accepted, served and llama.cpp's are three different sets, and
    /// the banner reads all three. `q5_1` is llama.cpp's and accepted
    /// and NOT served; `fp8` is served and not llama.cpp's.
    #[test]
    fn served_is_a_smaller_set_than_accepted() {
        for t in ["f16", "f32", "q8_0", "q4_0", "fp8"] {
            assert!(is_served(t), "{t} has a store");
        }
        for t in ["bf16", "q4_1", "iq4_nl", "q5_0", "q5_1"] {
            assert!(is_accepted(t), "{t} is llama.cpp's");
            assert!(!is_served(t), "{t} has no frink store and must say so");
        }
    }

    #[test]
    fn the_parser_normalises_what_it_accepts_and_names_what_it_does_not() {
        assert_eq!(parse_value("  Q4_0 ").unwrap(), "q4_0");
        assert_eq!(parse_value("FP8").unwrap(), "fp8");
        let e = parse_value("turbo4").unwrap_err();
        assert!(e.contains("unsupported cache type"), "{e}");
    }

    #[test]
    fn case_and_padding_do_not_change_the_answer() {
        assert!(is_accepted("  Q4_0 "));
        assert!(is_accepted("F16"));
    }

    /// Accepted is not the same as served, and the split is the point:
    /// every accepted value has to resolve to a store that exists.
    #[test]
    fn every_accepted_value_resolves_to_a_store() {
        for t in LLAMA_CPP_KV_CACHE_TYPES
            .iter()
            .chain(FRINK_EXTRA_KV_CACHE_TYPES.iter())
        {
            let e = KvElem::from_ctk(t);
            assert!(
                matches!(e, KvElem::F32 | KvElem::F16 | KvElem::Q8_0 | KvElem::Q4_0),
                "{t} resolved to {e:?}"
            );
        }
        assert_eq!(KvElem::from_ctk("q4_0"), KvElem::Q4_0);
        assert_eq!(KvElem::from_ctk("q8_0"), KvElem::Q8_0);
        assert_eq!(KvElem::from_ctk("fp8"), KvElem::Q8_0);
        // Valid upstream, no frink store: the nearest one that exists.
        assert_eq!(KvElem::from_ctk("q5_1"), KvElem::F16);
    }
}
