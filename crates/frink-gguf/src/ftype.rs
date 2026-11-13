//! `general.file_type`, as llama.cpp names it.
//!
//! The values are `llama_ftype` (`llama.h`) and the strings are
//! `llama_model_ftype_name`'s (`llama-model.cpp`), so a banner or a
//! `/props` reads the same as llama.cpp's for the same file. One table
//! for every caller: `frink`'s load banner and `frink-server`'s
//! `/props` both read it.

/// The name llama.cpp gives file type `v`, or `None` for a value this
/// table does not carry. A caller decides what an unknown value reads
/// as; this never guesses the nearest row.
pub fn llama_ftype_name(v: u64) -> Option<&'static str> {
    Some(match v {
        0 => "all F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K - Medium",
        11 => "Q3_K - Small",
        12 => "Q3_K - Medium",
        13 => "Q3_K - Large",
        14 => "Q4_K - Small",
        15 => "Q4_K - Medium",
        16 => "Q5_K - Small",
        17 => "Q5_K - Medium",
        18 => "Q6_K",
        19 => "IQ2_XXS - 2.0625 bpw",
        20 => "IQ2_XS - 2.3125 bpw",
        21 => "Q2_K - Small",
        22 => "IQ3_XS - 3.3 bpw",
        23 => "IQ3_XXS - 3.0625 bpw",
        24 => "IQ1_S - 1.5625 bpw",
        25 => "IQ4_NL - 4.5 bpw",
        26 => "IQ3_S - 3.4375 bpw",
        27 => "IQ3_M - 3.66 bpw",
        28 => "IQ2_S - 2.5 bpw",
        29 => "IQ2_M - 2.7 bpw",
        30 => "IQ4_XS - 4.25 bpw",
        31 => "IQ1_M - 1.75 bpw",
        32 => "BF16",
        36 => "TQ1_0 - 1.69 bpw ternary",
        37 => "TQ2_0 - 2.06 bpw ternary",
        38 => "MXFP4 MoE",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_known_type_has_llama_cpps_name_and_an_unknown_one_has_none() {
        assert_eq!(llama_ftype_name(15), Some("Q4_K - Medium"));
        assert_eq!(llama_ftype_name(7), Some("Q8_0"));
        assert_eq!(llama_ftype_name(99), None);
    }
}
