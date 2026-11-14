//! The GLSL matvec shaders (`crates/frink-vulkan/shaders/*.comp`) and
//! their committed SPIR-V.
//!
//! The beachhead emitted SPIR-V words from Rust, which needed no tool
//! at all and did not scale: ~560 lines for the simplest kernel
//! (`bc5074aa:docs/plans/archive/vulkan-beachhead-verdict.md`, §2), and it was deleted
//! once its Q8_0 had a GLSL twin here. These are GLSL,
//! compiled by `glslangValidator` (`shaders/build.sh`) into `shaders/spv/`,
//! and the `.spv` files are committed, so a frink build needs no shader
//! compiler and no C++ toolchain, which this repo has refused twice. Two
//! tests keep the bytes honest: every module validates as Vulkan 1.0
//! SPIR-V, and the committed module is what the source compiles to,
//! checked whenever the local `glslangValidator` is the version that
//! wrote them (`spv/GENERATOR`), as a golden is checked against the
//! oracle that produced it.
//!
//! Every shader keeps `Runtime::matvec`'s interface (one workgroup of
//! [`LOCAL_SIZE_X`] per row; `matvec_common.glsl`) and decodes one
//! `frink_quant` block layout; its scalar twin is that crate's
//! `dot_*_f32_scalar`.

/// One quant kind's matvec shader.
#[derive(Debug, Clone, Copy)]
pub struct MatvecShader {
    /// `QuantKind::name()`.
    pub kind: &'static str,
    /// The compiled module, as committed.
    pub spv: &'static [u8],
    /// The same source compiled with `-DGATHER`: one row of the table,
    /// dequantized (`shaders/matvec_common.glsl`).
    pub gather: &'static [u8],
    pub block_bytes: usize,
    pub block_elems: usize,
}

/// `local_size_x` in every shader (`matvec_common.glsl`).
pub const LOCAL_SIZE_X: u32 = 64;

/// Every kind the Vulkan matvec runs.
pub const MATVEC: &[MatvecShader] = &[
    MatvecShader {
        kind: "Q8_0",
        spv: include_bytes!("../shaders/spv/q8_0_matvec.spv"),
        gather: include_bytes!("../shaders/spv/q8_0_gather.spv"),
        block_bytes: 34,
        block_elems: 32,
    },
    MatvecShader {
        kind: "Q4_0",
        spv: include_bytes!("../shaders/spv/q4_0_matvec.spv"),
        gather: include_bytes!("../shaders/spv/q4_0_gather.spv"),
        block_bytes: 18,
        block_elems: 32,
    },
    MatvecShader {
        kind: "Q4_K",
        spv: include_bytes!("../shaders/spv/q4_k_matvec.spv"),
        gather: include_bytes!("../shaders/spv/q4_k_gather.spv"),
        block_bytes: 144,
        block_elems: 256,
    },
    MatvecShader {
        kind: "Q5_K",
        spv: include_bytes!("../shaders/spv/q5_k_matvec.spv"),
        gather: include_bytes!("../shaders/spv/q5_k_gather.spv"),
        block_bytes: 176,
        block_elems: 256,
    },
    MatvecShader {
        kind: "Q6_K",
        spv: include_bytes!("../shaders/spv/q6_k_matvec.spv"),
        gather: include_bytes!("../shaders/spv/q6_k_gather.spv"),
        block_bytes: 210,
        block_elems: 256,
    },
    MatvecShader {
        kind: "Q4_1",
        spv: include_bytes!("../shaders/spv/q4_1_matvec.spv"),
        gather: include_bytes!("../shaders/spv/q4_1_gather.spv"),
        block_bytes: 20,
        block_elems: 32,
    },
    MatvecShader {
        kind: "Q5_0",
        spv: include_bytes!("../shaders/spv/q5_0_matvec.spv"),
        gather: include_bytes!("../shaders/spv/q5_0_gather.spv"),
        block_bytes: 22,
        block_elems: 32,
    },
    MatvecShader {
        kind: "Q5_1",
        spv: include_bytes!("../shaders/spv/q5_1_matvec.spv"),
        gather: include_bytes!("../shaders/spv/q5_1_gather.spv"),
        block_bytes: 24,
        block_elems: 32,
    },
    MatvecShader {
        kind: "Q2_K",
        spv: include_bytes!("../shaders/spv/q2_k_matvec.spv"),
        gather: include_bytes!("../shaders/spv/q2_k_gather.spv"),
        block_bytes: 84,
        block_elems: 256,
    },
    MatvecShader {
        kind: "Q3_K",
        spv: include_bytes!("../shaders/spv/q3_k_matvec.spv"),
        gather: include_bytes!("../shaders/spv/q3_k_gather.spv"),
        block_bytes: 110,
        block_elems: 256,
    },
    MatvecShader {
        kind: "IQ4_NL",
        spv: include_bytes!("../shaders/spv/iq4_nl_matvec.spv"),
        gather: include_bytes!("../shaders/spv/iq4_nl_gather.spv"),
        block_bytes: 18,
        block_elems: 32,
    },
    MatvecShader {
        kind: "IQ4_XS",
        spv: include_bytes!("../shaders/spv/iq4_xs_matvec.spv"),
        gather: include_bytes!("../shaders/spv/iq4_xs_gather.spv"),
        block_bytes: 136,
        block_elems: 256,
    },
    MatvecShader {
        kind: "MXFP4",
        spv: include_bytes!("../shaders/spv/mxfp4_matvec.spv"),
        gather: include_bytes!("../shaders/spv/mxfp4_gather.spv"),
        block_bytes: 17,
        block_elems: 32,
    },
    MatvecShader {
        kind: "TQ1_0",
        spv: include_bytes!("../shaders/spv/tq1_0_matvec.spv"),
        gather: include_bytes!("../shaders/spv/tq1_0_gather.spv"),
        block_bytes: 54,
        block_elems: 256,
    },
    MatvecShader {
        kind: "PTQ1_0",
        spv: include_bytes!("../shaders/spv/ptq1_0_matvec.spv"),
        gather: include_bytes!("../shaders/spv/ptq1_0_gather.spv"),
        block_bytes: 28,
        block_elems: 128,
    },
    MatvecShader {
        kind: "IQ2_XXS",
        spv: include_bytes!("../shaders/spv/iq2_xxs_matvec.spv"),
        gather: include_bytes!("../shaders/spv/iq2_xxs_gather.spv"),
        block_bytes: 66,
        block_elems: 256,
    },
    MatvecShader {
        kind: "IQ2_XS",
        spv: include_bytes!("../shaders/spv/iq2_xs_matvec.spv"),
        gather: include_bytes!("../shaders/spv/iq2_xs_gather.spv"),
        block_bytes: 74,
        block_elems: 256,
    },
    MatvecShader {
        kind: "IQ2_S",
        spv: include_bytes!("../shaders/spv/iq2_s_matvec.spv"),
        gather: include_bytes!("../shaders/spv/iq2_s_gather.spv"),
        block_bytes: 82,
        block_elems: 256,
    },
    MatvecShader {
        kind: "IQ3_XXS",
        spv: include_bytes!("../shaders/spv/iq3_xxs_matvec.spv"),
        gather: include_bytes!("../shaders/spv/iq3_xxs_gather.spv"),
        block_bytes: 98,
        block_elems: 256,
    },
    MatvecShader {
        kind: "IQ3_S",
        spv: include_bytes!("../shaders/spv/iq3_s_matvec.spv"),
        gather: include_bytes!("../shaders/spv/iq3_s_gather.spv"),
        block_bytes: 110,
        block_elems: 256,
    },
    MatvecShader {
        kind: "IQ1_S",
        spv: include_bytes!("../shaders/spv/iq1_s_matvec.spv"),
        gather: include_bytes!("../shaders/spv/iq1_s_gather.spv"),
        block_bytes: 50,
        block_elems: 256,
    },
    MatvecShader {
        kind: "IQ1_M",
        spv: include_bytes!("../shaders/spv/iq1_m_matvec.spv"),
        gather: include_bytes!("../shaders/spv/iq1_m_gather.spv"),
        block_bytes: 56,
        block_elems: 256,
    },
    // Not a quant kind: the decode stack's F32 operand
    // (`WeightMatrix::vulkan_stack_job`). No `QuantKind` is named "F32",
    // so no claim table can reach it.
    MatvecShader {
        kind: "F32",
        spv: include_bytes!("../shaders/spv/f32_matvec.spv"),
        gather: include_bytes!("../shaders/spv/f32_gather.spv"),
        block_bytes: 4,
        block_elems: 1,
    },
];

/// Word offsets into the codebook table buffer (binding 3), the layout
/// `shaders/iq_common.glsl` reads. A 64-bit grid row is two words, low
/// first; a sign byte is one word.
pub mod iq_offsets {
    pub const KSIGNS: usize = 0;
    pub const IQ2XXS: usize = KSIGNS + 128;
    pub const IQ2XS: usize = IQ2XXS + 2 * 256;
    pub const IQ2S: usize = IQ2XS + 2 * 512;
    pub const IQ3XXS: usize = IQ2S + 2 * 1024;
    pub const IQ3S: usize = IQ3XXS + 256;
    pub const IQ1S: usize = IQ3S + 512;
    pub const END: usize = IQ1S + 2 * 2048;
}

/// The codebook kinds' tables as the words of that buffer, straight from
/// `frink_quant::iq_tables`: the one copy of each table in the tree.
pub fn iq_table_words() -> Vec<u32> {
    use frink_quant::iq_tables as t;
    let mut w = Vec::with_capacity(iq_offsets::END);
    let wide = |w: &mut Vec<u32>, rows: &[u64]| {
        for r in rows {
            w.push(*r as u32);
            w.push((*r >> 32) as u32);
        }
    };
    w.extend(t::KSIGNS_IQ2XS.iter().map(|&b| u32::from(b)));
    wide(&mut w, &t::IQ2XXS_GRID);
    wide(&mut w, &t::IQ2XS_GRID);
    wide(&mut w, &t::IQ2S_GRID);
    w.extend_from_slice(&t::IQ3XXS_GRID);
    w.extend_from_slice(&t::IQ3S_GRID);
    wide(&mut w, &t::IQ1S_GRID);
    assert_eq!(w.len(), iq_offsets::END);
    w
}

/// The shader for `kind`.
pub fn matvec_shader(kind: &str) -> Option<&'static MatvecShader> {
    MATVEC.iter().find(|s| s.kind == kind)
}

/// A shader that is not a matvec: it keeps the matvec interface's three
/// bindings and push-constant block, so the one pipeline layout serves
/// it, and decodes no block layout.
#[derive(Debug, Clone, Copy)]
pub struct ElementwiseShader {
    /// The name its pipeline is cached under, and its file stem.
    pub name: &'static str,
    pub spv: &'static [u8],
}

/// `h = silu(g) * u`, between a fused FFN's gate/up and its down.
pub const SWIGLU: ElementwiseShader = ElementwiseShader {
    name: "swiglu",
    spv: include_bytes!("../shaders/spv/swiglu.spv"),
};

macro_rules! elementwise {
    ($($konst:ident => $name:literal),* $(,)?) => {
        $(
            #[doc = concat!("`shaders/", $name, ".comp`.")]
            pub const $konst: ElementwiseShader = ElementwiseShader {
                name: $name,
                spv: include_bytes!(concat!("../shaders/spv/", $name, ".spv")),
            };
        )*
        /// Every elementwise shader, for the tests that cover every module.
        pub const ELEMENTWISE: &[ElementwiseShader] = &[SWIGLU, $($konst),*];
    };
}

elementwise! {
    RMS_NORM => "rms_norm",
    ADD => "add",
    ROPE => "rope",
    KV_STORE => "kv_store",
    ATTN_SCORES => "attn_scores",
    ATTN_VALUES => "attn_values",
}

/// Every committed module: its file name under `shaders/spv/` and its
/// bytes. The one list the module tests walk, so a shader cannot be
/// added to a table and skipped by validation.
pub fn modules() -> Vec<(String, &'static [u8])> {
    let matvecs = MATVEC.iter().flat_map(|s| {
        let stem = s.kind.to_lowercase();
        [
            (format!("{stem}_matvec.spv"), s.spv),
            (format!("{stem}_gather.spv"), s.gather),
        ]
    });
    let elementwise = ELEMENTWISE
        .iter()
        .map(|s| (format!("{}.spv", s.name), s.spv));
    matvecs.chain(elementwise).collect()
}

/// A module's bytes as the words `vkCreateShaderModule` takes.
pub fn words(spv: &[u8]) -> Vec<u32> {
    assert_eq!(spv.len() % 4, 0, "SPIR-V is whole words");
    spv.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::process::Command;

    fn shader_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders")
    }

    /// Every committed module is SPIR-V (magic, whole words) and, when
    /// `spirv-val` is installed, valid for Vulkan 1.0.
    #[test]
    fn every_committed_module_is_valid_vulkan_spirv() {
        for (name, spv) in modules() {
            let w = words(spv);
            assert_eq!(w[0], 0x0723_0203, "{name}: not SPIR-V");
            let path = shader_dir().join("spv").join(&name);
            match Command::new("spirv-val")
                .args(["--target-env", "vulkan1.0"])
                .arg(&path)
                .output()
            {
                Ok(out) => assert!(
                    out.status.success(),
                    "{name}: spirv-val rejected it: {}",
                    String::from_utf8_lossy(&out.stderr)
                ),
                Err(_) => crate::skipped("spirv-val not installed; magic checked only"),
            }
        }
    }

    /// The committed `.spv` is what `build.sh` writes from the source,
    /// byte for byte, when the local `glslangValidator` is the version
    /// recorded in `spv/GENERATOR`. Another version emits different (and
    /// equally valid) bytes, so it is not compared, and says so.
    #[test]
    fn committed_spirv_is_what_the_source_compiles_to() {
        let dir = shader_dir();
        let pinned = std::fs::read_to_string(dir.join("spv/GENERATOR")).unwrap();
        let local = match Command::new("glslangValidator").arg("--version").output() {
            Ok(o) => String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .trim_start_matches("Glslang Version: ")
                .to_string(),
            Err(_) => {
                eprintln!("glslangValidator not installed; drift not checked");
                return;
            }
        };
        if local.trim() != pinned.trim() {
            eprintln!(
                "glslangValidator {local} is not the pinned {}; drift not checked",
                pinned.trim()
            );
            return;
        }
        let out = std::env::temp_dir().join(format!("frink-spv-{}", std::process::id()));
        let status = Command::new(dir.join("build.sh"))
            .arg(&out)
            .status()
            .unwrap();
        assert!(status.success(), "build.sh failed");
        for (name, spv) in modules() {
            let fresh = std::fs::read(out.join(&name)).unwrap();
            assert_eq!(
                fresh, spv,
                "{name} is not what its source compiles to: run shaders/build.sh"
            );
        }
        let _ = std::fs::remove_dir_all(&out);
    }

    /// `iq_common.glsl` spells the table offsets as `#define`s; they are
    /// the Rust ones, and each table sits where its offset says.
    #[test]
    fn the_glsl_table_offsets_are_the_rust_ones() {
        use frink_quant::iq_tables as t;
        let src = include_str!("../shaders/iq_common.glsl");
        let glsl = |name: &str| -> usize {
            let line = src
                .lines()
                .find(|l| l.starts_with(&format!("#define {name} ")))
                .unwrap_or_else(|| panic!("no #define {name}"));
            line.rsplit(' ')
                .next()
                .unwrap()
                .trim_end_matches('u')
                .parse()
                .unwrap()
        };
        for (name, want) in [
            ("IQ_KSIGNS", iq_offsets::KSIGNS),
            ("IQ_GRID_IQ2XXS", iq_offsets::IQ2XXS),
            ("IQ_GRID_IQ2XS", iq_offsets::IQ2XS),
            ("IQ_GRID_IQ2S", iq_offsets::IQ2S),
            ("IQ_GRID_IQ3XXS", iq_offsets::IQ3XXS),
            ("IQ_GRID_IQ3S", iq_offsets::IQ3S),
            ("IQ_GRID_IQ1S", iq_offsets::IQ1S),
        ] {
            assert_eq!(glsl(name), want, "{name}");
        }
        let w = iq_table_words();
        let wide =
            |at: usize, i: usize| u64::from(w[at + 2 * i]) | u64::from(w[at + 2 * i + 1]) << 32;
        assert_eq!(w[iq_offsets::KSIGNS + 127], u32::from(t::KSIGNS_IQ2XS[127]));
        assert_eq!(wide(iq_offsets::IQ2XXS, 255), t::IQ2XXS_GRID[255]);
        assert_eq!(wide(iq_offsets::IQ2XS, 511), t::IQ2XS_GRID[511]);
        assert_eq!(wide(iq_offsets::IQ2S, 1023), t::IQ2S_GRID[1023]);
        assert_eq!(w[iq_offsets::IQ3XXS + 255], t::IQ3XXS_GRID[255]);
        assert_eq!(w[iq_offsets::IQ3S + 511], t::IQ3S_GRID[511]);
        assert_eq!(wide(iq_offsets::IQ1S, 2047), t::IQ1S_GRID[2047]);
    }

    #[test]
    fn every_shader_is_one_row_with_ggml_block_sizes() {
        // Sizes from frink_quant, the CPU's own constants, so a row
        // typed wrong here cannot agree with itself.
        for (kind, bytes, elems) in [
            (
                "Q8_0",
                frink_quant::Q8_0_BLOCK_BYTES,
                frink_quant::Q8_0_BLOCK_ELEMS,
            ),
            (
                "Q4_0",
                frink_quant::Q4_0_BLOCK_BYTES,
                frink_quant::Q4_0_BLOCK_ELEMS,
            ),
            (
                "Q4_K",
                frink_quant::Q4_K_BLOCK_BYTES,
                frink_quant::Q4_K_BLOCK_ELEMS,
            ),
            (
                "Q5_K",
                frink_quant::Q5_K_BLOCK_BYTES,
                frink_quant::Q5_K_BLOCK_ELEMS,
            ),
            (
                "Q6_K",
                frink_quant::Q6_K_BLOCK_BYTES,
                frink_quant::Q6_K_BLOCK_ELEMS,
            ),
            (
                "Q4_1",
                frink_quant::Q4_1_BLOCK_BYTES,
                frink_quant::Q4_1_BLOCK_ELEMS,
            ),
            (
                "Q5_0",
                frink_quant::Q5_0_BLOCK_BYTES,
                frink_quant::Q5_0_BLOCK_ELEMS,
            ),
            (
                "Q5_1",
                frink_quant::Q5_1_BLOCK_BYTES,
                frink_quant::Q5_1_BLOCK_ELEMS,
            ),
            (
                "Q2_K",
                frink_quant::Q2_K_BLOCK_BYTES,
                frink_quant::Q2_K_BLOCK_ELEMS,
            ),
            (
                "Q3_K",
                frink_quant::Q3_K_BLOCK_BYTES,
                frink_quant::Q3_K_BLOCK_ELEMS,
            ),
            (
                "IQ4_NL",
                frink_quant::IQ4_NL_BLOCK_BYTES,
                frink_quant::IQ4_NL_BLOCK_ELEMS,
            ),
            (
                "IQ4_XS",
                frink_quant::IQ4_XS_BLOCK_BYTES,
                frink_quant::IQ4_XS_BLOCK_ELEMS,
            ),
            (
                "MXFP4",
                frink_quant::MXFP4_GGUF_BLOCK_BYTES,
                frink_quant::MXFP4_GGUF_BLOCK_ELEMS,
            ),
            (
                "TQ1_0",
                frink_quant::ternary::TQ1_0_BLOCK_BYTES,
                frink_quant::ternary::TQ1_0_BLOCK_ELEMS,
            ),
            (
                "PTQ1_0",
                frink_quant::ternary::PTQ1_0_BLOCK_BYTES,
                frink_quant::ternary::PTQ1_0_BLOCK_ELEMS,
            ),
            (
                "IQ2_XXS",
                frink_quant::IQ2_XXS_BLOCK_BYTES,
                frink_quant::IQ2_XXS_BLOCK_ELEMS,
            ),
            (
                "IQ2_XS",
                frink_quant::IQ2_XS_BLOCK_BYTES,
                frink_quant::IQ2_XS_BLOCK_ELEMS,
            ),
            (
                "IQ2_S",
                frink_quant::IQ2_S_BLOCK_BYTES,
                frink_quant::IQ2_S_BLOCK_ELEMS,
            ),
            (
                "IQ3_XXS",
                frink_quant::IQ3_XXS_BLOCK_BYTES,
                frink_quant::IQ3_XXS_BLOCK_ELEMS,
            ),
            (
                "IQ3_S",
                frink_quant::IQ3_S_BLOCK_BYTES,
                frink_quant::IQ3_S_BLOCK_ELEMS,
            ),
            (
                "IQ1_S",
                frink_quant::IQ1_S_BLOCK_BYTES,
                frink_quant::IQ1_S_BLOCK_ELEMS,
            ),
            (
                "IQ1_M",
                frink_quant::IQ1_M_BLOCK_BYTES,
                frink_quant::IQ1_M_BLOCK_ELEMS,
            ),
            ("F32", 4, 1),
        ] {
            let s = matvec_shader(kind).unwrap();
            assert_eq!((s.block_bytes, s.block_elems), (bytes, elems), "{kind}");
        }
        // No orphan: every compiled module has its row, so a shader
        // cannot be built and then silently never dispatched.
        let compiled = std::fs::read_dir(shader_dir().join("spv"))
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .is_some_and(|x| x == "spv")
            })
            .count();
        assert_eq!(
            compiled,
            modules().len(),
            "a .spv in shaders/spv has no MATVEC or ELEMENTWISE row"
        );
    }
}
