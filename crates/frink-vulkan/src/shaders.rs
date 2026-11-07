//! The GLSL matvec shaders (`crates/frink-vulkan/shaders/*.comp`) and
//! their committed SPIR-V.
//!
//! The beachhead emitted SPIR-V words from Rust, which needed no tool
//! at all and did not scale: ~560 lines for the simplest kernel
//! (`docs/plans/vulkan-beachhead-verdict.md`, §2), and it was deleted
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
        block_bytes: 34,
        block_elems: 32,
    },
    MatvecShader {
        kind: "Q4_0",
        spv: include_bytes!("../shaders/spv/q4_0_matvec.spv"),
        block_bytes: 18,
        block_elems: 32,
    },
    MatvecShader {
        kind: "Q4_K",
        spv: include_bytes!("../shaders/spv/q4_k_matvec.spv"),
        block_bytes: 144,
        block_elems: 256,
    },
    MatvecShader {
        kind: "Q5_K",
        spv: include_bytes!("../shaders/spv/q5_k_matvec.spv"),
        block_bytes: 176,
        block_elems: 256,
    },
    MatvecShader {
        kind: "Q6_K",
        spv: include_bytes!("../shaders/spv/q6_k_matvec.spv"),
        block_bytes: 210,
        block_elems: 256,
    },
];

/// The shader for `kind`.
pub fn matvec_shader(kind: &str) -> Option<&'static MatvecShader> {
    MATVEC.iter().find(|s| s.kind == kind)
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
        for s in MATVEC {
            let w = words(s.spv);
            assert_eq!(w[0], 0x0723_0203, "{}: not SPIR-V", s.kind);
            let path = shader_dir()
                .join("spv")
                .join(format!("{}_matvec.spv", s.kind.to_lowercase()));
            match Command::new("spirv-val")
                .args(["--target-env", "vulkan1.0"])
                .arg(&path)
                .output()
            {
                Ok(out) => assert!(
                    out.status.success(),
                    "{}: spirv-val rejected it: {}",
                    s.kind,
                    String::from_utf8_lossy(&out.stderr)
                ),
                Err(_) => eprintln!("spirv-val not installed; magic checked only"),
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
        for s in MATVEC {
            let name = format!("{}_matvec.spv", s.kind.to_lowercase());
            let fresh = std::fs::read(out.join(&name)).unwrap();
            assert_eq!(
                fresh, s.spv,
                "{name} is not what its source compiles to: run shaders/build.sh"
            );
        }
        let _ = std::fs::remove_dir_all(&out);
    }

    #[test]
    fn every_shader_is_one_row_with_ggml_block_sizes() {
        for (kind, bytes, elems) in [
            ("Q8_0", 34, 32),
            ("Q4_0", 18, 32),
            ("Q4_K", 144, 256),
            ("Q5_K", 176, 256),
            ("Q6_K", 210, 256),
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
            MATVEC.len(),
            "a .spv in shaders/spv has no MATVEC row"
        );
    }
}
