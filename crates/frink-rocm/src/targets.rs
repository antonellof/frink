//! The AMD architectures frink compiles for, and the ones it refuses.
//!
//! `frink-cuda`'s kernels assume a 32-lane warp in 37 places (`% 32`,
//! `& 31`, `lane = threadIdx.x % 32`, `blockDim.x / 32`). RDNA GPUs run
//! HIP code as wave32, so those kernels are correct there as written.
//! CDNA (Instinct) runs only wave64, where the same code reads the wrong
//! lanes: those targets are refused by name until each assumption is
//! made width-aware, as llama.cpp's are (`WARP_SIZE`,
//! `ggml_cuda_get_physical_warp_size`). The families follow llama.cpp's
//! `GGML_CUDA_CC_*` table (`ggml/src/ggml-cuda/common.cuh:69-101`).

/// Whether frink compiles for an architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Served {
    Yes,
    /// Refused, with the reason the refusal prints.
    Refused(&'static str),
}

/// One `gfx` target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmdArch {
    /// The LLVM target, as `hipDeviceProp_t::gcnArchName` reports it
    /// before its `:feature` suffixes.
    pub gfx: &'static str,
    /// The product family, for messages.
    pub family: &'static str,
    /// Lanes per wavefront under HIP's default.
    pub wave: u32,
    pub served: Served,
}

const WAVE64: &str =
    "a CDNA GPU runs HIP as wave64, and frink's kernels still assume a 32-lane warp \
     in 37 places (docs/plans/amd.md, phase R3); running them would read the wrong lanes";

/// Every architecture frink knows about. RDNA2 is the oldest served: it
/// is the first RDNA with `dp4a`, and the oldest ROCm still lists.
pub const ARCHS: &[AmdArch] = &[
    AmdArch {
        gfx: "gfx1030",
        family: "RDNA2 (RX 6800/6900)",
        wave: 32,
        served: Served::Yes,
    },
    AmdArch {
        gfx: "gfx1100",
        family: "RDNA3 (RX 7900)",
        wave: 32,
        served: Served::Yes,
    },
    AmdArch {
        gfx: "gfx1101",
        family: "RDNA3 (RX 7800/7700)",
        wave: 32,
        served: Served::Yes,
    },
    AmdArch {
        gfx: "gfx1102",
        family: "RDNA3 (RX 7600)",
        wave: 32,
        served: Served::Yes,
    },
    AmdArch {
        gfx: "gfx1150",
        family: "RDNA3.5 (Radeon 890M, Strix Point)",
        wave: 32,
        served: Served::Yes,
    },
    AmdArch {
        gfx: "gfx1151",
        family: "RDNA3.5 (Radeon 8060S, Strix Halo)",
        wave: 32,
        served: Served::Yes,
    },
    AmdArch {
        gfx: "gfx1200",
        family: "RDNA4 (RX 9060)",
        wave: 32,
        served: Served::Yes,
    },
    AmdArch {
        gfx: "gfx1201",
        family: "RDNA4 (RX 9070)",
        wave: 32,
        served: Served::Yes,
    },
    AmdArch {
        gfx: "gfx90a",
        family: "CDNA2 (MI210/MI250)",
        wave: 64,
        served: Served::Refused(WAVE64),
    },
    AmdArch {
        gfx: "gfx942",
        family: "CDNA3 (MI300)",
        wave: 64,
        served: Served::Refused(WAVE64),
    },
    AmdArch {
        gfx: "gfx950",
        family: "CDNA4 (MI350/MI355)",
        wave: 64,
        served: Served::Refused(WAVE64),
    },
];

/// The served targets, in table order.
pub fn served() -> impl Iterator<Item = &'static AmdArch> {
    ARCHS.iter().filter(|a| a.served == Served::Yes)
}

/// Why an architecture cannot be compiled for.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ArchRefused {
    #[error("{gfx} ({family}) is refused: {why}")]
    Refused {
        gfx: String,
        family: &'static str,
        why: &'static str,
    },
    #[error(
        "{0} is not an architecture frink knows: frink compiles for RDNA2-4 (gfx1030, gfx110x, \
         gfx115x, gfx120x); set FRINK_ROCM_ARCH only to one of those"
    )]
    Unknown(String),
}

/// The served entry for `gcn_arch_name`, which may carry feature
/// suffixes (`gfx1100:sramecc-:xnack-`), or why it is refused.
pub fn lookup(gcn_arch_name: &str) -> Result<&'static AmdArch, ArchRefused> {
    let gfx = gcn_arch_name.split(':').next().unwrap_or("").trim();
    match ARCHS.iter().find(|a| a.gfx == gfx) {
        Some(a) => match a.served {
            Served::Yes => Ok(a),
            Served::Refused(why) => Err(ArchRefused::Refused {
                gfx: gfx.to_string(),
                family: a.family,
                why,
            }),
        },
        None => Err(ArchRefused::Unknown(gfx.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rdna_is_served_at_wave32_and_cdna_refused_at_wave64() {
        for a in ARCHS {
            match a.served {
                Served::Yes => assert_eq!(a.wave, 32, "{} served at wave{}", a.gfx, a.wave),
                Served::Refused(_) => assert_eq!(a.wave, 64, "{}", a.gfx),
            }
        }
        assert!(served().count() >= 8);
    }

    #[test]
    fn feature_suffixes_are_ignored_and_refusals_name_why() {
        assert_eq!(lookup("gfx1100:sramecc-:xnack-").unwrap().gfx, "gfx1100");
        assert_eq!(
            lookup("gfx1151").unwrap().family,
            "RDNA3.5 (Radeon 8060S, Strix Halo)"
        );
        let err = lookup("gfx942:sramecc+:xnack-").unwrap_err().to_string();
        assert!(err.contains("wave64") && err.contains("MI300"), "{err}");
        assert!(matches!(lookup("gfx803"), Err(ArchRefused::Unknown(_))));
    }

    #[test]
    fn every_target_is_listed_once() {
        let mut g: Vec<&str> = ARCHS.iter().map(|a| a.gfx).collect();
        g.sort_unstable();
        let n = g.len();
        g.dedup();
        assert_eq!(g.len(), n);
    }
}
