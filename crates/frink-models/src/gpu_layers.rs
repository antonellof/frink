//! llama.cpp's `-ngl` / `--n-gpu-layers` value, for every front end.
//!
//! `frink run` and `frink-server` each had their own copy of this type,
//! and only `frink run`'s refused a partial count: `frink-server -ngl
//! 10` on a 32-layer model offloaded all 32, the divergence the run
//! copy had been fixed for. One type, one refusal.

use std::fmt;
use std::str::FromStr;

/// What `-ngl` was asked for: llama.cpp's `auto`, `all`, or a count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuLayers {
    Auto,
    All,
    Count(u32),
}

impl GpuLayers {
    /// Whether any layer is to be offloaded.
    pub fn offload_enabled(self) -> bool {
        !matches!(self, Self::Count(0))
    }

    /// Reject a PARTIAL offload rather than silently offloading
    /// everything.
    ///
    /// llama.cpp's `-ngl N` puts exactly `N` layers in VRAM and runs the
    /// rest on the CPU (`common/arg.cpp`), which is how people fit a
    /// model that does not otherwise fit. frink parses the count and
    /// then reads only `offload_enabled()`, a bool -- so `--ngl 10` on a
    /// 32-layer model offloaded all 32.
    ///
    /// That is the worst shape of divergence: same flag, same value, no
    /// error, and the failure lands as an out-of-memory on the machine
    /// the flag existed to accommodate.
    ///
    /// Partial offload is a real feature and not implemented here, so
    /// this REFUSES and names it. `0` (all CPU) and any count at or
    /// above the layer count (all GPU) are exact, and stay accepted.
    pub fn check_supported(self, n_layers: usize) -> anyhow::Result<()> {
        if let Self::Count(n) = self {
            let n = n as usize;
            if n > 0 && n < n_layers {
                anyhow::bail!(
                    "--ngl {n} asks for a PARTIAL offload ({n} of {n_layers} layers), which \
                     frink does not implement -- it would silently offload all {n_layers}. \
                     Use `--ngl 0` for CPU only, or `--ngl {n_layers}` / `--ngl all` for \
                     every layer."
                );
            }
        }
        Ok(())
    }
}

impl FromStr for GpuLayers {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "all" => Ok(Self::All),
            _ => value
                .parse::<u32>()
                .map(Self::Count)
                .map_err(|_| "expected 0, a positive integer, 'auto', or 'all'".into()),
        }
    }
}

impl fmt::Display for GpuLayers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::All => f.write_str("all"),
            Self::Count(value) => value.fmt(f),
        }
    }
}
