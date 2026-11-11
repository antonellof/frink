//! `-dev` / `--device` and `-ngl 0`, turned into the backend switches,
//! in one place.
//!
//! `frink`, `frink-server` and `frink bench` each set `FRINK_METAL`,
//! `FRINK_METAL_ATTN` and `FRINK_CUDA` from their own copy of this
//! match. None of the copies knew `FRINK_VULKAN`, so `-ngl 0` and
//! `-dev cpu` left a user's `FRINK_VULKAN=1` in force and the "CPU" run
//! did its matvecs on the GPU. One copy, and [`SWITCHES`] lists every
//! variable it writes, so a fourth backend is one arm here.

use std::fmt;
use std::str::FromStr;

/// Where `-dev` asked the model to run.
///
/// llama.cpp's `-dev` takes a device list (`CUDA0,Vulkan1`); frink
/// places a model on one backend and takes its name. `docs/CLI.md`
/// lists that deviation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffloadDevice {
    Auto,
    None,
    Cpu,
    Metal,
    Cuda,
    Vulkan,
}

impl OffloadDevice {
    /// Every value, in the order `--help` lists them.
    pub const ALL: [OffloadDevice; 6] = [
        OffloadDevice::Auto,
        OffloadDevice::None,
        OffloadDevice::Cpu,
        OffloadDevice::Metal,
        OffloadDevice::Cuda,
        OffloadDevice::Vulkan,
    ];

    /// The spelling `-dev` accepts.
    pub fn as_str(self) -> &'static str {
        match self {
            OffloadDevice::Auto => "auto",
            OffloadDevice::None => "none",
            OffloadDevice::Cpu => "cpu",
            OffloadDevice::Metal => "metal",
            OffloadDevice::Cuda => "cuda",
            OffloadDevice::Vulkan => "vulkan",
        }
    }

    /// `-ngl 0` overrides `-dev`: no layer on a device means no device.
    pub fn with_offload(device: Option<Self>, offload_enabled: bool) -> Self {
        if offload_enabled {
            device.unwrap_or(OffloadDevice::Auto)
        } else {
            OffloadDevice::None
        }
    }
}

impl fmt::Display for OffloadDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for OffloadDevice {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|d| d.as_str().eq_ignore_ascii_case(s.trim()))
            .ok_or_else(|| {
                let names: Vec<&str> = Self::ALL.iter().map(|d| d.as_str()).collect();
                format!("expected one of {}", names.join(", "))
            })
    }
}

/// Every environment variable [`select`] writes.
pub const SWITCHES: [&str; 4] = [
    "FRINK_METAL",
    "FRINK_METAL_ATTN",
    "FRINK_CUDA",
    "FRINK_VULKAN",
];

/// What [`select`] writes for `device`, as `(variable, value)`. `None`
/// as the value means "leave it as the user set it". Pure, so a test
/// can hold every arm without touching the process environment.
pub fn switches_for(device: OffloadDevice) -> [(&'static str, Option<&'static str>); 4] {
    let [metal, attn, cuda, vulkan] = SWITCHES;
    match device {
        OffloadDevice::None | OffloadDevice::Cpu => [
            (metal, Some("0")),
            (attn, Some("0")),
            (cuda, Some("0")),
            (vulkan, Some("0")),
        ],
        // Vulkan stays opt-in under `auto`: unset means off, and a
        // user's `FRINK_VULKAN=1` is honoured. It is last in the
        // precedence, so CUDA or Metal still wins where present.
        OffloadDevice::Auto => [
            (metal, Some("auto")),
            (attn, None),
            (cuda, Some("auto")),
            (vulkan, None),
        ],
        OffloadDevice::Metal => [
            (metal, Some("1")),
            (attn, None),
            (cuda, Some("0")),
            (vulkan, Some("0")),
        ],
        OffloadDevice::Cuda => [
            (metal, Some("0")),
            (attn, Some("0")),
            (cuda, Some("1")),
            (vulkan, Some("0")),
        ],
        OffloadDevice::Vulkan => [
            (metal, Some("0")),
            (attn, Some("0")),
            (cuda, Some("0")),
            (vulkan, Some("1")),
        ],
    }
}

/// Refuse a named device this build or machine cannot run on, by name.
fn check_available(device: OffloadDevice) -> anyhow::Result<()> {
    match device {
        OffloadDevice::Metal => {
            #[cfg(not(feature = "metal"))]
            anyhow::bail!("Metal requested but this binary was built without --features metal");
            #[cfg(feature = "metal")]
            if !frink_metal::MetalProfile::detect().available {
                anyhow::bail!("Metal requested but no Metal device is available");
            }
        }
        OffloadDevice::Cuda => {
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("CUDA requested but this binary was built without --features cuda");
            #[cfg(feature = "cuda")]
            if !frink_cuda::HardwareProfile::detect().cuda_available {
                anyhow::bail!("CUDA requested but no CUDA device is available");
            }
        }
        OffloadDevice::Vulkan => {
            #[cfg(not(feature = "vulkan"))]
            anyhow::bail!("Vulkan requested but this binary was built without --features vulkan");
            #[cfg(feature = "vulkan")]
            if let Err(e) = frink_vulkan::device::probe() {
                anyhow::bail!("Vulkan requested but no Vulkan device is usable: {e}");
            }
        }
        OffloadDevice::Auto | OffloadDevice::None | OffloadDevice::Cpu => {}
    }
    Ok(())
}

/// Check `device` is usable, then set the backend switches for it.
///
/// # Safety
///
/// Writes the process environment. Call before any thread that reads
/// it exists: before the rayon pool, the server runtime and any model
/// load.
pub unsafe fn select(device: OffloadDevice) -> anyhow::Result<()> {
    check_available(device)?;
    for (name, value) in switches_for(device) {
        match value {
            // SAFETY: the caller guarantees no other thread exists.
            Some(v) => unsafe { std::env::set_var(name, v) },
            // `FRINK_METAL_ATTN` defaults on with Metal, but a pre-set
            // value is an ablation (`FRINK_METAL_ATTN=0 -ngl 99`) and
            // must hold.
            None if name == "FRINK_METAL_ATTN" && std::env::var_os(name).is_none() => {
                // SAFETY: as above.
                unsafe { std::env::set_var(name, "1") }
            }
            None => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(device: OffloadDevice, name: &str) -> Option<&'static str> {
        switches_for(device)
            .into_iter()
            .find(|(n, _)| *n == name)
            .unwrap()
            .1
    }

    #[test]
    fn every_value_round_trips_and_ignores_case() {
        for d in OffloadDevice::ALL {
            assert_eq!(d.as_str().parse::<OffloadDevice>(), Ok(d));
            assert_eq!(d.as_str().to_uppercase().parse::<OffloadDevice>(), Ok(d));
        }
        assert!("rocm".parse::<OffloadDevice>().is_err());
    }

    #[test]
    fn every_arm_writes_every_switch() {
        for d in OffloadDevice::ALL {
            let names: Vec<&str> = switches_for(d).iter().map(|(n, _)| *n).collect();
            assert_eq!(names, SWITCHES, "{d}");
        }
    }

    /// The bug this module exists for: `-ngl 0` and `-dev cpu` must
    /// turn Vulkan off as they turn Metal and CUDA off.
    #[test]
    fn the_host_turns_every_gpu_off_including_vulkan() {
        for d in [OffloadDevice::None, OffloadDevice::Cpu] {
            for name in SWITCHES {
                assert_eq!(value(d, name), Some("0"), "{d} {name}");
            }
        }
        assert_eq!(
            OffloadDevice::with_offload(Some(OffloadDevice::Vulkan), false),
            OffloadDevice::None
        );
    }

    #[test]
    fn a_named_gpu_turns_the_other_gpus_off() {
        let on = [
            (OffloadDevice::Metal, "FRINK_METAL"),
            (OffloadDevice::Cuda, "FRINK_CUDA"),
            (OffloadDevice::Vulkan, "FRINK_VULKAN"),
        ];
        for (device, own) in on {
            assert_eq!(value(device, own), Some("1"), "{device}");
            for (other, other_name) in on {
                if other != device {
                    assert_eq!(
                        value(device, other_name),
                        Some("0"),
                        "{device} {other_name}"
                    );
                }
            }
        }
    }

    #[test]
    fn auto_leaves_vulkan_opt_in() {
        assert_eq!(value(OffloadDevice::Auto, "FRINK_VULKAN"), None);
    }
}
