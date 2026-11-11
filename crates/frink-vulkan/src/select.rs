//! Which Vulkan devices frink may use, decided the way llama.cpp's
//! `ggml_vk_instance_init` decides it, with no device needed.
//!
//! llama.cpp at the pin (`ggml/src/ggml-vulkan/ggml-vulkan.cpp`,
//! `ggml_vk_instance_init`):
//!
//! - `GGML_VK_VISIBLE_DEVICES` set: exactly the listed indices into the
//!   loader's enumeration, in the order given, separated by commas or
//!   spaces. An index past the end is an error.
//! - Unset: every discrete and every integrated GPU, in enumeration
//!   order. A CPU implementation (lavapipe, SwiftShader) or a virtual
//!   GPU is left out, so a machine with a Radeon and lavapipe installed
//!   runs on the Radeon.
//!
//! `Vulkan0` is the first entry of the result. frink runs on that one.
//!
//! Two of llama.cpp's checks are not made here, and `docs/CLI.md`
//! lists them as deviations. It drops a device without 16-bit storage
//! buffers, which its shaders need and frink's (byte extraction only)
//! do not. It also keeps one entry per device UUID when two drivers
//! expose one GPU (RADV and AMDVLK), which needs a Vulkan 1.1 instance;
//! frink asks for 1.0.

/// What kind of device the loader reported (`VkPhysicalDeviceType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Discrete,
    Integrated,
    Virtual,
    Cpu,
    Other,
}

impl DeviceKind {
    /// Whether llama.cpp uses a device of this kind when
    /// `GGML_VK_VISIBLE_DEVICES` is unset.
    pub fn is_default_visible(self) -> bool {
        matches!(self, DeviceKind::Discrete | DeviceKind::Integrated)
    }

    pub fn name(self) -> &'static str {
        match self {
            DeviceKind::Discrete => "discrete GPU",
            DeviceKind::Integrated => "integrated GPU",
            DeviceKind::Virtual => "virtual GPU",
            DeviceKind::Cpu => "CPU",
            DeviceKind::Other => "other",
        }
    }
}

/// llama.cpp's environment variable, honoured unchanged (ADR 0005).
pub const VISIBLE_ENV: &str = "GGML_VK_VISIBLE_DEVICES";

/// Why the visible list could not be built.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SelectError {
    #[error("{VISIBLE_ENV}: {0:?} is not a device index")]
    NotAnIndex(String),
    #[error("{VISIBLE_ENV}: device index {index} is out of range; the loader reports {count}")]
    OutOfRange { index: usize, count: usize },
}

/// Indices into the loader's enumeration that frink may use, in
/// llama.cpp's order. `kinds` is the enumeration; `visible_env` is the
/// value of `GGML_VK_VISIBLE_DEVICES`, if set.
///
/// llama.cpp reads the variable with `operator>>`, which stops quietly
/// at the first token that is not a number. Here such a token is an
/// error: a typo must not silently pick fewer devices (ADR 0002).
pub fn visible(kinds: &[DeviceKind], visible_env: Option<&str>) -> Result<Vec<usize>, SelectError> {
    let Some(list) = visible_env else {
        return Ok(kinds
            .iter()
            .enumerate()
            .filter(|(_, k)| k.is_default_visible())
            .map(|(i, _)| i)
            .collect());
    };
    list.split([',', ' '])
        .filter(|t| !t.is_empty())
        .map(|t| {
            let index: usize = t
                .parse()
                .map_err(|_| SelectError::NotAnIndex(t.to_string()))?;
            if index >= kinds.len() {
                return Err(SelectError::OutOfRange {
                    index,
                    count: kinds.len(),
                });
            }
            Ok(index)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::DeviceKind::*;
    use super::*;

    #[test]
    fn unset_keeps_discrete_and_integrated_gpus_in_order() {
        let kinds = [Cpu, Integrated, Virtual, Discrete, Other];
        assert_eq!(visible(&kinds, None).unwrap(), vec![1, 3]);
    }

    #[test]
    fn unset_on_a_cpu_driver_alone_is_empty() {
        // A machine with only lavapipe: llama.cpp finds no device
        // unless GGML_VK_VISIBLE_DEVICES names it.
        assert!(visible(&[Cpu], None).unwrap().is_empty());
        assert_eq!(visible(&[Cpu], Some("0")).unwrap(), vec![0]);
    }

    #[test]
    fn the_variable_picks_exactly_the_listed_indices_in_its_order() {
        let kinds = [Discrete, Cpu, Integrated];
        assert_eq!(visible(&kinds, Some("2,0")).unwrap(), vec![2, 0]);
        assert_eq!(visible(&kinds, Some("1 2")).unwrap(), vec![1, 2]);
        assert!(visible(&kinds, Some("")).unwrap().is_empty());
    }

    #[test]
    fn a_bad_index_is_an_error_not_a_shorter_list() {
        let kinds = [Discrete, Integrated];
        assert_eq!(
            visible(&kinds, Some("0,2")),
            Err(SelectError::OutOfRange { index: 2, count: 2 })
        );
        assert_eq!(
            visible(&kinds, Some("0,x")),
            Err(SelectError::NotAnIndex("x".into()))
        );
    }
}
