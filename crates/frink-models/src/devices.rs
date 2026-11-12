//! What `--list-devices` prints, in one place.
//!
//! `frink --list-devices` and `frink-server --list-devices` are the
//! same llama.cpp-style flag and printed the same text from two
//! byte-identical copies, one in `frink-cli/src/run.rs` and one in
//! `frink-server/src/cli/mod.rs`. A backend added to one would have been
//! missing from the other, silently, and the user-visible symptom
//! would be two binaries of the same engine disagreeing about what
//! hardware exists.
//!
//! Vulkan is listed by a `--features vulkan` build, which can load the
//! Vulkan loader and ask it (`frink_vulkan::device::list`): one row per
//! device llama.cpp's rule makes visible, named `Vulkan<i>` as llama.cpp
//! names them, so `GGML_VK_VISIBLE_DEVICES` reads the same on both. A
//! build without the feature has no loader to ask, so it lists nothing
//! for Vulkan rather than guessing.

/// Print the detected devices, one per line, after a header.
pub fn print_available_devices() {
    println!("Available devices:");
    println!("  CPU");

    let metal = frink_metal::MetalProfile::detect();
    if let Some(name) = metal.device_name {
        println!("  Metal: {name}");
    }

    let cuda = frink_cuda::HardwareProfile::detect();
    if cuda.cuda_available {
        let name = cuda.cuda_device_name.as_deref().unwrap_or("unknown device");
        println!("  CUDA: {name}");
        if cuda.cuda_device_count > 1 {
            println!("        ({} devices detected)", cuda.cuda_device_count);
        }
    }

    #[cfg(feature = "vulkan")]
    for line in vulkan_lines(frink_vulkan::device::list()) {
        println!("  {line}");
    }

    // Listed so a Ryzen AI machine is not silent about its NPU, and says
    // plainly that frink does not use it yet.
    for npu in crate::amd_npu::detect() {
        println!("  {}", crate::amd_npu::describe(&npu));
    }
}

/// The `--list-devices` rows for what the Vulkan loader answered: each
/// visible device, or why there is none, so a Vulkan build on a machine
/// without a driver says so rather than printing nothing. frink runs on
/// `Vulkan0`.
#[cfg(feature = "vulkan")]
fn vulkan_lines(
    list: Result<Vec<frink_vulkan::device::Listed>, frink_vulkan::device::VulkanError>,
) -> Vec<String> {
    match list {
        Ok(devices) => devices
            .iter()
            .map(|d| {
                let (major, minor, patch) = d.api_version;
                let used = if d.index == 0 { ", used" } else { "" };
                format!(
                    "Vulkan{}: {} ({}, Vulkan {major}.{minor}.{patch}{used})",
                    d.index,
                    d.name,
                    d.kind.name()
                )
            })
            .collect(),
        Err(e) => vec![format!("Vulkan: none usable ({e})")],
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::*;

    #[test]
    fn every_visible_vulkan_device_is_a_row_named_as_llama_cpp_names_it() {
        use frink_vulkan::device::Listed;
        use frink_vulkan::select::DeviceKind;
        let rows = vulkan_lines(Ok(vec![
            Listed {
                index: 0,
                name: "AMD Radeon RX 7900 XTX".into(),
                kind: DeviceKind::Discrete,
                api_version: (1, 3, 287),
            },
            Listed {
                index: 1,
                name: "Intel(R) Graphics".into(),
                kind: DeviceKind::Integrated,
                api_version: (1, 3, 0),
            },
        ]));
        assert_eq!(
            rows,
            [
                "Vulkan0: AMD Radeon RX 7900 XTX (discrete GPU, Vulkan 1.3.287, used)",
                "Vulkan1: Intel(R) Graphics (integrated GPU, Vulkan 1.3.0)",
            ]
        );
        let none = vulkan_lines(Err(frink_vulkan::device::VulkanError::NoPhysicalDevice));
        assert_eq!(none.len(), 1);
        assert!(none[0].starts_with("Vulkan: none usable ("), "{}", none[0]);
    }
}
