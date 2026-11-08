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
//! Vulkan loader and ask it (`frink_vulkan::device::probe`). A build
//! without the feature has no loader to ask, so it lists nothing for
//! Vulkan rather than guessing.

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
    println!("  {}", vulkan_line(frink_vulkan::device::probe()));

    // Listed so a Ryzen AI machine is not silent about its NPU, and says
    // plainly that frink does not use it yet.
    for npu in crate::amd_npu::detect() {
        println!("  {}", crate::amd_npu::describe(&npu));
    }
}

/// The `--list-devices` row for what the Vulkan loader answered: the
/// device it would run on, or why there is none, so a Vulkan build on a
/// machine without a driver says so rather than printing nothing.
#[cfg(feature = "vulkan")]
fn vulkan_line(probe: Result<String, frink_vulkan::device::VulkanError>) -> String {
    match probe {
        Ok(device) => format!("Vulkan: {device}"),
        Err(e) => format!("Vulkan: none usable ({e})"),
    }
}

#[cfg(all(test, feature = "vulkan"))]
mod tests {
    use super::*;

    #[test]
    fn the_vulkan_row_names_the_device_or_why_there_is_none() {
        assert_eq!(
            vulkan_line(Ok("Apple M2 Pro (Vulkan 1.2.3)".into())),
            "Vulkan: Apple M2 Pro (Vulkan 1.2.3)"
        );
        let none = vulkan_line(Err(frink_vulkan::device::VulkanError::NoPhysicalDevice));
        assert!(none.starts_with("Vulkan: none usable ("), "{none}");
    }
}
