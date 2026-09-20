//! What `--list-devices` prints, in one place.
//!
//! `frink --list-devices` and `frink-server --list-devices` are the
//! same llama.cpp-style flag and printed the same text from two
//! byte-identical copies, one in `frink-cli/src/run.rs` and one in
//! `frink-server/src/cli.rs`. A backend added to one would have been
//! missing from the other, silently, and the user-visible symptom
//! would be two binaries of the same engine disagreeing about what
//! hardware exists.
//!
//! KNOWN GAP, recorded rather than fixed here: neither copy listed
//! Vulkan, which frink has a backend for (`--features vulkan`). It is
//! not added in this function yet because `frink-vulkan` is reachable
//! only through `frink-core`'s optional feature and wiring that is a
//! change of its own; `docs/ROADMAP.md` carries the row.

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
}
