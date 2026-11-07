//! Detecting an AMD Ryzen AI NPU (XDNA), so `--list-devices` can say
//! one is present and that frink does not use it yet.
//!
//! frink has no NPU kernels (`docs/plans/amd.md`, phase N): the open
//! route to one is AIE kernels written in `mlir-aie`/IRON and loaded
//! through XRT, and FastFlowLM's NPU kernels are closed binaries over
//! its own model format. Detection needs neither: the Linux `amdxdna`
//! driver binds a PCI function whose IDs it lists
//! (`drivers/accel/amdxdna/amdxdna_pci_drv.c`), and sysfs exposes the
//! vendor, device, revision and bound driver of every PCI function.

use std::path::Path;

/// One NPU found in sysfs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmdNpu {
    /// PCI address, `0000:c5:00.1`.
    pub address: String,
    /// `npu1` (XDNA, Phoenix/Hawk Point) or `npu4`/`npu5`/`npu6`
    /// (XDNA2: Strix, Strix Halo, Kraken, Gorgon Point).
    pub generation: &'static str,
    /// The kernel driver bound to it, `amdxdna` when it is usable.
    pub driver: Option<String>,
}

/// `amdxdna_pci_drv.c`'s `pci_ids` and its `(device, revision) -> info`
/// table at the pin read on 2026-11-07.
fn generation(device: u32, revision: u32) -> Option<&'static str> {
    match (device, revision) {
        (0x1502, _) => Some("npu1 (XDNA)"),
        (0x17f0, 0x10) => Some("npu4 (XDNA2)"),
        (0x17f0, 0x11) => Some("npu5 (XDNA2)"),
        (0x17f0, 0x20) => Some("npu6 (XDNA2)"),
        (0x17f0 | 0x17f2 | 0x17f3 | 0x1b0b | 0x1b0c, _) => Some("XDNA2"),
        _ => None,
    }
}

fn read_hex(p: &Path) -> Option<u32> {
    let s = std::fs::read_to_string(p).ok()?;
    u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()
}

/// Every AMD NPU under `pci_root` (`/sys/bus/pci/devices` on Linux).
pub fn scan(pci_root: &Path) -> Vec<AmdNpu> {
    let Ok(entries) = std::fs::read_dir(pci_root) else {
        return Vec::new();
    };
    let mut out: Vec<AmdNpu> = entries
        .flatten()
        .filter_map(|e| {
            let dir = e.path();
            if read_hex(&dir.join("vendor"))? != 0x1022 {
                return None;
            }
            let device = read_hex(&dir.join("device"))?;
            let revision = read_hex(&dir.join("revision")).unwrap_or(0);
            let generation = generation(device, revision)?;
            let driver = std::fs::read_link(dir.join("driver"))
                .ok()
                .and_then(|l| l.file_name().map(|n| n.to_string_lossy().into_owned()));
            Some(AmdNpu {
                address: e.file_name().to_string_lossy().into_owned(),
                generation,
                driver,
            })
        })
        .collect();
    out.sort_by(|a, b| a.address.cmp(&b.address));
    out
}

/// The NPUs on this machine: Linux sysfs, and none elsewhere (Windows
/// exposes the NPU through its own driver stack, which this does not
/// read).
pub fn detect() -> Vec<AmdNpu> {
    if cfg!(target_os = "linux") {
        scan(Path::new("/sys/bus/pci/devices"))
    } else {
        Vec::new()
    }
}

/// The `--list-devices` line for one NPU.
pub fn describe(npu: &AmdNpu) -> String {
    let driver = match npu.driver.as_deref() {
        Some("amdxdna") => "driver amdxdna",
        Some(other) => {
            return format!(
                "AMD Ryzen AI NPU {} at {}: bound to {other}, not amdxdna; not used",
                npu.generation, npu.address
            )
        }
        None => "no driver bound (amdxdna is in Linux 7.0+, or amdxdna-dkms)",
    };
    format!(
        "AMD Ryzen AI NPU {} at {} ({driver}): detected, not used; frink has no NPU kernels yet \
         (docs/plans/amd.md)",
        npu.generation, npu.address
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn function(
        root: &Path,
        addr: &str,
        vendor: &str,
        device: &str,
        rev: &str,
        driver: Option<&str>,
    ) {
        let d = root.join(addr);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("vendor"), format!("{vendor}\n")).unwrap();
        std::fs::write(d.join("device"), format!("{device}\n")).unwrap();
        std::fs::write(d.join("revision"), format!("{rev}\n")).unwrap();
        if let Some(drv) = driver {
            let target = root.join("drivers").join(drv);
            std::fs::create_dir_all(&target).unwrap();
            #[cfg(unix)]
            std::os::unix::fs::symlink(&target, d.join("driver")).unwrap();
        }
    }

    /// A fake sysfs with a Strix NPU (driver bound), a Phoenix NPU with
    /// no driver, an AMD GPU and a non-AMD device: only the two NPUs are
    /// reported, each with its generation and driver.
    #[test]
    fn the_npus_are_found_and_nothing_else() {
        let root = std::env::temp_dir().join(format!("frink-npu-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        function(
            &root,
            "0000:c5:00.1",
            "0x1022",
            "0x17f0",
            "0x10",
            Some("amdxdna"),
        );
        function(&root, "0000:07:00.1", "0x1022", "0x1502", "0x00", None);
        function(
            &root,
            "0000:03:00.0",
            "0x1002",
            "0x744c",
            "0xc8",
            Some("amdgpu"),
        );
        function(
            &root,
            "0000:01:00.0",
            "0x10de",
            "0x2684",
            "0xa1",
            Some("nvidia"),
        );
        let found = scan(&root);
        assert_eq!(found.len(), 2, "{found:?}");
        assert_eq!(found[0].address, "0000:07:00.1");
        assert_eq!(found[0].generation, "npu1 (XDNA)");
        assert_eq!(found[1].generation, "npu4 (XDNA2)");
        #[cfg(unix)]
        assert_eq!(found[1].driver.as_deref(), Some("amdxdna"));
        let line = describe(&found[1]);
        assert!(
            line.contains("not used") && line.contains("amd.md"),
            "{line}"
        );
        assert!(describe(&found[0]).contains("no driver bound"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn no_sysfs_is_no_npu() {
        assert!(scan(Path::new("/nonexistent/sys/bus/pci/devices")).is_empty());
    }
}
