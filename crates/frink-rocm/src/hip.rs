//! The HIP runtime, opened with `dlopen`: device discovery and one
//! launch, the Q8_0 matvec, checked against its scalar twin.
//!
//! A beachhead in the sense `frink-vulkan` is one: it proves frink can
//! reach an AMD GPU from Rust, compile its own kernel for it, upload a
//! quantized weight verbatim, launch, and read back the right answer.
//! It is not a backend; phase R2 (`docs/plans/amd.md`) wires one in.
//!
//! **Never executed on an AMD GPU.** The launch test is `#[ignore]`d
//! with that reason until it has run on one.

use std::ffi::{c_char, c_int, c_uint, c_void, CString};

use libloading::Library;

use crate::rtc::Hiprtc;
use crate::targets::lookup;
use crate::RocmError;

const LIB: &str = "libamdhip64";
const ENV: &str = "FRINK_HIP_LIB";
const CANDIDATES: &[&str] = &[
    "libamdhip64.so.6",
    "libamdhip64.so.7",
    "libamdhip64.so",
    "/opt/rocm/lib/libamdhip64.so.6",
    "/opt/rocm/lib/libamdhip64.so",
    "amdhip64_6.dll",
    "amdhip64_7.dll",
];

/// `hipDeviceProp_tR0600`, read as bytes at the offsets ROCm's own
/// header gives (measured with `offsetof` against ROCm 6.4's
/// `hip/hip_runtime_api.h`): the struct behind the versioned symbol
/// `hipGetDevicePropertiesR0600@@hip_6.0`, whose layout is fixed for
/// every ROCm that exports that symbol. Read by offset rather than
/// mirrored as a Rust struct because only four fields are needed and a
/// 1472-byte mirror would be four hundred lines nobody can check.
pub mod props_r0600 {
    pub const SIZE: usize = 1472;
    pub const NAME: (usize, usize) = (0, 256);
    pub const TOTAL_GLOBAL_MEM: usize = 288;
    pub const WARP_SIZE: usize = 308;
    pub const GCN_ARCH_NAME: (usize, usize) = (1160, 256);
}

/// One AMD GPU, as the HIP runtime reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    pub index: i32,
    pub name: String,
    /// `gfx1100:sramecc-:xnack-` and the like.
    pub gcn_arch: String,
    pub total_mem: u64,
    pub warp_size: i32,
}

/// The fields [`DeviceInfo`] reads, out of a `hipDeviceProp_tR0600`.
pub fn parse_props(index: i32, raw: &[u8]) -> DeviceInfo {
    use props_r0600::*;
    let cstr = |(off, len): (usize, usize)| {
        let s = &raw[off..off + len];
        let end = s.iter().position(|&b| b == 0).unwrap_or(len);
        String::from_utf8_lossy(&s[..end]).into_owned()
    };
    DeviceInfo {
        index,
        name: cstr(NAME),
        gcn_arch: cstr(GCN_ARCH_NAME),
        total_mem: u64::from_le_bytes(
            raw[TOTAL_GLOBAL_MEM..TOTAL_GLOBAL_MEM + 8]
                .try_into()
                .unwrap(),
        ),
        warp_size: i32::from_le_bytes(raw[WARP_SIZE..WARP_SIZE + 4].try_into().unwrap()),
    }
}

type Ptr = *mut c_void;

/// The HIP runtime library.
pub struct Hip {
    lib: Library,
}

impl Hip {
    /// Opens `libamdhip64` (`FRINK_HIP_LIB` overrides the search) and
    /// initialises the runtime.
    pub fn open() -> Result<Self, RocmError> {
        let hip = Self {
            lib: crate::open_library(LIB, ENV, CANDIDATES)?,
        };
        let init: unsafe extern "C" fn(c_uint) -> c_int = hip.sym("hipInit")?;
        // SAFETY: `hipInit(0)` takes no pointers; the signature is the
        // one `hip_runtime_api.h` declares.
        hip.check("hipInit", unsafe { init(0) })?;
        Ok(hip)
    }

    fn sym<T: Copy>(&self, name: &'static str) -> Result<T, RocmError> {
        let cname = format!("{name}\0");
        // SAFETY: `T` is the C signature `hip_runtime_api.h` declares for
        // `name`, and `self.lib` outlives every call made through it.
        unsafe {
            self.lib
                .get::<T>(cname.as_bytes())
                .map(|s| *s)
                .map_err(|e| RocmError::Symbol {
                    lib: LIB,
                    symbol: name,
                    detail: e.to_string(),
                })
        }
    }

    fn check(&self, call: &'static str, code: c_int) -> Result<(), RocmError> {
        if code == 0 {
            return Ok(());
        }
        let message = self
            .sym::<unsafe extern "C" fn(c_int) -> *const c_char>("hipGetErrorString")
            .ok()
            // SAFETY: hipGetErrorString returns a static C string for any
            // error code.
            .map(|f| {
                unsafe { std::ffi::CStr::from_ptr(f(code)) }
                    .to_string_lossy()
                    .into_owned()
            })
            .unwrap_or_default();
        Err(RocmError::Hip {
            call,
            code,
            message,
        })
    }

    /// Every AMD GPU the runtime sees.
    pub fn devices(&self) -> Result<Vec<DeviceInfo>, RocmError> {
        let count: unsafe extern "C" fn(*mut c_int) -> c_int = self.sym("hipGetDeviceCount")?;
        let props: unsafe extern "C" fn(*mut c_void, c_int) -> c_int =
            self.sym("hipGetDevicePropertiesR0600")?;
        let mut n: c_int = 0;
        // SAFETY: `n` is a valid out-pointer.
        self.check("hipGetDeviceCount", unsafe { count(&mut n) })?;
        let mut out = Vec::new();
        for i in 0..n {
            // u64 words so the buffer is 8-byte aligned, as the struct is.
            let mut buf = vec![0u64; props_r0600::SIZE / 8];
            // SAFETY: `buf` is `SIZE` bytes, the size of the struct this
            // versioned symbol writes.
            self.check("hipGetDevicePropertiesR0600", unsafe {
                props(buf.as_mut_ptr() as *mut c_void, i)
            })?;
            // SAFETY: reinterpreting the initialised u64 words as bytes.
            let raw =
                unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, props_r0600::SIZE) };
            out.push(parse_props(i, raw));
        }
        Ok(out)
    }

    /// `y[row] = dot(dequant(W[row]), x)` on device `dev`, with
    /// `frink-cuda`'s `q8_0_matvec` compiled for that device's
    /// architecture: one 256-thread block per row, as `frink-cuda`
    /// launches it. The kernel reduces through shared memory and assumes
    /// no warp width, so it is also correct on wave64.
    pub fn q8_0_matvec(
        &self,
        dev: &DeviceInfo,
        weights: &[u8],
        x: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>, RocmError> {
        let arch = lookup(&dev.gcn_arch)?;
        let blocks_per_row = x.len() / 32;
        let row_bytes = blocks_per_row * 34;
        assert_eq!(
            weights.len(),
            rows * row_bytes,
            "weights are rows x Q8_0 blocks"
        );
        let kind = frink_cuda::matvec_kinds::KINDS
            .iter()
            .find(|k| k.name == "Q8_0")
            .expect("frink-cuda has a Q8_0 matvec");
        let code = Hiprtc::open()?.compile("q8_0_matvec", kind.source(), arch.gfx)?;

        let set_device: unsafe extern "C" fn(c_int) -> c_int = self.sym("hipSetDevice")?;
        let load: unsafe extern "C" fn(*mut Ptr, *const c_void) -> c_int =
            self.sym("hipModuleLoadData")?;
        let get_fn: unsafe extern "C" fn(*mut Ptr, Ptr, *const c_char) -> c_int =
            self.sym("hipModuleGetFunction")?;
        let malloc: unsafe extern "C" fn(*mut Ptr, usize) -> c_int = self.sym("hipMalloc")?;
        let free: unsafe extern "C" fn(Ptr) -> c_int = self.sym("hipFree")?;
        let h2d: unsafe extern "C" fn(Ptr, *const c_void, usize) -> c_int =
            self.sym("hipMemcpyHtoD")?;
        let d2h: unsafe extern "C" fn(*mut c_void, Ptr, usize) -> c_int =
            self.sym("hipMemcpyDtoH")?;
        #[allow(clippy::type_complexity)]
        let launch: unsafe extern "C" fn(
            Ptr,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            c_uint,
            Ptr,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> c_int = self.sym("hipModuleLaunchKernel")?;
        let sync: unsafe extern "C" fn() -> c_int = self.sym("hipDeviceSynchronize")?;
        let unload: unsafe extern "C" fn(Ptr) -> c_int = self.sym("hipModuleUnload")?;

        let entry = CString::new(kind.fn_name).expect("no NUL");
        let mut module: Ptr = std::ptr::null_mut();
        let mut func: Ptr = std::ptr::null_mut();
        let (mut d_w, mut d_x, mut d_y): (Ptr, Ptr, Ptr) = (
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        let mut y = vec![0f32; rows];
        // SAFETY: every out-pointer is a live local; every device buffer
        // is allocated at exactly the byte length later copied into or out
        // of it; the kernel parameters are pointers to live locals whose
        // types match `q8_0_matvec`'s signature (three pointers, three
        // ints); the module and buffers are released on every path below.
        let result = unsafe {
            (|| {
                self.check("hipSetDevice", set_device(dev.index))?;
                self.check(
                    "hipModuleLoadData",
                    load(&mut module, code.as_ptr() as *const c_void),
                )?;
                self.check(
                    "hipModuleGetFunction",
                    get_fn(&mut func, module, entry.as_ptr()),
                )?;
                self.check("hipMalloc", malloc(&mut d_w, weights.len()))?;
                self.check("hipMalloc", malloc(&mut d_x, x.len() * 4))?;
                self.check("hipMalloc", malloc(&mut d_y, rows * 4))?;
                self.check(
                    "hipMemcpyHtoD",
                    h2d(d_w, weights.as_ptr() as *const c_void, weights.len()),
                )?;
                self.check(
                    "hipMemcpyHtoD",
                    h2d(d_x, x.as_ptr() as *const c_void, x.len() * 4),
                )?;
                let (r, rb, nb) = (rows as c_int, row_bytes as c_int, blocks_per_row as c_int);
                let mut params: [*mut c_void; 6] = [
                    &mut d_w as *mut Ptr as *mut c_void,
                    &mut d_x as *mut Ptr as *mut c_void,
                    &mut d_y as *mut Ptr as *mut c_void,
                    &r as *const c_int as *mut c_void,
                    &rb as *const c_int as *mut c_void,
                    &nb as *const c_int as *mut c_void,
                ];
                self.check(
                    "hipModuleLaunchKernel",
                    launch(
                        func,
                        rows as c_uint,
                        1,
                        1,
                        256,
                        1,
                        1,
                        0,
                        std::ptr::null_mut(),
                        params.as_mut_ptr(),
                        std::ptr::null_mut(),
                    ),
                )?;
                self.check("hipDeviceSynchronize", sync())?;
                self.check(
                    "hipMemcpyDtoH",
                    d2h(y.as_mut_ptr() as *mut c_void, d_y, rows * 4),
                )?;
                Ok(())
            })()
        };
        // SAFETY: freeing what was allocated above; null is skipped.
        unsafe {
            for p in [d_w, d_x, d_y] {
                if !p.is_null() {
                    free(p);
                }
            }
            if !module.is_null() {
                unload(module);
            }
        }
        result.map(|()| y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four offsets sit inside the struct and do not overlap, and a
    /// synthetic struct written at them reads back field for field.
    #[test]
    fn the_props_offsets_read_what_they_name() {
        use props_r0600::*;
        let mut raw = vec![0u8; SIZE];
        raw[NAME.0..NAME.0 + 13].copy_from_slice(b"AMD Radeon X\0");
        raw[GCN_ARCH_NAME.0..GCN_ARCH_NAME.0 + 23].copy_from_slice(b"gfx1100:sramecc-:xnack-");
        raw[TOTAL_GLOBAL_MEM..TOTAL_GLOBAL_MEM + 8].copy_from_slice(&(24u64 << 30).to_le_bytes());
        raw[WARP_SIZE..WARP_SIZE + 4].copy_from_slice(&32i32.to_le_bytes());
        let d = parse_props(0, &raw);
        assert_eq!(d.name, "AMD Radeon X");
        assert_eq!(d.gcn_arch, "gfx1100:sramecc-:xnack-");
        assert_eq!((d.total_mem, d.warp_size), (24 << 30, 32));
        assert!(GCN_ARCH_NAME.0 + GCN_ARCH_NAME.1 <= SIZE && NAME.1 <= TOTAL_GLOBAL_MEM);
    }

    /// **The R1 launch test.** The Q8_0 matvec on the first served AMD
    /// GPU against `frink_quant`'s scalar dot, at shapes that cover a
    /// non-4-byte-aligned row stride and several blocks per row.
    #[test]
    #[ignore = "NEVER RUN: needs an RDNA AMD GPU with ROCm. Run: cargo test -p frink-rocm --features rocm -- --ignored gpu"]
    fn gpu_q8_0_matvec_matches_the_scalar_twin() {
        let hip = Hip::open().expect("libamdhip64");
        let dev = hip
            .devices()
            .expect("devices")
            .into_iter()
            .find(|d| lookup(&d.gcn_arch).is_ok())
            .expect("a served (RDNA) AMD GPU");
        eprintln!("device: {} ({})", dev.name, dev.gcn_arch);
        for (rows, blocks) in [(1usize, 1usize), (9, 3), (65, 4), (200, 11)] {
            let cols = blocks * 32;
            let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.37).sin()).collect();
            let mut w = Vec::with_capacity(rows * blocks * 34);
            for r in 0..rows {
                for b in 0..blocks {
                    w.extend_from_slice(
                        &half::f16::from_f32(0.01 + (r + b) as f32 * 0.003).to_le_bytes(),
                    );
                    w.extend((0..32).map(|i| ((i * 7 + r * 3 + b) % 255) as u8));
                }
            }
            let got = hip.q8_0_matvec(&dev, &w, &x, rows).expect("launch");
            for r in 0..rows {
                let row = &w[r * blocks * 34..(r + 1) * blocks * 34];
                let want = frink_quant::dot_q8_0_f32_scalar(row, &x);
                let tol = 1e-4 * want.abs().max(1.0);
                assert!(
                    (got[r] - want).abs() <= tol,
                    "({rows},{blocks}) row {r}: {} vs {want}",
                    got[r]
                );
            }
        }
    }

    /// Lists the AMD GPUs the runtime sees and whether each is served.
    #[test]
    #[ignore = "needs ROCm's libamdhip64. Run: cargo test -p frink-rocm --features rocm -- --ignored gpu --nocapture"]
    fn gpu_devices_are_listed_with_their_verdict() {
        let hip = Hip::open().expect("libamdhip64");
        for d in hip.devices().expect("devices") {
            eprintln!(
                "{} {} wave{}: {:?}",
                d.name,
                d.gcn_arch,
                d.warp_size,
                lookup(&d.gcn_arch).map(|a| a.family)
            );
        }
    }
}
