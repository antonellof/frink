//! Every copy across the bus and every host wait in `frink-cuda`,
//! through one door so [`crate::counters`] sees all of them.
//!
//! A counter that one call site bypasses reads as a measurement and
//! is not one, which is the shape of the host-check tool that checked
//! nothing for a day. So the raw `cudarc` calls (`htod_copy`,
//! `htod_sync_copy_into`, `dtoh_sync_copy`, `dtod_copy`,
//! `synchronize`) appear in this file and in tests only, and
//! `counters::tests::no_transfer_bypasses_the_counted_door` fails if a
//! new one appears anywhere else in the crate's non-test code.

use crate::counters;
use crate::gpu::CudaError;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr};
use std::sync::Arc;

fn err(what: &str, e: impl std::fmt::Debug) -> CudaError {
    CudaError::Launch(format!("{what}: {e:?}"))
}

/// Uploads `src` into a new device buffer. Enqueued on the shared
/// stream; from pageable memory the driver stages the bytes before it
/// returns, but it does not wait for the stream.
pub(crate) fn htod<T: DeviceRepr + Unpin + Clone>(
    dev: &Arc<CudaDevice>,
    what: &str,
    src: &[T],
) -> Result<CudaSlice<T>, CudaError> {
    htod_owned(dev, what, src.to_vec())
}

/// [`htod`] for a vector the caller already owns, so it is not copied
/// a second time on the host.
pub(crate) fn htod_owned<T: DeviceRepr + Unpin>(
    dev: &Arc<CudaDevice>,
    what: &str,
    src: Vec<T>,
) -> Result<CudaSlice<T>, CudaError> {
    counters::record_h2d(std::mem::size_of_val(src.as_slice()));
    dev.htod_copy(src).map_err(|e| err(what, e))
}

/// Uploads `src` into an existing device range and WAITS for it (the
/// source is borrowed, so the copy must finish before this returns).
/// Counted as an upload and a sync.
pub(crate) fn htod_into_sync<T: DeviceRepr, D: DevicePtrMut<T>>(
    dev: &Arc<CudaDevice>,
    what: &str,
    src: &[T],
    dst: &mut D,
) -> Result<(), CudaError> {
    counters::record_h2d(std::mem::size_of_val(src));
    counters::record_sync();
    dev.htod_sync_copy_into(src, dst).map_err(|e| err(what, e))
}

/// Downloads `src` and waits for it: one D2H and one sync.
pub(crate) fn dtoh<T: DeviceRepr, S: DevicePtr<T>>(
    dev: &Arc<CudaDevice>,
    what: &str,
    src: &S,
) -> Result<Vec<T>, CudaError> {
    counters::record_d2h(src.len() * std::mem::size_of::<T>());
    dev.dtoh_sync_copy(src).map_err(|e| err(what, e))
}

/// The largest download staged through page-locked memory, in bytes.
/// Past it a download takes the pageable path ([`dtoh`]): page-locked
/// memory is taken from the whole machine, so it is bounded. A
/// Llama-3.1-8B `pp512` step downloads 143 MB.
pub const PINNED_MAX_BYTES: usize = 512 << 20;

/// The process's page-locked download buffer, `cap` floats at `ptr`.
struct Pinned {
    ptr: *mut f32,
    cap: usize,
}

// SAFETY: the buffer is host memory this process owns for its whole
// life, and every access goes through `PINNED`'s lock.
unsafe impl Send for Pinned {}

static PINNED: std::sync::Mutex<Option<Pinned>> = std::sync::Mutex::new(None);

/// A finished download: its floats, in page-locked memory or, when that
/// buffer was too small to grow or held by another download, in a host
/// vector. Holds the page-locked buffer until it is dropped.
pub struct Download {
    inner: DownloadInner,
}

enum DownloadInner {
    Pinned {
        guard: std::sync::MutexGuard<'static, Option<Pinned>>,
        len: usize,
    },
    Pageable(Vec<f32>),
}

impl Download {
    /// The downloaded floats.
    pub fn as_slice(&self) -> &[f32] {
        match &self.inner {
            DownloadInner::Pinned { guard, len } => {
                let p = guard.as_ref().expect("a pinned download holds its buffer");
                // SAFETY: `p.ptr` is a live page-locked allocation of
                // `p.cap >= len` floats, written by the download that
                // built this value and waited for; the guard keeps any
                // other download out of it while this borrow lives.
                unsafe { std::slice::from_raw_parts(p.ptr, *len) }
            }
            DownloadInner::Pageable(v) => v,
        }
    }
}

/// Downloads `src` and waits for it, through the page-locked buffer
/// when it fits: a pageable download is staged by the driver in small
/// pieces and ran at 1.9 GB/s for the 143 MB a prefill step returns,
/// where a page-locked one runs at the bus's speed. One D2H and one
/// sync either way.
///
/// The buffer is taken with `try_lock`, so a second download while
/// one is held (another request's, or a caller holding two) takes the
/// pageable path instead of waiting.
pub(crate) fn dtoh_staged(
    dev: &Arc<CudaDevice>,
    what: &str,
    src: &CudaSlice<f32>,
) -> Result<Download, CudaError> {
    let len = cudarc::driver::DeviceSlice::len(src);
    let bytes = std::mem::size_of::<f32>() * len;
    let guard = if bytes <= PINNED_MAX_BYTES {
        PINNED.try_lock().ok()
    } else {
        None
    };
    let Some(mut guard) = guard else {
        return Ok(Download {
            inner: DownloadInner::Pageable(dtoh(dev, what, src)?),
        });
    };
    if guard.as_ref().is_none_or(|p| p.cap < len) {
        dev.bind_to_thread().map_err(|e| err(what, e))?;
        if let Some(old) = guard.take() {
            // SAFETY: `old.ptr` came from `cuMemHostAlloc` below and no
            // `Download` borrows it: this thread holds the lock.
            unsafe { cudarc::driver::sys::lib().cuMemFreeHost(old.ptr.cast()) }
                .result()
                .map_err(|e| err(what, e))?;
        }
        let cap = len.next_power_of_two().min(PINNED_MAX_BYTES / 4).max(len);
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: `p` is a valid out-pointer and the context is bound.
        unsafe { cudarc::driver::sys::lib().cuMemHostAlloc(&mut p, cap * 4, 0) }
            .result()
            .map_err(|e| err(what, e))?;
        *guard = Some(Pinned { ptr: p.cast(), cap });
    }
    let pinned = guard.as_ref().expect("allocated above");
    // SAFETY: the destination is `len <= cap` floats of the page-locked
    // buffer, which this thread holds; `src` is a live device buffer of
    // `len` floats on `dev`, and the copy is waited for before the
    // buffer is read.
    let dst = unsafe { std::slice::from_raw_parts_mut(pinned.ptr, len) };
    counters::record_d2h(bytes);
    counters::record_sync();
    unsafe { cudarc::driver::result::memcpy_dtoh_async(dst, *src.device_ptr(), *dev.cu_stream()) }
        .map_err(|e| err(what, e))?;
    dev.synchronize().map_err(|e| err(what, e))?;
    Ok(Download {
        inner: DownloadInner::Pinned { guard, len },
    })
}

/// Device-to-device copy of equal-length ranges. Enqueued; no wait.
pub(crate) fn dtod<T: DeviceRepr, S: DevicePtr<T>, D: DevicePtrMut<T>>(
    dev: &Arc<CudaDevice>,
    what: &str,
    src: &S,
    dst: &mut D,
) -> Result<(), CudaError> {
    counters::record_d2d();
    dev.dtod_copy(src, dst).map_err(|e| err(what, e))
}

/// Waits for everything enqueued on the shared stream: one sync.
pub(crate) fn sync(dev: &Arc<CudaDevice>, what: &str) -> Result<(), CudaError> {
    counters::record_sync();
    dev.synchronize().map_err(|e| err(what, e))

#[cfg(test)]
mod tests {
    use super::*;

    /// A staged download returns what the device holds, through the
    /// page-locked buffer; one made while the buffer is held takes the
    /// pageable path instead of waiting; and a larger one grows the
    /// buffer. The buffer is the process's, and the other hardware
    /// tests download through it in parallel, so the test holds it to
    /// check the fallback and waits for it to be free to check the
    /// page-locked path.
    #[test]
    #[ignore = "requires real CUDA hardware"]
    fn a_staged_download_matches_and_a_held_buffer_falls_back() {
        let dev = crate::gpu::shared_device().expect("a CUDA device");
        let pinned = |what: &str, d: &CudaSlice<f32>| -> Download {
            for _ in 0..400 {
                let got = dtoh_staged(&dev, what, d).unwrap();
                if matches!(got.inner, DownloadInner::Pinned { .. }) {
                    return got;
                }
                drop(got);
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            panic!("{what}: the page-locked buffer never came free");
        };

        let host: Vec<f32> = (0..300_000).map(|i| i as f32 * 0.5 - 7.0).collect();
        let d = dev.htod_copy(host.clone()).unwrap();
        {
            let _held = PINNED.lock().unwrap_or_else(|p| p.into_inner());
            let b = dtoh_staged(&dev, "held", &d).unwrap();
            assert!(
                matches!(b.inner, DownloadInner::Pageable(_)),
                "a held buffer is not waited for"
            );
            assert_eq!(b.as_slice(), &host[..]);
        }
        assert_eq!(pinned("a", &d).as_slice(), &host[..]);

        let big: Vec<f32> = (0..1_500_000).map(|i| (i % 977) as f32).collect();
        let d2 = dev.htod_copy(big.clone()).unwrap();
        assert_eq!(pinned("big", &d2).as_slice(), &big[..]);
        // The grown buffer still serves the smaller download.
        assert_eq!(pinned("small", &d).as_slice(), &host[..]);
    }
}
