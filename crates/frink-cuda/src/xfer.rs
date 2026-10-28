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
