//! A process-wide observer of the f32 activations fed to every
//! [`WeightMatrix`] projection: the seam `frink imatrix` collects an
//! importance matrix through.
//!
//! llama.cpp collects its imatrix with a scheduler callback
//! (`cb_eval`) that sees every `MUL_MAT` node and its `src1` input
//! (`tools/imatrix/imatrix.cpp:219-237`). frink has no graph, so the
//! equivalent seam is the function every projection goes through:
//! [`WeightMatrix::apply`] for one activation and
//! [`WeightMatrix::apply_batch_with_acts`] for a batch. Both call
//! [`observe`] before they touch the weights.
//!
//! **A `WeightMatrix` carries no name**, and the loader that knows the
//! names lives in `frink-models`, so the observer is handed the matrix
//! by reference and it is the installer's job to know which one it is
//! -- `frink imatrix` walks the decoder's public weight fields and
//! keys on the address. That is why the callback takes `&WeightMatrix`
//! and not a string: adding a name field would have to be threaded
//! through thirty construction sites across seven loaders to change
//! nothing for inference.
//!
//! Cost when nothing is installed: one relaxed atomic load per
//! projection call, not per element. A decode step is a few dozen
//! calls, so this is not a hot-path concern.
//!
//! The tap sees the f32 input EXACTLY as the projection receives it,
//! before any activation quantization the CPU INT_DOT path does. That
//! is what llama.cpp's callback sees too (`src1->type == F32` is a
//! precondition for collection).
//!
//! What it does NOT see: a GPU batch that fails to launch and degrades
//! to per-row [`WeightMatrix::apply`] would fire the observer twice for
//! the same rows. `frink imatrix` pins the CPU backend, where no such
//! degradation path exists, and checks every dense entry's row count
//! against the token count so a double observation is a refusal rather
//! than a silently doubled matrix.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use crate::weight_matrix::WeightMatrix;

/// The observer: the matrix being applied, the activation rows laid
/// out `[n_rows][cols]`, and `n_rows`.
pub type Observer = dyn Fn(&WeightMatrix, &[f32], usize) + Send + Sync;

static INSTALLED: AtomicBool = AtomicBool::new(false);
static OBSERVER: RwLock<Option<Arc<Observer>>> = RwLock::new(None);

/// Installs `observer` for the life of the returned guard. Only one
/// may be installed at a time: a second install while one is live is
/// refused, because two observers would each see every row and neither
/// would know the other was counting.
pub fn install(observer: Arc<Observer>) -> Result<TapGuard, AlreadyInstalled> {
    let mut slot = OBSERVER.write().unwrap_or_else(|e| e.into_inner());
    if slot.is_some() {
        return Err(AlreadyInstalled);
    }
    *slot = Some(observer);
    INSTALLED.store(true, Ordering::Release);
    Ok(TapGuard(()))
}

/// Uninstalls the observer on drop.
#[must_use = "dropping the guard uninstalls the tap immediately"]
pub struct TapGuard(());

impl Drop for TapGuard {
    fn drop(&mut self) {
        INSTALLED.store(false, Ordering::Release);
        *OBSERVER.write().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlreadyInstalled;

impl std::fmt::Display for AlreadyInstalled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("an activation tap is already installed in this process")
    }
}

impl std::error::Error for AlreadyInstalled {}

/// Called by the projection entry points. The fast path is the
/// `INSTALLED` load; the observer is cloned out of the lock so a slow
/// observer never holds it across the call.
#[inline]
pub(crate) fn observe(matrix: &WeightMatrix, rows: &[f32], n_rows: usize) {
    if !INSTALLED.load(Ordering::Acquire) {
        return;
    }
    let observer = OBSERVER
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .cloned();
    if let Some(observer) = observer {
        observer(matrix, rows, n_rows);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use std::sync::Mutex;

    // Tests that install a tap share the one process-wide slot, so
    // they serialise on this rather than racing each other. That covers
    // the slot and nothing else: the tap also sees every projection any
    // other test in this binary runs while it is installed, so each
    // observer below keeps only its own matrix. A device build makes
    // the overlap likely, because the first projection in the process
    // probes the device while the tap is installed.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn small_matrix() -> WeightMatrix {
        WeightMatrix::F32(Tensor::new(vec![1.0; 8], vec![2, 4]))
    }

    /// The tap sees each `apply_batch` call once, with every row, and
    /// can tell which matrix was applied by address. Both halves are
    /// what `frink imatrix` relies on: a wrong address leaves `seen`
    /// empty here, and a second observation leaves it with two entries.
    #[test]
    fn the_observer_sees_every_batch_row_once_and_the_matrix_identity() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let m = small_matrix();
        let addr = &m as *const WeightMatrix as usize;
        type Seen = Mutex<Vec<(Vec<f32>, usize)>>;
        let seen: Arc<Seen> = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let guard = install(Arc::new(move |w: &WeightMatrix, rows: &[f32], n: usize| {
            if std::ptr::addr_eq(w, addr as *const WeightMatrix) {
                sink.lock().unwrap().push((rows.to_vec(), n));
            }
        }))
        .unwrap();
        let x: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let _ = m.apply_batch(&x, 3);
        drop(guard);
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1, "one batch call, one observation");
        assert_eq!(seen[0].0, x);
        assert_eq!(seen[0].1, 3);
    }

    /// After the guard drops nothing is observed, and a second install
    /// while one is live is refused rather than replacing it.
    #[test]
    fn the_tap_is_gone_after_the_guard_drops_and_cannot_be_installed_twice() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let m = small_matrix();
        let addr = &m as *const WeightMatrix as usize;
        let calls = Arc::new(Mutex::new(0usize));
        let c = calls.clone();
        let guard = install(Arc::new(move |w: &WeightMatrix, _: &[f32], _| {
            if std::ptr::addr_eq(w, addr as *const WeightMatrix) {
                *c.lock().unwrap() += 1;
            }
        }))
        .unwrap();
        assert_eq!(
            install(Arc::new(|_: &WeightMatrix, _: &[f32], _| {}))
                .err()
                .map(|_| ()),
            Some(()),
            "a second install must be refused while the first is live"
        );
        let _ = m.apply(&[1.0; 4]);
        drop(guard);
        let _ = m.apply(&[1.0; 4]);
        assert_eq!(*calls.lock().unwrap(), 1);
    }
}
