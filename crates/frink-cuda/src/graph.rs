//! CUDA graph capture and replay through the driver API, for a decode
//! token of the resident stack (`FRINK_CUDA_GRAPH=1`, #537).
//!
//! `cudarc` 0.11.9 has no safe graph API, so this wraps the raw
//! `cuStreamBeginCapture_v2` / `cuStreamEndCapture` /
//! `cuGraphInstantiateWithFlags` / `cuGraphLaunch` calls. What a decode
//! token captures, and when, is `crate::prefill::decode_graph`'s; this
//! module only records a body and replays it.
//!
//! # Where the capture happens
//!
//! Every launch in this crate goes to the shared device's stream, which
//! `CudaDevice::new` leaves as the legacy NULL stream, and a capture
//! "may not be initiated if `stream` is CU_STREAM_LEGACY"
//! (`cuStreamBeginCapture`). The body is therefore recorded on a second
//! `CudaDevice` over the same primary context with a non-blocking stream
//! of its own ([`capture`]), which nothing else enqueues on, so a
//! request decoding on another thread is neither captured into the
//! graph nor stopped by it. A capture executes nothing, so the second
//! stream never runs work and never races the first; the instantiated
//! graph is launched on the shared stream, in order with the uploads
//! before it and the download after it.
//!
//! # Memory
//!
//! The body's per-launch allocations (`alloc_out`, `alloc_zeros`) are
//! stream-ordered (`cuMemAllocAsync`), and a capture records each as a
//! graph allocation node and each drop as a free node. "Graph
//! allocations have fixed addresses over the life of a graph including
//! repeated instantiations and launches" (CUDA programming guide, graph
//! memory nodes), so they are the token's scratch arena with no change
//! to the enqueues. Two rules follow, and the decode graph keeps both:
//! everything the body allocates is dropped inside it (a graph with an
//! unfreed allocation is refused a relaunch), and nothing allocated
//! outside is dropped inside (a free node takes only a graph
//! allocation). A device without stream-ordered allocation would
//! allocate for real inside the capture, so [`device_supports`] refuses
//! it.
//!
//! # What may not happen inside
//!
//! A host wait or a host copy on a capturing stream invalidates the
//! capture, and after that every free returns an error, which `cudarc`
//! unwraps. Every copy and wait goes through `crate::xfer`, which
//! refuses one on a thread that is capturing ([`capturing`]) before it
//! reaches the driver, so a body that would upload a missing weight or
//! wait for a router's logits fails cleanly and the token runs eagerly.

use crate::gpu::{shared_device, CudaError};
use cudarc::driver::sys::{self, lib};
use cudarc::driver::CudaDevice;
use std::cell::Cell;
use std::sync::{Arc, Mutex};

/// The variable [`decode_graphs`] reads.
pub const SWITCH: &str = "FRINK_CUDA_GRAPH";

/// What `FRINK_CUDA_GRAPH` asks for: unset or `0` runs every decode
/// token eagerly, `1` replays it from a captured graph. Anything else is
/// refused by name (ADR 0005: a flag accepted but not honoured is an
/// error), not read as either. Pure, for the host test.
pub fn parse_switch(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!(
            "{SWITCH}={other:?}: expected 1 (replay a decode token from a CUDA graph) \
             or 0 / unset (run it eagerly)"
        )),
    }
}

/// [`parse_switch`] of the process's `FRINK_CUDA_GRAPH`, read once. A
/// value it refuses is an `Unsupported` error at every decode token, so
/// the resident decode declines and says why rather than running
/// either way.
pub(crate) fn decode_graphs() -> Result<bool, CudaError> {
    static ON: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    ON.get_or_init(|| parse_switch(std::env::var(SWITCH).ok().as_deref()))
        .clone()
        .map_err(CudaError::Unsupported)
}

/// Whether the device can replay a captured token: compute capability
/// 7.0 and up, below which llama.cpp turns its graphs off
/// (`ggml_cuda_graph_set_enabled`, `GGML_CUDA_CC_VOLTA`), and the
/// stream-ordered allocator the graph's memory nodes are. Asked once.
pub(crate) fn device_supports(dev: &Arc<CudaDevice>) -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| {
        let pools = dev
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED)
            .unwrap_or(0);
        crate::gpu::compute_capability_major(dev) >= 7 && pools > 0
    })
}

thread_local! {
    static CAPTURING: Cell<bool> = const { Cell::new(false) };
}

/// Whether this thread is recording a graph. `crate::xfer` refuses every
/// host copy and wait while it is.
pub(crate) fn capturing() -> bool {
    CAPTURING.with(Cell::get)
}

/// The device a body is recorded on: the shared device's context with a
/// non-blocking stream of its own, made once and kept for the process.
/// The lock is held for a whole capture, so one runs at a time.
static CAPTURE_DEVICE: Mutex<Option<Arc<CudaDevice>>> = Mutex::new(None);

fn driver(what: &str, r: sys::CUresult) -> Result<(), CudaError> {
    r.result()
        .map_err(|e| CudaError::Launch(format!("{what}: {e:?}")))
}

/// One instantiated graph and the graph it came from, destroyed
/// together.
pub(crate) struct Graph {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    /// The shared device, bound before the destroy calls.
    dev: Arc<CudaDevice>,
}

// SAFETY: the handles are owned by this value alone, and the driver
// lets any thread use them once the context is bound, which every call
// here does first.
unsafe impl Send for Graph {}
// SAFETY: graph objects are not internally synchronised, and nothing
// here touches the handles through `&self`: the launch and the destroy
// take `&mut self`, so a shared reference can do nothing with them.
unsafe impl Sync for Graph {}

/// Ends a capture still open when its scope is left, a panic in the body
/// included, so the capture stream is never left capturing.
struct Recording<'a> {
    stream: sys::CUstream,
    dev: &'a Arc<CudaDevice>,
    open: bool,
}

impl Recording<'_> {
    /// Ends the capture and takes its graph (null if the capture was
    /// invalidated).
    fn end(&mut self) -> Result<sys::CUgraph, CudaError> {
        self.open = false;
        CAPTURING.with(|c| c.set(false));
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        // SAFETY: `stream` is the capture device's own stream, capturing
        // since `capture` began it; `graph` is a valid out-pointer.
        let r = unsafe { lib().cuStreamEndCapture(self.stream, &mut graph) };
        driver("cuStreamEndCapture", r)?;
        Ok(graph)
    }
}

impl Drop for Recording<'_> {
    fn drop(&mut self) {
        if self.open {
            let _ = self.dev.bind_to_thread();
            if let Ok(g) = self.end() {
                destroy_graph(g);
            }
        }
    }
}

fn destroy_graph(graph: sys::CUgraph) {
    if !graph.is_null() {
        // SAFETY: `graph` came from `cuStreamEndCapture` and is
        // destroyed once, here.
        let _ = unsafe { lib().cuGraphDestroy(graph) };
    }
}

impl Graph {
    /// Records `body` on the capture device and instantiates it. The
    /// body enqueues on the device it is handed and must drop inside
    /// itself everything it allocates there (see the module doc); what
    /// it returns is host data. Nothing it enqueues runs until
    /// [`Self::launch`]. An error from the body, or a capture the driver
    /// invalidated, is returned and no graph is kept.
    pub(crate) fn capture<T>(
        body: impl FnOnce(&Arc<CudaDevice>) -> Result<T, CudaError>,
    ) -> Result<(Graph, T), CudaError> {
        let shared = shared_device()?;
        let mut slot = CAPTURE_DEVICE.lock().unwrap_or_else(|p| p.into_inner());
        if slot.is_none() {
            let dev = CudaDevice::new_with_stream(shared.ordinal())
                .map_err(|e| CudaError::DriverInit(format!("capture stream: {e:?}")))?;
            *slot = Some(dev);
        }
        let cap = slot.as_ref().expect("made above").clone();
        // Every module a warm token loaded, loaded here before the
        // capture, so none is compiled or loaded inside it.
        crate::gpu::preload_modules(&cap)?;
        cap.bind_to_thread()
            .map_err(|e| CudaError::Launch(format!("capture bind: {e:?}")))?;
        let stream = *cap.cu_stream();
        // Relaxed, as llama.cpp captures (`cudaStreamCaptureModeRelaxed`):
        // other threads keep the shared stream and their own calls, and
        // nothing they do reaches this stream.
        // SAFETY: `stream` is the capture device's non-blocking stream,
        // not capturing (the lock above admits one capture at a time and
        // `Recording` ends every one).
        let r = unsafe {
            lib().cuStreamBeginCapture_v2(
                stream,
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            )
        };
        driver("cuStreamBeginCapture", r)?;
        let mut rec = Recording {
            stream,
            dev: &cap,
            open: true,
        };
        CAPTURING.with(|c| c.set(true));
        let out = body(&cap);
        let graph = rec.end();
        let (out, graph) = match (out, graph) {
            (Ok(out), Ok(g)) if !g.is_null() => (out, g),
            (out, graph) => {
                if let Ok(g) = graph {
                    destroy_graph(g);
                }
                out?;
                return Err(CudaError::Launch(
                    "the decode token's capture was invalidated".to_string(),
                ));
            }
        };
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        // SAFETY: `graph` is the capture's live graph and `exec` a valid
        // out-pointer; flags 0 is the default instantiation.
        let r = unsafe { lib().cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        if let Err(e) = driver("cuGraphInstantiateWithFlags", r) {
            destroy_graph(graph);
            return Err(e);
        }
        Ok((
            Graph {
                graph,
                exec,
                dev: shared,
            },
            out,
        ))
    }

    /// Launches the graph on `dev`'s stream, ordered after everything
    /// already enqueued there and before everything enqueued after.
    pub(crate) fn launch(&mut self, dev: &Arc<CudaDevice>) -> Result<(), CudaError> {
        dev.bind_to_thread()
            .map_err(|e| CudaError::Launch(format!("graph launch bind: {e:?}")))?;
        // SAFETY: `exec` is this value's live instantiation; the stream
        // is the shared device's, on the same context.
        let r = unsafe { lib().cuGraphLaunch(self.exec, *dev.cu_stream()) };
        driver("cuGraphLaunch", r)
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        let _ = self.dev.bind_to_thread();
        // SAFETY: both handles are this value's and destroyed once. An
        // exec still in flight "will not be terminated, but rather freed
        // asynchronously on completion" (`cuGraphExecDestroy`).
        unsafe {
            let _ = lib().cuGraphExecDestroy(self.exec);
        }
        destroy_graph(self.graph);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unset and `0` are off, `1` is on, and every other spelling is
    /// refused by the variable's name, not read as on or off.
    #[test]
    fn the_switch_takes_unset_zero_and_one_and_refuses_the_rest_by_name() {
        assert_eq!(parse_switch(None), Ok(false));
        assert_eq!(parse_switch(Some("0")), Ok(false));
        assert_eq!(parse_switch(Some("1")), Ok(true));
        for bad in ["", "true", "on", "yes", "2", " 1", "1 ", "off"] {
            let e = parse_switch(Some(bad)).expect_err(bad);
            assert!(e.starts_with("FRINK_CUDA_GRAPH="), "{bad:?}: {e}");
            assert!(e.contains(&format!("{bad:?}")), "{bad:?}: {e}");
        }
    }

    #[test]
    fn no_thread_is_capturing_unless_a_capture_says_so() {
        assert!(!capturing());
        let other = std::thread::spawn(capturing).join().unwrap();
        assert!(!other);
    }
}
