//! A decode token of the resident stack replayed from a CUDA graph
//! (`FRINK_CUDA_GRAPH=1`, #537, #133).
//!
//! A Qwen3-30B-A3B decode token is about a thousand kernel launches and
//! as many stream-ordered allocations, resident-cache lookups and
//! module-table lookups, with no host wait between them since the
//! routing moved onto the device (`super::moe_device`). Each launch
//! costs host time and a gap on the GPU, and the host is the slower of
//! the two. llama.cpp's CUDA backend captures a decode graph and replays
//! it (`ggml_backend_cuda_graph_compute` at the pin); this is that for
//! the stack, behind `crate::graph`'s capture.
//!
//! # Static replay, not a capture per token
//!
//! A graph's arguments are frozen at capture, and three things change
//! every token: the position (RoPE's angle, the mirror row the token's
//! K and V land in, the attention's key range), the input (the hidden
//! row, or the token id the embedding gather reads), and nothing else.
//! llama.cpp re-captures when a node's properties change and updates
//! the executable with `cudaGraphExecUpdate`; its per-token values are
//! input tensors uploaded before the graph runs, and its KV view is
//! padded so the properties change rarely. Re-capturing every token
//! would keep every launch's host cost (a capture records each launch
//! through the same calls, plus the update), which is the cost this
//! removes. So the graph is captured once and replayed: the position is
//! a device word ([`super::enqueue::StepPos::Device`]) the RoPE, append
//! and attention kernels read, and the input a fixed device buffer, both
//! written by the token before its replay ([`replay`]). A replay costs
//! two small uploads, one `cuGraphLaunch` and the download the stack
//! always made.
//!
//! The one shape that grows with the position is the one-row
//! attention's grid, a block per 64 keys. A replay launches the grid
//! the history's capacity needs ([`crate::decode_attn::replay_slots`])
//! and the chunks past the position's own return at once. The capacity
//! doubles from 256 rows (`super::kv_mirror::grown_capacity`), and a
//! growth moves the mirror, which changes the key: the token after a
//! growth runs eagerly and the next one captures again, about
//! `log2(context / 256)` captures a sequence.
//!
//! # When a token captures
//!
//! [`plan`] is the rule. A run captures only after a token of the same
//! [`key`] ran eagerly with every routed layer routed on the device: so
//! every weight, bias, expert table and kernel module the capture reads
//! is already on the device, and no layer waits for its router's logits
//! on the host. A capture that still meets a host copy or wait is
//! stopped by `crate::xfer` before the driver sees it, and that key
//! decodes eagerly from then on. Everything else (a prefill batch, a
//! stack that uploads its prefix, a device below `sm_70` or without
//! stream-ordered allocation) runs eagerly, as it always has.
//!
//! # Where it lives
//!
//! In the run's first KV mirror ([`CudaKvMirror::graph`]): a graph
//! freezes that sequence's mirror addresses, and a mirror that stops
//! matching its host history is rebuilt and takes the graph with it.

mod key;
#[cfg(test)]
pub(super) mod tests;

use super::held::Held;
use super::stage::{StackOutBufs, StageLayout};
use super::{
    enqueue_body, finish, run_eager, BodyInput, CudaKvMirror, PrefillDenseLayerCuda, PrefillParams,
    PrefillStackOut, RouteFn, StackFfn, StackInput, StackKv,
};
use crate::gpu::CudaError;
use crate::graph::Graph;
use crate::xfer;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;

/// How a token of the stack ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphUse {
    /// Enqueued launch by launch as it ran.
    Eager,
    /// Recorded into a graph, and run by the graph's first replay.
    Captured,
    /// Run by replaying the graph an earlier token captured.
    Replayed,
}

/// What a run's first mirror knows about graphs for the run.
#[derive(Default)]
pub(crate) enum GraphSlot {
    /// Nothing yet.
    #[default]
    Cold,
    /// A token of this key ran eagerly and could be captured: the next
    /// one captures.
    Warm(Vec<u64>),
    /// Tokens of this key run eagerly: one routed a layer on the host,
    /// or its capture failed.
    Refused(Vec<u64>),
    /// The graph captured for its key.
    Ready(Box<DecodeGraph>),
}

/// A slot as [`plan`] reads it: its state and its key, nothing else,
/// so the rule is tested without a device.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SlotKey<'a> {
    Cold,
    Warm(&'a [u64]),
    Refused(&'a [u64]),
    Ready(&'a [u64]),
}

impl GraphSlot {
    fn view(&self) -> SlotKey<'_> {
        match self {
            GraphSlot::Cold => SlotKey::Cold,
            GraphSlot::Warm(k) => SlotKey::Warm(k),
            GraphSlot::Refused(k) => SlotKey::Refused(k),
            GraphSlot::Ready(g) => SlotKey::Ready(&g.key),
        }
    }
}

/// What a token does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Plan {
    Eager,
    Capture,
    Replay,
}

/// THE rule: replay the graph captured for this key; capture after a
/// token of this key ran eagerly and could be ([`after_eager`]);
/// otherwise run eagerly. A key never seen, a key whose capture failed
/// or that routes on the host, and a key that changed (a mirror grew, a
/// run took other layers) all run eagerly.
pub(crate) fn plan(slot: SlotKey<'_>, key: &[u64]) -> Plan {
    match slot {
        SlotKey::Ready(k) if k == key => Plan::Replay,
        SlotKey::Warm(k) if k == key => Plan::Capture,
        _ => Plan::Eager,
    }
}

/// Whether a key that just ran eagerly may capture next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Settle {
    Warm,
    Refused,
}

/// The slot after an eager token of `key`, `capturable` when every
/// routed layer of it routed on the device: warm, unless this key was
/// already refused (its capture failed), which it stays.
pub(crate) fn after_eager(was: SlotKey<'_>, key: &[u64], capturable: bool) -> Settle {
    match was {
        SlotKey::Refused(k) if k == key => Settle::Refused,
        _ if capturable => Settle::Warm,
        _ => Settle::Refused,
    }
}

/// One run's captured token: its graph, the buffers it reads and
/// writes at fixed addresses, and everything resident it reads, held.
pub(crate) struct DecodeGraph {
    key: Vec<u64>,
    exec: Graph,
    bufs: StepBuffers,
    /// The weights, vectors and expert tables the graph reads, held so
    /// none is freed under it.
    _held: Held,
    /// Which layers the capture routed on the device: every routed one.
    on_device: Vec<bool>,
}

/// The buffers a replay writes before its launch and reads after: the
/// position word, the input, and the stack's output.
struct StepBuffers {
    pos: CudaSlice<i32>,
    input: StepInput,
    out: StackOutBufs,
}

/// A replay's input buffer: the hidden row, or the ids to gather.
enum StepInput {
    Hidden(CudaSlice<f32>),
    Gather(CudaSlice<u32>),
}

fn slot_of<'a>(kv: &'a mut StackKv<'_, '_>) -> Option<&'a mut GraphSlot> {
    match kv {
        StackKv::Mirrors(slots) => slots.first_mut()?.as_mut().map(|m| &mut m.graph),
        StackKv::UploadPrefix => None,
    }
}

/// The run's key: the layers' frozen values, then each mirror's device
/// addresses and capacity.
fn key_of(
    input: &StackInput<'_>,
    layers: &[(&PrefillDenseLayerCuda<'_>, &PrefillParams<'_>)],
    kv: &StackKv<'_, '_>,
) -> Vec<u64> {
    let mut w = key::Words::default();
    w.stack(input, layers);
    if let StackKv::Mirrors(slots) = kv {
        for m in slots.iter().map(|s| s.as_ref()) {
            match m {
                Some(m) => {
                    w.n(*m.k.device_ptr() as usize);
                    w.n(*m.v.device_ptr() as usize);
                    w.n(m.cap_rows());
                }
                None => w.n(0),
            }
        }
    }
    w.0
}

/// A one-row step over the mirrors (`launch_stack_in` has validated and
/// reserved them): replayed, captured and replayed, or run eagerly, as
/// [`plan`] says, and the run's slot left as the token leaves it.
#[allow(clippy::too_many_arguments)]
pub(super) fn token<'e>(
    dev: &Arc<CudaDevice>,
    input: StackInput<'_>,
    layers: &[(&PrefillDenseLayerCuda<'_>, &PrefillParams<'_>)],
    kv: &mut StackKv<'_, '_>,
    layout: &StageLayout,
    mut route: Option<&mut RouteFn<'_, 'e>>,
    start_pos: usize,
) -> Result<PrefillStackOut, CudaError> {
    let key = key_of(&input, layers, kv);
    let Some(slot) = slot_of(kv).map(std::mem::take) else {
        return Err(CudaError::Unsupported(
            "decode graph: a step with no KV mirror".to_string(),
        ));
    };
    let put = |kv: &mut StackKv<'_, '_>, s: GraphSlot| {
        if let Some(slot) = slot_of(kv) {
            *slot = s;
        }
    };
    match plan(slot.view(), &key) {
        Plan::Replay => {
            let GraphSlot::Ready(mut g) = slot else {
                unreachable!("plan replays only a ready slot")
            };
            let out = replay(
                dev,
                &mut g,
                input,
                layers,
                kv,
                layout,
                start_pos,
                GraphUse::Replayed,
            )?;
            put(kv, GraphSlot::Ready(g));
            Ok(out)
        }
        Plan::Capture => {
            match capture(
                dev,
                &input,
                layers,
                kv,
                layout,
                route.as_deref_mut(),
                key.clone(),
            ) {
                Ok(mut g) => {
                    let out = replay(
                        dev,
                        &mut g,
                        input,
                        layers,
                        kv,
                        layout,
                        start_pos,
                        GraphUse::Captured,
                    )?;
                    put(kv, GraphSlot::Ready(g));
                    Ok(out)
                }
                Err(e) => {
                    say_once(&e);
                    let out = run_eager(dev, input, layers, 1, kv, layout, route, start_pos)?;
                    put(kv, GraphSlot::Refused(key));
                    Ok(out)
                }
            }
        }
        Plan::Eager => {
            let out = run_eager(dev, input, layers, 1, kv, layout, route, start_pos)?;
            let capturable = layers
                .iter()
                .zip(&out.device_routed)
                .all(|((l, _), ids)| !matches!(l.ffn, StackFfn::Moe { .. }) || ids.is_some());
            put(
                kv,
                match after_eager(slot.view(), &key, capturable) {
                    Settle::Warm => GraphSlot::Warm(key),
                    Settle::Refused => GraphSlot::Refused(key),
                },
            );
            Ok(out)
        }
    }
}

/// The first capture that fails says why, once a process: a key that
/// fails decodes eagerly, which a measurement must not mistake for a
/// replay.
fn say_once(e: &CudaError) {
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| {
        eprintln!("frink-cuda: a decode token could not be captured into a CUDA graph ({e}); it decodes eagerly");
    });
}

/// Records the run's token: buffers for it on the shared device, then
/// the stack's body on the capture device, the position read from the
/// buffers' word.
fn capture<'e>(
    dev: &Arc<CudaDevice>,
    input: &StackInput<'_>,
    layers: &[(&PrefillDenseLayerCuda<'_>, &PrefillParams<'_>)],
    kv: &mut StackKv<'_, '_>,
    layout: &StageLayout,
    route: Option<&mut RouteFn<'_, 'e>>,
    key: Vec<u64>,
) -> Result<Box<DecodeGraph>, CudaError> {
    let alloc = |e: cudarc::driver::DriverError| {
        CudaError::Launch(format!("decode graph buffers alloc: {e:?}"))
    };
    let mut bufs = StepBuffers {
        pos: dev.alloc_zeros::<i32>(1).map_err(alloc)?,
        input: match input {
            StackInput::Hidden(h) => {
                StepInput::Hidden(dev.alloc_zeros::<f32>(h.len()).map_err(alloc)?)
            }
            StackInput::Gather { ids, .. } => {
                StepInput::Gather(dev.alloc_zeros::<u32>(ids.len()).map_err(alloc)?)
            }
        },
        out: StackOutBufs::alloc(dev, layout)?,
    };
    let mut held = Held::default();
    let (exec, on_device) = {
        let _t = crate::phases::phase("graph capture");
        let StepBuffers {
            pos,
            input: step_in,
            out,
        } = &mut bufs;
        Graph::capture(|cap| {
            let body_in = match (input, step_in) {
                (StackInput::Hidden(_), StepInput::Hidden(h)) => BodyInput::Hidden(h),
                (StackInput::Gather { table, .. }, StepInput::Gather(ids)) => {
                    BodyInput::Gather { table: *table, ids }
                }
                _ => {
                    return Err(CudaError::Launch(
                        "decode graph: an input buffer of another kind".to_string(),
                    ))
                }
            };
            enqueue_body(
                cap,
                body_in,
                layers,
                1,
                kv,
                layout,
                Some(&*pos),
                out,
                route,
                &mut held,
            )
        })?
    };
    // A routed layer routed on the host would have waited for its
    // logits mid-token, which `xfer` refuses inside a capture; a graph
    // without that layer's routing is not this token's.
    if layers
        .iter()
        .zip(&on_device)
        .any(|((l, _), &on)| matches!(l.ffn, StackFfn::Moe { .. }) && !on)
    {
        return Err(CudaError::Unsupported(
            "decode graph: a routed layer the device did not route".to_string(),
        ));
    }
    Ok(Box::new(DecodeGraph {
        key,
        exec,
        bufs,
        _held: held,
        on_device,
    }))
}

/// One replay: the position and the input written into the graph's
/// buffers, the graph launched, and the stack's one wait.
#[allow(clippy::too_many_arguments)]
fn replay(
    dev: &Arc<CudaDevice>,
    g: &mut DecodeGraph,
    input: StackInput<'_>,
    layers: &[(&PrefillDenseLayerCuda<'_>, &PrefillParams<'_>)],
    kv: &mut StackKv<'_, '_>,
    layout: &StageLayout,
    start_pos: usize,
    how: GraphUse,
) -> Result<PrefillStackOut, CudaError> {
    // The kernels index the mirrors by the word: every mirror must hold
    // its row (the reserve before this made room for it).
    let fits = match &*kv {
        StackKv::Mirrors(slots) => slots.iter().all(|s| {
            s.as_ref()
                .is_some_and(|m: &CudaKvMirror| start_pos < m.cap_rows())
        }),
        StackKv::UploadPrefix => false,
    };
    let word = i32::try_from(start_pos)
        .ok()
        .filter(|_| fits)
        .ok_or_else(|| {
            CudaError::Launch(format!(
                "decode graph: position {start_pos} past a mirror's capacity"
            ))
        })?;
    xfer::htod_into(dev, "graph position", &[word], &mut g.bufs.pos)?;
    match (input, &mut g.bufs.input) {
        (StackInput::Hidden(h), StepInput::Hidden(b)) => {
            xfer::htod_into(dev, "graph hidden", h, b)?
        }
        (StackInput::Gather { ids, .. }, StepInput::Gather(b)) => {
            xfer::htod_into(dev, "graph ids", ids, b)?
        }
        _ => {
            return Err(CudaError::Launch(
                "decode graph: an input of another kind than its capture's".to_string(),
            ))
        }
    }
    {
        let _t = crate::phases::phase("graph launch");
        g.exec.launch(dev)?;
    }
    finish(
        dev,
        &g.bufs.out,
        &g.on_device,
        layout,
        kv,
        layers,
        1,
        start_pos,
        how,
    )
}
