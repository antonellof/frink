//! The device ops a decode layer chains between its matvecs: RMSNorm,
//! the residual add, RoPE, the f16 KV append and attention. Each records
//! into a [`Batch`] that [`Runtime::submit_batch`] runs as ONE
//! submission, so a layer can stay on the device from its input norm to
//! its residual (#409, step 5).
//!
//! The data lives in [`DeviceBuf`]s, host-visible buffers the runtime
//! owns: written and read through their mappings, bound by the ops. Each
//! op is its own stage, so a barrier separates it from the next and an
//! op may read what the one before it wrote.
//!
//! Every op is held to a scalar twin here and, through `frink-core`'s
//! seam tests, to the CPU function it replaces.

use super::memory::{alloc, free, write_padded, Buf};
use super::{grid, push_words, Dispatch, MatvecJob, Runtime, MAX_DISPATCHES};
use crate::device::VulkanError;
use crate::shaders::{
    ElementwiseShader, ADD, ATTN_SCORES, ATTN_VALUES, KV_STORE, RMS_NORM, ROPE, SWIGLU,
};
use frink_quant::fingerprint::WeightFingerprint;
use std::sync::Mutex;

/// Buffers dropped while the runtime may be locked, freed at the next
/// [`Runtime::submit_batch`]. A KV mirror is dropped with its sequence's
/// cache, which can happen on a thread that holds the runtime's lock, so
/// its `Drop` queues the buffer here instead of locking.
///
/// Each entry names the runtime that allocated it ([`Runtime::id`]): a
/// [`DeviceBuf`] is an index, and freeing it in another runtime would
/// free an unrelated buffer.
static PENDING_FREE: Mutex<Vec<(u64, DeviceBuf)>> = Mutex::new(Vec::new());

/// Free `b`, which runtime `owner` allocated, at that runtime's next
/// submission. Safe to call from a `Drop` on any thread.
pub fn free_later(owner: u64, b: DeviceBuf) {
    if let Ok(mut v) = PENDING_FREE.lock() {
        v.push((owner, b));
    }
}

pub(super) fn next_runtime_id() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// A runtime is going away and frees all its buffers itself.
pub(super) fn forget_pending(owner: u64) {
    if let Ok(mut v) = PENDING_FREE.lock() {
        v.retain(|(o, _)| *o != owner);
    }
}

/// The buffers queued for runtime `owner`, taken off the queue.
fn take_pending(owner: u64) -> Vec<DeviceBuf> {
    let Ok(mut v) = PENDING_FREE.lock() else {
        return Vec::new();
    };
    let mut mine = Vec::new();
    v.retain(|(o, b)| {
        if *o == owner {
            mine.push(*b);
        }
        *o != owner
    });
    mine
}
use ash::vk;

/// A buffer the runtime allocated with [`Runtime::alloc_f32`], by index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBuf(usize);

/// Which mean a norm divides by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    /// `x / sqrt(mean(x^2) + eps)`.
    Rms,
    /// `(x - mean) / sqrt(var + eps)`.
    Layer,
}

/// The shape of one decode attention: `q` is `[n_heads, head_dim]`, the
/// cache holds `seq_len` positions of `[n_kv_heads, head_dim]` K at word
/// `k_off` and V at word `v_off`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnShape {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub k_off: usize,
    pub v_off: usize,
    pub scale: f32,
    /// A sliding window: each query attends over its own position and
    /// the `window - 1` before it. `None` is everything cached.
    pub window: Option<usize>,
}

/// What one bound op reads and writes: three `(buffer, f32 offset,
/// f32 length)` views, one per binding.
type Views = [(DeviceBuf, usize, usize); 3];

pub(super) enum Op<'w> {
    Shader {
        shader: ElementwiseShader,
        views: Views,
        push: [u32; 8],
        grid: (u32, u32),
        /// The grid's z: 1, or the batch positions of a shader that
        /// reads one from it (RoPE, attention).
        depth: u32,
    },
    Matvec {
        job: MatvecJob<'w>,
        x: (DeviceBuf, usize, usize),
        y: (DeviceBuf, usize, usize),
        /// Batch positions: a GEMM when above 1.
        depth: usize,
    },
    /// One row of a table, dequantized (`runtime::gather`).
    Gather {
        table: MatvecJob<'w>,
        row: usize,
        out: (DeviceBuf, usize, usize),
    },
}

/// Ops for one submission, in order. Build with the methods, run with
/// [`Runtime::submit_batch`].
///
/// A batch is recorded for a number of POSITIONS, one by default (a
/// decode token). [`Batch::for_positions`] records the same ops over
/// `p` consecutive positions (a prefill): every buffer then holds `p`
/// rows, position after position, and each op's sizes below are per
/// position.
pub struct Batch<'w> {
    ops: Vec<Op<'w>>,
    positions: usize,
}

impl Default for Batch<'_> {
    fn default() -> Self {
        Self::for_positions(1)
    }
}

/// The grid for one invocation per value, 64 to a workgroup.
fn per_value(n: usize) -> (u32, u32) {
    grid(n.div_ceil(crate::shaders::LOCAL_SIZE_X as usize))
}

fn words(w: &[u32]) -> [u32; 8] {
    let mut out = [0u32; 8];
    out[..w.len()].copy_from_slice(w);
    out
}

impl<'w> Batch<'w> {
    pub fn new() -> Self {
        Self::default()
    }

    /// A batch whose ops each cover `positions` consecutive positions.
    pub fn for_positions(positions: usize) -> Self {
        assert!(positions >= 1, "a batch of no positions");
        Self {
            ops: Vec::new(),
            positions,
        }
    }

    pub(super) fn push_op(&mut self, op: Op<'w>) {
        assert_eq!(
            self.positions, 1,
            "this op is for the batch of one position"
        );
        self.ops.push(op);
    }

    /// Dispatches this batch will record.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// `rows` rows of `n`: `out = x * w / sqrt(mean(x^2) + eps)`, `w`
    /// of length `n` shared by every row.
    pub fn rms_norm(
        &mut self,
        w: DeviceBuf,
        x: DeviceBuf,
        out: DeviceBuf,
        n: usize,
        rows: usize,
        eps: f32,
    ) {
        self.norm(NormKind::Rms, Some(w), x, out, n, rows, eps);
    }

    /// A norm over `rows` rows of `n`: RMS, or LayerNorm (the mean
    /// subtracted, the divisor the variance), times `w` when there is
    /// one. A bias is [`Self::add_bias`] after it.
    #[allow(clippy::too_many_arguments)]
    pub fn norm(
        &mut self,
        kind: NormKind,
        w: Option<DeviceBuf>,
        x: DeviceBuf,
        out: DeviceBuf,
        n: usize,
        rows: usize,
        eps: f32,
    ) {
        let rows = rows * self.positions;
        let mode = (kind == NormKind::Layer) as u32 | (w.is_none() as u32) << 1;
        // Binding 0 is not read without a weight; it must still be a
        // buffer, and `x` holds at least `n` values.
        self.ops.push(Op::Shader {
            shader: RMS_NORM,
            views: [(w.unwrap_or(x), 0, n), (x, 0, n * rows), (out, 0, n * rows)],
            push: words(&[n as u32, rows as u32, eps.to_bits(), mode]),
            grid: grid(rows),
            depth: 1,
        });
    }

    /// `h = silu(g) * u` over `n` values (the SwiGLU combine).
    pub fn swiglu(&mut self, g: DeviceBuf, u: DeviceBuf, out: DeviceBuf, n: usize) {
        self.glu(g, u, out, n, false);
    }

    /// The gated combine `h = act(g) * u` over `n` values: SiLU, or the
    /// tanh-approximation GELU of a GeGLU FFN when `gelu`.
    pub fn glu(&mut self, g: DeviceBuf, u: DeviceBuf, out: DeviceBuf, n: usize, gelu: bool) {
        let n = n * self.positions;
        self.ops.push(Op::Shader {
            shader: SWIGLU,
            views: [(g, 0, n), (u, 0, n), (out, 0, n)],
            push: words(&[n as u32, gelu as u32]),
            grid: per_value(n),
            depth: 1,
        });
    }

    /// `out = a + b` over `n` values.
    pub fn add(&mut self, a: DeviceBuf, b: DeviceBuf, out: DeviceBuf, n: usize) {
        let n = n * self.positions;
        self.ops.push(Op::Shader {
            shader: ADD,
            views: [(a, 0, n), (b, 0, n), (out, 0, n)],
            push: words(&[n as u32]),
            grid: per_value(n),
            depth: 1,
        });
    }

    /// `out = a + bias` for every position: `bias` is `n` values, added
    /// to each position's `n` values of `a`.
    pub fn add_bias(&mut self, a: DeviceBuf, bias: DeviceBuf, out: DeviceBuf, n: usize) {
        let total = n * self.positions;
        self.ops.push(Op::Shader {
            shader: ADD,
            views: [(a, 0, total), (bias, 0, n), (out, 0, total)],
            push: words(&[total as u32, n as u32]),
            grid: per_value(total),
            depth: 1,
        });
    }

    /// RoPE over `n_heads` heads of `head_dim`, the first `rot_dim`
    /// channels rotated by the `rot_dim / 2` `(cos, sin)` pairs in `tab`
    /// (see `shaders/rope.comp`); `neox` picks split-half pairing. Each
    /// position has its own `rot_dim` values of `tab`, one after another.
    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &mut self,
        tab: DeviceBuf,
        x: DeviceBuf,
        out: DeviceBuf,
        n_heads: usize,
        head_dim: usize,
        rot_dim: usize,
        neox: bool,
    ) {
        let (n, p) = (n_heads * head_dim, self.positions);
        self.ops.push(Op::Shader {
            shader: ROPE,
            views: [(tab, 0, rot_dim * p), (x, 0, n * p), (out, 0, n * p)],
            push: words(&[
                n_heads as u32,
                head_dim as u32,
                rot_dim as u32,
                neox as u32,
                rot_dim as u32,
            ]),
            grid: per_value(n),
            depth: p as u32,
        });
    }

    /// Append `n` values of `src` to the f16 `cache` at word `dst`.
    pub fn kv_store(
        &mut self,
        src: DeviceBuf,
        cache: DeviceBuf,
        n: usize,
        dst: usize,
        cache_words: usize,
    ) {
        let n = n * self.positions;
        self.ops.push(Op::Shader {
            shader: KV_STORE,
            views: [(src, 0, n), (src, 0, n), (cache, 0, cache_words)],
            push: words(&[n as u32, dst as u32]),
            grid: per_value(n / 2),
            depth: 1,
        });
    }

    /// Causal attention: `out[h] = softmax(scale * q_h . K) . V`. The
    /// first position attends over `shape.seq_len` cached positions and
    /// each later one over one more, so the batch's own K and V must be
    /// in the cache already. `scores` is scratch for
    /// `n_heads * (seq_len + positions - 1)` values per position.
    pub fn attention(
        &mut self,
        cache: DeviceBuf,
        cache_words: usize,
        q: DeviceBuf,
        scores: DeviceBuf,
        out: DeviceBuf,
        s: AttnShape,
    ) {
        let p = self.positions;
        let stride = s.seq_len + p - 1;
        assert!(
            s.n_heads < 1 << 16 && s.n_kv_heads < 1 << 16,
            "head counts share a word"
        );
        assert!(s.window != Some(0), "a window of no positions");
        let push = words(&[
            s.n_heads as u32 | (s.n_kv_heads as u32) << 16,
            s.window.unwrap_or(0) as u32,
            s.head_dim as u32,
            s.seq_len as u32,
            s.k_off as u32,
            s.v_off as u32,
            s.scale.to_bits(),
            stride as u32,
        ]);
        let n_scores = s.n_heads * stride * p;
        let qn = s.n_heads * s.head_dim * p;
        self.ops.push(Op::Shader {
            shader: ATTN_SCORES,
            views: [(cache, 0, cache_words), (q, 0, qn), (scores, 0, n_scores)],
            push,
            grid: grid(s.n_heads),
            depth: p as u32,
        });
        self.ops.push(Op::Shader {
            shader: ATTN_VALUES,
            views: [(cache, 0, cache_words), (scores, 0, n_scores), (out, 0, qn)],
            push,
            grid: grid(s.n_heads),
            depth: p as u32,
        });
    }

    /// A matvec reading `x[x_off..]` and writing `y[y_off..]`, its
    /// weights resident as in [`Runtime::matvec_many`].
    pub fn matvec(&mut self, job: MatvecJob<'w>, x: (DeviceBuf, usize), y: (DeviceBuf, usize)) {
        let cols = job.weights.len() / job.rows / job.shader.block_bytes * job.shader.block_elems;
        let p = self.positions;
        self.ops.push(Op::Matvec {
            job,
            x: (x.0, x.1, cols * p),
            y: (y.0, y.1, job.rows * p),
            depth: p,
        });
    }
}

impl Runtime {
    /// A host-visible buffer of `n` f32 values (zeroed), owned by the
    /// runtime for its lifetime.
    pub fn alloc_f32(&mut self, n: usize) -> Result<DeviceBuf, VulkanError> {
        let buf = alloc(
            &self.ctx.device,
            &self.mem_props,
            (n * 4).max(4),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        write_padded(&buf, &[]);
        self.bufs.push(Some(buf));
        Ok(DeviceBuf(self.bufs.len() - 1))
    }

    fn buf(&self, b: DeviceBuf) -> &Buf {
        self.bufs[b.0]
            .as_ref()
            .expect("a DeviceBuf used after it was freed")
    }

    /// This runtime's id, the owner [`free_later`] takes.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The f32 capacity of `b`.
    pub fn capacity_f32(&self, b: DeviceBuf) -> usize {
        self.buf(b).size / 4
    }

    /// A buffer of at least `n` f32 values kept under `(name, index)`
    /// and reused by every later call: a decode token's activations
    /// allocate nothing. Grown (and its contents lost) when too small.
    pub fn scratch(
        &mut self,
        name: &'static str,
        index: usize,
        n: usize,
    ) -> Result<DeviceBuf, VulkanError> {
        if let Some(&b) = self.scratch.get(&(name, index)) {
            if self.capacity_f32(b) >= n {
                return Ok(b);
            }
            self.free_now(b);
        }
        let b = self.alloc_f32(n.next_power_of_two())?;
        self.scratch.insert((name, index), b);
        Ok(b)
    }

    /// `data` on the device, uploaded on first use and served by address
    /// after, checked against its fingerprint on every hit as resident
    /// weights are. For model constants such as norm weights.
    pub fn constant_f32(&mut self, data: &[f32]) -> Result<DeviceBuf, VulkanError> {
        let key = (data.as_ptr() as usize, data.len());
        // SAFETY: an f32 slice is plain bytes, 4 per value, alive for
        // this call.
        let bytes =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        let fp = WeightFingerprint::of(bytes);
        if let Some((b, seen)) = self.constants.get(&key) {
            let b = *b;
            if *seen == fp {
                return Ok(b);
            }
            self.free_now(b);
        }
        let b = self.alloc_f32(data.len())?;
        self.write_f32(b, 0, data)?;
        self.constants.insert(key, (b, fp));
        Ok(b)
    }

    /// Free `b` now. Every submission has been waited on, so it is idle.
    fn free_now(&mut self, b: DeviceBuf) {
        if let Some(buf) = self.bufs.get_mut(b.0).and_then(Option::take) {
            // SAFETY: every submission is waited on before its call
            // returns, so no command buffer references `buf`.
            unsafe { free(&self.ctx.device, buf) };
        }
    }

    /// Write raw u32 words (f16 pairs) at word offset `at`.
    pub fn write_u32(&mut self, b: DeviceBuf, at: usize, data: &[u32]) -> Result<(), VulkanError> {
        let buf = self.buf(b);
        if (at + data.len()) * 4 > buf.size {
            return Err(VulkanError::Shape(format!(
                "write of {} words at {at} into a {}-byte buffer",
                data.len(),
                buf.size
            )));
        }
        let ptr = buf.mapped.expect("host-visible");
        // SAFETY: bounds checked above; the device is idle between
        // submissions; `data` is a distinct host allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                ptr.add(at * 4),
                data.len() * 4,
            )
        };
        Ok(())
    }

    /// Raw u32 words from word offset `at`.
    pub fn read_u32(&self, b: DeviceBuf, at: usize, n: usize) -> Result<Vec<u32>, VulkanError> {
        Ok(self
            .read_f32(b, at, n)?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    /// Write `data` at f32 offset `at`. Between submissions only.
    pub fn write_f32(&mut self, b: DeviceBuf, at: usize, data: &[f32]) -> Result<(), VulkanError> {
        let buf = self.buf(b);
        if (at + data.len()) * 4 > buf.size {
            return Err(VulkanError::Shape(format!(
                "write of {} values at {at} into a {}-byte buffer",
                data.len(),
                buf.size
            )));
        }
        let ptr = buf.mapped.expect("host-visible");
        // SAFETY: the range was checked against the mapping's size just
        // above; every submission is waited on, so the device is not
        // using the buffer; `data` is a distinct host allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                ptr.add(at * 4),
                data.len() * 4,
            )
        };
        Ok(())
    }

    /// `n` f32 values from f32 offset `at`. Between submissions only.
    pub fn read_f32(&self, b: DeviceBuf, at: usize, n: usize) -> Result<Vec<f32>, VulkanError> {
        let buf = self.buf(b);
        if (at + n) * 4 > buf.size {
            return Err(VulkanError::Shape(format!(
                "read of {n} values at {at} from a {}-byte buffer",
                buf.size
            )));
        }
        let ptr = buf.mapped.expect("host-visible");
        let mut bytes = vec![0u8; n * 4];
        // SAFETY: bounds checked above; the last submission was waited
        // on with a HOST_READ barrier (`run`), so its writes are visible.
        unsafe { std::ptr::copy_nonoverlapping(ptr.add(at * 4), bytes.as_mut_ptr(), n * 4) };
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// Run every op of `batch` in order as ONE submission: one stage per
    /// op, a barrier between each, one wait.
    pub fn submit_batch(&mut self, batch: &Batch<'_>) -> Result<(), VulkanError> {
        for b in take_pending(self.id) {
            self.free_now(b);
        }
        if batch.ops.is_empty() || batch.ops.len() > MAX_DISPATCHES {
            return Err(VulkanError::Shape(format!(
                "{} ops in one submission (1..={MAX_DISPATCHES})",
                batch.ops.len()
            )));
        }
        let align = self.min_offset_align / 4;
        let mut dispatches = Vec::with_capacity(batch.ops.len());
        for (set, op) in batch.ops.iter().enumerate() {
            let (pipeline, ranges, push, grid, depth) = match op {
                Op::Shader {
                    shader,
                    views,
                    push,
                    grid,
                    depth,
                } => {
                    let ranges = [
                        self.view(views[0])?,
                        self.view(views[1])?,
                        self.view(views[2])?,
                    ];
                    (
                        self.pipeline(shader.spv)?,
                        ranges,
                        push_words(push),
                        *grid,
                        *depth,
                    )
                }
                Op::Gather { table, row, out } => {
                    let (_, dispatch_of) = self.gather_shape(table, *row)?;
                    let w = self.resident(table.weights)?;
                    let o = self.view(*out)?;
                    // Binding 1 is not read by a gather.
                    let ranges = [(w, 0, vk::WHOLE_SIZE), o, o];
                    let d = dispatch_of(self.pipeline(table.shader.gather)?, set);
                    (d.pipeline, ranges, d.push, d.grid, d.depth)
                }
                Op::Matvec { job, x, y, depth } => {
                    let shape = self.check(job, x.2 / depth)?;
                    let w = self.resident(job.weights)?;
                    let ranges = [(w, 0, vk::WHOLE_SIZE), self.view(*x)?, self.view(*y)?];
                    let (rows, n_blocks, row_bytes) = shape;
                    (
                        self.kernel(job.shader)?,
                        ranges,
                        push_words(&[
                            rows as u32,
                            n_blocks as u32,
                            row_bytes as u32,
                            (x.2 / depth) as u32,
                        ]),
                        grid(rows),
                        *depth as u32,
                    )
                }
            };
            for (_, offset, _) in ranges {
                if offset % (align * 4) != 0 {
                    return Err(VulkanError::Shape(format!(
                        "op {set}: a view at byte {offset} is not at a {}-byte storage offset",
                        align * 4
                    )));
                }
            }
            self.write_set(set, ranges);
            dispatches.push(Dispatch {
                pipeline,
                set,
                push,
                grid,
                depth,
            });
        }
        let stages: Vec<&[Dispatch]> = dispatches.chunks(1).collect();
        self.run(&stages)
    }

    /// A `(buffer, byte offset, byte range)` for an f32 view, checked
    /// against the buffer's size.
    fn view(
        &self,
        (b, at, n): (DeviceBuf, usize, usize),
    ) -> Result<(vk::Buffer, u64, u64), VulkanError> {
        let buf = self.buf(b);
        if n == 0 || (at + n) * 4 > buf.size {
            return Err(VulkanError::Shape(format!(
                "a view of {n} values at {at} does not fit a {}-byte buffer",
                buf.size
            )));
        }
        Ok((buf.buffer, at as u64 * 4, n as u64 * 4))
    }
}

#[cfg(test)]
mod tests;
