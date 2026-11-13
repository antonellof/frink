//! GPU time per dispatch KIND inside one command buffer, measured with
//! the GPU's own timestamp counters rather than inferred from a diff.
//!
//! `crate::timing` clocks a whole command buffer. That was enough to
//! show that Gemma-2-2B's GPU time alone exceeds llama.cpp's whole
//! token (PR #202), and not enough to say WHICH kernel. Apple GPUs
//! sample the timestamp counter only at encoder boundaries
//! (`MTLCounterSamplingPointAtStageBoundary`; dispatch-boundary
//! sampling is unsupported on every Apple family), so the only way to
//! time one dispatch kind is to give it an encoder of its own. With
//! `FRINK_METAL_KERNEL_TIMING=1` the dense decode stack does exactly
//! that: every hazard-tracked op group ends the running encoder and
//! opens a fresh one whose start and end are sampled under a label.
//!
//! What the instrument costs, and why it is still an instrument: each
//! encoder boundary is a full pipeline drain, so the sum of spans is
//! LARGER than the same command buffer un-instrumented. The per-span
//! figures are therefore upper bounds for the small kernels and close
//! to exact for the large ones, and the gaps between spans are reported
//! as their own row rather than folded into a neighbour. The ratio
//! between two models measured the same way is the number this exists
//! for. It is off by default and costs nothing when off: [`SpanClock`]
//! is a `None` and every call on it is one branch.

use crate::gpu::MetalError;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLComputePassDescriptor,
    MTLCounterResultTimestamp, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSamplingPoint, MTLCounterSet, MTLDevice, MTLDispatchType, MTLStorageMode,
};
use std::collections::BTreeMap;
use std::sync::Mutex;

/// True when `FRINK_METAL_KERNEL_TIMING` is set (cached; read once).
pub(crate) fn kernel_timing_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FRINK_METAL_KERNEL_TIMING").is_some())
}

type Encoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

/// The timestamp sample buffer of one command buffer and the label of
/// every span sampled into it. `None` inside when the instrument is off.
pub(crate) struct SpanClock(Option<Inner>);

struct Inner {
    buf: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    labels: Vec<&'static str>,
    cap: usize,
}

impl SpanClock {
    /// A clock for a command buffer that will open at most `max_spans`
    /// spans. Off (a no-op clock) unless `FRINK_METAL_KERNEL_TIMING` is
    /// set; also off, with one warning, on a device that cannot sample
    /// at stage boundaries.
    pub(crate) fn begin(device: &ProtocolObject<dyn MTLDevice>, max_spans: usize) -> Self {
        if !kernel_timing_enabled() {
            return Self(None);
        }
        if !device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary) {
            warn_once("this device cannot sample GPU timestamps at encoder boundaries");
            return Self(None);
        }
        let Some(sets) = device.counterSets() else {
            warn_once("this device exposes no counter sets");
            return Self(None);
        };
        // The timestamp set's name is the string Metal documents for
        // `MTLCommonCounterSetTimestamp`, compared by value so the
        // extern static need not be read.
        let timestamp_set = sets.iter().find(|s| s.name().to_string() == "timestamp");
        let Some(set) = timestamp_set else {
            warn_once("this device has no timestamp counter set");
            return Self(None);
        };
        let desc = MTLCounterSampleBufferDescriptor::new();
        desc.setCounterSet(Some(&set));
        desc.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: two samples per span; the buffer is sized for the
        // caller's declared maximum and `span` refuses to exceed it.
        unsafe { desc.setSampleCount(2 * max_spans) };
        match device.newCounterSampleBufferWithDescriptor_error(&desc) {
            Ok(buf) => Self(Some(Inner {
                buf,
                labels: Vec::with_capacity(max_spans),
                cap: max_spans,
            })),
            Err(e) => {
                warn_once(&format!("counter sample buffer refused: {e}"));
                Self(None)
            }
        }
    }

    /// Close `encoder` and replace it with a fresh concurrent encoder
    /// whose GPU start and end are sampled under `label`. A no-op when
    /// the clock is off, so the encode loop reads the same either way.
    ///
    /// Past `max_spans` the encoder is left alone and the work joins the
    /// previous span, which the report cannot tell apart from a real
    /// span, so `finish` says when that happened.
    pub(crate) fn span(
        &mut self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        encoder: &mut Encoder,
        label: &'static str,
    ) -> Result<(), MetalError> {
        let Some(inner) = self.0.as_mut() else {
            return Ok(());
        };
        if inner.labels.len() >= inner.cap {
            return Ok(());
        }
        let i = inner.labels.len();
        encoder.endEncoding();
        let desc = MTLComputePassDescriptor::new();
        desc.setDispatchType(MTLDispatchType::Concurrent);
        let attachments = desc.sampleBufferAttachments();
        // SAFETY: index 0 is always a valid attachment slot, and the two
        // sample indices are below the count the buffer was created with.
        unsafe {
            let att = attachments.objectAtIndexedSubscript(0);
            att.setSampleBuffer(Some(&inner.buf));
            att.setStartOfEncoderSampleIndex(2 * i);
            att.setEndOfEncoderSampleIndex(2 * i + 1);
        }
        *encoder = cmd_buf
            .computeCommandEncoderWithDescriptor(&desc)
            .ok_or(MetalError::CommandFailed)?;
        inner.labels.push(label);
        Ok(())
    }

    /// After the command buffer has completed: read every span's
    /// timestamps and account them under `tag`, printing a table every
    /// `every` command buffers.
    pub(crate) fn finish(
        self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        tag: &'static str,
        every: u64,
    ) {
        let Some(inner) = self.0 else {
            return;
        };
        let n = inner.labels.len();
        if n == 0 {
            return;
        }
        if n >= inner.cap {
            warn_once(&format!(
                "kernel timing: {tag} opened {n} spans, the cap; later work joined the last span"
            ));
        }
        // SAFETY: the range is within the sample count the buffer was
        // created with, and the command buffer that wrote it has completed.
        let Some(data) = (unsafe {
            inner.buf.resolveCounterRange(NSRange {
                location: 0,
                length: 2 * n,
            })
        }) else {
            warn_once("resolveCounterRange returned nothing");
            return;
        };
        let bytes = data.to_vec();
        let stride = std::mem::size_of::<MTLCounterResultTimestamp>();
        if bytes.len() < 2 * n * stride {
            warn_once("resolved fewer timestamps than spans");
            return;
        }
        let ts = |k: usize| -> u64 {
            let mut raw = [0u8; 8];
            raw.copy_from_slice(&bytes[k * stride..k * stride + 8]);
            u64::from_le_bytes(raw)
        };
        let mut spans = Vec::with_capacity(n);
        for (i, label) in inner.labels.iter().enumerate() {
            spans.push((*label, ts(2 * i), ts(2 * i + 1)));
        }
        let gpu_ns = ((cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e9).max(0.0) as u64;
        let report = match LEDGER.lock() {
            Ok(mut l) => l.note(tag, &spans, gpu_ns, every),
            Err(_) => None,
        };
        if let Some(r) = report {
            r.print();
        }
    }
}

fn warn_once(msg: &str) {
    static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    WARNED.get_or_init(|| eprintln!("frink: metal kernel timing off: {msg}"));
}

static LEDGER: Mutex<SpanLedger> = Mutex::new(SpanLedger::new());

/// Per-tag totals over every command buffer finished so far.
struct SpanLedger {
    slots: Vec<TagSlot>,
}

struct TagSlot {
    tag: &'static str,
    n: u64,
    /// Ticks per label, and how many spans carried it, summed over `n`.
    by_label: BTreeMap<&'static str, (u64, u64)>,
    /// Ticks from the first span's start to the last span's end.
    extent_ticks: u64,
    /// Ticks inside spans (extent minus the gaps between encoders).
    span_ticks: u64,
    /// GPU-clock nanoseconds of the whole command buffer.
    gpu_ns: u64,
}

/// One tag's per-command-buffer means, in GPU-clock milliseconds: the
/// ticks of each label scaled by `gpu_ns / extent_ticks`, so the rows
/// sum to the command buffer's own GPU time plus nothing.
#[derive(Debug, Clone, PartialEq)]
struct SpanReport {
    tag: &'static str,
    n: u64,
    gpu_ms: f64,
    /// (label, ms per command buffer, spans per command buffer)
    rows: Vec<(&'static str, f64, f64)>,
    /// Between-encoder time per command buffer, ms.
    gaps_ms: f64,
    /// Instrument overhead: how many ns one tick is, from the last note.
    ns_per_tick: f64,
}

impl SpanLedger {
    const fn new() -> Self {
        Self { slots: Vec::new() }
    }

    fn note(
        &mut self,
        tag: &'static str,
        spans: &[(&'static str, u64, u64)],
        gpu_ns: u64,
        every: u64,
    ) -> Option<SpanReport> {
        let slot = match self.slots.iter_mut().position(|s| s.tag == tag) {
            Some(i) => &mut self.slots[i],
            None => {
                self.slots.push(TagSlot {
                    tag,
                    n: 0,
                    by_label: BTreeMap::new(),
                    extent_ticks: 0,
                    span_ticks: 0,
                    gpu_ns: 0,
                });
                self.slots.last_mut().expect("just pushed")
            }
        };
        slot.n += 1;
        slot.gpu_ns += gpu_ns;
        // An encoder that ran nothing samples 0 for both ends on an M5
        // (and `MTLCounterErrorValue` where sampling failed). Such a span
        // took no time, and must not stretch the extent back to tick 0,
        // which scaled every row to zero.
        let sampled =
            |s: &&(&'static str, u64, u64)| !(s.1 == 0 && s.2 == 0) && s.2 != u64::MAX && s.2 >= s.1;
        let first = spans.iter().filter(sampled).map(|s| s.1).min().unwrap_or(0);
        let last = spans.iter().filter(sampled).map(|s| s.2).max().unwrap_or(0);
        slot.extent_ticks += last.saturating_sub(first);
        for span in spans {
            let (label, start, end) = span;
            let d = if sampled(&span) { end - start } else { 0 };
            slot.span_ticks += d;
            let e = slot.by_label.entry(label).or_insert((0, 0));
            e.0 += d;
            e.1 += 1;
        }
        if slot.n.is_multiple_of(every.max(1)) {
            Some(slot.report())
        } else {
            None
        }
    }
}

impl TagSlot {
    fn report(&self) -> SpanReport {
        let n = self.n.max(1) as f64;
        let ns_per_tick = self.gpu_ns as f64 / self.extent_ticks.max(1) as f64;
        let ms = |ticks: u64| ticks as f64 * ns_per_tick / 1e6 / n;
        let mut rows: Vec<(&'static str, f64, f64)> = self
            .by_label
            .iter()
            .map(|(label, (ticks, count))| (*label, ms(*ticks), *count as f64 / n))
            .collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        SpanReport {
            tag: self.tag,
            n: self.n,
            gpu_ms: self.gpu_ns as f64 / 1e6 / n,
            rows,
            gaps_ms: ms(self.extent_ticks.saturating_sub(self.span_ticks)),
            ns_per_tick,
        }
    }
}

impl SpanReport {
    fn print(&self) {
        eprintln!(
            "frink: metal kernels[{}] {:.3} ms/cb over {} cbs ({:.3} ns/tick):",
            self.tag, self.gpu_ms, self.n, self.ns_per_tick
        );
        eprintln!(
            "  {:<16} {:>9} {:>7} {:>8} {:>9}",
            "kind", "ms/cb", "%", "n/cb", "us/each"
        );
        for (label, ms, count) in &self.rows {
            eprintln!(
                "  {:<16} {:>9.3} {:>6.1}% {:>8.1} {:>9.1}",
                label,
                ms,
                100.0 * ms / self.gpu_ms.max(f64::MIN_POSITIVE),
                count,
                1000.0 * ms / count.max(f64::MIN_POSITIVE)
            );
        }
        eprintln!(
            "  {:<16} {:>9.3} {:>6.1}%",
            "(encoder gaps)",
            self.gaps_ms,
            100.0 * self.gaps_ms / self.gpu_ms.max(f64::MIN_POSITIVE)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rows are the command buffer's GPU time divided among the
    /// labels in proportion to their ticks, with the between-encoder
    /// gaps as their own row, so the rows plus the gap row sum to the
    /// command buffer's GPU time. A ledger that scaled by span ticks
    /// instead of extent ticks would hide the instrument's own overhead
    /// inside the kernels it is supposed to measure.
    #[test]
    fn rows_and_gaps_sum_to_the_command_buffers_gpu_time() {
        let mut l = SpanLedger::new();
        // Two spans of 300 and 100 ticks with a 100-tick gap: extent
        // 500 ticks, and the command buffer took 5 ms, so 10 us/tick.
        let spans = [("matvec", 1000, 1300), ("norm", 1400, 1500)];
        let r = l.note("t", &spans, 5_000_000, 1).unwrap();
        assert!((r.ns_per_tick - 10_000.0).abs() < 1e-6);
        let total: f64 = r.rows.iter().map(|r| r.1).sum::<f64>() + r.gaps_ms;
        assert!((total - 5.0).abs() < 1e-9, "rows + gaps = {total}, not 5.0");
        assert_eq!(r.rows[0], ("matvec", 3.0, 1.0));
        assert_eq!(r.rows[1], ("norm", 1.0, 1.0));
        assert!((r.gaps_ms - 1.0).abs() < 1e-9);
    }

    /// An M5 samples an encoder that ran nothing as 0..0. That span
    /// costs nothing, and the extent starts at the first real sample:
    /// counting it from tick 0 scaled every row of the table to zero.
    #[test]
    fn an_unsampled_empty_encoder_does_not_stretch_the_extent() {
        let mut l = SpanLedger::new();
        let spans = [("(empty encoder)", 0, 0), ("matvec", 1000, 1300), ("norm", 1400, 1500)];
        let r = l.note("t", &spans, 5_000_000, 1).unwrap();
        assert!((r.ns_per_tick - 10_000.0).abs() < 1e-6, "{}", r.ns_per_tick);
        assert_eq!(r.rows.iter().find(|r| r.0 == "matvec").unwrap().1, 3.0);
    }

    /// Per-command-buffer means: the same label across two buffers
    /// reports the mean per buffer and the mean span count per buffer,
    /// not the sum.
    #[test]
    fn a_label_seen_twice_per_buffer_reports_the_per_buffer_mean() {
        let mut l = SpanLedger::new();
        assert!(l
            .note("t", &[("norm", 0, 10), ("norm", 10, 30)], 4_000, 2)
            .is_none());
        let r = l
            .note("t", &[("norm", 0, 10), ("norm", 10, 30)], 4_000, 2)
            .unwrap();
        assert_eq!(r.n, 2);
        assert_eq!(r.rows.len(), 1);
        let (label, ms, per_cb) = r.rows[0];
        assert_eq!(label, "norm");
        assert!(
            (ms - 0.004).abs() < 1e-12,
            "4000 ns per buffer, got {ms} ms"
        );
        assert_eq!(per_cb, 2.0, "two spans per buffer, not four");
        assert_eq!(r.gaps_ms, 0.0);
    }
}
