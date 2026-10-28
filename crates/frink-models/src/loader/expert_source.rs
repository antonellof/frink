//! Routed experts left on disk: the byte layout of one expert's three
//! matrices, the `ExpertSource` that reads them out of the GGUF shards
//! on demand, and the per-tensor specs it is built from.

use frink_core::expert_store::{ExpertKey, ExpertSource};
use frink_core::weight_matrix::quant_kind_for;
use frink_core::weight_matrix::{QuantKind, WeightBytes, WeightMatrix};
use frink_gguf::ShardedGguf;
use frink_moe::ExpertWeights;

use super::tensors::find_info;
use super::LoadError;

/// One matrix's place inside a store-backed expert's combined byte
/// buffer (gate bytes, then up, then down, concatenated by
/// `GgufExpertSource::read_expert`).
#[derive(Debug, Clone, Copy)]
pub struct StoredMatrixSpec {
    pub offset: usize,
    pub len: usize,
    pub rows: usize,
    pub cols: usize,
    pub kind: QuantKind,
}

/// Byte-range layout of one store-backed routed expert.
#[derive(Debug, Clone, Copy)]
pub struct StoredExpertLayout {
    pub gate: StoredMatrixSpec,
    pub up: StoredMatrixSpec,
    pub down: StoredMatrixSpec,
}

impl StoredExpertLayout {
    pub fn total_bytes(&self) -> usize {
        self.gate.len + self.up.len + self.down.len
    }

    /// Builds temporary zero-copy `WeightMatrix` views over a leased
    /// buffer. Each view's `WeightBytes::Shared` clone of the lease's
    /// `Arc` keeps the cache entry pinned for the view's lifetime.
    pub fn materialize(&self, lease: &frink_core::expert_store::ExpertLease) -> ExpertWeights {
        let mk = |spec: &StoredMatrixSpec| WeightMatrix::Quantized {
            data: WeightBytes::Shared {
                buf: lease.shared_buf(),
                range: spec.offset..spec.offset + spec.len,
            },
            rows: spec.rows,
            cols: spec.cols,
            kind: spec.kind,
        };
        ExpertWeights {
            gate: mk(&self.gate),
            up: mk(&self.up),
            down: mk(&self.down),
        }
    }
}

/// [`ExpertSource`] over a (possibly sharded) GGUF checkpoint: each
/// expert's gate/up/down byte ranges are read positionally from the
/// owning shard file and concatenated, so a store miss touches exactly
/// that expert's bytes -- no mmap of the expert region, no shared seek
/// cursor.
pub struct GgufExpertSource {
    pub(super) files: Vec<std::fs::File>,
    /// (layer, expert) -> the three (file index, offset, len) segments
    /// in gate/up/down order.
    pub(super) segments: std::collections::HashMap<ExpertKey, [(usize, u64, usize); 3]>,
}

impl ExpertSource for GgufExpertSource {
    fn expert_len(&self, key: ExpertKey) -> Option<usize> {
        self.segments
            .get(&key)
            .map(|segs| segs.iter().map(|&(_, _, len)| len).sum())
    }

    fn read_expert(&self, key: ExpertKey) -> std::io::Result<Vec<u8>> {
        let segs = self
            .segments
            .get(&key)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, format!("{key:?}")))?;
        let total: usize = segs.iter().map(|&(_, _, len)| len).sum();
        let mut buf = vec![0u8; total];
        let mut written = 0;
        for &(fi, offset, len) in segs {
            let dst = &mut buf[written..written + len];
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileExt;
                self.files[fi].read_exact_at(dst, offset)?;
            }
            #[cfg(not(unix))]
            {
                use std::io::{Read, Seek, SeekFrom};
                let mut f = &self.files[fi];
                f.seek(SeekFrom::Start(offset))?;
                f.read_exact(dst)?;
            }
            written += len;
        }
        Ok(buf)
    }
}

/// Collects the per-expert `(file, offset, len)` segments and layout
/// for one packed 3D expert tensor -- the store-backed counterpart of
/// `split_expert_tensor`, sharing its shape/offset math. Only
/// quantized dtypes are supported (an F32/BF16 expert tensor keeps the
/// resident path; the store exists for the quantized multi-hundred-GB
/// case).
/// One packed 3D expert tensor's store-backed description: the owning
/// shard index, each expert's `(offset, len)` within that shard file,
/// and the matrix spec shared by every expert's slice.
pub(super) struct StoredTensorSpecs {
    pub(super) shard: usize,
    pub(super) per_expert: Vec<(u64, usize)>,
    pub(super) spec: StoredMatrixSpec,
}

pub(super) fn stored_expert_specs(
    file: &ShardedGguf,
    name: &str,
    n_experts: usize,
) -> Result<Option<StoredTensorSpecs>, LoadError> {
    let info = find_info(file, name)?;
    if info.shape.len() != 3 || info.shape[2] as usize != n_experts {
        let file_experts = info.shape.last().map(|&d| d as usize).unwrap_or(0);
        return Err(LoadError::ExpertCountMismatch(
            name.to_string(),
            file_experts,
            n_experts,
        ));
    }
    let out_dim = info.shape[1] as usize;
    let in_dim = info.shape[0] as usize;
    let Some(kind) = quant_kind_for(info.dtype) else {
        return Ok(None); // F32/BF16 (or unsupported): resident fallback
    };
    let shard = file
        .tensor_shard_index(name)
        .expect("find_info succeeded, shard index must exist");
    // The mmap range of a tensor within a GgufFile IS its byte offset
    // range within that shard file (the mmap covers the whole file).
    let (_, full_range) = file.tensor_mapped_range(name)?;
    let total_len = full_range.end - full_range.start;
    let bytes_per_expert = total_len / n_experts;
    let per_expert: Vec<(u64, usize)> = (0..n_experts)
        .map(|e| {
            (
                (full_range.start + e * bytes_per_expert) as u64,
                bytes_per_expert,
            )
        })
        .collect();
    let spec = StoredMatrixSpec {
        offset: 0, // caller assigns the position within the combined buffer
        len: bytes_per_expert,
        rows: out_dim,
        cols: in_dim,
        kind,
    };
    Ok(Some(StoredTensorSpecs {
        shard,
        per_expert,
        spec,
    }))
}
