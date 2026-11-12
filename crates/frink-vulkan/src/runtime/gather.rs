//! The embedding gather: one row of a quantized table, dequantized on
//! the device. It is the kind's matvec shader compiled with `-DGATHER`,
//! which swaps the activation for a one-hot (`shaders/matvec_common.glsl`),
//! so a gather is its matvec's decode and nothing is written twice.
//!
//! The price of that is work: every output column walks the whole row,
//! where a dedicated dequant would decode it once. It is one dispatch a
//! token, the size of a square matvec, and can be tuned on hardware.

use super::ops::{Batch, DeviceBuf, Op};
use super::{grid, push_words, Dispatch, MatvecJob, Runtime};
use crate::device::VulkanError;
use ash::vk;

impl<'w> Batch<'w> {
    /// Row `row` of `table` into `out`, `cols` values. For the batch of
    /// one position: a prefill embeds on the host.
    pub fn gather(&mut self, table: MatvecJob<'w>, row: usize, out: DeviceBuf) {
        let cols =
            table.weights.len() / table.rows / table.shader.block_bytes * table.shader.block_elems;
        self.push_op(Op::Gather {
            table,
            row,
            out: (out, 0, cols),
        });
    }
}

impl Runtime {
    /// Row `row` of `table`, dequantized: what `frink_quant`'s dequant
    /// of that row gives, to the rounding of one multiply.
    pub fn get_row(&mut self, table: MatvecJob<'_>, row: usize) -> Result<Vec<f32>, VulkanError> {
        let (cols, dispatch_of) = self.gather_shape(&table, row)?;
        let pipeline = self.pipeline(table.shader.gather)?;
        let w_buf = self.resident(table.weights)?;
        let y_buf = Self::grow(&self.ctx.device, &self.mem_props, &mut self.y, cols * 4)?;
        // Binding 1 is not read by a gather; it must still be a buffer.
        self.write_set(
            0,
            [
                (w_buf, 0, vk::WHOLE_SIZE),
                (y_buf, 0, vk::WHOLE_SIZE),
                (y_buf, 0, (cols * 4) as u64),
            ],
        );
        let dispatch = [dispatch_of(pipeline, 0)];
        self.run(&[&dispatch])?;
        Ok(self.read_y(0, cols))
    }

    /// The row width, and the dispatch for a pipeline and a set, after
    /// checking that the table is whole rows and holds `row`.
    #[allow(clippy::type_complexity)]
    pub(super) fn gather_shape(
        &self,
        table: &MatvecJob<'_>,
        row: usize,
    ) -> Result<(usize, impl Fn(vk::Pipeline, usize) -> Dispatch), VulkanError> {
        let s = table.shader;
        let row_bytes = table.weights.len().checked_div(table.rows).unwrap_or(0);
        let n_blocks = row_bytes / s.block_bytes;
        if n_blocks == 0
            || row_bytes != n_blocks * s.block_bytes
            || table.weights.len() != table.rows * row_bytes
            || row >= table.rows
        {
            return Err(VulkanError::Shape(format!(
                "{} gather: row {row} of {} rows in {} weight bytes",
                s.kind,
                table.rows,
                table.weights.len()
            )));
        }
        let cols = n_blocks * s.block_elems;
        // The same limits a matvec of this table has.
        self.check(table, cols)?;
        let push = push_words(&[
            table.rows as u32,
            n_blocks as u32,
            row_bytes as u32,
            row as u32,
            cols as u32,
        ]);
        Ok((cols, move |pipeline, set| Dispatch {
            pipeline,
            set,
            push,
            grid: grid(cols),
            depth: 1,
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::dispatch::tests::{kind_row, runtime, twin};
    use crate::runtime::ops::Batch;
    use crate::runtime::MatvecJob;
    use crate::shaders::MATVEC;

    /// Every kind's gather against its scalar twin with a one-hot
    /// activation, which is that row's dequantized value: the first row,
    /// a middle one and the last, at a width past one workgroup of
    /// units. Then the same row through a recorded batch.
    #[test]
    fn gpu_every_gather_is_the_row_its_twin_dequantizes() {
        let Some(mut rt) = runtime() else { return };
        for shader in MATVEC {
            let (rows, blocks) = (5usize, if shader.block_elems == 1 { 300 } else { 3 });
            let cols = blocks * shader.block_elems;
            let row_bytes = blocks * shader.block_bytes;
            let weights: Vec<u8> = (0..rows)
                .flat_map(|r| kind_row(shader.kind, blocks, r as u64 + 9))
                .collect();
            let table = MatvecJob {
                shader,
                weights: &weights,
                rows,
            };
            let dot = twin(shader.kind);
            for row in [0, 2, rows - 1] {
                let got = rt.get_row(table, row).unwrap();
                assert_eq!(got.len(), cols);
                let bytes = &weights[row * row_bytes..(row + 1) * row_bytes];
                let mut e = vec![0f32; cols];
                for c in 0..cols {
                    e[c] = 1.0;
                    let want = dot(bytes, &e);
                    e[c] = 0.0;
                    assert!(
                        (got[c] - want).abs() <= 1e-6 * want.abs().max(1e-3),
                        "{} row {row} column {c}: gather {} vs twin {want}",
                        shader.kind,
                        got[c]
                    );
                }
                let out = rt.alloc_f32(cols).unwrap();
                let mut batch = Batch::new();
                batch.gather(table, row, out);
                rt.submit_batch(&batch).unwrap();
                assert_eq!(
                    rt.read_f32(out, 0, cols)
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{} row {row}: the batched gather",
                    shader.kind
                );
            }
            assert!(rt.get_row(table, rows).is_err(), "a row past the table");
        }
    }
}
