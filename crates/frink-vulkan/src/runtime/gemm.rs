//! A batched matmul: the matvec shaders over a grid whose z is the batch
//! position (`shaders/matvec_common.glsl`). One upload of the batch, one
//! dispatch, one wait, where a prefill through [`Runtime::matvec`] would
//! pay a queue round trip per position.
//!
//! Every (row, position) workgroup decodes its row's blocks again, as
//! llama.cpp's `mul_mat_vec` does when it serves a small batch. Decoding
//! a row once for the whole batch is tuning, and waits for hardware to
//! measure it on.

use super::{grid, push_words, Dispatch, MatvecJob, Runtime};
use crate::device::VulkanError;
use ash::vk;

/// `maxComputeWorkGroupCount[2]`'s guaranteed minimum.
const MAX_DEPTH: usize = 65535;

impl Runtime {
    /// `job`'s matrix applied to `batch` activations: `x_batch` is
    /// `[batch][cols]`, the answer `[batch][rows]`, the layout of
    /// `WeightMatrix::apply_batch`. Position `b`'s row is the value
    /// [`Runtime::matvec`] gives for `x_batch[b]`, bit for bit: the same
    /// shader, the same invocations, the same reduction.
    ///
    /// A batch whose activations or outputs would not fit one binding is
    /// run as several submissions. A shape that does not fit is an
    /// error, as for a matvec.
    pub fn gemm(
        &mut self,
        job: MatvecJob<'_>,
        x_batch: &[f32],
        batch: usize,
    ) -> Result<Vec<f32>, VulkanError> {
        self.gemm_in_chunks(job, x_batch, batch, MAX_DEPTH)
    }

    /// [`Self::gemm`] with at most `max_depth` positions a submission.
    fn gemm_in_chunks(
        &mut self,
        job: MatvecJob<'_>,
        x_batch: &[f32],
        batch: usize,
        max_depth: usize,
    ) -> Result<Vec<f32>, VulkanError> {
        if batch == 0 || !x_batch.len().is_multiple_of(batch) {
            return Err(VulkanError::Shape(format!(
                "{} GEMM: {} activations in a batch of {batch}",
                job.shader.kind,
                x_batch.len()
            )));
        }
        let cols = x_batch.len() / batch;
        let (rows, n_blocks, row_bytes) = self.check(&job, cols)?;
        // The shader indexes both buffers in a 32-bit `uint`, and a
        // binding cannot exceed the device's range.
        let limit = self.max_storage_range.min(u64::from(u32::MAX)) as usize / 4;
        let per_submission = (limit / cols.max(rows)).min(max_depth);
        if per_submission == 0 {
            return Err(VulkanError::TooLarge {
                bytes: cols.max(rows) as u64 * 4,
                limit: limit as u64 * 4,
            });
        }
        let pipeline = self.kernel(job.shader)?;
        let w_buf = self.resident(job.weights)?;
        let mut out = Vec::with_capacity(batch * rows);
        for chunk in x_batch.chunks(per_submission * cols) {
            let depth = chunk.len() / cols;
            let x_buf = self.upload_x(chunk)?;
            let y_buf = Self::grow(
                &self.ctx.device,
                &self.mem_props,
                &mut self.y,
                depth * rows * 4,
            )?;
            self.write_set(
                0,
                [
                    (w_buf, 0, vk::WHOLE_SIZE),
                    (x_buf, 0, vk::WHOLE_SIZE),
                    (y_buf, 0, (depth * rows * 4) as u64),
                ],
            );
            let dispatch = [Dispatch {
                pipeline,
                set: 0,
                push: push_words(&[rows as u32, n_blocks as u32, row_bytes as u32, cols as u32]),
                grid: grid(rows),
                depth: depth as u32,
            }];
            self.run(&[&dispatch])?;
            out.extend(self.read_y(0, depth * rows));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use crate::dispatch::tests::{kind_row, pseudo_random, runtime};
    use crate::runtime::MatvecJob;
    use crate::shaders::MATVEC;

    /// Every kind's GEMM is, position by position, the matvec of that
    /// position bit for bit, and the matvec is what `dispatch::tests`
    /// hold to the scalar twin. Shapes with a batch of one, an odd batch
    /// and rows that are not a multiple of anything.
    #[test]
    fn gpu_every_gemm_is_its_twin_and_its_matvec_per_position() {
        let Some(mut rt) = runtime() else { return };
        for shader in MATVEC {
            for (rows, blocks, batch) in [(3usize, 1usize, 1usize), (5, 2, 7), (67, 3, 33)] {
                let cols = blocks * shader.block_elems;
                let weights: Vec<u8> = (0..rows)
                    .flat_map(|r| kind_row(shader.kind, blocks, r as u64 + 1))
                    .collect();
                let x = pseudo_random(rows as u64 * 31 + batch as u64, batch * cols);
                let job = MatvecJob {
                    shader,
                    weights: &weights,
                    rows,
                };
                let before = rt.submissions();
                let got = rt.gemm(job, &x, batch).unwrap();
                assert_eq!(
                    rt.submissions() - before,
                    1,
                    "{}: one submission",
                    shader.kind
                );
                assert_eq!(got.len(), batch * rows);
                for b in 0..batch {
                    let xb = &x[b * cols..(b + 1) * cols];
                    let one = rt.matvec(shader, &weights, xb, rows).unwrap();
                    for r in 0..rows {
                        let g = got[b * rows + r];
                        assert_eq!(
                            g.to_bits(),
                            one[r].to_bits(),
                            "{} position {b} row {r}: gemm {g} vs matvec {}",
                            shader.kind,
                            one[r]
                        );
                    }
                }
            }
        }
    }

    /// A batch cut into several submissions is the batch.
    #[test]
    fn gpu_a_gemm_in_chunks_is_the_gemm() {
        let Some(mut rt) = runtime() else { return };
        let shader = crate::shaders::matvec_shader("Q4_K").unwrap();
        let (rows, batch) = (9usize, 11usize);
        let weights: Vec<u8> = (0..rows)
            .flat_map(|r| kind_row("Q4_K", 1, r as u64 + 5))
            .collect();
        let job = MatvecJob {
            shader,
            weights: &weights,
            rows,
        };
        let x = pseudo_random(77, batch * 256);
        let whole = rt.gemm(job, &x, batch).unwrap();
        let before = rt.submissions();
        let cut = rt.gemm_in_chunks(job, &x, batch, 4).unwrap();
        assert_eq!(rt.submissions() - before, 3, "4 + 4 + 3 positions");
        assert_eq!(
            cut.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            whole.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn gpu_a_gemm_shape_that_does_not_fit_is_an_error() {
        let Some(mut rt) = runtime() else { return };
        let shader = crate::shaders::matvec_shader("Q8_0").unwrap();
        let weights = kind_row("Q8_0", 2, 1);
        let job = MatvecJob {
            shader,
            weights: &weights,
            rows: 1,
        };
        let x = pseudo_random(3, 64 * 3);
        assert!(rt.gemm(job, &x, 0).is_err(), "an empty batch");
        assert!(rt.gemm(job, &x[..100], 3).is_err(), "a ragged batch");
        assert!(
            rt.gemm(job, &x[..96], 3).is_err(),
            "columns the row does not have"
        );
        assert_eq!(rt.gemm(job, &x, 3).unwrap().len(), 3);
    }
}
