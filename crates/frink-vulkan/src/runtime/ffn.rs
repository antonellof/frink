//! The fused dense FFN: `down(silu(gate(x)) * up(x))` in ONE submission.
//!
//! Through [`Runtime::matvec_many`] a dense layer's FFN is two queue
//! round trips: gate and up together, then the host's SwiGLU, then down.
//! Here gate and up run as one stage, the SwiGLU shader
//! ([`crate::shaders::SWIGLU`]) as the next, and down as the last,
//! reading the SwiGLU output where it was written. Barriers separate the
//! stages; the activation is uploaded once and only down's output comes
//! back. It is the Vulkan twin of `frink-cuda`'s and `frink-metal`'s
//! `launch_dense_ffn_swiglu`.

use super::{grid, push_constants, Dispatch, MatvecJob, Runtime};
use crate::device::VulkanError;
use crate::shaders::SWIGLU;
use ash::vk;

/// The grid for an elementwise shader over `n` values, 64 per
/// workgroup: 1-D up to 65535 workgroups, then 2-D, which the shader
/// folds back (`y * NumWorkGroups.x * 64 + x`).
fn elementwise_grid(n: usize) -> (u32, u32) {
    grid(n.div_ceil(crate::shaders::LOCAL_SIZE_X as usize))
}

impl Runtime {
    /// `down(silu(gate(x)) * up(x))`, one upload, one submission, one
    /// download. `gate` and `up` have one row per FFN unit and cover
    /// `x`; `down` covers the units. A shape that does not fit is an
    /// error, as in [`Runtime::matvec_many`].
    pub fn dense_ffn_swiglu(
        &mut self,
        gate: MatvecJob<'_>,
        up: MatvecJob<'_>,
        down: MatvecJob<'_>,
        x: &[f32],
    ) -> Result<Vec<f32>, VulkanError> {
        let n = gate.rows;
        if up.rows != n {
            return Err(VulkanError::Shape(format!(
                "dense FFN: gate has {n} rows and up {}",
                up.rows
            )));
        }
        let g_shape = self.check(&gate, x.len())?;
        let u_shape = self.check(&up, x.len())?;
        let d_shape = self.check(&down, n)?;

        // Four regions of the output buffer: gate, up, the SwiGLU
        // result, down's output.
        let align = self.min_offset_align;
        let units = n as u64 * 4;
        let (off_g, off_u) = (0, units.next_multiple_of(align));
        let off_h = (off_u + units).next_multiple_of(align);
        let off_o = (off_h + units).next_multiple_of(align);
        let end = off_o + down.rows as u64 * 4;

        let (p_g, w_g) = (self.kernel(gate.shader)?, self.resident(gate.weights)?);
        let (p_u, w_u) = (self.kernel(up.shader)?, self.resident(up.weights)?);
        let (p_d, w_d) = (self.kernel(down.shader)?, self.resident(down.weights)?);
        let p_s = self.pipeline(SWIGLU.name, SWIGLU.spv)?;
        let x_buf = self.upload_x(x)?;
        let y = Self::grow(&self.ctx.device, &self.mem_props, &mut self.y, end as usize)?;

        let whole = vk::WHOLE_SIZE;
        self.write_set(0, [(w_g, 0, whole), (x_buf, 0, whole), (y, off_g, units)]);
        self.write_set(1, [(w_u, 0, whole), (x_buf, 0, whole), (y, off_u, units)]);
        self.write_set(2, [(y, off_g, units), (y, off_u, units), (y, off_h, units)]);
        self.write_set(
            3,
            [
                (w_d, 0, whole),
                (y, off_h, units),
                (y, off_o, down.rows as u64 * 4),
            ],
        );

        let matvec = |pipeline, set, (rows, n_blocks, row_bytes): (usize, usize, usize)| Dispatch {
            pipeline,
            set,
            push: push_constants(rows, n_blocks, row_bytes),
            grid: grid(rows),
            depth: 1,
        };
        let gate_up = [matvec(p_g, 0, g_shape), matvec(p_u, 1, u_shape)];
        let combine = [Dispatch {
            pipeline: p_s,
            set: 2,
            push: push_constants(n, 0, 0),
            grid: elementwise_grid(n),
            depth: 1,
        }];
        let project = [matvec(p_d, 3, d_shape)];
        self.run(&[&gate_up, &combine, &project])?;
        Ok(self.read_y(off_o, down.rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::tests::{kind_row, pseudo_random};
    use crate::shaders::matvec_shader;

    fn silu(v: f32) -> f32 {
        v / (1.0 + (-v).exp())
    }

    fn twin(kind: &str) -> fn(&[u8], &[f32]) -> f32 {
        crate::dispatch::tests::twin(kind)
    }

    fn rows_of(kind: &str, rows: usize, cols: usize, seed: u64) -> (Vec<u8>, usize) {
        let s = matvec_shader(kind).unwrap();
        let blocks = cols / s.block_elems;
        let w = (0..rows)
            .flat_map(|r| kind_row(kind, blocks, seed + 13 * r as u64))
            .collect();
        (w, blocks * s.block_bytes)
    }

    #[test]
    fn elementwise_grid_covers_every_value() {
        for n in [1usize, 64, 65, 8192, 64 * 65535 + 1] {
            let (gx, gy) = elementwise_grid(n);
            assert!(gx as usize * gy as usize * 64 >= n, "{n}");
            assert!(gx <= 65535 && gy <= 65535, "{n}");
        }
    }

    /// The fused FFN against the scalar twins and the CPU's SiLU, at a
    /// Q4_K_M layer's kinds (Q4_K gate/up, Q6_K down) and an all-Q8_0
    /// one, in ONE submission, twice: the second is served resident.
    #[test]
    fn gpu_dense_ffn_matches_gate_up_swiglu_down_in_one_submission() {
        let Some(mut rt) = crate::dispatch::tests::runtime() else {
            return;
        };
        // `scale` puts each kind's gate values where they straddle zero,
        // where SiLU is neither the identity nor zero.
        for (gk, dk, hidden, units, scale) in [
            ("Q4_K", "Q6_K", 512, 768, 0.01f32),
            ("Q8_0", "Q8_0", 96, 160, 1.0),
        ] {
            let x: Vec<f32> = pseudo_random(0x5eed, hidden)
                .iter()
                .map(|v| v * scale)
                .collect();
            let (wg, rb_g) = rows_of(gk, units, hidden, 3);
            let (wu, rb_u) = rows_of(gk, units, hidden, 7);
            let (wd, rb_d) = rows_of(dk, hidden, units, 11);
            let g: Vec<f32> = (0..units)
                .map(|r| twin(gk)(&wg[r * rb_g..(r + 1) * rb_g], &x))
                .collect();
            let u: Vec<f32> = (0..units)
                .map(|r| twin(gk)(&wu[r * rb_u..(r + 1) * rb_u], &x))
                .collect();
            let h: Vec<f32> = g.iter().zip(&u).map(|(g, u)| silu(*g) * u).collect();
            let want: Vec<f32> = (0..hidden)
                .map(|r| twin(dk)(&wd[r * rb_d..(r + 1) * rb_d], &h))
                .collect();
            let job = |kind, weights, rows| MatvecJob {
                shader: matvec_shader(kind).unwrap(),
                weights,
                rows,
            };
            // Not a vacuous comparison: SiLU must have seen both signs
            // and the output must not be ~0.
            assert!(
                g.iter().any(|v| *v > 0.5) && g.iter().any(|v| *v < -0.5),
                "{gk}: {g:?}"
            );
            assert!(want.iter().any(|v| v.abs() > 1.0), "{gk}/{dk}: output ~0");
            for pass in ["upload", "resident"] {
                let before = rt.submissions();
                let got = rt
                    .dense_ffn_swiglu(
                        job(gk, &wg, units),
                        job(gk, &wu, units),
                        job(dk, &wd, hidden),
                        &x,
                    )
                    .expect("fused FFN");
                assert_eq!(
                    rt.submissions() - before,
                    1,
                    "{gk}/{dk} {pass}: one submission"
                );
                assert_eq!(got.len(), hidden);
                for (r, (got, want)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                        "{gk}/{dk} {pass} row {r}: gpu {got} vs cpu {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn gpu_a_dense_ffn_shape_that_does_not_fit_is_an_error() {
        let Some(mut rt) = crate::dispatch::tests::runtime() else {
            return;
        };
        let x = pseudo_random(1, 256);
        let (w, _) = rows_of("Q8_0", 64, 256, 1);
        let (wd, _) = rows_of("Q8_0", 256, 64, 2);
        let job = |weights, rows| MatvecJob {
            shader: matvec_shader("Q8_0").unwrap(),
            weights,
            rows,
        };
        // up with fewer rows than gate.
        assert!(rt
            .dense_ffn_swiglu(job(&w, 64), job(&w[..32 * 34 * 8], 32), job(&wd, 256), &x)
            .is_err());
        // down sized for a different unit count.
        assert!(rt
            .dense_ffn_swiglu(job(&w, 64), job(&w, 64), job(&wd[..34], 1), &x)
            .is_err());
    }
}
