//! `frink caps`: what this host has, and with `--kernels`, what each
//! backend can run (the kernel coverage matrix, #405).
//!
//! Also the kernel-coverage section of `frink inspect-plan`, which asks
//! the same matrix about the tensors one GGUF actually holds.

use frink_core::kernel_matrix::{Cell, KernelMatrix, Op};
use frink_core::kernel_registry::Backend;
use frink_core::weight_matrix::quant_kind_for;
use frink_models::BudgetBackend;
use std::collections::BTreeMap;

/// `frink caps [--kernels [--write PATH]]`.
pub fn run(kernels: bool, write: Option<String>) -> anyhow::Result<()> {
    if write.is_some() && !kernels {
        anyhow::bail!("--write needs --kernels: only the kernel matrix is written to a file");
    }
    if kernels {
        let report = KernelMatrix::derive().markdown();
        match write {
            Some(path) => {
                std::fs::write(&path, &report)?;
                println!("wrote kernel coverage matrix to {path}");
            }
            None => print!("{report}"),
        }
        return Ok(());
    }
    print_host();
    Ok(())
}

fn print_host() {
    let profile = frink_cuda::HardwareProfile::detect();
    println!("Hardware capabilities (detected, not assumed):");
    println!("  CPU logical cores : {}", profile.cpu_logical_cores);
    if profile.host_ram_total_bytes > 0 {
        println!(
            "  Host RAM total    : {:.2} GiB",
            profile.host_ram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    } else {
        println!("  Host RAM total    : could not detect (non-Linux host?)");
    }
    println!("  SIMD              : {}", profile.simd.label());
    println!(
        "    avx2={} avx512f={} fma={} neon={}",
        profile.simd.avx2, profile.simd.avx512f, profile.simd.fma, profile.simd.neon
    );
    if profile.cuda_available {
        println!(
            "  CUDA              : {} device(s), first = {:?}, {:.2} GiB VRAM",
            profile.cuda_device_count,
            profile.cuda_device_name,
            profile.cuda_vram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    } else {
        println!("  CUDA              : not available (no device found, or built without --features cuda)");
    }
    let metal = frink_metal::MetalProfile::detect();
    if metal.available {
        println!("  Metal             : {:?}", metal.device_name);
    } else {
        println!("  Metal             : not available (no device found, or built without --features metal)");
    }
    println!("  Kernels per backend: `frink caps --kernels` (docs/KERNELS.md)");
}

/// The registry backend a budget backend plans for.
fn registry_backend(b: BudgetBackend) -> Backend {
    match b {
        BudgetBackend::Cpu => Backend::Cpu,
        BudgetBackend::Metal => Backend::Metal,
        BudgetBackend::Cuda => Backend::Cuda,
    }
}

/// The gap a cell stands for on `backend`, or `None` when a kernel (or,
/// for an accelerator, a loop that stays on it) runs it.
fn gap(backend: Backend, cell: Cell) -> Option<&'static str> {
    match cell {
        Cell::Host(_) if backend.is_accelerator() => Some("host"),
        Cell::F32Dot => Some("f32 dot"),
        Cell::Loop(_) => Some("matvec loop"),
        // The weight still runs, on the host: the kind has no reference
        // kernel on any accelerator to match.
        Cell::NotApplicable => Some("host (activation format)"),
        Cell::Kernel | Cell::Host(_) => None,
    }
}

/// One line per (quant kind, op) in `tensors` that has no kernel on
/// `backend`, for `frink inspect-plan`. `tensors` is (name, dtype); a
/// dtype with no quantized kernel at all (F32, F16) is not a gap the
/// matrix covers and is skipped.
pub fn plan_lines(tensors: &[(&str, frink_gguf::GgmlType)], budget: BudgetBackend) -> Vec<String> {
    let backend = registry_backend(budget);
    let m = KernelMatrix::derive();
    // A tied model reads its embedding as the output head too.
    let tied = !tensors.iter().any(|(n, _)| *n == "output.weight");
    let mut gaps: BTreeMap<(&'static str, &'static str, &'static str), usize> = BTreeMap::new();
    let mut quantized = 0usize;
    for &(name, dtype) in tensors {
        let Some(kind) = quant_kind_for(dtype) else {
            continue;
        };
        quantized += 1;
        let embd = name == "token_embd.weight";
        let ops: &[Op] = match (embd, tied) {
            (true, true) => &[Op::GetRows, Op::Matvec, Op::Gemm],
            (true, false) => &[Op::GetRows],
            (false, _) => &[Op::Matvec, Op::Gemm],
        };
        for &op in ops {
            if let Some(what) = gap(backend, m.cell(backend, op, kind)) {
                *gaps.entry((kind.name(), op.name(), what)).or_default() += 1;
            }
        }
    }
    let mut out = vec![format!(
        "Kernel coverage on {budget} ({quantized} quantized tensors, docs/KERNELS.md):"
    )];
    if gaps.is_empty() {
        out.push("  every quantized tensor has a kernel for every op it runs".into());
    }
    for ((kind, op, what), n) in gaps {
        let unit = if n == 1 { "tensor" } else { "tensors" };
        out.push(format!("  {kind:<8} {op:<8} -> {what} ({n} {unit})"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use frink_gguf::GgmlType;

    /// A Q4_1 projection has no CUDA kernel: the plan must say it leaves
    /// the GPU, by kind and op, and say nothing about the Q4_K one.
    #[test]
    fn a_kind_with_no_kernel_is_named_in_the_plan() {
        let tensors = [
            ("token_embd.weight", GgmlType::Q4K),
            ("output.weight", GgmlType::Q6K),
            ("blk.0.attn_q.weight", GgmlType::Q4_1),
            ("blk.0.ffn_up.weight", GgmlType::Q4K),
            ("blk.0.attn_norm.weight", GgmlType::F32),
        ];
        let lines = plan_lines(&tensors, BudgetBackend::Cuda);
        let text = lines.join("\n");
        assert!(text.contains("4 quantized tensors"), "{text}");
        assert!(
            text.contains("Q4_1     matvec   -> host (1 tensor)"),
            "{text}"
        );
        assert!(
            text.contains("Q4_1     gemm     -> host (1 tensor)"),
            "{text}"
        );
        // CUDA gathers a Q4_K embedding on the device (#405).
        assert!(!text.contains("get_rows"), "{text}");
        assert!(!text.contains("Q4_K     matvec"), "{text}");
        assert!(!text.contains("Q6_K"), "{text}");
    }

    #[test]
    fn a_fully_covered_plan_says_so() {
        let tensors = [
            ("token_embd.weight", GgmlType::Q4K),
            ("output.weight", GgmlType::Q6K),
            ("blk.0.ffn_up.weight", GgmlType::Q4K),
        ];
        let lines = plan_lines(&tensors, BudgetBackend::Metal);
        assert!(lines[1].contains("every quantized tensor"), "{lines:?}");
    }
}
