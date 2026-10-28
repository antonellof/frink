//! `frink quantize`: read a GGUF, write a GGUF whose eligible tensors
//! are re-encoded to a quantization frink can actually produce.
//!
//! Today that is Q8_0, Q4_K, Q5_K and Q6_K, and the six llama.cpp mixes
//! built from them. See [`policy`] for why the other targets refuse by
//! name instead of being approximated and for the tensor-eligibility
//! rules this shares with llama.cpp, and [`recipe`] for the per-tensor
//! MIX -- the reason a `Q4_K_M` file has a Q6_K output head and Q6_K
//! `ffn_down` on a quarter of its layers.
//!
//! `--imatrix` feeds an importance matrix (`frink imatrix` or
//! `llama-imatrix`, either format) to the K-quant encoders, the way
//! `llama-quantize --imatrix` does (`src/llama-quant.cpp:913-934`):
//! each tensor looks its own name up, a tensor with no entry is
//! quantized unweighted with a printed notice, and an entry of the
//! wrong width is a refusal -- except on `token_embd.weight`, which
//! upstream exempts because its imatrix is routinely the wrong shape.
//! Q8_0 ignores the matrix (`ggml-quants.c:2088-2093`), and none of the
//! mixes this tool writes read `has_imatrix` (upstream consults it only
//! for the IQ tiers and Q4_0/Q5_0, `llama-quant.cpp:287,343,366,374`),
//! so the plan is the same with and without one; only the bytes
//! inside each K-quant tensor change. The four `quantize.imatrix.*`
//! keys llama.cpp records in the output (`tools/quantize/quantize.cpp:
//! 532-566`) are written too, with its 127-byte truncation.
//!
//! The pass is streaming: the input is mmap'd, tensors are re-encoded
//! one at a time, and the output is written through a `BufWriter`. A
//! 70B checkpoint costs the output's page cache plus one tensor's f32
//! expansion, not the model.

pub mod imatrix_input;
pub mod policy;
pub mod recipe;
#[cfg(test)]
mod recipe_golden;

use std::collections::BTreeMap;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use frink_gguf::{GgmlType, GgufFile, GgufValue, GgufWriter, TensorPlan};
use rayon::prelude::*;

use imatrix_input::{imatrix_for_tensor, imatrix_metadata, load_imatrix};
use policy::{allows_quantization, disposition, parse_target, Disposition, Target};
use recipe::{ModelShape, Recipe};

/// `general.quantization_version`, ggml's `GGML_QNT_VERSION`. Every
/// tool in the ecosystem reads it; a file without it looks pre-2023.
const GGML_QNT_VERSION: u32 = 2;

/// Rows re-encoded per rayon task. Big enough that the per-task
/// overhead disappears, small enough that a 128k-row embedding table
/// does not become 128k allocations.
const ROWS_PER_TASK: usize = 64;

#[derive(Parser, Debug)]
pub struct QuantizeArgs {
    /// Source GGUF. Its quantizable tensors must be F32, F16 or BF16:
    /// re-quantizing an already-quantized checkpoint compounds two
    /// roundings and is refused rather than done quietly.
    pub input: PathBuf,

    /// Destination GGUF. Defaults to `<input stem>-<TYPE>.gguf` beside
    /// the input. (llama-quantize's default is `ggml-model-<TYPE>.gguf`
    /// in the same directory, which collides the moment two models
    /// share one; this one does not.)
    pub output: Option<PathBuf>,

    /// Quantization to write: Q8_0, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M or
    /// Q6_K. Every other llama.cpp target is refused BY NAME -- frink
    /// reads them all and encodes four, and a subcommand that pretended
    /// otherwise would hand back a file that loads and is worse.
    #[arg(long = "type", default_value = "Q8_0")]
    pub ty: String,

    /// Skip llama.cpp's per-tensor mix and write every quantizable
    /// tensor as `--type`, the way `llama-quantize --pure` does.
    ///
    /// Optional, and it changes the file: without it a `Q4_K_M` output
    /// head is Q6_K and a quarter of its `ffn_down` tensors are too.
    /// This used to be MANDATORY for the K-quant mixes, because frink
    /// had no Q5_K or Q6_K encoder to promote anything to.
    #[arg(long)]
    pub pure: bool,

    /// Print the plan (per tensor: quantize or copy, and why) and the
    /// resulting size, without writing anything.
    #[arg(long)]
    pub dry_run: bool,

    /// Overwrite the output if it already exists.
    #[arg(long)]
    pub force: bool,

    /// Importance matrix from `frink imatrix` or `llama-imatrix`, in
    /// either the GGUF or the legacy `.dat` format. Weights the K-quant
    /// fits by measured activation energy; Q8_0 tensors are unaffected.
    #[arg(long)]
    pub imatrix: Option<PathBuf>,
}

/// One tensor's decided fate, and the numbers that follow from it.
pub(crate) struct Planned {
    pub(crate) name: String,
    pub(crate) shape: Vec<u64>,
    source_dtype: GgmlType,
    out_dtype: GgmlType,
    source_bytes: usize,
    out_bytes: usize,
    /// `None` for a tensor being copied; `Some(reason)` is the reason.
    copy_reason: Option<&'static str>,
}

pub fn run(args: QuantizeArgs) -> Result<()> {
    let target = parse_target(&args.ty).map_err(|e| anyhow::anyhow!("{e}"))?;

    let file =
        GgufFile::open(&args.input).with_context(|| format!("opening {}", args.input.display()))?;

    // A split checkpoint's shards each carry a slice of the tensors and
    // a `split.*` header that would be a lie on a single output file.
    // Refuse by name rather than silently quantizing one thirteenth of
    // a model.
    if file
        .metadata_u64(frink_gguf::sharded::SPLIT_COUNT_KEY)
        .is_some_and(|n| n > 1)
    {
        bail!(
            "{} is one shard of a split GGUF. `frink quantize` writes a single file and has no \
             --keep-split; merge the shards first, or quantize the unsplit source.",
            args.input.display()
        );
    }

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| default_output_path(&args.input, target));
    if !args.dry_run {
        if output == args.input {
            bail!("output would overwrite the input ({})", output.display());
        }
        if output.exists() && !args.force {
            bail!(
                "{} already exists (pass --force to overwrite)",
                output.display()
            );
        }
    }

    let planned = plan(&file, target, args.pure)?;

    let imatrix = match &args.imatrix {
        Some(p) => Some(load_imatrix(p)?),
        None => None,
    };

    let mut metadata: BTreeMap<String, GgufValue> = file
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    metadata.insert(
        "general.file_type".to_string(),
        GgufValue::U32(target.llama_ftype()),
    );
    metadata.insert(
        "general.quantization_version".to_string(),
        GgufValue::U32(GGML_QNT_VERSION),
    );
    if let (Some(im), Some(p)) = (&imatrix, &args.imatrix) {
        imatrix_metadata(&mut metadata, im, p);
    }

    let src_total: u64 = planned.iter().map(|p| p.source_bytes as u64).sum();
    let out_total: u64 = planned.iter().map(|p| p.out_bytes as u64).sum();
    let n_quantized = planned.iter().filter(|p| p.copy_reason.is_none()).count();

    println!(
        "quantize: {} -> {}",
        args.input.display(),
        if args.dry_run {
            "(dry run, nothing written)".to_string()
        } else {
            output.display().to_string()
        }
    );
    println!("  target: {}", target.name());
    for (i, p) in planned.iter().enumerate() {
        match p.copy_reason {
            Some(reason) => println!(
                "  [{:>4}/{}] {:<44} {:?} {:>10} B  copy ({reason})",
                i + 1,
                planned.len(),
                p.name,
                p.source_dtype,
                p.source_bytes
            ),
            None => println!(
                "  [{:>4}/{}] {:<44} {:?} {:>10} B -> {:?} {:>10} B",
                i + 1,
                planned.len(),
                p.name,
                p.source_dtype,
                p.source_bytes,
                p.out_dtype,
                p.out_bytes
            ),
        }
    }
    println!(
        "  {n_quantized}/{} tensors quantized; {:.2} MiB -> {:.2} MiB ({:.2}x)",
        planned.len(),
        src_total as f64 / (1024.0 * 1024.0),
        out_total as f64 / (1024.0 * 1024.0),
        src_total as f64 / out_total.max(1) as f64,
    );

    if args.dry_run {
        return Ok(());
    }

    let plan_entries: Vec<TensorPlan> = planned
        .iter()
        .map(|p| TensorPlan {
            name: p.name.clone(),
            shape: p.shape.clone(),
            dtype: p.out_dtype,
            byte_len: p.out_bytes,
        })
        .collect();

    let out_file =
        std::fs::File::create(&output).with_context(|| format!("creating {}", output.display()))?;
    let mut writer = GgufWriter::create(
        BufWriter::with_capacity(4 << 20, out_file),
        &metadata,
        plan_entries,
    )?;

    for p in &planned {
        let src = file.tensor_bytes(&p.name)?;
        if p.copy_reason.is_some() {
            writer.write_tensor(&p.name, src)?;
        } else {
            let qw = match &imatrix {
                Some(im) => imatrix_for_tensor(im, p)?,
                None => None,
            };
            let encoded = encode_tensor(p, src, qw)?;
            writer.write_tensor(&p.name, &encoded)?;
        }
    }
    writer.finish()?.into_inner()?;

    println!("wrote {}", output.display());
    Ok(())
}

fn default_output_path(input: &Path, target: Target) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "ggml-model".to_string());
    input.with_file_name(format!("{stem}-{}.gguf", target.name()))
}

/// Decides every tensor's fate and sizes the output, refusing before a
/// single byte is written. The refusals here are the point: a tensor
/// this build cannot encode must stop the run, not be quietly copied
/// through at F16 into a file labelled Q8_0.
fn plan(file: &GgufFile, target: Target, pure: bool) -> Result<Vec<Planned>> {
    let mut out = Vec::with_capacity(file.tensors.len());

    // The mix is positional -- `i_attention_wv` and `i_ffn_down` are
    // running counters -- and the order they advance in is llama.cpp's
    // `weight_name_comparer`, NOT this file's tensor order. So the
    // whole map is resolved up front by `resolve_all`, which owns that
    // ordering, and this loop only looks types up. Walking
    // `file.tensors` and asking per tensor was the obvious thing and it
    // was wrong for 12 of Llama-3.2-1B's 113 tensors.
    //
    // `--pure` is llama.cpp's `if (!params->pure && ...)`: the mix is
    // not consulted AND its counters do not advance, so there is
    // nothing to resolve.
    let chosen_types = if pure {
        BTreeMap::new()
    } else {
        let tensors: Vec<(String, Vec<u64>)> = file
            .tensors
            .iter()
            .map(|t| (t.name.clone(), t.shape.clone()))
            .collect();
        Recipe::resolve_all(
            target,
            ModelShape::from_header(file),
            &tensors,
            |name, shape| allows_quantization(name, shape).is_none(),
        )
    };

    for t in &file.tensors {
        let source_bytes = t.byte_len().ok_or_else(|| {
            anyhow::anyhow!(
                "tensor '{}' has dtype {:?}, whose block layout this build does not know, so it \
                 cannot even be copied through",
                t.name,
                t.dtype
            )
        })?;
        let copy_through = |reason: &'static str| Planned {
            name: t.name.clone(),
            shape: t.shape.clone(),
            source_dtype: t.dtype,
            out_dtype: t.dtype,
            source_bytes,
            out_bytes: source_bytes,
            copy_reason: Some(reason),
        };

        if let Some(reason) = allows_quantization(&t.name, &t.shape) {
            out.push(copy_through(reason));
            continue;
        }

        let chosen = if pure {
            target.ggml_type()
        } else {
            // Every eligible tensor is in the map: `resolve_all` was
            // given the same `allows_quantization` this loop just
            // consulted. `expect` rather than a fallback, because a
            // fallback here would silently write the target's block
            // format for a tensor the mix meant to promote.
            *chosen_types
                .get(&t.name)
                .expect("resolve_all and allows_quantization disagree about a tensor")
        };

        match disposition(t.dtype, chosen) {
            Disposition::Copy(reason) => out.push(copy_through(reason)),
            Disposition::Quantize(ty) => {
                if !matches!(t.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
                    bail!(
                        "tensor '{}' is {:?}. `frink quantize` reads F32/F16/BF16 sources only: \
                         re-quantizing an already-quantized tensor stacks a second rounding on \
                         the first, and the result is worse than quantizing the original \
                         checkpoint once. Convert from the original weights instead.",
                        t.name,
                        t.dtype
                    );
                }
                let (block_bytes, block_elems) = ty.block_layout();
                let n_cols = t.shape[0] as usize;
                // The block size that matters is the CHOSEN type's, not
                // the target's: under Q4_K_M an `output.weight` is Q6_K
                // and a Q4_K-sized check would pass a row Q6_K cannot
                // tile. Both are 256 today and would be the same
                // number, which is exactly why reading it off the wrong
                // one is a bug nothing would notice.
                if block_elems == 0 || !n_cols.is_multiple_of(block_elems) {
                    bail!(
                        "tensor '{}' has {n_cols} columns, which is not a multiple of {:?}'s \
                         block size ({block_elems}). {}",
                        t.name,
                        ty,
                        target.fallback_note()
                    );
                }
                let n_elements = t.element_count().ok_or_else(|| {
                    anyhow::anyhow!("tensor '{}' declares an unrepresentable shape", t.name)
                })?;
                out.push(Planned {
                    name: t.name.clone(),
                    shape: t.shape.clone(),
                    source_dtype: t.dtype,
                    out_dtype: ty,
                    source_bytes,
                    out_bytes: n_elements / block_elems * block_bytes,
                    copy_reason: None,
                });
            }
        }
    }
    Ok(out)
}

/// Re-encodes one tensor, row by row, in parallel over row groups.
///
/// Row-wise and not whole-tensor because a Q8_0 block must not straddle
/// two rows: the reader walks a row at a time, so a block spanning the
/// boundary would be decoded with the wrong scale for half its values.
/// `n_cols % block_elems == 0` (checked in `plan`) is what makes the
/// row-wise and flat tilings identical -- but relying on that instead
/// of tiling per row is how the next format, with a 256-element block,
/// would silently break.
///
/// `imatrix`, when present, is `n_cols * ne2` weights: one per column,
/// and for a 3-D expert stack one set per expert, so row `r`'s set is
/// the one for expert `r / ne1` (`llama-quant.cpp:984`, `imatrix +
/// i03 * n_per_row`). For a 2-D tensor that index is always 0.
fn encode_tensor(p: &Planned, src: &[u8], imatrix: Option<&[f32]>) -> Result<Vec<u8>> {
    let n_cols = p.shape[0] as usize;
    let n_rows = (p.shape.iter().product::<u64>() as usize)
        .checked_div(n_cols)
        .unwrap_or(0);
    let ne1 = p.shape.get(1).copied().unwrap_or(1).max(1) as usize;
    let src_row_bytes = source_bytes_per_element(p.source_dtype) * n_cols;
    let out_row_bytes = p.out_bytes / n_rows.max(1);

    let groups: Vec<Vec<u8>> = (0..n_rows)
        .collect::<Vec<_>>()
        .par_chunks(ROWS_PER_TASK)
        .map(|rows| {
            let mut buf = Vec::with_capacity(rows.len() * out_row_bytes);
            let mut scratch = vec![0f32; n_cols];
            for &r in rows {
                let row = &src[r * src_row_bytes..(r + 1) * src_row_bytes];
                decode_source_row(p.source_dtype, row, &mut scratch)?;
                let qw = imatrix.map(|im| {
                    let mat = r / ne1;
                    &im[mat * n_cols..(mat + 1) * n_cols]
                });
                encode_row(p.out_dtype, &scratch, qw, &mut buf)?.ok_or_else(|| {
                    anyhow::anyhow!(
                        "tensor '{}' row length {n_cols} is not a whole number of {:?} blocks",
                        p.name,
                        p.out_dtype
                    )
                })?;
            }
            Ok(buf)
        })
        .collect::<Result<Vec<Vec<u8>>>>()?;

    let mut out = Vec::with_capacity(p.out_bytes);
    for g in groups {
        out.extend_from_slice(&g);
    }
    debug_assert_eq!(out.len(), p.out_bytes);
    Ok(out)
}

/// The one place a ggml type is turned into the encoder that writes it.
///
/// Keyed on the tensor's CHOSEN type, not on the target, because under
/// a mix they differ per tensor: one `Q4_K_M` run writes Q4_K, Q5_K,
/// Q6_K and Q8_0 rows. `Err` for a type `plan` should have refused
/// before reaching here -- an encoder dispatch whose fallthrough
/// silently copies or zero-fills is how a format becomes "supported" in
/// a table and nowhere else.
///
/// `qw` is the row's importance weights. Q8_0 takes none, because
/// `quantize_q8_0` discards its `quant_weights` (`ggml-quants.c:2089`),
/// and passing them through would read as coverage.
fn encode_row(
    ty: GgmlType,
    row: &[f32],
    qw: Option<&[f32]>,
    out: &mut Vec<u8>,
) -> Result<Option<()>> {
    Ok(match ty {
        GgmlType::Q8_0 => frink_quant::encode_row_q8_0(row, out),
        GgmlType::Q4K => frink_quant::encode_row_q4_k(row, qw, out),
        GgmlType::Q5K => frink_quant::encode_row_q5_k(row, qw, out),
        GgmlType::Q6K => frink_quant::encode_row_q6_k(row, qw, out),
        other => bail!(
            "the mix chose {other:?} for a tensor and `frink quantize` has no encoder for it. \
             This is a bug in the recipe table, not in the checkpoint: `plan` refuses an \
             unwritable type before any byte is written."
        ),
    })
}

fn source_bytes_per_element(dtype: GgmlType) -> usize {
    match dtype {
        GgmlType::F32 => 4,
        GgmlType::F16 | GgmlType::BF16 => 2,
        // `plan` refuses every other source dtype before this is
        // reached; 0 would produce a zero-length row and a silently
        // wrong file, so make it impossible to compute one.
        other => unreachable!("source dtype {other:?} reached the encoder"),
    }
}

fn decode_source_row(dtype: GgmlType, row: &[u8], out: &mut [f32]) -> Result<()> {
    match dtype {
        GgmlType::F32 => {
            for (o, c) in out.iter_mut().zip(row.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*c);
            }
        }
        GgmlType::F16 => {
            for (o, c) in out.iter_mut().zip(row.as_chunks::<2>().0) {
                *o = half::f16::from_le_bytes(*c).to_f32();
            }
        }
        GgmlType::BF16 => {
            // BF16 is f32's top 16 bits, so the widening is a shift,
            // not a format conversion.
            for (o, c) in out.iter_mut().zip(row.as_chunks::<2>().0) {
                *o = f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16);
            }
        }
        other => bail!("source dtype {other:?} reached the encoder"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "frink-quantize-{}-{}-{:?}",
            tag,
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// One place the test `QuantizeArgs` are built.
    ///
    /// Six hand-written literals of this struct is the shape ADR 0004
    /// names first: adding `--pure` had to be spelled six times, and
    /// six copies is five chances to give one of them a different
    /// default from the CLI's. The builder takes what each test varies
    /// and nothing else.
    fn args(input: &Path, output: &Path, ty: &str) -> QuantizeArgs {
        QuantizeArgs {
            input: input.to_path_buf(),
            output: Some(output.to_path_buf()),
            ty: ty.into(),
            pure: false,
            dry_run: false,
            force: true,
            imatrix: None,
        }
    }

    /// Writes a tiny but structurally real F16 GGUF: a 2-D weight to
    /// quantize, a 1-D tensor (the dimension rule), a 2-D norm and a
    /// 2-D router gate (the keep-list rules).
    ///
    /// The norm is 2-D on purpose. A 1-D one would be kept by the
    /// dimension rule alone, so deleting `_norm.weight` from the
    /// keep-list would leave this test green -- which it did, until the
    /// sabotage pass found it.
    /// `n_cols` is a parameter because the targets have different block
    /// sizes: 64 columns is a whole number of Q8_0 blocks and NOT of
    /// Q4_K super-blocks, and a fixture that only ever had one width
    /// could not tell the two refusals apart.
    ///
    /// `output.weight` is here so the MIX has something to promote. It
    /// is the tensor every K-quant mix sends to Q6_K, so without it a
    /// `--pure` run and a mixed run of this fixture would produce
    /// identical files and the mix would be untested end to end.
    fn write_f16_source(path: &Path, n_cols: usize) -> Vec<f32> {
        let n = n_cols;
        let values: Vec<f32> = (0..n * 2)
            .map(|i| ((i as f32) * 0.037).sin() * 0.8)
            .collect();
        let f16_bytes = |vals: &[f32]| -> Vec<u8> {
            vals.iter()
                .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                .collect()
        };
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("llama".into()),
        );
        metadata.insert("general.file_type".to_string(), GgufValue::U32(1));

        let w = f16_bytes(&values);
        let one_d = f16_bytes(&values[..n]);
        let norm = f16_bytes(&values);
        let gate = f16_bytes(&values);
        let head = f16_bytes(&values);
        let plan = vec![
            TensorPlan {
                name: "blk.0.attn_q.weight".into(),
                shape: vec![n as u64, 2],
                dtype: GgmlType::F16,
                byte_len: w.len(),
            },
            TensorPlan {
                name: "blk.0.attn_q.bias".into(),
                shape: vec![n as u64],
                dtype: GgmlType::F16,
                byte_len: one_d.len(),
            },
            TensorPlan {
                name: "blk.0.attn_norm.weight".into(),
                shape: vec![n as u64, 2],
                dtype: GgmlType::F16,
                byte_len: norm.len(),
            },
            TensorPlan {
                name: "blk.0.ffn_gate_inp.weight".into(),
                shape: vec![n as u64, 2],
                dtype: GgmlType::F16,
                byte_len: gate.len(),
            },
            TensorPlan {
                name: "output.weight".into(),
                shape: vec![n as u64, 2],
                dtype: GgmlType::F16,
                byte_len: head.len(),
            },
        ];
        let f = std::fs::File::create(path).unwrap();
        let mut wr = GgufWriter::create(BufWriter::new(f), &metadata, plan).unwrap();
        wr.write_tensor("blk.0.attn_q.weight", &w).unwrap();
        wr.write_tensor("blk.0.attn_q.bias", &one_d).unwrap();
        wr.write_tensor("blk.0.attn_norm.weight", &norm).unwrap();
        wr.write_tensor("blk.0.ffn_gate_inp.weight", &gate).unwrap();
        wr.write_tensor("output.weight", &head).unwrap();
        wr.finish().unwrap().into_inner().unwrap();
        values
    }

    /// End to end: an F16 GGUF in, a Q8_0 GGUF out, read back by the
    /// same reader the engine loads models with. The tensors that
    /// llama.cpp keeps at source precision are still F16, the one it
    /// quantizes is Q8_0, and the values survive within Q8_0's error.
    #[test]
    fn an_f16_gguf_round_trips_through_quantize_and_reads_back_as_q8_0() {
        let dir = tmp_dir("roundtrip");
        let src = dir.join("src.gguf");
        let dst = dir.join("dst.gguf");
        let values = write_f16_source(&src, 64);

        run(args(&src, &dst, "Q8_0")).unwrap();

        let out = GgufFile::open(&dst).unwrap();
        assert_eq!(out.metadata_u64("general.file_type"), Some(7));
        assert_eq!(out.metadata_u64("general.quantization_version"), Some(2));
        assert_eq!(out.metadata_str("general.architecture"), Some("llama"));

        let q = out.find_tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(q.dtype, GgmlType::Q8_0);
        assert_eq!(q.shape, vec![64, 2]);
        // The 2-D norm and the router gate are llama.cpp's keep-list;
        // the bias is the dimension rule and the not-a-weight rule.
        for kept in [
            "blk.0.attn_q.bias",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_gate_inp.weight",
        ] {
            assert_eq!(
                out.find_tensor(kept).unwrap().dtype,
                GgmlType::F16,
                "{kept} should have been kept at source precision"
            );
        }

        let back =
            frink_quant::dequant_q8_0(out.tensor_bytes("blk.0.attn_q.weight").unwrap()).unwrap();
        assert_eq!(back.len(), values.len());
        for (i, (&want, &have)) in values.iter().zip(back.iter()).enumerate() {
            assert!((want - have).abs() < 0.01, "element {i}: {want} -> {have}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The quantized bytes are the ones llama.cpp's encoder would have
    /// written for the same row, not merely bytes that decode to
    /// similar numbers. `frink-quant`'s golden pins the encoder; this
    /// pins the pipeline that feeds it -- row tiling, F16 decode, and
    /// the writer's data-section layout all have to be right for these
    /// bytes to land where the reader looks.
    #[test]
    fn the_quantized_tensor_bytes_are_what_the_encoder_produces_for_those_rows() {
        let dir = tmp_dir("bytes");
        let src = dir.join("src.gguf");
        let dst = dir.join("dst.gguf");
        let values = write_f16_source(&src, 64);
        run(args(&src, &dst, "q8_0")).unwrap();

        // Re-encode independently, from the f16 the source stored (not
        // from `values`, which is f32 and would round differently).
        let mut want = Vec::new();
        let f16_roundtrip: Vec<f32> = values
            .iter()
            .map(|v| half::f16::from_f32(*v).to_f32())
            .collect();
        for row in f16_roundtrip.chunks(64) {
            frink_quant::encode_row_q8_0(row, &mut want).unwrap();
        }
        let out = GgufFile::open(&dst).unwrap();
        assert_eq!(out.tensor_bytes("blk.0.attn_q.weight").unwrap(), &want[..]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The refusal this subcommand is scoped around, exercised through
    /// the real entry point rather than only through `parse_target`.
    #[test]
    fn asking_for_a_target_with_no_encoder_refuses_before_touching_the_filesystem() {
        let dir = tmp_dir("refuse");
        let src = dir.join("src.gguf");
        let dst = dir.join("dst.gguf");
        write_f16_source(&src, 64);
        // Q6_K used to be the example here and is writable now, which
        // is what this change is for. Q3_K_M is the next mix up with no
        // encoder.
        let err = run(args(&src, &dst, "Q3_K_M")).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("cannot WRITE Q3_K_M"), "{msg}");
        assert!(msg.contains("Q8_0"), "{msg}");
        assert!(!dst.exists(), "a refused run must not leave a file behind");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A source tensor that is already quantized is refused by name.
    /// Requantizing stacks two roundings, and the file would carry no
    /// sign that it happened.
    #[test]
    fn an_already_quantized_source_tensor_is_refused_by_name() {
        let dir = tmp_dir("requant");
        let src = dir.join("src.gguf");
        let bytes = vec![0u8; 144 * 2]; // two Q4_K super-blocks
        let plan = vec![TensorPlan {
            name: "blk.0.attn_q.weight".into(),
            shape: vec![256, 2],
            dtype: GgmlType::Q4K,
            byte_len: bytes.len(),
        }];
        let f = std::fs::File::create(&src).unwrap();
        let mut wr = GgufWriter::create(BufWriter::new(f), &BTreeMap::new(), plan).unwrap();
        wr.write_tensor("blk.0.attn_q.weight", &bytes).unwrap();
        wr.finish().unwrap().into_inner().unwrap();

        let err = run(QuantizeArgs {
            dry_run: true,
            ..args(&src, &dir.join("dst.gguf"), "Q8_0")
        })
        .unwrap_err();
        assert!(
            err.to_string().contains("F32/F16/BF16 sources only"),
            "{err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A row length that is not a whole number of blocks stops the run.
    /// The alternative -- padding the row -- writes more elements than
    /// the shape declares, and every later row decodes shifted.
    #[test]
    fn a_row_length_that_is_not_a_multiple_of_the_block_size_is_refused() {
        let dir = tmp_dir("ragged");
        let src = dir.join("src.gguf");
        let bytes = vec![0u8; 33 * 2 * 2];
        let plan = vec![TensorPlan {
            name: "blk.0.attn_q.weight".into(),
            shape: vec![33, 2],
            dtype: GgmlType::F16,
            byte_len: bytes.len(),
        }];
        let f = std::fs::File::create(&src).unwrap();
        let mut wr = GgufWriter::create(BufWriter::new(f), &BTreeMap::new(), plan).unwrap();
        wr.write_tensor("blk.0.attn_q.weight", &bytes).unwrap();
        wr.finish().unwrap().into_inner().unwrap();

        let err = run(QuantizeArgs {
            dry_run: true,
            ..args(&src, &dir.join("dst.gguf"), "Q8_0")
        })
        .unwrap_err();
        assert!(err.to_string().contains("not a multiple of"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_existing_output_is_not_overwritten_without_force() {
        let dir = tmp_dir("clobber");
        let src = dir.join("src.gguf");
        let dst = dir.join("dst.gguf");
        write_f16_source(&src, 64);
        std::fs::write(&dst, b"precious").unwrap();
        let err = run(QuantizeArgs {
            force: false,
            ..args(&src, &dst, "Q8_0")
        })
        .unwrap_err();
        assert!(err.to_string().contains("--force"), "{err}");
        assert_eq!(std::fs::read(&dst).unwrap(), b"precious");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `--type q4_k_m` names a MIX, and the mix is now applied: the
    /// same file quantized with and without `--pure` must DIFFER, and
    /// differ in the tensor the mix promotes.
    ///
    /// This replaces the refusal that used to stand here. It is the
    /// assertion most likely to be argued away later ("Q4_K everywhere
    /// is close enough"), so it goes through the real entry point and
    /// names the tensor.
    #[test]
    fn the_q4_k_m_mix_promotes_the_output_head_and_pure_does_not() {
        let dir = tmp_dir("mix");
        let src = dir.join("src.gguf");
        write_f16_source(&src, 256);

        let mixed = dir.join("mixed.gguf");
        run(args(&src, &mixed, "q4_k_m")).unwrap();
        let pure = dir.join("pure.gguf");
        run(QuantizeArgs {
            pure: true,
            ..args(&src, &pure, "q4_k_m")
        })
        .unwrap();

        let m = GgufFile::open(&mixed).unwrap();
        let p = GgufFile::open(&pure).unwrap();
        // The mix sends `output.weight` to Q6_K; `--pure` leaves it at
        // the target's block format.
        assert_eq!(m.find_tensor("output.weight").unwrap().dtype, GgmlType::Q6K);
        assert_eq!(p.find_tensor("output.weight").unwrap().dtype, GgmlType::Q4K);
        // And the tensor the mix does NOT touch is Q4_K in both.
        for f in [&m, &p] {
            assert_eq!(
                f.find_tensor("blk.0.attn_q.weight").unwrap().dtype,
                GgmlType::Q4K
            );
        }
        // Both still declare Q4_K_M, which is precisely why the bytes
        // having to differ is worth asserting: `general.file_type`
        // alone cannot tell the two files apart.
        assert_eq!(m.metadata_u64("general.file_type"), Some(15));
        assert_eq!(p.metadata_u64("general.file_type"), Some(15));
        assert_ne!(
            std::fs::read(&mixed).unwrap(),
            std::fs::read(&pure).unwrap()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// End to end for the K-quant: an F16 GGUF in, a `--pure` Q4_K
    /// GGUF out, read back by the reader the engine loads models with.
    ///
    /// The bytes are compared against the encoder rather than against a
    /// tolerance, because `frink-quant`'s golden already pins the
    /// encoder to llama.cpp's; what this adds is the pipeline around it
    /// -- 256-wide row tiling, F16 decode, the writer's data-section
    /// layout, and the `general.file_type` that tells every other tool
    /// what the file is.
    #[test]
    fn an_f16_gguf_round_trips_through_quantize_and_reads_back_as_q4_k() {
        let dir = tmp_dir("q4k");
        let src = dir.join("src.gguf");
        let dst = dir.join("dst.gguf");
        let values = write_f16_source(&src, 256);

        run(QuantizeArgs {
            pure: true,
            ..args(&src, &dst, "q4_k_s")
        })
        .unwrap();

        let out = GgufFile::open(&dst).unwrap();
        // 14 is LLAMA_FTYPE_MOSTLY_Q4_K_S. `q4_k_m` would write 15 from
        // the same bytes, which is the whole reason the two are
        // separate targets.
        assert_eq!(out.metadata_u64("general.file_type"), Some(14));
        let q = out.find_tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(q.dtype, GgmlType::Q4K);
        assert_eq!(q.shape, vec![256, 2]);
        // `--pure` skips llama.cpp's per-layer mix, NOT its keep-list:
        // a norm quantized to Q4_K is a broken model either way.
        for kept in [
            "blk.0.attn_q.bias",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_gate_inp.weight",
        ] {
            assert_eq!(
                out.find_tensor(kept).unwrap().dtype,
                GgmlType::F16,
                "{kept} should have been kept at source precision"
            );
        }

        let mut want = Vec::new();
        let f16_roundtrip: Vec<f32> = values
            .iter()
            .map(|v| half::f16::from_f32(*v).to_f32())
            .collect();
        for row in f16_roundtrip.chunks(256) {
            frink_quant::encode_row_q4_k(row, None, &mut want).unwrap();
        }
        assert_eq!(out.tensor_bytes("blk.0.attn_q.weight").unwrap(), &want[..]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The block-size refusal has to say the right thing for the right
    /// target. 64 columns is a whole number of Q8_0 blocks and not of
    /// Q4_K super-blocks, and llama.cpp's answers differ: it throws for
    /// Q8_0 and silently rewrites the tensor to Q5_0 for Q4_K. One
    /// sentence covering both was true of only Q8_0.
    #[test]
    fn a_row_too_narrow_for_a_super_block_is_refused_and_names_llama_cpps_fallback() {
        let dir = tmp_dir("narrow");
        let src = dir.join("src.gguf");
        write_f16_source(&src, 64);
        let err = run(QuantizeArgs {
            pure: true,
            dry_run: true,
            ..args(&src, &dir.join("dst.gguf"), "q4_k_s")
        })
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not a multiple of Q4K's block size (256)"),
            "{msg}"
        );
        assert!(msg.contains("Q4_K -> Q5_0"), "{msg}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `--imatrix` end to end: the weighted bytes are what the encoder
    /// produces for that row with that slice, they differ from the
    /// unweighted run, and the output records the imatrix keys. The
    /// whole-model byte identity is measured outside CI; this pins the
    /// plumbing from flag to encoder.
    #[test]
    fn an_imatrix_reaches_the_encoder_and_is_recorded_in_the_metadata() {
        let dir = tmp_dir("imatrix");
        let src = dir.join("src.gguf");
        let values = write_f16_source(&src, 256);
        let im_path = dir.join("im.gguf");
        let mut stats = BTreeMap::new();
        stats.insert(
            "blk.0.attn_q.weight".to_string(),
            crate::imatrix::file::Stats {
                values: (0..256).map(|i| 1.0 + (i % 7) as f32).collect(),
                counts: vec![4],
            },
        );
        crate::imatrix::file::write(
            &im_path,
            crate::imatrix::file::OutputFormat::Gguf,
            &stats,
            &["calib.txt".to_string()],
            2,
            512,
        )
        .unwrap();

        let plain = dir.join("plain.gguf");
        run(QuantizeArgs {
            pure: true,
            ..args(&src, &plain, "q4_k_s")
        })
        .unwrap();
        let weighted = dir.join("weighted.gguf");
        run(QuantizeArgs {
            pure: true,
            imatrix: Some(im_path.clone()),
            ..args(&src, &weighted, "q4_k_s")
        })
        .unwrap();

        let qw: Vec<f32> = (0..256).map(|i| (1.0 + (i % 7) as f32) / 4.0).collect();
        let mut want = Vec::new();
        for row in values.chunks(256) {
            let f16_row: Vec<f32> = row
                .iter()
                .map(|v| half::f16::from_f32(*v).to_f32())
                .collect();
            frink_quant::encode_row_q4_k(&f16_row, Some(&qw), &mut want).unwrap();
        }
        let w = GgufFile::open(&weighted).unwrap();
        let p = GgufFile::open(&plain).unwrap();
        assert_eq!(w.tensor_bytes("blk.0.attn_q.weight").unwrap(), &want[..]);
        assert_ne!(
            w.tensor_bytes("blk.0.attn_q.weight").unwrap(),
            p.tensor_bytes("blk.0.attn_q.weight").unwrap()
        );
        assert_eq!(w.metadata_u64("quantize.imatrix.entries_count"), Some(1));
        assert_eq!(w.metadata_u64("quantize.imatrix.chunks_count"), Some(2));
        assert_eq!(
            w.metadata_str("quantize.imatrix.dataset"),
            Some("calib.txt")
        );
        assert!(p.metadata_str("quantize.imatrix.file").is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_default_output_name_carries_the_target_and_sits_beside_the_input() {
        assert_eq!(
            default_output_path(Path::new("/m/Llama-3.2-1B-F16.gguf"), Target::Q8_0),
            PathBuf::from("/m/Llama-3.2-1B-F16-Q8_0.gguf")
        );
    }
}
