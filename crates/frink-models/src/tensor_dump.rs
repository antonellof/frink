//! The frink half of `frink parity --dump-tensors` (#476): the batched
//! prefill's intermediates, written under the names llama.cpp's eval
//! callback gives the same tensors, so the two dumps compare file by
//! file.
//!
//! The llama.cpp half is `tools/llama_logits.c` under
//! `LLAMA_LOGITS_DUMP_TENSORS`, and the format is its format: each
//! tensor is raw host-order f32 in `<dir>/<name>.f32`, row-major
//! `[rows, width]`, and `<dir>/manifest.tsv` holds one line
//! `<name>\t<width>\t<rows>` per tensor. A name is `<base>-<layer>`, as
//! `llm_graph_context::cb` spells it, and the second tensor the graph
//! gives the same name is `<name>#2` (`Qcur` after RoPE, `Kcur` after
//! RoPE).
//!
//! First-token KL says THAT two engines disagree; this says at which
//! layer and which op the disagreement enters, which is the question
//! #476 could not answer: a quantizer change moved KL the wrong way, and
//! KL alone cannot tell a second deviation that cancels it from noise.
//!
//! Off unless [`dump_to`] named a directory, and nothing reads the
//! environment: the one caller is the parity command. The hooks sit in
//! the host body of the contiguous batched prefill
//! (`Decoder::forward_hidden_batch_inner`), the body `frink parity`
//! runs; the Metal and CUDA arms leave the layer before reaching them.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// One open dump: the directory, its manifest, and the first write that
/// failed, so a dump that lost a tensor fails the command that asked for
/// it instead of comparing what is left.
pub struct TensorDump {
    dir: PathBuf,
    state: Mutex<DumpState>,
}

struct DumpState {
    manifest: std::fs::File,
    error: Option<std::io::Error>,
}

static DUMP: OnceLock<TensorDump> = OnceLock::new();

/// Dumps every later batched prefill into `dir`, for the rest of the
/// process. Fails if a dump is already open, rather than splitting one
/// forward pass across two directories.
pub fn dump_to(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let manifest = std::fs::File::create(dir.join("manifest.tsv"))?;
    DUMP.set(TensorDump {
        dir: dir.to_path_buf(),
        state: Mutex::new(DumpState {
            manifest,
            error: None,
        }),
    })
    .map_err(|_| std::io::Error::other("a tensor dump is already open in this process"))
}

/// The open dump, if any. Read once per forward pass, not per tensor.
pub(crate) fn active() -> Option<&'static TensorDump> {
    DUMP.get()
}

/// The first write that failed in the open dump, if any. `Ok` when no
/// dump is open.
pub fn finish() -> std::io::Result<()> {
    match DUMP.get() {
        None => Ok(()),
        Some(d) => match lock(&d.state).error.take() {
            None => Ok(()),
            Some(e) => Err(e),
        },
    }
}

fn lock(m: &Mutex<DumpState>) -> std::sync::MutexGuard<'_, DumpState> {
    // A poisoned lock only means another hook panicked mid-write; the
    // manifest is still a file and the error slot still a slot.
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// llama.cpp's name for the `occurrence`-th tensor called `base` in
/// layer `layer` (`None`: a graph-level tensor such as `result_norm`).
pub(crate) fn tensor_name(base: &str, layer: Option<usize>, occurrence: usize) -> String {
    let mut name = match layer {
        Some(l) => format!("{base}-{l}"),
        None => base.to_string(),
    };
    if occurrence > 1 {
        name.push_str(&format!("#{occurrence}"));
    }
    name
}

impl TensorDump {
    /// Writes `data`, row-major `[data.len() / width, width]`.
    pub(crate) fn write(
        &self,
        base: &str,
        layer: Option<usize>,
        occurrence: usize,
        width: usize,
        data: &[f32],
    ) {
        let name = tensor_name(base, layer, occurrence);
        let mut state = lock(&self.state);
        if state.error.is_some() {
            return;
        }
        let written = (|| {
            if width == 0 || !data.len().is_multiple_of(width) {
                return Err(std::io::Error::other(format!(
                    "{name}: {} values is not a whole number of rows of {width}",
                    data.len()
                )));
            }
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for v in data {
                bytes.extend_from_slice(&v.to_ne_bytes());
            }
            std::fs::write(self.dir.join(format!("{name}.f32")), bytes)?;
            writeln!(state.manifest, "{name}\t{width}\t{}", data.len() / width)
        })();
        if let Err(e) = written {
            state.error = Some(e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_llama_graph_callbacks() {
        assert_eq!(tensor_name("l_out", Some(15), 1), "l_out-15");
        assert_eq!(tensor_name("Qcur", Some(0), 2), "Qcur-0#2");
        assert_eq!(tensor_name("result_norm", None, 1), "result_norm");
    }

    /// A dump of its own, not the process's: opening the global one here
    /// would make every other prefill in this test binary write files.
    fn local(tag: &str) -> (PathBuf, TensorDump) {
        let dir = std::env::temp_dir().join(format!("frink-dump-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = std::fs::File::create(dir.join("manifest.tsv")).unwrap();
        let dump = TensorDump {
            dir: dir.clone(),
            state: Mutex::new(DumpState {
                manifest,
                error: None,
            }),
        };
        (dir, dump)
    }

    /// The format `tools/llama_logits.c` writes: raw host-order f32 and
    /// one `name\twidth\trows` line per tensor.
    #[test]
    fn writes_the_dumpers_format() {
        let (dir, dump) = local("fmt");
        dump.write("Kcur", Some(3), 2, 2, &[1.0, -2.0, 3.5, 0.25]);
        dump.write("result_norm", None, 1, 4, &[9.0; 4]);
        let manifest = std::fs::read_to_string(dir.join("manifest.tsv")).unwrap();
        let bytes = std::fs::read(dir.join("Kcur-3#2.f32")).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(manifest, "Kcur-3#2\t2\t2\nresult_norm\t4\t1\n");
        let back: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_ne_bytes(*c))
            .collect();
        assert_eq!(back, [1.0, -2.0, 3.5, 0.25]);
    }

    /// A tensor that is not whole rows is an error the caller sees, and
    /// nothing after it is written: a comparison over what is left would
    /// read as a complete one.
    #[test]
    fn a_ragged_tensor_fails_the_dump() {
        let (dir, dump) = local("ragged");
        dump.write("l_out", Some(0), 1, 3, &[1.0; 4]);
        dump.write("l_out", Some(1), 1, 2, &[1.0; 4]);
        let manifest = std::fs::read_to_string(dir.join("manifest.tsv")).unwrap();
        let err = lock(&dump.state).error.take();
        std::fs::remove_dir_all(&dir).ok();
        assert!(err.unwrap().to_string().contains("whole number of rows"));
        assert_eq!(manifest, "");
    }
}
