//! The sampling a GGUF recommends for itself (`general.sampling.*`).

use frink_gguf::TensorSource;

impl crate::sampling::RecommendedSampling {
    /// The sampling a GGUF recommends for itself, from the
    /// `general.sampling.*` metadata keys llama.cpp's converter writes
    /// when the source checkpoint carried a `generation_config.json`.
    ///
    /// This is the GGUF half of FreeToken's `load_generation_sampling`
    /// (`python/freetoken/utils/hf.py:92`), which checks the GGUF
    /// metadata *first* and only falls back to a `generation_config.json`
    /// sidecar for non-GGUF checkpoints -- a GGUF is a single file and
    /// has no sidecar to read.
    ///
    /// Key names are llama.cpp's own (`general.sampling.temp`, not
    /// `temperature`). Each key is independent: a file that names only
    /// `top_k` recommends only `top_k`, and the two fields it did not
    /// mention stay `None` so the server's own defaults keep speaking
    /// for them.
    ///
    /// `temp` / `top_p` are read as float *or* integer, because a
    /// converter that wrote `temp = 1` stores a GGUF integer and
    /// dropping that value would silently serve the checkpoint greedy --
    /// the exact repetition-loop failure the recommendation exists to
    /// prevent.
    pub fn from_gguf(file: &impl TensorSource) -> Self {
        let number = |k: &str| -> Option<f32> {
            file.metadata(k)
                .and_then(|v| v.as_f32().or_else(|| v.as_u64().map(|u| u as f32)))
        };
        crate::sampling::RecommendedSampling {
            temperature: number("general.sampling.temp"),
            top_p: number("general.sampling.top_p"),
            top_k: file
                .metadata("general.sampling.top_k")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
        }
    }
}
