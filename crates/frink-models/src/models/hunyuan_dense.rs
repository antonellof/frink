//! `hunyuan-dense`: `src/models/hunyuan-vl.cpp` at the pinned commit.
//!
//! It has no graph of its own: `src/models/models.h:1830-1834` derives
//! `llama_model_hunyuan_dense` from `llama_model_hunyuan_vl` and reuses
//! its hparams and graph. Two rows follow from that:
//!
//! - **The NTK-alpha RoPE base rescale** (`hunyuan-vl.cpp:8-12`,
//!   [`crate::rope_ntk_alpha`]): `{arch}.rope.scaling.alpha`, when
//!   present and positive, multiplies the trained base by
//!   `alpha^(dim / (dim - 2))`.
//! - **The Q/K norm after RoPE**: `:56-66` rope, then `:73-81` norm
//!   (see [`super::maincoder`] for the order and why the default is
//!   "before").
//!
//! Evidence: `tests/one_match_arm_graphs.rs`, against libllama.

use super::facts::{ArchFacts, ScalingFacts, TensorFacts};
use super::ModelDef;

pub const FACTS: ArchFacts = ArchFacts {
    scaling: ScalingFacts {
        rope_ntk_alpha: true,
        ..ArchFacts::DEFAULT.scaling
    },
    tensors: TensorFacts {
        qk_norm_after_rope: true,
        ..ArchFacts::DEFAULT.tensors
    },
    ..ArchFacts::DEFAULT
};

pub const DEF: ModelDef = ModelDef {
    names: &["hunyuan-dense"],
    source: "src/models/hunyuan-vl.cpp",
    facts: FACTS,
    evidence: Some(
        "tests/one_match_arm_graphs.rs: the NTK-alpha RoPE base and QK norm after RoPE, a \
         fixture against libllama's logits",
    ),
};
