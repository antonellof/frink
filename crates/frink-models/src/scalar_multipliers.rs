//! The four scalar multipliers a checkpoint can declare in METADATA,
//! and which architectures apply which of them.
//!
//! `{arch}.logit_scale`, `{arch}.residual_scale`,
//! `{arch}.embedding_scale` and `{arch}.attention.scale`
//! (`llama-arch.cpp`: `LLM_KV_LOGIT_SCALE`, `LLM_KV_RESIDUAL_SCALE`,
//! `LLM_KV_EMBEDDING_SCALE`, `LLM_KV_ATTENTION_SCALE`) are the blind
//! spot [`crate::loader::assert_every_tensor_consumed`] cannot cover:
//! they are hyper-parameters, not weights, so a checkpoint carrying one
//! leaves no unread tensor. Before this module frink REFUSED any file
//! declaring one, by name, because the alternative was loading it and
//! computing a differently-scaled graph than it was trained as.
//!
//! **One implementation, parameterised by architecture.** llama.cpp
//! spreads these over four unrelated places -- the shared
//! `build_inp_embd` for the embedding scale (llama-graph.cpp:2337-2342),
//! `kq_scale` for the attention scale, a `ggml_scale` before each
//! residual add, and one more after the lm_head -- and each
//! architecture picks a subset. frink resolves the subset ONCE here,
//! into plain `Option<f32>` fields on [`crate::ModelConfig`] that the
//! decoder reads as data. `granite` and `granitemoe` differ in the FFN
//! and not in the scaling, and `granite-moe` is a frink-only alias for
//! `granitemoe`, so all three share one [`MultiplierSupport`] constant
//! and cannot drift apart.
//!
//! **What each architecture reads, against the C.**
//!
//! | arch | llama.cpp | embedding | residual | logit | attention |
//! |---|---|---|---|---|---|
//! | `granite` | `granite.cpp:5-10` | yes | yes | divide | yes |
//! | `granitemoe` | `granite-moe.cpp:3-10` | yes | yes | divide | yes |
//! | `minicpm` | `minicpm.cpp:5-14` | yes | yes | divide | **no** |
//! | `grok` | `grok.cpp:5-27` | yes | **no** | **multiply** | `attention.output_scale` |
//!
//! **Gemma is not in that table, and that is the interesting part.** It
//! scales its embeddings by `sqrt(n_embd)` and, at 27B, overrides its
//! attention scale -- but it reads NEITHER KEY: `gemma3.cpp:31` and
//! `gemma2.cpp:27` assign `f_attention_scale` from the model type, and
//! the embedding scale is arithmetic in the graph. The whole family used
//! to be exempted from the refusal list wholesale, on the strength of
//! implementing two of the four, which meant a hand-written
//! `gemma3.residual_scale` would have loaded and been ignored. It is
//! refused now, along with the two keys Gemma's own scales are NOT read
//! from, because a file declaring one describes something llama.cpp does
//! not do either. The Gemma scales come from
//! `capability::attention_scale_override` and `loader.rs`'s family
//! branch, which is where an arch-computed value belongs.
//!
//! **MiniCPM is the row that made the DEFAULTS column real.** It runs
//! `llama_model_granite::graph` verbatim (`models.h:1594-1601`) -- the
//! same graph object, not a similar one -- so it needs no arithmetic of
//! its own. What it adds is DEFAULTS: `minicpm.cpp:5-7` hardcodes
//! `f_embedding_scale = 12.0`, `f_residual_scale = 1.4/sqrt(n_layer)`
//! and `f_logit_scale = 256/n_embd` and only THEN reads the three keys
//! with `required = false` (`:12-14`), so an older MiniCPM export
//! carrying none of them is still scaled by all three. A key-presence
//! gate sees nothing in such a file, which is why MiniCPM used to be
//! refused by NAME rather than detected: nothing in the metadata
//! reveals it. [`MultiplierDefaults`] is that hook, and it is a FIELD
//! of [`MultiplierSupport`] rather than a second table, so an
//! architecture cannot be given a default for a key its graph does not
//! apply.
//!
//! MiniCPM differs from Granite in exactly one column: it never reads
//! `{arch}.attention.scale` (`minicpm.cpp:3-24` contains no
//! `LLM_KV_ATTENTION_SCALE`), so `hparams.f_attention_scale` keeps its
//! `0.0f` and `granite.cpp:225` falls back to `1/sqrt(n_embd_head)`.
//! That key is still refused for `minicpm`, by the derived list, which
//! is what deriving it is for.
//!
//! **Grok is the MiniCPM shape with two more columns moved.**
//! `grok.cpp:5-12` seeds SEVEN hyper-parameters before `:14-27` let the
//! file override each with `required = false`, so a Grok-1 export that
//! declares none of them is still scaled by all of them and
//! `unsupported_scaling_keys` sees an ordinary file. The converter
//! (`conversion/grok.py:34-57`) writes every one of them for a fresh
//! export; the defaults are llama.cpp's own comment, "defaults for old
//! GGUFs". Four of the seven are this module's business:
//!
//! ```text
//! hparams.f_logit_scale            = 0.5773502691896257f;   // 1/sqrt(3)
//! hparams.f_embedding_scale        = 78.38367176906169f;    // sqrt(6144)
//! hparams.f_attn_out_scale         = 0.08838834764831845f;  // 1/sqrt(128)
//! hparams.f_attn_logit_softcapping = 30.0f;
//! ```
//!
//! and the graph applies them differently from Granite in two places.
//! `grok.cpp:211` is `ggml_scale(cur, f_logit_scale)` -- a MULTIPLY,
//! [`LogitScaleUse::AsIs`], the third variant this header used to name
//! as deliberately absent. And the attention scale is not
//! `{arch}.attention.scale` at all: `grok.cpp` never reads that key. It
//! reads `{arch}.attention.output_scale` (`LLM_KV_ATTENTION_OUTPUT_SCALE`,
//! `:18`) and applies it INSIDE the softcap, passing `kq_scale = 1.0f`
//! to `build_attn` (`:137`) and letting `llama-graph.cpp:2572-2582` do
//! `kq = 30 * tanh(kq * f_attn_out_scale / 30)` before the softmax.
//! That is arithmetically "pre-scale Q by `f_attn_out_scale`, then
//! softcap at 30", which is what `ModelConfig::attention_scale` plus
//! `ModelConfig::attn_logit_softcap` already compute, so the key
//! resolves into the SAME slot as Granite's `attention.scale` and
//! [`AttentionScaleKey`] records which spelling each architecture
//! reads. (llama.cpp forces flash attention OFF for Grok,
//! `llama-context.cpp:3544-3547`, precisely because its FA path has no
//! room for that fold; the non-FA branch is the reference.)
//!
//! The other three of the seven are not multipliers. The softcap
//! default of 30 is [`MultiplierDefaults::attn_logit_softcap`], on the
//! same variant so that ONE hook carries everything `grok.cpp:5-12`
//! seeds; `yarn_beta_fast = 8.0f` (:5, against `llama-hparams.h:137`'s
//! 32) is [`MultiplierDefaults::yarn_beta_fast`] for the same reason;
//! and `f_router_logit_softcapping` (:10,:20) plus `attn_temp_length`
//! (:23) are read and then applied NOWHERE -- no other reference to
//! either field exists under `src/` (measured), so frink reads neither
//! and refuses neither.
//!
//! `residual_scale` stays refused for `grok`: the graph has no residual
//! multiplier, and the derived list keeps saying so.
//!
//! * **Command-R** applies `f_logit_scale` as a MULTIPLY
//!   (`command-r.cpp:4,137-138`, optional, skipped at zero), which is
//!   [`LogitScaleUse::AsIs`], the `grok` / `talkie` use; it is a row
//!   since its other blocker, the shared-norm parallel residual over a
//!   weighted LayerNorm, landed (`crate::parallel_residual`,
//!   `capability::WEIGHTED_LAYER_NORM`). **Cohere2** reads the same key
//!   REQUIRED (`cohere2.cpp:14`, `get_key` with no `false`) and applies
//!   it the same way (`:153-154`), so it is the `talkie` shape,
//!   [`LogitScaleUse::AsIs`], under its own row.
//!
//! **Neither attention key lives in this module's output.** Both
//! `{arch}.attention.scale` and `{arch}.attention.output_scale` resolve
//! into the `ModelConfig::attention_scale` slot Gemma-27B already uses,
//! because that slot's contract -- "pre-scale Q and pass 1.0 to the
//! kernel" -- is exactly what llama.cpp's `kq_scale` needs and having
//! two fields for one number is the shape this repo keeps paying for.

/// How an architecture's graph turns `{arch}.logit_scale` into a
/// multiplier on the lm_head's output.
///
/// The direction is a per-architecture fact with no key, so it is
/// resolved HERE and the decoder only ever multiplies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogitScaleUse {
    /// The graph never scales its logits.
    #[default]
    NotApplied,
    /// Granite / MiniCPM: `ggml_scale(cur, 1.0f / f_logit_scale)`
    /// (`granite.cpp:180`).
    ///
    /// Whether the KEY is required is not part of this variant: it
    /// follows from [`MultiplierSupport::defaults`]. `granite.cpp:7`
    /// reads it with no default, so a Granite file omitting it is
    /// refused; `minicpm.cpp:7` seeds `256/n_embd` first, so a MiniCPM
    /// file omitting it is scaled by that. One fact, derived, rather
    /// than a `required` flag beside the table that could come to
    /// disagree with it.
    Reciprocal,
    /// Grok / Talkie: `ggml_scale(cur, f_logit_scale)` (`grok.cpp:211`,
    /// `talkie.cpp:141`). The value IS the multiplier, and the key is
    /// REQUIRED (or seeded, for Grok).
    ///
    /// The same positivity rule as [`Self::Reciprocal`], for the same
    /// reason: a Metal decode stack may fold the lm_head into an argmax
    /// only while every post-head transform is monotone increasing, and
    /// a zero would blank the whole vocabulary rather than divide by it.
    AsIs,
    /// Command-R: the same multiply behind `if (f_logit_scale)`
    /// (`command-r.cpp:4,137-138`; `get_key(..., false)` leaves the
    /// `llama-hparams.h` zero when the file has no key). Absent or zero
    /// is "no scale"; a negative value is refused, because the graph
    /// would apply it and the argmax fold could not.
    AsIsOptional,
}

/// Which GGUF key, if any, an architecture reads its attention scale
/// from.
///
/// Two spellings, one slot. Granite reads `{arch}.attention.scale`
/// (`LLM_KV_ATTENTION_SCALE`, `granite.cpp:8`) and passes it to
/// `build_attn` as `kq_scale`. Grok reads
/// `{arch}.attention.output_scale` (`LLM_KV_ATTENTION_OUTPUT_SCALE`,
/// `grok.cpp:18`), passes `kq_scale = 1.0f`, and applies the value
/// inside its tanh softcap (`llama-graph.cpp:2579`). Both are "the
/// score is `q . k * s`", and both resolve into
/// `ModelConfig::attention_scale`. Which key is a per-architecture fact
/// with no key of its own, so it is here, and
/// [`crate::capability::unsupported_scaling_keys`] derives from it that
/// the spelling an architecture does NOT read stays refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionScaleKey {
    /// The graph uses the kernels' own `1/sqrt(head_dim)` and reads no
    /// key.
    #[default]
    NotRead,
    /// `{arch}.attention.scale`, with `0.0` as llama.cpp's own "unset"
    /// sentinel (`granite.cpp:225`).
    Scale,
    /// `{arch}.attention.output_scale`, applied as-is: `grok.cpp` has no
    /// sentinel test on it, so a declared zero really scales every
    /// score by zero there, and does here.
    OutputScale,
}

impl AttentionScaleKey {
    /// The key suffix this architecture reads, for the loader.
    pub fn suffix(self) -> Option<&'static str> {
        match self {
            Self::NotRead => None,
            Self::Scale => Some("attention.scale"),
            Self::OutputScale => Some("attention.output_scale"),
        }
    }
}

/// The value a multiplier takes when the file declares no key.
///
/// llama.cpp spells this as plain assignment before a `required = false`
/// `get_key`, so the default and the override are one statement apart
/// and easy to read past. Here it is a variant, because "the file said
/// nothing" and "the file said 1.0" are the same input to [`resolve`]
/// and must not be the same output.
///
/// It is a field of [`MultiplierSupport`] rather than a table beside it:
/// a default for a key the graph does not apply would be arithmetic
/// nothing performs, and this way that combination is not expressible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MultiplierDefaults {
    /// The file is the only source. A key it omits is not applied, and
    /// a `logit_scale` it omits is an error where the graph divides by
    /// it (`granite.cpp:7` reads that key with no default and throws).
    #[default]
    FromFileOnly,
    /// MiniCPM: `minicpm.cpp:5-7` assigns all three multipliers BEFORE
    /// `:12-14` lets the file override them.
    ///
    /// ```text
    /// f_embedding_scale = 12.0f;
    /// f_residual_scale  = 1.4f / sqrtf(float(n_layer));
    /// f_logit_scale     = n_embd ? (256.0f / float(n_embd)) : 1.0f;
    /// ```
    ///
    /// No `attention.scale` default: MiniCPM does not read that key at
    /// all, so `f_attention_scale` keeps llama.cpp's own `0.0f`.
    MiniCpm,
    /// Grok: `grok.cpp:5-12` assigns seven hyper-parameters BEFORE
    /// `:14-27` let the file override them. The four this module
    /// resolves:
    ///
    /// ```text
    /// f_logit_scale            = 0.5773502691896257f;   // multiplied, :211
    /// f_embedding_scale        = 78.38367176906169f;
    /// f_attn_out_scale         = 0.08838834764831845f;  // inside the softcap
    /// f_attn_logit_softcapping = 30.0f;
    /// ```
    ///
    /// No `residual_scale`: the graph has none. The softcap and
    /// `yarn_beta_fast = 8.0f` (:5) are not multipliers and are exposed
    /// as [`Self::attn_logit_softcap`] / [`Self::yarn_beta_fast`] rather
    /// than as fields of [`DeclaredMultipliers`], so that ONE variant
    /// still carries everything the C file seeds.
    Grok,
}

impl MultiplierDefaults {
    /// What this architecture applies for each key the file leaves out.
    ///
    /// `None` in a field means "nothing to fall back on", which for
    /// [`LogitScaleUse::Reciprocal`] is what makes the key required.
    pub fn values(self, dims: MultiplierDims) -> DeclaredMultipliers {
        match self {
            Self::FromFileOnly => DeclaredMultipliers::default(),
            Self::MiniCpm => DeclaredMultipliers {
                // `n_embd ? 256/n_embd : 1.0` -- the ternary is
                // llama.cpp's own guard against a zero embedding width,
                // kept because dropping it turns a malformed file into a
                // division by zero instead of the missing-hyper-parameter
                // error the loader already raises for it.
                logit: Some(if dims.n_embd == 0 {
                    1.0
                } else {
                    256.0 / dims.n_embd as f32
                }),
                residual: Some(1.4 / (dims.n_layer as f32).sqrt()),
                embedding: Some(12.0),
                attention: None,
            },
            Self::Grok => DeclaredMultipliers {
                logit: Some(0.577_350_3),
                residual: None,
                embedding: Some(78.383_67),
                attention: Some(0.088_388_35),
            },
        }
    }

    /// The attention logit softcap the graph applies when the file
    /// declares no `{arch}.attn_logit_softcapping`.
    ///
    /// `grok.cpp:9` seeds 30.0 before `:19` reads the key with
    /// `required = false`; `llama-hparams.h:110` gives every other
    /// architecture 50.0, which only Gemma-2 (`attn_soft_cap = true`)
    /// ever applies -- and Gemma-2's converter always writes the key.
    /// `None` here means "the file is the only source", which for the
    /// generic path means no softcap.
    pub fn attn_logit_softcap(self) -> Option<f32> {
        match self {
            Self::FromFileOnly | Self::MiniCpm => None,
            Self::Grok => Some(30.0),
        }
    }

    /// YaRN's `beta_fast` when the file declares
    /// `rope.scaling.type = yarn` and no `rope.scaling.yarn.beta_fast`.
    ///
    /// `grok.cpp:5` seeds 8.0 against `llama-hparams.h:137`'s 32.0
    /// before `:26` lets the file override it. Only Grok-2 exports
    /// declare YaRN (`conversion/grok.py:42-49`) and they write the key,
    /// so this matters for a hand-written file; it is here so that the
    /// default is ONE fact beside the others rather than a literal in
    /// the RoPE loader that a future row would restate.
    pub fn yarn_beta_fast(self) -> Option<f32> {
        match self {
            Self::FromFileOnly | Self::MiniCpm => None,
            Self::Grok => Some(8.0),
        }
    }

    /// The file's declaration where it has one, this architecture's
    /// default where it does not.
    ///
    /// Destructured exhaustively with no `..` on purpose: a fifth
    /// multiplier must not be able to slip through unmerged.
    fn merge(self, declared: DeclaredMultipliers, dims: MultiplierDims) -> DeclaredMultipliers {
        let DeclaredMultipliers {
            logit,
            residual,
            embedding,
            attention,
        } = declared;
        let d = self.values(dims);
        DeclaredMultipliers {
            logit: logit.or(d.logit),
            residual: residual.or(d.residual),
            embedding: embedding.or(d.embedding),
            attention: attention.or(d.attention),
        }
    }
}

/// The model dimensions the defaults and the sentinels are computed
/// from.
///
/// One struct rather than three positional `usize` arguments, because
/// `resolve(support, declared, 6, 2, 24)` is three chances to swap two
/// of them and no way for the compiler to notice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiplierDims {
    /// `n_embd_head`, for the `attention.scale` that restates the
    /// kernels' own `1/sqrt(head_dim)`.
    pub head_dim: usize,
    /// `n_layer`, for MiniCPM's `1.4/sqrt(n_layer)` residual default.
    pub n_layer: usize,
    /// `n_embd`, for MiniCPM's `256/n_embd` logit default.
    pub n_embd: usize,
}

/// Which of the four multipliers this architecture's reference graph
/// applies -- and therefore which frink implements for it.
///
/// The same value drives BOTH halves: what the loader reads and applies,
/// and what [`crate::capability::unsupported_scaling_keys`] still
/// refuses. Deriving the refusal list from this struct is the point --
/// a hand-written second list is how frink once refused a key it
/// implemented and implemented a key it refused.
/// What a graph does with `{arch}.residual_scale`.
///
/// Three answers rather than a bool, because `minimax-01` reads the
/// same key and multiplies something else by it. It is the one graph of
/// the 155 that does: `minimax-01.cpp:249` makes `residual` the
/// ATTENTION PRE-NORM's output, `:428-431` add `scale * residual` to
/// the attention branch, and `:440,455-458` do the same with the FFN
/// pre-norm's. The layer input itself -- `inpSA`, `:244` -- is never
/// added to anything; it survives only to be sliced by `inp_out_ids`
/// at `:424` and is then dropped, which is how a reader can check that
/// the stream really is the normed value and not a second residual.
///
/// Measured over the pin: `grep -ln f_residual_scale src/models/*.cpp`
/// is five files, and `minimax-01.cpp` is the only one whose
/// `ggml_scale` takes a norm's output rather than a branch's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResidualScaleUse {
    /// The graph never reads the key.
    #[default]
    NotRead,
    /// `granite.cpp:213,238`: every BRANCH output is multiplied before
    /// it rejoins the stream, read with `required = false`.
    BranchOutput,
    /// `minimax-01.cpp:428,455`: the sublayer's own PRE-NORM OUTPUT,
    /// multiplied, REPLACES the stream the branch joins. REQUIRED
    /// (`minimax-01.cpp:6` reads it with no default), so a file
    /// without it is refused here as llama.cpp refuses it.
    NormedInputRequired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MultiplierSupport {
    /// `{arch}.embedding_scale` multiplies every token embedding row.
    pub embedding: bool,
    /// What the graph does with `{arch}.residual_scale`.
    pub residual: ResidualScaleUse,
    /// What the graph does with `{arch}.logit_scale`.
    pub logit: LogitScaleUse,
    /// Which key, if any, replaces the kernels' `1/sqrt(head_dim)`.
    pub attention: AttentionScaleKey,
    /// What the graph applies for a key the file does NOT declare.
    pub defaults: MultiplierDefaults,
}

impl MultiplierSupport {
    /// Nothing declared and nothing applied: the generic decoder's own
    /// graph.
    pub const NONE: Self = Self {
        embedding: false,
        residual: ResidualScaleUse::NotRead,
        logit: LogitScaleUse::NotApplied,
        attention: AttentionScaleKey::NotRead,
        defaults: MultiplierDefaults::FromFileOnly,
    };

    /// `granite`, `granitemoe` and the `granite-moe` alias. One
    /// constant, so the three rows cannot disagree about the scaling
    /// they share.
    pub const GRANITE: Self = Self {
        embedding: true,
        residual: ResidualScaleUse::BranchOutput,
        logit: LogitScaleUse::Reciprocal,
        attention: AttentionScaleKey::Scale,
        defaults: MultiplierDefaults::FromFileOnly,
    };

    /// `minicpm`. The same graph as [`Self::GRANITE`]
    /// (`models.h:1594-1601` is `using graph = llama_model_granite::graph`)
    /// with two differences, both from `minicpm.cpp:3-24`: it never
    /// reads `{arch}.attention.scale`, and it seeds the other three
    /// before the file is consulted.
    pub const MINICPM: Self = Self {
        embedding: true,
        residual: ResidualScaleUse::BranchOutput,
        logit: LogitScaleUse::Reciprocal,
        attention: AttentionScaleKey::NotRead,
        defaults: MultiplierDefaults::MiniCpm,
    };

    /// `grok`. Embedding and logit multipliers with the logit one
    /// MULTIPLIED (`grok.cpp:211`), the attention scale from
    /// `attention.output_scale` (`:18`, applied at
    /// `llama-graph.cpp:2579`), no residual multiplier, and every one
    /// of them seeded before the file is read.
    pub const GROK: Self = Self {
        embedding: true,
        residual: ResidualScaleUse::NotRead,
        logit: LogitScaleUse::AsIs,
        attention: AttentionScaleKey::OutputScale,
        defaults: MultiplierDefaults::Grok,
    };

    /// `talkie`. `{arch}.logit_scale` REQUIRED (`talkie.cpp:5`) and
    /// MULTIPLIED onto the logits (`:141`, `ggml_scale(cur,
    /// f_logit_scale)`), the `grok` use; none of the other three keys
    /// is read, and nothing is seeded before the file.
    pub const TALKIE: Self = Self {
        embedding: false,
        residual: ResidualScaleUse::NotRead,
        logit: LogitScaleUse::AsIs,
        attention: AttentionScaleKey::NotRead,
        defaults: MultiplierDefaults::FromFileOnly,
    };

    /// `command-r`. `{arch}.logit_scale` OPTIONAL (`command-r.cpp:4`,
    /// `required = false`) and MULTIPLIED onto the logits when nonzero
    /// (`:137-138`); every real export writes it (`conversion/
    /// command_r.py:19`, `0.0625` for the 35B). None of the other three
    /// keys is read.
    pub const COMMAND_R: Self = Self {
        embedding: false,
        residual: ResidualScaleUse::NotRead,
        logit: LogitScaleUse::AsIsOptional,
        attention: AttentionScaleKey::NotRead,
        defaults: MultiplierDefaults::FromFileOnly,
    };

    /// `cohere2`. `{arch}.logit_scale` REQUIRED (`cohere2.cpp:14`) and
    /// MULTIPLIED onto the logits (`:153-154`); every export writes it
    /// (`conversion/command_r.py:27`, `0.25` for Command-R7B). None of
    /// the other three keys is read.
    /// An `embedding_scale` and nothing else: `hrm-text.cpp:8`.
    pub const EMBEDDING_ONLY: Self = Self {
        embedding: true,
        residual: ResidualScaleUse::NotRead,
        logit: LogitScaleUse::NotApplied,
        attention: AttentionScaleKey::NotRead,
        defaults: MultiplierDefaults::FromFileOnly,
    };

    /// `minimax-01`. ONE key, REQUIRED, and it multiplies each
    /// sublayer's pre-norm output rather than its branch output
    /// ([`ResidualScaleUse::NormedInputRequired`]). None of the other
    /// three is read: `minimax-01.cpp:5-6` is the whole of its
    /// `load_arch_hparams` beside the RMS epsilon.
    pub const MINIMAX_01: Self = Self {
        embedding: false,
        residual: ResidualScaleUse::NormedInputRequired,
        logit: LogitScaleUse::NotApplied,
        attention: AttentionScaleKey::NotRead,
        defaults: MultiplierDefaults::FromFileOnly,
    };

    pub const COHERE2: Self = Self {
        embedding: false,
        residual: ResidualScaleUse::NotRead,
        logit: LogitScaleUse::AsIs,
        attention: AttentionScaleKey::NotRead,
        defaults: MultiplierDefaults::FromFileOnly,
    };
}

/// The GGUF architectures whose graph applies one or more of the four
/// multipliers, outside the Gemma family (which [`multiplier_support`]
/// keys off [`DecoderFamily::GemmaFamily`] instead of naming five
/// strings that would then have to be kept in step with the catalog).
///
/// `granite-moe` has no llama.cpp spelling -- `llama-arch.cpp:101` is
/// `granitemoe` -- and exists only because frink's catalog carries the
/// hyphenated alias. It is here so a file declaring it cannot get
/// different arithmetic from the row it is an alias FOR.
const MULTIPLIER_ARCHITECTURES: &[(&str, MultiplierSupport)] = &[
    ("granite", MultiplierSupport::GRANITE),
    ("granitemoe", MultiplierSupport::GRANITE),
    ("granite-moe", MultiplierSupport::GRANITE),
    // `granite-hybrid.cpp:4-7` reads the same four, all optional, and
    // `:113,145-147,175` apply them at the same four sites.
    ("granitehybrid", MultiplierSupport::GRANITE),
    ("granite-hybrid", MultiplierSupport::GRANITE),
    // `granite-swa.cpp:7-10` reads the same four -- logit REQUIRED, the
    // other three optional -- and `:170-171,192,231,247-249,308-310`
    // apply them at the same four sites. Landed upstream after the
    // 2026-08-04 pin.
    ("granite_swa", MultiplierSupport::GRANITE),
    ("minicpm", MultiplierSupport::MINICPM),
    ("grok", MultiplierSupport::GROK),
    ("talkie", MultiplierSupport::TALKIE),
    ("command-r", MultiplierSupport::COMMAND_R),
    ("cohere2", MultiplierSupport::COHERE2),
    ("minimax-01", MultiplierSupport::MINIMAX_01),
    // `muse-glimmer.cpp:8` reads `logit_scale` REQUIRED and `:186`
    // MULTIPLIES by it, which is `cohere2`'s shape rather than
    // Granite's divide.
    ("muse-glimmer", MultiplierSupport::COHERE2),
    // `hrm-text.cpp:8` reads `embedding_scale` OPTIONAL and
    // `build_inp_embd` applies it; the other three it never reads.
    ("hrm_text", MultiplierSupport::EMBEDDING_ONLY),
    // `cohere2moe.cpp:14,287-289`: the same REQUIRED key, multiplied
    // when nonzero.
    ("cohere2moe", MultiplierSupport::COHERE2),
];

/// Which multipliers frink applies for `arch`.
///
/// This is about the KEYS, not about whether the architecture scales
/// anything. Gemma is the case that makes the distinction load-bearing:
/// it scales its embeddings and, at 27B, its attention scores, but it
/// reads neither key -- `gemma3.cpp:31` and `gemma2.cpp:27` ASSIGN
/// `f_attention_scale` from the model type, and the embedding scale is
/// `sqrt(n_embd)` computed in the graph. So a Gemma file declaring
/// `gemma3.embedding_scale` describes something llama.cpp does not do,
/// and frink refuses it here rather than honouring a number its own
/// reference ignores. The Gemma scales themselves come from
/// `capability::attention_scale_override` and `loader.rs`'s family
/// branch, which is where an arch-computed value belongs.
pub fn multiplier_support(arch: &str) -> MultiplierSupport {
    MULTIPLIER_ARCHITECTURES
        .iter()
        .find(|(n, _)| *n == arch)
        .map_or(MultiplierSupport::NONE, |(_, s)| *s)
}

/// The raw values a file declares, before llama.cpp's per-key sentinels
/// are applied.
///
/// Destructured exhaustively by [`resolve`] with no `..`, so a fifth
/// multiplier cannot be added here and silently ignored there.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DeclaredMultipliers {
    pub logit: Option<f32>,
    pub residual: Option<f32>,
    pub embedding: Option<f32>,
    /// Whichever key [`MultiplierSupport::attention`] names -- the
    /// loader reads exactly that one.
    pub attention: Option<f32>,
}

/// The resolved multipliers, in the form [`crate::ModelConfig`] carries
/// them: `None` means "this graph does not do that".
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ResolvedMultipliers {
    /// [`crate::ModelConfig::embedding_scale`].
    pub embedding_scale: Option<f32>,
    /// [`crate::ModelConfig::residual_scale`]: the multiplier on every
    /// branch OUTPUT.
    pub residual_scale: Option<f32>,
    /// [`crate::ModelConfig::normed_residual_scale`]: the multiplier on
    /// each sublayer's PRE-NORM OUTPUT, which then replaces the
    /// residual stream ([`ResidualScaleUse::NormedInputRequired`]).
    ///
    /// Never `Some` together with [`Self::residual_scale`]: one column
    /// resolves both, and a `1.0` here is kept where a `1.0` there is
    /// dropped, because this field also carries the TOPOLOGY and the
    /// identity scale does not make it the ordinary one.
    pub normed_residual_scale: Option<f32>,
    /// [`crate::ModelConfig::logit_multiplier`], already inverted where
    /// the architecture divides.
    pub logit_multiplier: Option<f32>,
    /// [`crate::ModelConfig::attention_scale`].
    pub attention_scale: Option<f32>,
}

/// Why a declared multiplier cannot be honoured.
#[derive(Debug, Clone, PartialEq)]
pub enum MultiplierError {
    /// The architecture reads `{arch}.logit_scale` as REQUIRED
    /// (`granite.cpp:7`, `granite-moe.cpp:5`) and the file has no such
    /// key. llama.cpp throws on this file too. Unreachable for an
    /// architecture with a default, which is what makes the default a
    /// default.
    MissingRequiredLogitScale,
    /// A `logit_scale` of zero would divide by zero, and a negative one
    /// would REORDER the vocabulary -- which matters beyond the logits
    /// themselves, because a Metal decode stack is allowed to fold the
    /// lm_head and return an argmax id only while every post-head
    /// transform is monotone increasing.
    NonPositiveLogitScale(f32),
    /// The architecture reads `{arch}.residual_scale` as REQUIRED
    /// (`minimax-01.cpp:6`) and the file does not declare it.
    /// llama.cpp throws on the same file.
    MissingRequiredResidualScale,
}

impl MultiplierError {
    /// The sentence the loader puts in its error, naming the key.
    pub fn message(&self, arch: &str) -> String {
        match self {
            MultiplierError::MissingRequiredLogitScale => format!(
                "`{arch}.logit_scale` is REQUIRED for this architecture (src/models/granite.cpp:7 \
                 reads it with no default) and the file does not declare it; llama.cpp refuses \
                 the same file"
            ),
            MultiplierError::MissingRequiredResidualScale => format!(
                "`{arch}.residual_scale` is REQUIRED for this architecture \
                 (src/models/minimax-01.cpp:6 reads it with no default) and the file does not \
                 declare it; llama.cpp refuses the same file"
            ),
            MultiplierError::NonPositiveLogitScale(v) => format!(
                "`{arch}.logit_scale` = {v}: the graph scales every logit by it \
                 (src/models/granite.cpp:180 divides, src/models/grok.cpp:211 multiplies), so \
                 zero blanks or divides away the whole vocabulary and a negative value \
                 reorders it"
            ),
        }
    }
}

/// The no-op sentinel for `embedding_scale`, `residual_scale` and the
/// already-inverted `logit_scale`.
///
/// Two values are inert for these three, for two different reasons.
/// `1.0` is the arithmetic identity. `0.0` is llama.cpp's own "off":
/// `llama-graph.cpp:2337` tests `f_embedding_scale != 0.0f` and
/// `granite.cpp:235` tests `if (hparams.f_residual_scale)`, so a file
/// writing zero there means "do not scale" rather than "multiply
/// everything by zero", and reading it literally would blank the whole
/// residual stream.
///
/// **This is deliberately NOT applied to `attention.scale`**, and the
/// difference is the point. `f_attention_scale` uses `0.0` as its
/// "unset, use `1/sqrt(n_embd_head)`" sentinel (`granite.cpp:225`) and
/// `1.0` as a perfectly ordinary override -- llama.cpp passes it
/// straight to `build_attn` as `kq_scale`. Folding the two keys' rules
/// into one predicate silently dropped a declared `attention.scale` of
/// 1.0 while this module was being written, which is what
/// `each_keys_own_no_op_value_is_what_switches_it_off` is for.
fn scale_or_none(v: Option<f32>) -> Option<f32> {
    v.filter(|&v| v != 0.0 && v != 1.0)
}

/// Turn what the file declared -- plus what this architecture applies
/// when it declared nothing -- into what the decoder applies.
///
/// The defaults are merged FIRST, in llama.cpp's own order: assignment,
/// then the optional key read, then the graph's sentinel tests. Doing it
/// the other way round would let a MiniCPM file declaring
/// `residual_scale = 1.0` fall back to `1.4/sqrt(n_layer)` instead of
/// switching the scaling off, which is the opposite of what
/// `granite.cpp:235`'s `if (hparams.f_residual_scale)` does with it.
///
/// `dims.head_dim` is only used to drop an `attention.scale` that
/// restates the kernels' own `1/sqrt(head_dim)`:
/// `ModelConfig::attention_scale` means "pre-scale Q and pass 1.0 to the
/// kernel", so restating the default would be arithmetically identical
/// but would fence the layer off every fused Metal launch for nothing.
pub fn resolve(
    support: MultiplierSupport,
    declared: DeclaredMultipliers,
    dims: MultiplierDims,
) -> Result<ResolvedMultipliers, MultiplierError> {
    let DeclaredMultipliers {
        logit,
        residual,
        embedding,
        attention,
    } = support.defaults.merge(declared, dims);

    let logit_multiplier = match support.logit {
        LogitScaleUse::NotApplied => None,
        LogitScaleUse::Reciprocal => {
            let v = logit.ok_or(MultiplierError::MissingRequiredLogitScale)?;
            if v <= 0.0 {
                return Err(MultiplierError::NonPositiveLogitScale(v));
            }
            // `1.0` inverts to `1.0`, which `scale_or_none` then drops:
            // a file declaring the identity gets the graph frink
            // already computes, with no needless multiply per token.
            scale_or_none(Some(1.0 / v))
        }
        LogitScaleUse::AsIs => {
            let v = logit.ok_or(MultiplierError::MissingRequiredLogitScale)?;
            if v <= 0.0 {
                return Err(MultiplierError::NonPositiveLogitScale(v));
            }
            scale_or_none(Some(v))
        }
        LogitScaleUse::AsIsOptional => match logit {
            None | Some(0.0) => None,
            Some(v) if v < 0.0 => return Err(MultiplierError::NonPositiveLogitScale(v)),
            Some(v) => scale_or_none(Some(v)),
        },
    };

    // An override that restates the kernels' own `1/sqrt(head_dim)`
    // resolves to `None` for either key: arithmetically identical, and
    // `Some` here fences the model off every fused Metal attention
    // launch for nothing. Real Grok-1 is exactly this case --
    // `0.0883883... == 1/sqrt(128)` at head_dim 128.
    let restates_kernel = |v: &f32| {
        let kernel = 1.0 / (dims.head_dim as f32).sqrt();
        (v - kernel).abs() > f32::EPSILON * kernel.max(1.0)
    };
    let attention_scale = match support.attention {
        AttentionScaleKey::NotRead => None,
        // `0.0` is this key's ONLY sentinel (`granite.cpp:225`); 1.0 is
        // a real override. See `scale_or_none`, which must not be used
        // here.
        AttentionScaleKey::Scale => attention.filter(|&v| v != 0.0).filter(restates_kernel),
        // No sentinel at all: `grok.cpp` applies whatever it holds.
        AttentionScaleKey::OutputScale => attention.filter(restates_kernel),
    };

    Ok(ResolvedMultipliers {
        embedding_scale: support
            .embedding
            .then(|| scale_or_none(embedding))
            .flatten(),
        residual_scale: match support.residual {
            ResidualScaleUse::BranchOutput => scale_or_none(residual),
            ResidualScaleUse::NotRead | ResidualScaleUse::NormedInputRequired => None,
        },
        normed_residual_scale: match support.residual {
            // NOT `scale_or_none`: the value carries the topology as
            // well as the multiplier, and `minimax-01` is a different
            // graph at a scale of exactly 1.0 -- it discards the layer
            // input either way.
            ResidualScaleUse::NormedInputRequired => {
                Some(residual.ok_or(MultiplierError::MissingRequiredResidualScale)?)
            }
            ResidualScaleUse::NotRead | ResidualScaleUse::BranchOutput => None,
        },
        logit_multiplier,
        attention_scale,
    })
}

/// `hidden += scale.unwrap_or(1.0) * branch`, the ONE residual add in
/// the generic decoder.
///
/// Every `hidden[i] += branch[i]` in `decoder.rs` goes through here, and
/// that is the whole point of the function existing. `residual_scale`
/// multiplies BOTH branch outputs of EVERY layer (`granite.cpp:235-238`,
/// `:288-292`), and `decoder.rs` spells the residual add out eighteen
/// times across prefill, decode, paged decode and continuous batching.
/// Eighteen hand-written adds that must all agree about one scalar is
/// precisely the shape that has cost this repo eight model features, so
/// the scalar is a parameter of a shared function rather than a rule
/// eighteen call sites are trusted to remember.
#[inline]
pub fn residual_add(hidden: &mut [f32], branch: &[f32], scale: Option<f32>) {
    debug_assert_eq!(hidden.len(), branch.len());
    match scale {
        None => {
            for (h, b) in hidden.iter_mut().zip(branch.iter()) {
                *h += *b;
            }
        }
        Some(s) => {
            for (h, b) in hidden.iter_mut().zip(branch.iter()) {
                *h += s * *b;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dimensions for the rows whose arithmetic does not depend on
    /// them. Only the MiniCPM defaults read `n_layer` / `n_embd`, and
    /// those tests spell their own out.
    fn dims(head_dim: usize) -> MultiplierDims {
        MultiplierDims {
            head_dim,
            n_layer: 2,
            n_embd: 24,
        }
    }

    /// The three Granite rows share one support constant, so they cannot
    /// be given different arithmetic by an edit to one of them.
    ///
    /// `granite-moe` is the row this matters most for: no llama.cpp GGUF
    /// spells it that way, so nothing outside frink would ever notice
    /// it drifting.
    #[test]
    fn the_three_granite_rows_have_identical_multiplier_support() {
        let dense = multiplier_support("granite");
        assert_eq!(dense, MultiplierSupport::GRANITE);
        for alias in ["granitemoe", "granite-moe"] {
            assert_eq!(
                multiplier_support(alias),
                dense,
                "`{alias}` must scale exactly like `granite`"
            );
        }
    }

    /// An architecture nobody read must apply NOTHING, so that adding a
    /// row to the catalog cannot silently start scaling it.
    #[test]
    fn an_architecture_that_was_not_read_applies_no_multipliers() {
        for arch in ["llama", "qwen3", "deepseek", "not-an-architecture"] {
            assert_eq!(
                multiplier_support(arch),
                MultiplierSupport::NONE,
                "`{arch}` must not scale"
            );
        }
    }

    /// Gemma reads NONE of the four keys, even though it scales two of
    /// the things they name.
    ///
    /// The distinction this pins is between "the architecture scales
    /// this" and "the architecture reads this key". Gemma's embedding
    /// scale is `sqrt(n_embd)` computed in the graph and its 27B
    /// attention scale is assigned from the model type
    /// (`gemma3.cpp:31`, `gemma2.cpp:27`), so a file declaring either
    /// key describes something llama.cpp does not do.
    ///
    /// The whole family used to be exempted from the refusal list
    /// wholesale, which meant a hand-written `gemma3.residual_scale`
    /// would have loaded and been silently ignored -- exactly the
    /// blind spot that list exists to close.
    #[test]
    fn the_gemma_family_reads_none_of_the_four_keys() {
        for arch in ["gemma", "gemma2", "gemma3"] {
            assert_eq!(
                multiplier_support(arch),
                MultiplierSupport::NONE,
                "`{arch}` computes its scales; it does not read them"
            );
        }
    }

    /// Granite DIVIDES by `logit_scale`; the config carries the already
    /// inverted multiplier so the decoder only ever multiplies.
    ///
    /// Getting the direction backwards is invisible in a smoke test --
    /// the logits are still finite, still ordered the same way, and only
    /// the temperature of the distribution moves.
    #[test]
    fn granites_logit_scale_is_inverted_at_load_time() {
        let got = resolve(
            MultiplierSupport::GRANITE,
            DeclaredMultipliers {
                logit: Some(8.0),
                ..Default::default()
            },
            dims(64),
        )
        .expect("8.0 resolves");
        assert_eq!(got.logit_multiplier, Some(0.125));
    }

    /// The REQUIRED half of `logit_scale`, and the reason it is an error
    /// rather than a default of 1.0: llama.cpp cannot load such a file
    /// either, so silently running it would mean frink answering where
    /// its own reference refuses.
    #[test]
    fn a_granite_file_with_no_logit_scale_is_refused_rather_than_defaulted() {
        assert_eq!(
            resolve(
                MultiplierSupport::GRANITE,
                DeclaredMultipliers::default(),
                dims(64),
            ),
            Err(MultiplierError::MissingRequiredLogitScale)
        );
        assert!(
            MultiplierError::MissingRequiredLogitScale
                .message("granite")
                .contains("granite.logit_scale"),
            "the message must name the key"
        );
    }

    /// Zero divides by zero; a negative value reorders the vocabulary
    /// and would break the Metal decode stack's right to fold the
    /// lm_head into an argmax.
    #[test]
    fn a_non_positive_logit_scale_is_refused() {
        for bad in [0.0f32, -2.0] {
            assert_eq!(
                resolve(
                    MultiplierSupport::GRANITE,
                    DeclaredMultipliers {
                        logit: Some(bad),
                        ..Default::default()
                    },
                    dims(64),
                ),
                Err(MultiplierError::NonPositiveLogitScale(bad)),
                "logit_scale {bad} must be refused"
            );
        }
    }

    /// The two sentinels are different values and each key is judged
    /// against its own.
    ///
    /// A single "1.0 means off" rule would leave `attention.scale = 0.0`
    /// looking like a real override and pre-scale every Q by zero; a
    /// single "0.0 means off" rule would leave `residual_scale = 1.0`
    /// costing a multiply per element per branch per layer forever.
    #[test]
    fn each_keys_own_no_op_value_is_what_switches_it_off() {
        let got = resolve(
            MultiplierSupport::GRANITE,
            DeclaredMultipliers {
                logit: Some(1.0),
                residual: Some(1.0),
                embedding: Some(1.0),
                attention: Some(0.0),
            },
            dims(64),
        )
        .expect("all no-ops resolve");
        assert_eq!(got, ResolvedMultipliers::default(), "{got:?}");

        // ... and the OTHER key's sentinel is not treated as a no-op.
        let got = resolve(
            MultiplierSupport::GRANITE,
            DeclaredMultipliers {
                logit: Some(2.0),
                residual: Some(0.0),
                embedding: Some(0.0),
                attention: Some(1.0),
            },
            dims(64),
        )
        .expect("resolves");
        assert_eq!(
            got.residual_scale, None,
            "llama.cpp's `if (f_residual_scale)` guard makes 0.0 mean off"
        );
        assert_eq!(got.embedding_scale, None, "llama-graph.cpp:2337 likewise");
        assert_eq!(
            got.attention_scale,
            Some(1.0),
            "1.0 is a real attention-scale override, not its sentinel"
        );
    }

    /// An `attention.scale` that restates `1/sqrt(head_dim)` resolves to
    /// `None`.
    ///
    /// Arithmetically it makes no difference; operationally it does.
    /// `Some` here fences the whole model off every fused Metal
    /// attention launch (`Decoder::layer_supports_metal_attn`), so
    /// restating the default would cost a real checkpoint the GPU path
    /// for nothing.
    #[test]
    fn an_attention_scale_equal_to_the_kernels_own_resolves_to_none() {
        let head_dim = 64;
        let kernel = 1.0 / (head_dim as f32).sqrt();
        let got = resolve(
            MultiplierSupport::GRANITE,
            DeclaredMultipliers {
                logit: Some(2.0),
                attention: Some(kernel),
                ..Default::default()
            },
            dims(head_dim),
        )
        .expect("resolves");
        assert_eq!(got.attention_scale, None);

        // A value that really differs survives.
        let got = resolve(
            MultiplierSupport::GRANITE,
            DeclaredMultipliers {
                logit: Some(2.0),
                attention: Some(0.015_625),
                ..Default::default()
            },
            dims(head_dim),
        )
        .expect("resolves");
        assert_eq!(got.attention_scale, Some(0.015_625));
    }

    /// An architecture that does not apply a multiplier ignores the
    /// value even when the file declares it.
    ///
    /// It cannot reach here in practice -- the loader refuses such a
    /// file first -- but the two halves have to agree about which keys
    /// are live, and this is the half that says so in code.
    #[test]
    fn support_gates_the_value_rather_than_the_value_gating_itself() {
        let got = resolve(
            MultiplierSupport::NONE,
            DeclaredMultipliers {
                logit: Some(8.0),
                residual: Some(0.22),
                embedding: Some(12.0),
                attention: Some(0.015_625),
            },
            dims(64),
        )
        .expect("an unsupported logit_scale is not even read");
        assert_eq!(got, ResolvedMultipliers::default());
    }

    /// The residual add, both arms, against arithmetic written out
    /// separately.
    #[test]
    fn the_residual_add_scales_the_branch_and_not_the_stream() {
        let mut hidden = vec![1.0f32, 2.0, 3.0];
        residual_add(&mut hidden, &[10.0, 20.0, 30.0], Some(0.5));
        assert_eq!(hidden, vec![6.0, 12.0, 18.0]);

        let mut hidden = vec![1.0f32, 2.0, 3.0];
        residual_add(&mut hidden, &[10.0, 20.0, 30.0], None);
        assert_eq!(
            hidden,
            vec![11.0, 22.0, 33.0],
            "no scale must be exactly the unscaled add, not a multiply by 1.0"
        );
    }

    /// A Grok file declaring NOTHING resolves to `grok.cpp:5-12`'s
    /// seven seeds -- the four this module carries, with the logit one
    /// NOT inverted.
    ///
    /// `head_dim` is 6 here so that the attention default (`1/sqrt(128)`)
    /// does not restate the kernels' own scale and survives into the
    /// config; on real Grok-1 (head_dim 128) it does restate it and
    /// resolves to `None`, which the second half pins.
    #[test]
    fn a_grok_file_declaring_nothing_is_scaled_by_all_of_grok_cpps_defaults() {
        let got = resolve(
            MultiplierSupport::GROK,
            DeclaredMultipliers::default(),
            dims(6),
        )
        .expect("defaults resolve");
        assert_eq!(got.embedding_scale, Some(78.383_67));
        assert_eq!(
            got.logit_multiplier,
            Some(0.577_350_3),
            "grok.cpp:211 MULTIPLIES by f_logit_scale; a reciprocal here would be 1.732"
        );
        assert_eq!(got.attention_scale, Some(0.088_388_35));
        assert_eq!(got.residual_scale, None, "grok has no residual multiplier");

        // Real Grok-1: head_dim 128, and 1/sqrt(128) IS the kernels'
        // scale, so the slot stays free and the fused Metal attention
        // stays eligible.
        let real = resolve(
            MultiplierSupport::GROK,
            DeclaredMultipliers::default(),
            MultiplierDims {
                head_dim: 128,
                n_layer: 64,
                n_embd: 6144,
            },
        )
        .expect("resolves");
        assert_eq!(real.attention_scale, None);

        // The non-multiplier seeds ride the same variant.
        assert_eq!(MultiplierDefaults::Grok.attn_logit_softcap(), Some(30.0));
        assert_eq!(MultiplierDefaults::Grok.yarn_beta_fast(), Some(8.0));
        assert_eq!(MultiplierDefaults::MiniCpm.attn_logit_softcap(), None);
        assert_eq!(MultiplierDefaults::FromFileOnly.yarn_beta_fast(), None);
    }

    /// Command-R's `logit_scale` is a multiply that the graph skips at
    /// zero and when the key is absent (`command-r.cpp:4,137`), where
    /// Talkie's same multiply is REQUIRED (`talkie.cpp:5`): one variant
    /// each, so the two cannot be confused, and a negative value is
    /// refused on both.
    #[test]
    fn command_rs_logit_scale_is_optional_and_talkies_is_not() {
        let with = |logit: Option<f32>| DeclaredMultipliers {
            logit,
            ..Default::default()
        };
        let cr = |logit| resolve(MultiplierSupport::COMMAND_R, with(logit), dims(6));
        assert_eq!(cr(None).expect("absent is no scale").logit_multiplier, None);
        assert_eq!(
            cr(Some(0.0)).expect("zero is no scale").logit_multiplier,
            None
        );
        assert_eq!(
            cr(Some(0.0625)).expect("resolves").logit_multiplier,
            Some(0.0625)
        );
        assert!(matches!(
            cr(Some(-1.0)),
            Err(MultiplierError::NonPositiveLogitScale(_))
        ));
        assert!(matches!(
            resolve(MultiplierSupport::TALKIE, with(None), dims(6)),
            Err(MultiplierError::MissingRequiredLogitScale)
        ));
        // Neither reads the other three keys, so a Command-R file
        // declaring `residual_scale` is refused as it always was.
        let cr_keys = crate::capability::unsupported_scaling_keys("command-r");
        assert!(cr_keys
            .iter()
            .any(|(k, _, _)| k == "command-r.residual_scale"));
        assert!(!cr_keys.iter().any(|(k, _, _)| k == "command-r.logit_scale"));
    }

    /// The file wins over the Grok defaults, key by key.
    ///
    /// A hook merged the other way round agrees with llama.cpp on every
    /// file that omits the keys and disagrees on every file that carries
    /// them -- which is every fresh export, since `conversion/grok.py`
    /// writes all of them.
    #[test]
    fn a_grok_file_declaring_its_keys_overrides_every_default() {
        let got = resolve(
            MultiplierSupport::GROK,
            DeclaredMultipliers {
                logit: Some(2.5),
                residual: None,
                embedding: Some(3.0),
                attention: Some(0.25),
            },
            dims(6),
        )
        .expect("resolves");
        assert_eq!(got.logit_multiplier, Some(2.5), "multiplied as-is");
        assert_eq!(got.embedding_scale, Some(3.0));
        assert_eq!(got.attention_scale, Some(0.25));
    }

    /// `attention.output_scale` has no "off" sentinel, unlike
    /// `attention.scale`: `grok.cpp` applies whatever it holds.
    ///
    /// A declared zero therefore scales every score by zero here, as it
    /// does in llama.cpp, rather than falling back to the kernels' scale
    /// the way Granite's key does at `granite.cpp:225`.
    #[test]
    fn the_output_scale_key_has_no_sentinel_and_the_scale_key_does() {
        let grok = resolve(
            MultiplierSupport::GROK,
            DeclaredMultipliers {
                attention: Some(0.0),
                ..Default::default()
            },
            dims(6),
        )
        .expect("resolves");
        assert_eq!(grok.attention_scale, Some(0.0));

        let granite = resolve(
            MultiplierSupport::GRANITE,
            DeclaredMultipliers {
                logit: Some(2.0),
                attention: Some(0.0),
                ..Default::default()
            },
            dims(6),
        )
        .expect("resolves");
        assert_eq!(granite.attention_scale, None);

        assert_eq!(
            AttentionScaleKey::OutputScale.suffix(),
            Some("attention.output_scale")
        );
        assert_eq!(AttentionScaleKey::Scale.suffix(), Some("attention.scale"));
        assert_eq!(AttentionScaleKey::NotRead.suffix(), None);
    }

    /// An `AsIs` logit scale is refused when non-positive, like the
    /// reciprocal one: zero blanks the vocabulary and a negative value
    /// reorders it.
    #[test]
    fn a_non_positive_as_is_logit_scale_is_refused() {
        for bad in [0.0f32, -0.5] {
            assert_eq!(
                resolve(
                    MultiplierSupport::GROK,
                    DeclaredMultipliers {
                        logit: Some(bad),
                        ..Default::default()
                    },
                    dims(6),
                ),
                Err(MultiplierError::NonPositiveLogitScale(bad))
            );
        }
    }
}
