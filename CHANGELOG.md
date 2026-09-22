# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Every crate in the workspace shares one version and is published
together, so a version number describes the whole engine, not a
single crate. `frink` is pre-1.0: minor versions may change
behaviour, and a refusal that becomes a supported path counts as a
feature rather than a break.

Entries name what changed and, where it matters, what was wrong
before. A fix that closed a silent-wrong-answer class says so — those
are the ones worth reading twice.

## [Unreleased]

## [0.36.0] - 2026-09-22

### Added

- **`best_of` on both OpenAI routes.** Generates `best_of` completions
  from one shared prefill and returns the `n` best by **summed
  log-probability**.

  ```
  best_of: 5  ->  choices: 1
                  prompt_tokens 4, completion_tokens 20
  ```

  That pair is the whole contract: five were generated and billed,
  one came back, and the prompt was still prefilled once.

  The row waited on a scoring rule, and the rule is upstream's:
  the sum of `ln p` over the generated tokens, under the distribution
  each was actually drawn from. It became available when the sampler
  learned to publish that distribution (0.34.0); before it there was
  nothing to rank by except length.

  **The sum is length-sensitive and favours short answers**, and the
  module says so rather than implying otherwise. A per-token mean
  ranks differently -- one token at `p = 0.5` sums to `-0.69` while ten
  at `p = 0.9` sum to `-1.05`, so the sum picks the single token and a
  mean picks the ten -- and is deliberately not used, because giving a
  different answer from every other engine for the same request is
  worse than a rule with a known bias. Ties keep generation order, so
  a seeded `best_of` returns the same completion every run.

  `best_of` below `n` is a **400** naming both numbers: the field is
  implemented, and asking for the best 3 of 2 is not a request any
  server can serve.

## [0.35.0] - 2026-09-22

### Added

- **`logprobs` on `/v1/chat/completions`** too, in OpenAI's chat shape.
  On a real model:

  ```
  'Blue'  logp=-0.0545  p=0.947  bytes=[66,108,117,101]
          top: 'Blue':-0.05, 'Red':-3.26, 'Purple':-4.80
  ```

  `choices[].logprobs.content[]` of `{token, logprob, bytes,
  top_logprobs[]}` -- a different object from the completions wire's
  four parallel arrays, and rendered by its own function rather than
  translated from the other, because a translation layer would have to
  agree with two upstream shapes at once.

  **Such a request is uncacheable and always misses.**
  `CachedCompletion` stores text and finish reasons, never
  distributions, so replaying an entry for a logprobs request would
  return a completion with no logprobs and a 200. Sabotage confirms
  it: letting the cache serve one produces exactly
  `"logprobs":{"content":[]}` beside `"frink_cache":"hit"`.

  `top_logprobs` without `logprobs: true` is a **400**, not an implied
  `true`: guessing which of two fields the caller meant is how a
  server answers a question nobody asked. Above the cap of 20 is a
  400 on the value rather than a 501 on the field.

## [0.34.0] - 2026-09-22

### Added

- **`logprobs` on `/v1/completions`.** Per-token log-probabilities with
  up to five alternatives per position, over the distribution the
  sampler **actually drew from** -- penalties applied over the
  `penalty_last_n` window, llama.cpp's chain run in
  `sampler_order`, a grammar's mask included.

  ```
  ' Paris'    logp=-0.3489  p=0.705   top: ' Paris':-0.35, ' located':-2.85, ' a':-3.44
  '.'         logp=-0.8410  p=0.431   top: '.':-0.84, ',':-1.06, '.\n':-2.08
  ```

  The reason every logprobs field was refused until now was written in
  the refusal: *"the sampler does not publish the candidate
  distribution"*. `Sampler::sample_reporting` does, and it returns the
  very vector the draw came off rather than a second opinion computed
  beside it -- one pipeline, parameterised, so the report and the draw
  cannot disagree.

  A candidate the chain REMOVED is **omitted**, not reported as `null`
  or as a large negative stand-in: `ln(0)` is not a number JSON can
  carry, and it was not a candidate. The sampled token can never be one
  of these, so `token_logprobs` always holds a real number.

  `text_offset` is computed from the same pieces `tokens` reports
  rather than by re-tokenizing the finished string, because a
  detokenize-then-retokenize round trip is not the identity for every
  vocabulary. The synthetic-weights demo clears the distributions along
  with the text it replaces, at the one site that replaces it.

  `logprobs: N` above 5 is a **400** naming the field, as upstream caps
  it. Still **refused on `/v1/chat/completions`**, which caches one
  answer per key: replaying stored text for a request that asked for
  the distributions would return a completion with no logprobs and a
  200.

## [0.33.0] - 2026-09-22

### Added

- **`n` > 1 on `/v1/chat/completions`** as well, from one prefill.
  Verified on a real model: `n: 3` returns three choices and bills
  `prompt_tokens: 39`, the same as `n: 1`.

  Each choice is parsed for tool calls and reasoning **in its own
  right**: a tool call in choice 2 is a tool call, and reading only
  choice 0 would have returned the others as raw marker text.

  `CachedCompletion` holds every choice rather than one, which is what
  the `n` already in the cache key was promising -- keying `n` and then
  storing the first of three would have been a key stricter than the
  cache actually is. `cacheable()` now requires EVERY choice to be
  complete, not just the first, so a run with one cancelled choice is
  not stored under a key that promises three whole ones.

  **`n` > 1 with `stream` is refused by name.** Round-robin
  interleaving of `choices[].index` needs a sampler that can be stepped
  one token at a time per choice; emitting choice 0 to its end and then
  choice 1 would be sequential delivery wearing an `index` field, and a
  client reading those indices would be misled.

## [0.32.0] - 2026-09-22

### Added

- **`n` > 1 on `/v1/completions`**: several completions of one prompt,
  with the prompt **prefilled once**. Verified on a real model --
  `n: 3` returns three distinct completions and bills
  `prompt_tokens: 6`, not 18.

  That billing is the acceptance test, not the wall clock. A caller who
  wanted `k` independent generations could already send `k` requests;
  the only thing the field buys is the shared prefill, so a version
  that re-prefilled per choice would be the feature in name only with
  nothing in the response saying so. The forks are taken from the
  post-prefill KV state, before choice 0 decodes into it.

  Choice `i` samples from `seed + i`, derived rather than drawn, so a
  seeded request stays reproducible and choice 0 is byte-identical to
  the single answer at the same seed. `completion_tokens` sums over
  choices; the prefix cache still stores choice 0's continuation,
  which is the one a later `n = 1` reproduces.

  **Refused by name where it has no home**, rather than collapsed to
  one: llama.cpp's native `/completion` returns a single `content`,
  and neither the Anthropic nor the Responses wire has a `choices`
  array. `/v1/chat/completions` is the remaining row -- its
  non-streaming path goes through a response cache that stores one
  answer per key, and its streaming path needs interleaved
  `choices[].index`. Also refused by name: the paged KV store, whose
  block lists have no copy-on-write, and the Kimi/MLA engines, whose
  recurrent state has no cache to clone.

## [0.31.0] - 2026-09-22

### Fixed

- **A hybrid model's recurrent layers were charged for a KV cache they
  do not keep**, so every hybrid was priced at up to four times its real
  per-token cost and refused prompts the box could serve.

  Found on a real deployment: Ternary-Bonsai-2-27B is a `qwen35` graph
  with `full_attention_interval 4` over 64 blocks, so **16 layers cache
  and 48 run a gated delta net** whose state is fixed-size rather than
  one K and one V per position. The budget multiplied by `n_layers` and
  priced it at **524288 bytes/token instead of 131072**. A CPU-only host
  refused a 6415-token prompt at a derived ceiling of **4096** that
  should have been four times that. On an M2 Pro the same checkpoint's
  `ctx auto` goes from **30464 to 122368** tokens with no other change.

  The repo's dominant defect shape once more: two structures that must
  agree about how wide a layer's cache is.
  `ModelConfig::new_kv_caches` asks `AttnShape::cache_geometry` per
  layer and builds an EMPTY cache for a recurrent one; `kv_budget`
  multiplied by a scalar. Nothing compared them, and every
  pure-attention model made the two agree, which is why it went
  unnoticed.

  `KvShape` now carries `kv_layers` beside `n_layers` and prices the
  former, read through the same per-layer question the cache
  constructors ask. The plan line says so rather than printing a
  true-sounding total: `16 of 64 layers x [...]`. Reaches every hybrid
  frink serves -- the Qwen3.5 family, `minimax-01`, Granite 4.0, LFM2,
  Jamba, Mamba, PLaMo-2, Nemotron-H, Falcon-H1 -- and is largest for the
  pure recurrent rows, which were charged a full attention cache for
  layers that keep no rows at all.

## [0.30.0] - 2026-09-22

### Fixed

- **Eleven request fields that change the answer were accepted and
  ignored.** `n`, `best_of`, `prompt_logprobs`, `echo`,
  `use_beam_search`, `truncate_prompt_tokens`, `prompt_embeds`,
  `allowed_token_ids`, `bad_words`, `skip_special_tokens: false` and
  `return_tokens_as_token_ids` each returned **200** with an answer
  computed under different rules than the caller asked for. They now
  return **501 naming the field**, on all three generation routes.

  Two of them were worse than absent. `n: 3` was a **501 on
  `/v1/chat/completions` and a 200 on `/v1/completions`**, because the
  chat route hand-wrote its own check and the other two never learned
  it -- the same split `logit_bias` had, and the same one
  `sampling_knobs::ExtraSamplerFields` was built to close for the knobs
  that ARE implemented. `truncate_prompt_tokens` was the most
  dangerous: ignoring it answers a **different prompt** than the caller
  believes they sent, silently.

  The decision is one flattened struct shared by
  `/v1/chat/completions`, `/v1/completions` and llama.cpp's native
  `/completion`, destructured exhaustively with no `..`, so a field
  added to the wire and not answered **fails the build** rather than
  returning a 200. A test drives every field against every route;
  removing the refusal from one route turns it red, confirmed.

  A default a caller may legitimately spell out (`n: 1`, `echo: false`,
  `skip_special_tokens: true`) is **served**, not refused: it describes
  what this server already does.

  `cache_salt` is deliberately NOT in the table and is recorded as a
  row of its own: it selects which cached prefixes a request may reuse,
  so a server that ignores it can serve one caller from another's
  cached prefix. That names an isolation property, not a missing knob.

## [0.29.0] - 2026-09-20

### Added

- **MiniMax-Text-01 runs (`minimax-01`)**, the sixth of the eight rows
  the moved llama.cpp pin brought in, and the ninety-ninth architecture
  with a logit comparison behind it. Its recurrent layers run lightning
  attention: a linear attention whose state is one `head_dim x
  head_dim` KV per head, decayed by `exp(-c s_h)` per token, with the
  per-head slopes a geometric ladder and `c` falling with the layer
  index. Which layers those are comes from the same two keys Qwen3.5
  reads (`attention.recurrent_layers`, else
  `full_attention_interval`, seeded 8 instead of 4); the rest are
  ordinary GQA with partial NEOX RoPE, and every layer carries a
  four-expert softmax MoE.

  **Two facts the tensor shapes do not show**, both from
  `src/models/minimax-01.cpp:303-309`, and both wrong here for a day
  before a libllama golden said so: the fused `attn_qkv` runs through
  SiLU BEFORE it is split, and it is HEAD-major -- head `h` owns the
  contiguous run `[q | k | v]` -- rather than three
  `n_head * head_dim` blocks. Reading it the other way is a
  permutation of the rows, which is the identity at one head, so the
  fixture has four.

  **Its residual is not the layer input.** Each sublayer's own pre-norm
  output, times a REQUIRED `{arch}.residual_scale`, REPLACES the stream
  its branch joins (`:249,428-431,440,455-458`); the layer input is
  bound, sliced by `inp_out_ids`, and never added to anything. That is
  ONE graph of the 155 -- measured by reading the `ggml_scale` argument
  in each of the five files that read the key, where the four Granite
  rows scale a branch OUTPUT under the same key name -- so one column
  answers both meanings and an architecture cannot be given the key
  twice. A scale of exactly `1.0` is still this topology, so the value
  is NOT dropped as an identity the way every other multiplier here
  is, and a fixture declaring `1.0` pins that against libllama.

  Every host body now takes its sublayer pre-norm through one
  function, because a site that normed the stream by hand would read
  the right vector and keep the wrong topology -- a difference no shape
  check can see. A test greps the four bodies for that, with the
  whitespace stripped so a formatter cannot silence it.

  Five fixtures against libllama (the interval, the array spelling, a
  fused QKV on the attention layers, a separate `output.weight`, the
  unit scale), KL 1.3e-11 to 1.3e-9; four sabotages each move the
  logits by more than 3. Every fused Metal launch refuses the model:
  each bakes `x + branch` into its kernel, and here the residual never
  leaves the pre-norm.

  Four triaged refusals are left, and both cheap classes stay empty:
  nothing is a fixture or a match arm away.

## [0.28.0] - 2026-09-20

### Added

- **`frink-server` speculates.** A request whose sampling the
  verification can reproduce now drafts tokens with prompt-lookup (an
  n-gram match over its own history, no second checkpoint, so no
  memory and no load time) and verifies them in one batched forward.
  The `usage` block reports what it bought, filling four fields that
  had carried a wire contract and no producer since they were written.

  **The answer does not change.** Verification draws with the server's
  own sampler at every position and accepts a drafted token only if it
  EQUALS that draw, so the emitted token is always the sampler's and a
  drafter can only ever save a forward pass. That is lossless by
  construction rather than by proof: there is no `p(x)`/`q(x)`
  bookkeeping to get wrong, and no rollback, because state advances
  only over committed tokens and the walk stops at the first
  disagreement.

  Refused rather than silently skipped, each for a reason: a
  grammar-constrained request (verification draws through the same
  grammar machine, and a rejected block would leave it advanced over
  tokens that were never emitted), a paged or recurrent KV store
  (a rejected draft must be rolled back and neither can), and a
  request with no room. The `usage` fields stay ABSENT in those cases
  rather than reporting zeros, because an absent field reads as "this
  request did not speculate" where a zero reads as "the drafter was
  useless".

  Measured on a scripted engine: identical ids and text with 2 forward
  passes against 6 when drafts are right, and 6 against 6 when they
  are all wrong.

### Changed

- **`acceptance_length` counts forward passes, not speculative
  rounds.** It is tokens per forward now, so 1.0 means speculation
  bought nothing and 2.0 means half the forwards. The first
  implementation divided by speculative rounds alone and reported 8.0
  for a run that had taken 19 forwards to write 24 tokens -- a number
  that flatters the drafter. The field had never been populated, so
  nothing can have depended on the old reading.


## [0.27.0] - 2026-09-20

### Added

- **`--ctk q4_0`: 4-bit KV with a Hadamard rotation on K.** The 4-bit KV
  store rotates K before quantizing it (a deterministic per-head,
  per-channel sign flip, then a normalized Walsh-Hadamard transform
  over the head vector) and puts the query through the same matrix at
  read time, so `q . k` is unchanged and the rotation only spreads the
  outlier channels that make 4 bits lossy. V is left alone, measured:
  the rotation is worth 39% of the error on K and 12% on V.

  `frink-quant`'s `kv_rotation` module is the host definition the Metal
  kernel is checked against. The wire is unchanged: an f16 scale per 32
  elements and 16 nibble bytes, 18 bytes a block.

  Measured as next-token agreement with an f16 store across sixty
  long-context windows, 2,000 to 20,172 characters of the corpus, on
  Llama-3.2-3B-Instruct Q4_K_M: **52/60** for the shipped build,
  against 45/60 for the same wire without the rotation. It costs about
  5% of 4-bit decode (37.8 to 35.8 tok/s at a 5,110-token context on an
  M2 Pro, against f16's 51.8).

  **How much of that is noise, measured rather than argued.** An
  earlier build of the same scheme, differing only in the arbitrary
  per-channel sign pattern, scored 48/60. Nothing about the algorithm
  changed between them, so four windows in sixty is the spread of the
  metric itself, not a difference between schemes. That is worth more
  than either number: it says a comparison of two KV schemes at this
  sample size can only be believed when it is wider than about four
  windows. The rotation against no rotation (52 against 45) is; the
  scale-granularity question that looked settled at 48 against 52 is
  NOT, and is recorded as open rather than acted on.

  Two metrics were discarded before that one: a free-running greedy
  generation is chaotic, so one flipped token makes the rest unrelated
  and six prompts scored anywhere from 0 to 134 characters for the same
  build; and `frink perplexity` returns byte-identical numbers for
  every KV dtype, because that path never touches the Metal store.

  Two host paths had to learn the rotation with it, and neither would
  have failed loudly: `MetalKvBuffers::upload_from_host` seeds the
  device store after a CPU prefill, and `tokens_host` fills the host
  cache when the dense stack runs ahead of it. A rotated row handed to
  a host kernel whose query is not rotated is a different model, so the
  round trip has a test of its own, bounded in per-head L2 because an
  orthogonal transform preserves the norm and not the element.

  A head width the butterfly cannot serve (not a power of two, or not
  a multiple of 32) keeps the unrotated wire rather than being
  refused. `MetalKvBuffers::k_rotated` is the one field that answers
  the question, set once at construction.

### Changed

- **`--ctk` / `FRINK_CTK` take llama.cpp's value set, and refuse what
  it refuses.** The accepted spellings are llama.cpp's nine (`f32`,
  `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`,
  `common/arg.cpp:304`) plus frink's own `fp8` / `e4m3`, so a command
  line carries over unchanged. Served are `f16`, `q8_0`, `fp8` (the
  Q8_0 wire) and `q4_0`; the rest resolve to the nearest store that
  exists and the banner SAYS so rather than printing a substitution
  silently. A value outside the set is now refused with the list, where
  it used to become `f16` without a word, so a typo asked for a smaller
  KV cache and got the largest one.

  The previous `turbo8` / `turbo4` / `turbo3` spellings are gone, and
  two of the three carried no information: `turbo8` was a second name
  for `q8_0`'s identical 34-byte store, and `turbo3` had no
  implementation at all and fell back to f16.

- **The banner stopped calling honoured flags ignored.** It compared
  the resolved dtype name to the typed string, so every alias printed
  `(--ctk fp8 ignored: ...)` while doing exactly what was asked.

- **`frink-server --ctk` validates what `frink run --ctk` validates.**
  Both set `FRINK_CTK`, and the server accepted any string and passed
  it through, so `frink-server --ctk nonsense` served f16 without a
  word while `frink run --ctk nonsense` refused. One vocabulary
  (`frink_models::ctk`), one parser, and a test that holds them
  together.

- **The documented architecture count is checked against the table.**
  `README.md` said 47 architectures run with a logit comparison,
  `CLAUDE.md` said 98 and `capability::AUDITED_GENERIC_GQA` holds 98:
  three numbers for one fact, drifting further apart with every row
  that closed. README is corrected, and
  `crates/frink-models/tests/documented_counts.rs` fails when the prose
  and the table disagree again. The same test found
  `docs/manifests/architecture_manifest.md` stale by five architectures
  (`spark2_5`, `maple`, `granite_swa`, `muse-glimmer`, `hrm_text`),
  which is regenerated.

- **Three byte-identical functions collapsed onto one each**, found by
  hashing every function body in the workspace and comparing across
  files:
  - `print_available_devices` existed twice, once per binary, printing
    the user-visible `--list-devices` text. A backend added to one copy
    would have been missing from the other. It is
    `frink_models::devices` now, and the audit that found it also found
    that NEITHER copy lists Vulkan although frink has a Vulkan backend
    (recorded in `docs/ROADMAP.md`).
  - `process_stamp` was duplicated between `frink-api::request_id` and
    `frink-server`'s conversation ids, with a doc comment on the copy
    saying it mirrored the original. A copy that says it is a copy is
    still a copy; the original is `pub` now.
  - The `--ctk` value parser, which this release introduced, was
    written into both front ends before being collapsed into
    `frink_models::ctk::parse_value`.

- **`cargo test` optimises the numeric crates, and the suite got 64x
  faster where it mattered.** `kv_window_real_checkpoint` ran for
  **1,266 seconds** and failed twice under the parallel workspace
  suite while passing alone. The checkpoint was not the problem: the
  test profile compiled the quantized matmul kernels at opt-level 0.
  The same test is **19.9 seconds** with `opt-level = 3` for
  `frink-core`, `frink-quant`, `frink-moe`, `frink-gguf`,
  `frink-models` and `frink-metal` in the `dev` and `test` profiles
  (20.9s in `--release`, so this is the whole gap), and the entire
  `cargo test --workspace` now finishes in **3m52s** green.

  `debug-assertions` stays on: this repo relies on `debug_assert!`, and
  raising the optimiser does not disable it. The test crates themselves
  are left unoptimised so they stay cheap to compile.

  The first fix attempted was `#[ignore]` on the slow test, matching
  its sibling against the same checkpoint. That was treating the
  symptom, and it is reverted: at 20 seconds the test belongs in the
  default suite.

- **The KV wire left `attn.rs` for `frink-metal/src/kv_wire.rs`.** The
  dtype table, the append and dequant kernels and the store geometry
  are one concept and the attention kernels are another; the split came
  first, as the contribution rules ask, and `attn.rs` re-exports the
  names `frink-models` and the tests read. The prefill warmup used to
  restate the dtype-to-kernel mapping that the append table already
  held, which is the two-structures-that-must-agree shape; it is
  derived from that table now.

### Fixed

- **`FRINK_CTK` from the environment works through `frink run`.** It is
  documented in `docs/CONFIG.md` as "same as `--ctk`" and could not be:
  the resolution writes `args.ctk` into the variable unconditionally
  and that field's default is `f16`, so an environment saying `q4`
  was overwritten before `frink_metal::attn::metal_kv_dtype` looked.
  clap reads the variable as the argument's default now, so the flag
  wins when given and the environment when it is not, and the
  write-back is idempotent rather than destructive. Nothing had ever
  checked it; there is a test. (#297)

## [0.26.0] - 2026-09-19

### Changed

- **The project is Frink.** Every crate is `frink-*`, the binaries are
  `frink` and `frink-server`, the environment variables are `FRINK_*`,
  and the repository is `antonellof/frink`. 722 files and about 11,000
  occurrences; the crate directories, the six `frink_real_*.gguf`
  fixtures, the logo assets and the on-hold plan file were renamed on
  disk with `git mv` so history follows them.

  Two things did NOT change, and both are recorded where they live:

  - **The `ferroxtest*` architecture strings.** They are
    `general.architecture` VALUES inside committed binary GGUF
    fixtures, and a GGUF string is length-prefixed, so renaming them
    means regenerating the fixtures and the goldens that go with them.
    A wire value is data, not branding (`capability.rs`).
  - **Nothing on crates.io under the old name.** The `ferrox-*` crates
    stay published at 0.25.0; `frink-*` starts fresh at the same
    version rather than claiming continuity it does not have.

  One thing changed that a rename normally must not: `HASH_DOMAIN` in
  `frink-core/src/kv_block.rs` went from `ferrox-kv-block-v1` to
  `frink-kv-block-v1`, which changes every KV-block content address.
  That is the bump the test's own comment asks for -- blocks written by
  an older build become unreachable rather than read under a name that
  no longer describes them -- and the three pinned digests were
  re-derived in Python against the same length-prefixed encoding, not
  copied from the failing assertion.

- The README logo is the new Frink mark (`docs/assets/frink-logo.webp`,
  with the PNG beside it). The Studio UI keeps its own inline-SVG mark:
  it is monochrome `currentColor` geometry that serves the sidebar, the
  avatar and a 16px favicon in both themes, which a raster wordmark
  cannot do.

### Changed

- **The llama.cpp pin moved from 2026-08-04 to `5b59b83`** -- 792
  commits, fifteen new graphs -- and every census that is a grep over
  `src/models/*.cpp` was re-run against the new tree rather than having
  its denominator edited. Most answers are unchanged and now say 155
  where they said 140. Four changed, and each is corrected where it is
  claimed: the attention gate is FIFTEEN graphs, not six
  (`crate::attn_gate`); `attn_kv_a_mqa` is ten, not six
  (`crate::mla_q_proj`); the SwiGLU clamp arrays are read by six, not
  three (`crate::act_layers`); and the weightless RMSNorm is three
  architectures, not `talkie` alone (`capability::
  NON_PARAMETRIC_RMS_NORM`).

- **The count of triaged refusals went UP, 2 to 10**, because twelve of
  the fifteen new graphs are text generation. Eight are generic-path
  candidates and are triaged with the llama.cpp line that decides each
  (`granite_swa`, `graniteswitch`, `muse-glimmer`, `maple`, `spark2_5`,
  `hrm_text`, `minimax-01`, `qwen4exp`); four need an attention this
  engine does not have and are refused by name (`bailingmoe3`,
  `dots3note`, `hy_v4`, `kimi-k3`); two are text-to-speech and are
  deferred with the audio scope. **Two of the eight are ONE MATCH ARM**
  -- `maple` needs one `rope_layers` row and `spark2_5` one `attn_gate`
  row -- so that class is not empty for the first time since
  2026-09-12.

### Added

- **`spark2_5` runs (Spark-2.5 1.7B)**, the first architecture closed
  against the moved pin and the whole point of triaging the new rows
  the same day they arrive: its verdict said ONE `attn_gate` table row,
  and that is exactly what it cost. `src/models/spark2-5.cpp:41,97-105`
  projects a per-head gate from the normed attention input, sigmoids it
  and multiplies it into the attention output before `wo` --
  `step35`'s corner of the seam's two axes with the tensor REQUIRED
  rather than optional. Everything else in the graph was served and
  each existing table gained one name: the window ARRAY read with no
  scalar attempt (`crate::swa_layers`), per-layer head counts that SIZE
  the gate (`crate::layer_shapes`), and a gated GELU FFN
  (`capability::uses_geglu`, which the family rule would have given
  SwiGLU). KL 7.70e-7 against libllama, and 3.34e-12 with frink's GELU
  made to emulate ggml's f16 table -- so the residual is the
  reference's own approximation, measured rather than assumed
  (`tests/gated_attention_graphs.rs`).

- **`neo-bert` and `eurobert` embed**, together, because they are ONE
  topology: `neo-bert.cpp:59-118` and `eurobert.cpp:55-114` are RMSNorm
  BEFORE each block, a bare residual after it, and one final norm --
  where every other row on this loader is LayerNorm AFTER each add.
  `bert_encoder::BertTopology` is the two shapes and
  `bert_gguf_loader::EncoderSpec` the three columns they differ in: the
  QKV spelling (fused for `neo-bert`, split for `eurobert`), the FFN's
  (a `2 * n_ff`-wide `ffn_up` against a separate `ffn_gate`), the
  rotation (NORM against NEOX) and the tensor the final norm is stored
  under (`enc.output_norm` against `output_norm`). They read the RMS
  epsilon key where the post-norm rows read the LayerNorm one, which is
  the same fact said twice and is why the key is chosen by topology.

- **`jina-bert-v2` embeds (jina-embeddings-v2 base / small)**, the row
  on `bert.cpp`'s graph whose position is neither a table nor a
  rotation: `jina-bert-v2.cpp:5` sets `f_max_alibi_bias = 8.0f` as a
  LITERAL and `bert.cpp:78-80` builds no positions for it at all. The
  bias is SYMMETRIC on an encoder -- `llama-graph.cpp:442` fills a
  non-causal model's mask with `-|p0 - p1|` where the decoder's is
  `p_key - p_query` -- so the encoder computes its own rather than
  calling the decoder's row helper, over the slopes
  `frink_core::alibi` already builds. With it: GEGLU in BOTH
  spellings (a separate `ffn_gate`, or one fused into a
  `2 * n_ff`-wide `ffn_up` whose first half is the gate, decided per
  FILE at `bert.cpp:189`), the whole-projection QK LayerNorm
  (`:109-123`, not per head) and the second attention norm
  (`:156-159`), both optional and both carried by the fixture.

- **`jina-bert-v3` embeds (jina-embeddings-v3)**, one line of
  `bert_gguf_loader::ENCODER_ARCHS` after `nomic-bert`: it reuses
  `llama_model_bert::graph` verbatim (`models.h:314-322`) and is the
  OTHER combination of the two facts that table holds -- RoPE on Q/K
  with `bert`'s ungated GELU FFN. Its refusal had named "RoPE and
  per-projection QK norm", and the second half was wrong:
  `jina-bert-v3.cpp:25-43` creates no `attn_q_norm` at all, so that
  branch of the shared graph is dead for it. A verdict read from the
  graph's branches rather than from the architecture's own tensor
  loader named a blocker it does not have.

- **`nomic-bert` embeds (nomic-embed-text v1 / v1.5)**, on the SAME
  encoder `bert` has used since it landed. It shares
  `src/models/bert.cpp`'s graph and differs in two lines of it: NEOX
  RoPE on Q and K (`:126-133`) where `bert` adds a learned position
  table instead (`:90`, gated on the architecture), and a gated SiLU
  FFN with no biases (`:195-201`) where `bert`'s is an ungated GELU
  with both. `bert_encoder::BertFfn` and `BertHparams::rope_theta`
  are those two facts and `bert_gguf_loader::ENCODER_ARCHS` is the
  table that decides them -- by ARCHITECTURE, because a `bert` file
  that happens to carry a gate tensor would otherwise run gated here
  and ungated in llama.cpp. A rotating file carries no
  `position_embd` (measured: libllama never asks for one) and a
  rotating file that does carry one is refused rather than ignored.
  Checked against llama.cpp's own MEAN-pooled embedding on a
  committed fixture (`tests/nomic_bert_graphs.rs`), so it runs in CI
  rather than behind `--ignored` like the real-checkpoint BERT test.

- **`hrm_text` runs (DFM Mimir 1B)**, the fifth row closed against the
  moved pin and the first decoder here that is not a walk down ONE
  residual stream. `src/models/hrm-text.cpp:183-196` runs `h_cycles`
  cycles of `l_cycles` LOW stacks and one HIGH stack, every stack
  reading `zH + zL` and replacing one of the two; `zH` starts as the
  embeddings, `zL` as the learned `hrm.z_l_init` row, and the lm_head
  reads `zH` with no final norm because every stack ends with its own
  weightless RMS (`norm_sites::NO_OUTPUT_NORM`). The stacks are
  ALIASES: the file holds `2 * layers_per_stack` blocks while
  `block_count` is the expanded slot count (`:22-23` asserts it), and
  each slot keeps its own KV.

  `layer_loops::LayerLoops` is an enum now -- `Repeat` (nanbeige's one
  stream) and `Hrm` -- so the physical mapping, the pass norm and the
  stream schedule are one value the four host bodies ask, and
  `frink_models::hrm` is the two-stream state: one type with two
  methods rather than four copies of "hold two vectors and add them
  here". Two fixtures, because the alternating schedule (LOW HIGH LOW
  HIGH) never reaches the case where the LOW stack runs twice in a
  row: the first matches EXACTLY and the deep one at 1.6e-4, which
  three measured depths show is weightless renormalisations amplifying
  f32 reduction order and not a structural difference: two stacks are
  exact, three measure 3.5e-5 and six 1.6e-4 on arm64, and the
  six-stack file measured 1.1e-3 on x86_64 CI -- which is why the deep
  fixture is the THREE-stack schedule, the shallowest one that runs
  the LOW stack twice in a row (`tests/hrm_text_graphs.rs`).

- **`muse-glimmer` runs**, the fourth row closed against the moved pin
  and two norm facts nothing else upstream has.
  `src/models/muse-glimmer.cpp:69` norms the EMBEDDINGS with a
  weightless RMS (`bloom`'s embedding norm, the only other one of the
  155 graphs, has a weight; every other weightless RMS is a layer
  slot), and `:63` runs the post-attention and post-FFN norms at a
  LITERAL eps of 1e-8 where the pre-norms use the model's key.
  `norm_sites::WEIGHTLESS_EMBEDDING_NORM` and
  `norm::POST_NORM_EPS_LITERAL` are the tables,
  `ModelConfig::post_norm_eps()` the one accessor the three host
  post-norm sites read, and every fused Metal launch and the CUDA
  prefill refuse a model whose two epsilons differ because each bakes
  one epsilon into its kernel. The fixture declares the model epsilon
  four orders larger than the literal, so the two cannot be swapped
  without moving the logits (`tests/muse_glimmer_graphs.rs`).

- **`granite_swa` runs (Granite 4.1)**, the third row closed against
  the moved pin and the first architecture anywhere that lets the FILE
  say which layers rotate. `src/models/granite-swa.cpp:43` reads
  `{arch}.attention.rope_pattern`, one entry per layer, and
  `llama_hparams::has_rope` (`llama-hparams.cpp:333-343`) is what the
  graph asks at `:212`; `frink_models::rope_layers::RopeLayers::
  FileMask` is that mask and `ROPE_PATTERN_READERS` is the census (one
  line of 155, so the key stays dead metadata everywhere else, exactly
  as `llama-model.cpp:1314` leaves it). Everything else it needed was
  served and each table gained one name: Granite's four multipliers,
  the window ARRAY, the REQUIRED per-layer attention sinks, all four
  optional projection biases, and the `attention.scale` override. The
  fixture's window array and rope pattern DISAGREE about which layer is
  special, so a loader that read one into the other is caught
  (`tests/granite_swa_graphs.rs`).

- **`maple` runs (Maple-20B)**, the second row closed against the moved
  pin, and the one that says what a verdict read from a single graph
  file can miss. The row itself was one `crate::rope_layers` entry --
  `src/models/maple.cpp:88` rotates the sliding layers and not the full
  ones, `RopeLayers::SlidingOnly` -- and with the clamp arrays zeroed
  the fixture matched libllama on the first run. With them nonzero it
  was 0.12 off, because `llama-graph.cpp:2228` sends four architectures
  (`maple`, `deepseek4`, `hy_v4`, `dflash` with hyper-connections) to
  `ggml_swiglu_clamp`, whose kernel clamps the gate BEFORE the SiLU
  where every other graph clamps the SiLU's output. `frink_moe::
  ClampForm` is the two forms, carried ON `SwigluClamps` so a limit
  cannot be read without the form that says what it means, and
  `act_layers::CLAMP_BEFORE_SILU` is the list with the line. The two
  agree wherever `silu(x) <= limit`, so a fixture whose clamp never
  binds cannot tell them apart -- this one binds on two layers of four
  (`tests/no_rope_layer_graphs.rs`).

### Fixed

- **`expert_feed_forward_length` is scalar OR an array too**, and the
  array spelling silently sized every expert at `feed_forward_length /
  n_experts_used`. Same key shape and same converter as
  `expert_used_count` below (`conversion/nemotron.py:573` writes a list
  for Nemotron-H Puzzle), found because the `maple` fixture declares
  the array and frink built its experts 24 wide where the file said
  16. A uniform array is honoured; a varying one stops by name.

- **`expert_used_count` is scalar OR an array, and the array spelling
  silently became 2.** `llama-model.cpp:1266` reads the key with
  `get_key_or_arr` in llama.cpp's COMMON loader, for every
  architecture, and `conversion/nemotron.py:574` writes a list for
  Nemotron-H Puzzle -- whose architecture, `nemotron_h`, frink serves.
  frink read a scalar, got `None` for an array, and fell into a
  default of top-2 on every layer whatever the file said. A uniform
  array is now that value; a varying one stops by name, because the MoE
  layer carries one top-k for the model and routing every layer to the
  first entry would answer something else.

- **`kimi_k3` in the architecture catalog was spelled with an
  underscore** while `llama-arch.cpp:155` writes `kimi-k3`, so the
  refusal naming the Kimi loader could not fire on any real file and a
  Kimi-K3 export got the unknown-architecture message instead. frink's
  own preset and Kimi loader had the hyphen all along.

### Documentation

- `docs/plans/parity-audit-2026-09-19.md`: a re-measurement against
  BOTH reference engines. Against the moved llama.cpp pin the
  text-generation gap is the twelve new rows above; the remaining
  scopes are encoder/embedding (11), multimodal (10), diffusion (4) and
  audio (3). On the serving side an architecture count is the wrong
  comparison (HF class names map several-to-one onto a GGUF
  architecture) and the gap is in features, where the finding is that **speculative
  decoding is built, lossless, tested and unreachable from the
  server** -- `frink_models::speculative` has one caller,
  `frink-cli`, while `frink-server` carries an acceptance-rate metric
  with no producer.

- `crate::mtp_blocks` records that upstream now reads
  `nextn_predict_layers` centrally for every architecture (commit
  9d81721), which makes frink's seventeen-reader gate an
  over-refusal, and says what closing it needs: a libllama golden built
  from the moved pin on an architecture outside the seventeen.

## [0.25.0] - 2026-09-19

### Added

- **`plamo2` runs (PLaMo-2 1B / 2B / 8B).** PLaMo-2's own state-space
  block (`frink_models::plamo2_ssm`: Mamba-1's dt / B / C path in
  the order B, C, dt with REQUIRED norms, feeding Mamba-2's per-head
  scan; z and x interleaved per head; no conv bias) where the KV count
  is zero, attention with a per-head QK RMSNorm carrying a distinct
  weight row per head (`QkNormStyle::PerHeadDistinct`) elsewhere, and
  the `plamo2` tokenizer (a port of llama.cpp's suffix-table
  segmenter, byte-identical on the parity corpus). KL 3.0e-12 against
  libllama on the fixture where libllama rotates. Finding: llama.cpp
  seeds `n_rot` from layer 0's head count and so runs every current
  PLaMo-2 export unrotated (`print_info: n_rot = 0`); frink rotates,
  as the model does, and the deviation is pinned as a number in
  `tests/plamo2_graphs.rs`.

- **A decode token that does not come back to the host.** On a hybrid,
  a four-layer group -- one attention layer and three recurrent -- is
  now ONE Metal submission, and nothing inside it returns to the CPU.
  It arrived in five steps, each measured against the one before:

  | | waits/token | tg32 |
  |---|---|---|
  | 0.24.0 | 69 | 10.12 |
  | the attention layer's TAIL fused (`wo` + residual + norm + FFN) | 51 | 10.59 |
  | the decode attention parallel over heads | 51 | 10.59 (and 9.44 to 10.19 at 300 tokens) |
  | the tail riding in the NEXT recurrent run | 36 | 10.6 - 10.8 |
  | the layer's projections riding in the PREVIOUS one | 20 | 11.06 - 11.17 |
  | the attention itself on the device | 20 | **11.17** |

- **`KvCache::metal_attn`**, a sequence's own device KV mirror. It lives
  on the CACHE and not on the model because it is per-sequence state:
  two requests in flight have two histories, and a mirror hung off a
  shared `Decoder` would hand one sequence the other's keys. It is a
  mirror and not the authority -- the host `k`/`v` stay complete, so
  truncation, the prefix cache, slot files and every host reader are
  untouched -- and it is trusted only while its own `seq_len` equals
  the cache's `rows()`, re-uploading from the authority whenever
  anything has moved the cache backwards.

### Changed

- **Decode is FLAT with context, which it was not.** 11.17 / 11.19 /
  11.16 tok/s at 32 / 300 / 600 tokens on Bonsai-2-27B, against the
  PrismML fork's 11.46 / 11.50. Before this release it sloped -- 11.1
  at 32 tokens and 10.95 at 300 -- and the slope was exactly a host
  attention whose work grows with `seq_len`.
- The decode attention runs its heads in parallel. They share nothing,
  and the loop was serial on a six-core machine.
- `frink-metal/src/scratch_pool.rs` reuses shared-storage buffers
  across launches.

### Fixed

- **The TEMPLATE decides the tool-call grammar, not the served name.**
  A checkpoint served under `--alias bonsai-2-27b` is a Qwen3.5 file
  whose template prints `<tool_call><function=…><parameter=…>`, and
  `ToolCallFormat::infer` reads the NAME, finds no "qwen" in it and
  falls through to the Llama 3 fallback, which reads none of those
  calls. Measured against the running server before the fix:
  `finish_reason: stop`, `tool_calls: null`, and the whole call
  delivered as raw markup in `content`. **Tool calling was broken over
  the OpenAI API for that model**, which is what a coding agent
  pointed at it uses. `probe_implied_tool_format` renders the template
  with a tool in hand at load and reads the grammar out of what it
  prints; `PromptTemplate::tool_call_format` prefers that over the
  name.
- **The PTQ1_0 bandwidth probe reported roughly a third of the truth**,
  measuring wall time around a whole command buffer. It subtracts its
  own launch cost now -- which reversed a conclusion that had already
  been committed, and put the matvec back to four rows a threadgroup.

### Documentation

- `docs/plans/gdn-resident-state.md` records the bug that would have
  shipped: the device attention was first encoded on a CONCURRENT
  encoder while its tail expects a serial one, and the model generated
  fluent nonsense while `frink parity` stayed MATCH -- parity reads
  the FIRST token, which is prefill, and that path is decode. Nothing
  in the suite covers a decode step against a reference. A greedy
  100-token generation compared against the same build with the device
  path off is what proves the fix, and the plan says that comparison
  should be a test.

## [0.24.0] - 2026-09-19

### Added

- **A recurrent decoder layer end to end in ONE Metal submission**
  (`frink-metal/src/gdn_branch.rs`, `gdn_head.rs`,
  `frink-models/src/fused_layer.rs`,
  `decoder/fused_recurrent.rs`): `attn_norm`, the four projections, the
  two gates, the causal convolution with its SiLU, the per-head l2
  norms, the delta rule, the gated output norm, the folded rotation,
  `ssm_out`, the residual add, `ffn_norm`, the SwiGLU FFN and the
  second residual add. Nothing returns to the host inside a layer.

  And then a RUN of layers: consecutive recurrent layers hand each
  other a residual stream the host never looks at, and one Metal queue
  is ordered, so they commit back to back against one device buffer and
  wait ONCE (`GdnRun`). Qwen3.5 puts a full-attention layer every
  fourth, so the runs are three layers long.

  Bonsai-2-27B decode, `tg32`, interleaved on one M2 Pro:

  | | tok/s | command buffers waited on, per token |
  |---|---|---|
  | branch on the host | 7.10 | 192 |
  | branch fused | 7.27 | 192 |
  | whole layer fused | 7.95 | 144 |
  | head fused too | 8.93 | 97 |
  | runs of three | **10.12** | **69** |

  Which layers this serves is answered in two halves, because they
  fail for different reasons. `LayerFfnParts::for_layer` answers for
  the WEIGHTS and destructures `MoeWeights` with no `..`, so a field
  added to that struct does not compile until somebody says whether
  this path serves it -- which caught two fields on its first build.
  `Decoder::fused_layer_parts` answers for the MODEL: a residual
  scale, a skip stream, a gpt-oss block, an FFN-free block or a
  non-SwiGLU activation each take the host bodies.

- **The chunked gated delta rule on the device**
  (`frink-metal/src/gdn_chunk.rs`), which a prefill batch takes above
  32 rows. Three earlier attempts to put this recurrence on the GPU
  lost, and they share a cause that is not "the GPU is bad at this":
  every one moved the 3.1 MB state once per ROW, which is 38 GB for a
  128-token prefill whatever it computes. Chunking removes that, and
  only then is there anything for a GPU to be good at.

  | rows | 8 | 32 | 64 | 128 | 512 | 1201 |
  |---|---|---|---|---|---|---|
  | host | 1.24 | 3.43 | 5.48 | 10.43 | 43.38 | 96.10 ms |
  | device | 1.13 | 2.49 | 2.48 | 5.41 | 13.86 | 32.15 ms |

  The first version read 1.05x at 512 rows: `m[t]` and `n[t]` are
  per-thread arrays indexed by a loop bounded by the RUNTIME chunk
  length, so the compiler could not unroll it and spilled both into
  device memory. Padding the tiles to the compile-time `CHUNK` and
  running every hot loop to that constant is 50 GFLOP/s to 160.

- **The blocked prefill attention on the GPU for layers the fused
  attention block cannot take** (`Decoder::prefill_attention_blocked`).
  `launch_gqa_prefill_host_ex` had existed with no caller outside its
  own tests, while the layers that want it -- Bonsai's sixteen, whose Q
  is gated -- fell all the way back to the Rayon kernel.

- **`frink-metal/src/scratch_pool.rs`**, shared-storage buffers reused
  across launches. A launch that allocates its own scratch pays Metal
  for it every time, and the `FRINK_METAL_GPU_TIMING` ledger cannot
  see that because it times `commit` to completion.

- **`QuantKind::metal_kind_name`**, exhaustive with no `_` arm, and
  `frink-models/src/metal_launch.rs`, which asks the BACKEND's own
  table by that name.

### Changed

- **Prefill on Bonsai: 36.5 to 43.1 tok/s** on a 2420-token prompt
  (interleaved: 36.49, 36.60 base; 41.27, 41.66 with the device
  recurrence; 43.04, 43.12 with the device attention). Prefill's host
  side is finished -- a `sample` of it has nothing above the noise, and
  what remains is the GEMM.
- **Four keys per read of a state row** in the host chunked rule.
  Chunking traded 1.5x the multiply-adds for a 32nd of the traffic, so
  the step became compute-bound and a row dotted against one vector at
  a time left the pipeline waiting on the load: 18.1 ms sequential and
  13.0 chunked becomes 10.6.
- **The device delta step reads the state COALESCED.** It gave each
  thread a whole state ROW and walked it, so adjacent threads touched
  addresses `head_dim` floats apart and every 128-byte transaction
  carried 4 useful bytes. That is why the three earlier device attempts
  measured no better than six CPU cores. A head width that is not a
  power of two is refused, since the reduction halves its stride from
  it.
- **`RecurrentState::conv` is page-aligned**, as `ssm` already was, so
  a kernel wraps the host's bytes instead of copying 11.8 MB a token.
- **The PTQ1_0 matvec** requests four rows' bytes before decoding any,
  takes one aligned 16-bit load for the two adjacent bytes a lane owns,
  and is back to four rows a threadgroup -- the reference's own
  geometry, and what its FFN shapes measure fastest at.
- **The prefill GEMM submission is timed** like every other, so a
  batch's GPU time no longer reads as host time in the ledger.

### Fixed

- **`Q5_0` and `PTQ1_0` were unreachable from every fused Metal path in
  `frink-models`.** `frink_metal::gpu::MATVEC_KINDS` served both
  while a hand-written match in `decoder.rs` listed six kinds and
  neither, so the fused recurrent branch silently refused the very
  model it was written for. Two tables that had to agree with nothing
  enforcing it -- this repo's dominant bug shape, for the fifth
  recorded time.
- **The PTQ1_0 bandwidth probe reported roughly a third of the truth.**
  It measured wall time around an entire command buffer, so on small
  shapes most of the number was host cost. It now measures that cost on
  a shape whose kernel is negligible and reports the rest net of it --
  which reversed a conclusion that had already been committed. Even
  net, it is only good for comparing kernel variants: nine experiments
  were run against it and it has never predicted production.

### Documentation

- `docs/plans/gdn-resident-state.md` carries the floor this work ran
  into and every dead end with its number. **A decode token's GPU time
  is 88.8 ms and the reference's WHOLE token is 86.7 to 87.3**, so
  removing every remaining submission and every host microsecond
  converges to 11.26 tok/s against the reference's 11.45 to 11.54.
  Scheduling cannot reach parity from here; the ~3% that is left is in
  the PTQ1_0 matvec's inner loop. Recorded against it: three
  measured-neutral memory hypotheses, two concurrency schemes (a
  scope-Buffers encoder at 10.10 against 10.20, and a resource-scoped
  one that is correct and FLAT), and the reason -- one matvec already
  dispatches 4352 threadgroups and fills the part, so a second finds no
  idle capacity. The per-stage breakdown is there too: per recurrent
  layer the head is 0.286 ms, the branch 0.251 and the FFN 0.762.

### Added

- **The chunked gated delta rule** (`frink_core::gdn_chunk`), which a
  prefill batch takes: the same recurrence with the state read once per
  CHUNK of rows instead of once per row. The sequential step is
  bandwidth-bound (36 GB/s of state, measured by its own probe) and a
  128-token Bonsai prefill moves 38 GB through it, so trading 1.5x the
  multiply-adds for a 32nd of the traffic is **2.1x on the step** and
  ~7% end to end (interleaved A/B on a 1201-token prompt: branch 33.45,
  35.49, base 33.19, 33.00, branch 35.63). Chunk size swept: 16 gives
  1.96x, 32 gives 2.09x, 64 gives 1.66x.

  Correctness rests on the sequential step as an oracle, since that one
  is already verified against libllama: the two are compared row by row
  across five shapes, three chunk-boundary positions and decays from
  "barely forgets" to "forgets at once". Every decay ratio in the
  unrolled form is a PRODUCT, never a quotient, so a chunk of tiny
  decays underflows to zero instead of dividing by it; a test pins the
  `exp(-90)` case that the textbook `d_u / A_u` form cannot take
  without renormalising.

### Changed

- **A folded pair rotates its batch once, not twice.**
  `WeightMatrix::apply_batch_pair_with_acts` is the entry `gate`/`up`
  and a delta-net's `qkv`/`z` take: they read the same batch and
  carried the same fold, so each was rotating the same
  `[rows][n_embd]` block for itself. Bonsai `pp128` 32.2 to 33.2 tok/s,
  parity unmoved. `transform_rows` also stopped allocating a `Vec` per
  ROW inside its parallel region (`transform_into` writes into the
  caller's slice), which on a 128-token prefill was some fifty thousand
  multi-kilobyte allocations.

### Added

- **A page-aligned recurrent state**
  (`frink_core::recurrent_state::AlignedF32`), so Metal can wrap the
  host's own bytes for a kernel instead of copying them. It is what
  turned the second and third measurements below into measurements
  rather than guesses.
- **The gated delta-net recurrence and its gated output norm as Metal
  kernels** (`frink-metal/src/gdn.rs`), pinned against
  `frink_core::gdn::delta_step` on three shapes including Bonsai's;
  swapping the head map in the kernel turns the test red. They are NOT
  wired, and the reason is measured three ways against a 7.3 tok/s host
  baseline: 6.0 with the state copied both ways (3.1 MB a layer, 300 MB
  a token, more traffic than the whole weight read), 6.6 wrapped in
  place with no copy at all, and 6.9 with the wrapper cached so the
  host pages are mapped once. With every copy gone it is still behind,
  so the difference is the kernel, and the answer is the chunked delta
  rule rather than this loop on the GPU. `docs/plans/gdn-resident-state.md` carries the per-token
  ledger the next attempt has to beat (159 submissions, 66 ms GPU,
  34 ms submission overhead, 41 ms host) and four measured
  non-results, so none of them is tried twice.

## [0.23.1] - 2026-09-18

A packaging release: 0.22.0 and 0.23.0 shipped binaries but never
reached crates.io, because `frink-models` packaged 22 MB of golden
GGUF fixtures and the registry refuses an upload over 10 MB. Every
crate downstream of it then had no registry version to build against,
so `cargo install frink-cli` still installed 0.21.0. The fixtures are
`cargo test` inputs and are excluded from the package.

### Fixed

- **The fused dense FFN's command buffer is timed.** It carries a whole
  feed-forward block and was the largest submission the GPU ledger could
  not see, so its GPU time read as host time (the confusion issue #149
  was about). With it, a Bonsai decode token accounts for itself: 159
  submissions, 66 ms of GPU, 34 ms of submission overhead beyond it and
  41 ms of host compute, against a 141 ms token.

### Changed

- Two more measured non-results, recorded where the next attempt will
  look: `commandBufferWithUnretainedReferences` on the matvec path
  (7.00 tok/s against 7.1, for an `unsafe`), and sending a delta-net
  layer's gate projections in the same fused launch as `qkv`/`z`
  (neutral, and it made the launch all-or-nothing across two folds).

## [0.23.0] - 2026-09-18

### Added

- **PrismML's `PTQ1_0` ternary format and the folded Hadamard rotation
  behind Ternary-Bonsai-2-27B.** `frink_quant::ternary` is one trit
  codec for `PTQ1_0` (ggml 143) and `TQ1_0`; `frink-metal/src/
  ternary.rs` is the Metal matvec (the fork's byte-owning, float-pipe
  shape) and the simdgroup GEMM functor; `frink_models::hadamard_fold`
  reads `prism.hadamard.*` and `WeightMatrix::Folded` applies the
  rotation to the activation before every launch and undoes it on the
  embedding row. Verified on the real 27B checkpoint against PrismML's
  llama.cpp fork: first-token KL 2.1e-5 (CPU), 2.3e-5 (Metal decode),
  2.2e-6 (Metal GEMM), tokenizer MATCH with the new `qwen35`
  pre-tokenizer pattern. M2 Pro: pp128 34.0 / tg32 7.0 tok/s (fork
  66.6 / 11.5). `PQ2_0` (142) is recognised and refused.
- **PrismML's Hadamard rotation runs on the GPU** for a decode token
  (`frink-metal/src/hadamard.rs`): `perm -> signs -> FWHT` encoded into
  the SAME command buffer as the matvecs it feeds, so the host neither
  runs the butterfly nor ships a second vector. With it, a folded FFN
  can take the fused `gate -> SwiGLU -> down` launch, which halves a
  Bonsai layer's submissions. Interleaved A/B on an M2 Pro, three reps
  each, raw sequence `base 6.69, branch 7.30, base 6.68, branch 7.25`:
  **tg32 +9%**, `pp128` unchanged. A batch keeps the HOST rotation and
  the module says why: the same prologue cost prefill 6% there, because
  `transform_rows` is already parallel across cores while the kernel
  serialises into the GEMM's command buffer.
- `WeightMatrix::apply_pair`, `apply_many`, and a GPU arm in
  `apply_three`: every projection of ONE activation goes into one
  command buffer through `apply_gpu_multi` instead of a round trip
  each. A gated delta-net layer now sends `qkv`, `z` and its gate
  logits together, which is one submission per layer rather than two
  or three; on Bonsai that was NEUTRAL (tg32 7.2 either way, those
  buffers are small), and it is kept for collapsing the split and
  fused gate spellings onto one call rather than for speed.

### Fixed

- **Four hand-written copies of the Metal kind table, each lagging
  it.** `rows_per_threadgroup` walked its own list of matvec kinds, so
  a kernel absent from the copy dispatched at one row per threadgroup
  and returned zeros for most rows (the PTQ1_0 matvec, on its first
  run); `apply_gpu_multi`'s Metal arm and `apply_gpu_batch`'s GEMM
  dispatch each matched kinds by hand and lacked Q5_0 and PTQ1_0, so a
  Q5_0 gate/up pair ran as two launches and a Q5_0 prefill as N
  matvecs while `gemm_supported` claimed, and the kernel registry
  recorded, a GEMM; and `apply_gpu_dense_ffn_swiglu`'s own six-row
  match kept both kinds off the fused FFN entirely. All four read the
  one table now
  (`MATVEC_KINDS`, `Metal::matvec_kernel`, `launch_mul_mm_sg` off
  `mul_mm_sg_meta`), with tests holding each to it.
- **`frink bench` refused every hybrid model** (`layer 0 consumed 0
  of 128 prompt tokens`): the cache probe read KV rows, which a
  recurrent layer never has; it reads the position counter there.
- `Hadamard` fold memo was thread-local, so gate and up loaded on
  different threads carried different `Arc`s and never fused.
- `delta_step` (the gated delta-net recurrence) runs its heads in
  parallel with vectorisable reductions; `matmul_f32` dots through the
  resolved SIMD kernel attention already used; `HadamardFold::
  transform_rows` is parallel over rows with a vectorised butterfly.
  Together 2.4 to 7.0 tok/s decode and 2.9 to 34 prefill on Bonsai.

- **Chat templates that call `str.startswith` / `str.endswith` render.**
  Qwen3.5's template tests a user turn for `<tool_response>` wrapping
  with both (`chat:72`), and the Python-method shim had neither, so
  every chat request against a Qwen3.5 GGUF failed with `unknown
  method: string has no method named startswith`. Both take a string
  or a tuple of strings, as in Python.

## [0.22.0] - 2026-09-18

The largest span between two tags this repo has had: 28 architecture
closures (92 architectures on the audited generic path, the Mamba-1,
Mamba-2 and gated-delta-net engines, Llama 4), the CUDA GEMM verified
on hardware, CUDA prefill kept on the device and moved to the tensor
cores (pp512 305 to 1932 tok/s on an RTX 3090), the IQ4_XS and Q5_K
CPU paths fixed, and x86 and CUDA re-measured on rented hardware.

Where this leaves the north star, measured: Metal at parity with
llama.cpp on every ledger row; x86 CPU 1.0x to 1.4x behind; CUDA
prefill 4.3x and decode 2.75x behind on the RTX 3090 row, from 25.5x
and 3.4x when that card was first measured on 2026-09-15. `docs/plans/cpu-cuda-parity.md` has the order of
the remaining work.

### Added

- **CUDA `mul_mm` runs on the tensor cores.** `frink_cuda::mul_mm_tc`
  is the same per-kind dequantization as the SIMT body (the kind's own
  `frink_dequant_sub`, through one shared preamble) with the product
  on `mma.sync.m16n8k16`: weight tiles dequantized into shared memory
  as f16, activations converted on the way in, f32 accumulation.
  Selected on `sm_80` and up, the SIMT body below that or under
  `FRINK_CUDA_MUL_MM=simt`. Held against the scalar twin exactly, on
  fixtures whose every value is representable in f16, and on every
  kind within the f16 bound of the products' L1 norm. On an RTX 3090
  pp512 went 975 to 1932 tok/s on Llama-3.2-3B Q4_K_M and 2368 to 4577
  on Llama-3.2-1B, three interleaves, `frink verify` token-identical
  on both and on Qwen3-0.6B; the `q4_k` GEMM fell from 1.88 to 0.70 ms
  a call. The prefill attention kernel reads `float4` slices with four
  query rows per block, 2.5 to 0.96 ms a layer.

### Fixed

- **CUDA resident weights were trusted by host address.** The device
  copy of a weight matrix was cached on `(pointer, len)`, and a buffer
  freed and reallocated at the same address with the same length was
  served the previous tensor's weights; the tensor-core twin test
  caught it multiplying one fixture by another. Latent in production
  (a GGUF's weights are one mmap for the process), live for any
  short-lived matrix. Entries carry a byte sample of what they were
  uploaded from and a mismatching hit re-uploads.
- **CUDA prefill keeps the dense layer on the device.**
  `frink_cuda::prefill` runs a run of dense layers resident: the
  hidden batch goes up once, the norms, QKV biases, QK norms, RoPE, the
  causal GQA, SwiGLU and the residual adds are kernels between GEMMs
  that take device pointers, and what comes back is the hidden batch
  after the last layer plus each layer's K/V rows for the host cache.
  Before, every one of a layer's seven matmuls was a synchronous round
  trip with everything else on the host: 111 MB over PCIe per
  Llama-3.2-3B layer at pp512, 3.1 GB and 196 syncs per step, measured
  with `nsys` as two thirds of the step (#259). On an RTX 3090, pp512
  on Llama-3.2-3B Q4_K_M went 305 to 912 tok/s, `frink verify` is
  token-identical on Llama-3.2-3B, Llama-3.2-1B and Qwen3-0.6B, and
  every kernel has a hardware test against a host twin. Which layers
  the stack may take is decided once for Metal and CUDA in
  `decoder/fused_view.rs` (the exhaustive `AttnWeights` destructure,
  the model-wide fence, the dense-layer rule), so the two backends
  cannot admit different sets.
- **IQ4_XS on the CPU dots against Q8_K activations.**
  `frink_quant::iq4_xs_q8` is llama.cpp's `ggml_vec_dot_iq4_xs_q8_K`
  (a scalar twin, an SDOT arm, an AVX2 arm), and both the single-vector
  and the batched matmul take it for a 256-multiple width, quantizing
  the activations once per matmul as the K-quants do; the batched path
  had been the generic fallback, an f32 dot per (row, activation) that
  re-decoded each row's nibbles `batch` times, measured 4.45x behind
  llama.cpp on a Ryzen 9 3900X. On the M2 Pro, interleaved twice,
  Llama-3.2-1B IQ4_XS prefill 53 to 170 tok/s and decode 40 to 85;
  `frink parity` MATCH at KL 3.8e-5 against libllama.
- **x86 CPU and CUDA re-measured on rented hardware; the Q5_K batch
  gate asks the kernels.** `benchmarks/RESULTS.md` has a Ryzen 9 3900X
  section (CPU and RTX 3090 CUDA, 26 receipts), the first x86 rows
  since #159's AVX2 GEMMs: K-quant prefill 1.04x to 1.36x, decode
  1.04x to 1.17x. The first run found Q5_K_M prefill at 8.56x: the
  Q5_K batched matmul gated its Kx8 path on `cfg!(target_arch =
  "aarch64")` where the Q4_K and Q6_K arms asked the kernels, so x86
  ran the per-row GEMM (`weight_matrix::q5k_batch_takes_kx8` now;
  0.91x on a 5950X, 1.04x on the 3900X; Phi-4-mini 3.24x to 1.18x).
  On CUDA the K-quant GEMM ran on a GPU for the first time: all 13
  hardware tests pass and `frink verify --backend cuda` is
  token-identical on five quant kinds; `launch_mul_mm_matches_the_
  scalar_twin`'s tolerance gained the absolute floor FMA drift over
  256 columns needs (measured 4.8e-4 worst). Utilization sampled
  during the runs: prefill 30% to 39% at 175 W, decode 45% to 75%,
  which reframes the 25x to 43x CUDA prefill gap as launch- or
  host-bound before it is arithmetic (`docs/plans/cpu-cuda-parity.md`,
  `benchmarks/HISTORY.md`). Still open on x86: IQ4_XS prefill 4.45x
  (no batch kernel) and the small-model per-op constant (SmolLM2 2.08x).
- **`cohere2moe` runs: Cohere2 MoE 30B-A3B.** `rope_layers::
  RopeLayers::SlidingOrLeadingDense`, `parallel_dense_ffn::
  SHARED_EXPERT_SUM_SCALE`, `norm::NORM_BY_RMS_EPS_KEY` (`ModelConfig::
  norm_function`), `swa_layers::ARRAY_AT_TRUNK_LENGTH`. KL 1.7e-14
  (`tests/cohere2moe_graphs.rs`, `make_cohere2moe_fixture.py`). 92
  audited; no parallel-residual row refuses.
- **`llama4` runs: Llama 4 Scout 17B-16E and Maverick 17B-128E.**
  `chunked_swa` (the 8192-position chunk, per-query windows,
  `BatchWindow`), `attn_temperature::LITERAL_ATTN_TEMPERATURE` with
  `unrotated_layers_only`, `weightless_qk_norm`, `routed_weight_site`
  (the sigmoid weight on the expert's input, `llama-graph.cpp:1947`),
  `moe_interleave::INTERLEAVE_STEP_HONOURED_BY_LOADER`. KL 1.1e-12
  (`tests/llama4_graphs.rs`, `make_llama4_fixture.py`), the chunk
  boundary measured at position 8199. `route_top_k_sigmoid` honours
  `norm_topk_prob` (it renormalised unconditionally). 91 audited.
- **`qwen3next` runs: Qwen3-Next-80B-A3B.** `gdn::GROUPED_HEAD_
  ARCHITECTURES` / `GdnHparams::map` (`HeadMap::Grouped`) and
  `gdn::BetaAlpha::Fused` (the `ssm_ba` projection); catalog and
  `norm_sites` rows. KL 8.7e-12 (`tests/qwen35_graphs.rs`,
  `make_qwen35_fixture.py --next`). 90 audited.
- **`qwen35moe` runs: Qwen3.5-35B-A3B, 122B-A10B, 397B-A17B.** No
  code: `qwen2moe`'s FFN under Qwen3.5's layers; catalog, `norm_sites`
  and `mrope` rows. KL 2.9e-11 (`tests/qwen35_graphs.rs`,
  `make_qwen35_fixture.py --moe`). 89 audited.
- **`qwen35` runs: Qwen3.5 dense 0.8B to 27B.** `frink_core::gdn` is
  the autoregressive delta rule (`delta_step`, `l2_normalize`,
  `HeadMap::{Tiled, Grouped}`); `frink_models::gdn` the block
  (`Gdn`, `GdnHparams`, `recurrent_layers` from
  `attention.recurrent_layers` / `full_attention_interval`);
  `AttnShape::Gdn`, `SsmBlock::Gdn`; `LayerShapes::resolve` takes the
  recurrent mask; `AttnWeights::q_gate_interleaved` with
  `attn_gate::split_interleaved_q_gate` / `apply_interleaved_gate` in
  the three host bodies; `qwen35` rows in `norm_sites::
  PRE_FFN_NORM_IS_POST_ATTENTION_NORM` and `mrope::MROPE_READERS`.
  KL 4.1e-13 (`tests/qwen35_graphs.rs`, `scripts/make_qwen35_fixture.py`).
  88 audited.
- **Removed:** `frink-models/src/gdn.rs` (the FreeToken GDN port) and
  `hybrid_gguf_loader.rs`, 1.8k lines that never met libllama;
  `hybrid_engine.rs` is the refusal for the hybrid rows still off the
  generic path (`plamo2`, `qwen3next`, `qwen35moe`) and nothing else.
- **`jamba`, `mamba` and `mamba2` run: Jamba, Mamba / FalconMamba,
  Mamba-Codestral.** `frink_models::mamba1` is `build_mamba_layer`
  (`frink_core::mamba2::Decay::PerState` is the scan's per-state
  decay arm; `DtBcNorm` the three norm spellings); `ssm_block::
  SsmBlock` holds either generation on `AttnWeights::ssm`;
  `layer_shapes::PURE_RECURRENT` makes a model with no heads every
  layer the block (head_dim 0); `moe_interleave::
  DENSE_LAYER_BY_ROUTER_ABSENCE` makes Jamba's FFN dense or MoE per
  layer by the file; `rope_layers` answers `Never` for all three.
  KL 7.3e-12 / 2.3e-12 / 1.8e-12 / 3.6e-13 (`tests/mamba_graphs.rs`,
  `scripts/make_mamba_fixture.py`). 87 audited.
- **`falcon-h1` runs: Falcon-H1 0.5B to 34B.** The Mamba-2 block in
  parallel with attention on every layer (`falcon-h1.cpp:137-161`):
  `mamba2::PARALLEL_WITH_ATTENTION`, `ModelConfig::parallel_ssm`,
  `AttnWeights::mamba2` on a GQA layer, `KvStep::recurrent_slot`,
  `Decoder::mamba2_state_step` / `parallel_ssm_rows` /
  `add_parallel_ssm` for the three host bodies; the fused Metal
  launches refuse the model; `attn_output.bias` recorded as unread
  (`:76,154`). KL 1.3e-13 / 6.2e-13 / 3.2e-13
  (`tests/falcon_h1_graphs.rs`, `scripts/make_falcon_h1_fixture.py`).
  84 audited.
- **`nemotron_h_moe` runs: Nemotron-3 Nano 30B-A3B.** The routed and
  shared experts of an ungated architecture alias their gate to `up`
  as the dense loader does (`GluAct::ReluSqr` never reads it); a layer
  with `ffn_dim 0` takes the dense arm (`absent_ffn`) whatever the
  model's MoE says; `nemotron_h_moe` joins `GATING_LITERAL_
  ARCHITECTURES` (sigmoid), `EXPERT_WEIGHTS_SCALE_READERS` and
  `EXPERT_WEIGHTS_NORM_READERS`; `moe_latent_size` is refused by name
  (`capability::unsupported_feature_keys`). KL 3.6e-13
  (`tests/nemotron_h_graphs.rs`, `make_nemotron_h_fixture.py --moe`).
  83 audited.
- **`nemotron_h` runs: Nemotron-H 8B / 47B / 56B, Nemotron-3 Nano
  dense.** One block per layer on the Mamba-2 seam:
  `layer_shapes::ZeroKvLayer::Mamba2UnlessFfn` reads both per-layer
  arrays, `BLOCK_WITHOUT_FFN_KEEPS_ITS_OUTPUT` admits a block with no
  FFN (deci's is discarded and stays refused), `norm_sites::
  ONE_NORM_PER_LAYER` norms the FFN-only layer with `attn_norm`,
  `rope_layers` answers `Never` (no `ggml_rope_ext` in the graph),
  optional `attn_output.bias` / FFN biases, ungated ReLU-squared FFN.
  KL 2.0e-13 / 1.4e-12 / 7.5e-14 (`tests/nemotron_h_graphs.rs`,
  `scripts/make_nemotron_h_fixture.py`). 82 audited.
- **`granitehybrid` runs: Granite 4.0 (H-Micro, H-Tiny, H-Small), the
  first Mamba-2 row, on the generic path.** `frink_core::mamba2` is
  ggml's `ssm_conv` and `ssm_scan` steps; `frink_models::mamba2` is
  `build_mamba2_layer` once (the `ssm.*` hparams, the eight tensors,
  the block); `layer_shapes::AttnShape::Mamba2` / `ZeroKvLayer::Mamba2`
  put it where attention stands; `Decoder::recurrent_block` dispatches
  the conv and the Mamba-2 block on the three cache backings.
  `frink_core::recurrent_state::RecurrentState` rides on `KvCache` /
  `PagedKvCache` (`recurrent`), cloned and cleared with the cache;
  `KvCache::truncate` refuses a middle position on it
  (`can_truncate_to`), the prefix cache refuses to store such a cache,
  `--model-draft` refuses such a model. `ssm_conv1d.bias` is required
  (libllama segfaults without it, measured). Granite's
  `rope.scaling.finetuned = false` is SERVED as `RopeLayers::Never`
  (`rope_finetuned::unrotated`), on `granite` / `granitemoe` too; the
  fixture that evidenced the refusal has its golden. KL 1.9e-13 /
  7.9e-13 / 1.0e-13 (`tests/granite_hybrid_graphs.rs`,
  `scripts/make_granite_hybrid_fixture.py`). 81 audited.
- **`pangu-embedded` runs: openPangu-Embedded-1B / 7B.** A decoder LLM
  (`PanguEmbeddedForCausalLM`) that had been filed as "embedding
  variant; deferred" in the catalog and in `embedding_model::NOT_YET`
  from its name. `pangu-embed.cpp` is `llama.cpp`'s graph with a
  REQUIRED `attn_output.bias` (`:37`): one `proj_bias::
  ATTN_OUT_BIAS_CREATORS` row, NEOX RoPE. KL 1.5e-13 on three fixtures
  (`tests/pangu_embedded_graphs.rs`, `scripts/make_pangu_fixture.py`).
  79 audited.
- **`lfm2` and `lfm2moe` run: LFM2-350M / 700M / 1.2B / 2.6B and
  LFM2-8B-A1B, the first hybrid rows, on the generic path.** `lfm2.cpp:192-208` is the generic layer with a
  short convolution where attention would be on the layers whose
  `head_count_kv` is 0, so it is a third `layer_shapes::AttnShape`
  (`ShortConv`) served by `frink_models::shortconv` and
  `Decoder::shortconv_block`, with the conv state kept as the layer's
  KV history (one `n_embd` row per token, no V; `AttnShape::
  cache_geometry`) on the contiguous, paged and multi-seq backings.
  `layer_shapes::ZeroKvLayer` says what a zero-KV layer IS per
  architecture (deci's `wo`-only block, LFM2's conv, or a Mamba-2 /
  KDA block refused by name), where every architecture had been read
  as deci. `norm_sites::OUTPUT_NORM_UNDER_EMBEDDING_NAME`: LFM2's
  output norm is stored as `token_embd_norm` (llama-arch.cpp:384).
  `MultiSeqKv::step` is the one place the batched path picks a
  sequence's cache. KL 3.2e-12 on three fixtures
  (`tests/lfm2_graphs.rs`, `scripts/make_lfm2_fixture.py`); a window
  is refused by name (`lfm2.cpp:24-29` windows the attention layers
  alone). `lfm2moe` is the same graph (`models.h:1899`) with leading
  dense layers and a sigmoid MoE with `exp_probs_b` required, KL
  6.0e-13. 78 audited.
- **`minimax-m2` runs: MiniMax-M2.** No code: the row's refusal had
  said "unaudited, not unimplemented, a fixture away" while
  `tests/fixtures/minimax_m2_tiny.gguf` sat in the tree; its libllama
  golden matches at KL 3.44e-15 (`tests/minimax_m2_graphs.rs`). Plain
  GQA, whole-vector Q/K norm, partial NEOX RoPE, a sigmoid MoE with
  `exp_probs_b` on every layer. `minimax_engine.rs` refuses `minimax-m3`
  alone now. 76 audited.
- **ALiBi; `refact`, `bloom`, `mpt`, `jais` and Baichuan-13B run.**
  `frink_core::alibi::slopes` is llama.cpp's per-head slope formula
  (`ggml-cpu/ops.cpp:5489-5508`), and the row, paged and batched
  prefill kernels take the slopes as one additive `slope_h * (p_key -
  p_query)` on every score after the scale and the softcap
  (`online_attn_accumulate`'s visitor carries the bias); a kernel test
  pins the three against a naive reference. `frink_models::alibi` is
  the table of the five generic-path graphs that set
  `f_max_alibi_bias` (measured over all 140: the literal 8 for
  `bloom` / `refact`, the literal at 40 layers only for `baichuan`,
  `attention.max_alibi_bias` for `mpt` / `jais`), and
  `rope_layers::RopeLayers::Never` is derived from it, so the bias and
  the absence of rotation cannot disagree about a layer count.
  `ModelConfig::alibi_max_bias` fences every fused Metal launch and the
  CUDA resident attention; `Decoder::alibi_slopes` is derived from it
  in one place. Also: `norm_sites::EMBEDDING_NORM_ARCHITECTURES` and
  `Decoder::embedding_norm` for `bloom`'s `token_embd_norm`; `jais`'s
  `kq_scale = 1/d` (`jais.cpp:83`) in `attention_scale_override`;
  `mpt` in `CLAMPED_QKV_ARCHITECTURES` and its optional `position_embd`
  served; `mpt`'s whole-vector LayerNorm QK norm refused by name. The
  Baichuan-13B layer-count refusal is gone. `tests/alibi_graphs.rs`:
  KL 6.4e-13 (refact), 8.3e-8 (bloom), 3.6e-7 (mpt, `clamp_kqv 4`),
  1.6e-7 (mpt with a position table), 7.4e-13 (jais), 1.1e-12
  (Baichuan-13B); dropping the slopes, rotating, or the wrong slope
  table each diverge. 75 audited.
- **`gpt2` and `starcoder` run: the learned position table.**
  `frink_models::position_embd` adds `position_embd.weight`'s row
  `pos` to the token embedding at the one embedding site
  (`Decoder::embed_token` takes the position now; `gpt2.cpp:19,74-77`,
  `starcoder.cpp:19,75-78`), and `rope_layers::RopeLayers::Never` is
  the rule that rotates nothing, the `LLAMA_ROPE_TYPE_NONE` group as a
  value rather than a refusal. Three graphs of 140 create the tensor on
  the generic path (`gpt2`, `starcoder` REQUIRED; `mpt` optional,
  still refused for its ALiBi). The two graphs are one: the biased
  LayerNorm, a fused `attn_qkv` with bias, REQUIRED `attn_output` / FFN
  biases with the ungated GELU, `output` tied when absent; StarCoder
  is multi-query. Every fused Metal launch is fenced off a
  learned-position model (the GPU gather has no add), and a position
  past the table is refused rather than clamped.
  `tests/position_embd_graphs.rs`: KL 1.85e-7 (gpt2) and 1.89e-7
  (starcoder) at the f16 GELU-table line; dropping the table or
  rotating the layers diverges by more than 1. The bias group of
  `tests/attn_bias.rs` is empty. 71 audited.
- **`phimoe` runs: Phi-3.5-MoE.** `phi3`'s graph (`models.h:632`) on
  `phimoe.cpp`'s tensors, which differ from a Phi-3 file in biases
  only: an RMSNorm WITH a bias at every norm site (`phi3.cpp:99-102,
  137-139,174-177` under `LLM_NORM_RMS`; `NormOp::RmsBias`,
  `NormFunction::RmsBias`, `capability::BIASED_RMS_NORM`, one graph of
  140 on the generic path, measured) plus `attn_output.bias` and
  `output.bias`, slots that already existed. The old refusal had
  called the norm biases LayerNorm biases. `capability::
  swa_window_override` drops its window key as `phi3`'s
  (`phimoe.cpp:3-10` never read it; libllama `n_swa = 0` on a file
  declaring one, measured) where a test had asserted the opposite. The
  loader's norm-function census covers all five function lists now
  (it listed two). `tests/phimoe_graphs.rs`: KL 1.91e-11 with the
  LongRoPE long pair in use (`attn_factor 1.0955`), 1.86e-12 plain, at
  the `orion` tolerance; zeroing the norm bias, reading it as a
  LayerNorm bias, or dropping the output bias each diverge. 69 audited.
- **`cohere2` runs: Command-R7B and Command-A.** `command-r`'s graph
  with a REQUIRED sliding window whose SLIDING layers alone are rotated
  (`cohere2.cpp:72,91`, `if (is_swa)` around `ggml_rope_ext`): that is
  `rope_layers::RopeLayers::SlidingOnly`, the `exaone-moe` rule, which
  the module's census had missed by grepping for `use_rope`; the
  census is eight graphs now (`cohere2moe`'s `|| il <
  n_layer_dense_lead` variant recorded, no arm yet). Also: the third
  `WEIGHTED_LAYER_NORM` row, `MultiplierSupport::COHERE2` (`logit_scale`
  REQUIRED, multiplied), and `swa_geometry::window_required`: a
  `cohere2` or `exaone-moe` file without `attention.sliding_window` is
  refused by name, as libllama refuses it (`key not found`, measured),
  where it would have run with no layer rotated.
  `tests/cohere2_graphs.rs`: KL 1.03e-14 (seeded period 4, window 3
  inside the prompt), 8.95e-14 with `sliding_window_pattern = 2`
  (libllama's goldens differ by 0.36, so the scalar key is live);
  rotating the full layer diverges. 68 audited.
- **`phi2` runs: Phi-2 and Phi-1.5; the LM head has a bias slot.**
  `output.bias` (`phi2.cpp:22,136`, REQUIRED) is `Decoder::output_bias`,
  read by `proj_bias::load_output_bias` for the three graphs of 140
  that create it (`phi2`, `phimoe` required; `qwen2` optional, whose
  files with the tensor used to be refused as unread) and added in
  `decoder::lm_head::Logits::from_output_head`, per row, before the
  multiplier and the cap. `FoldedLmHead::permit` refuses a head with a
  bias: no fused Metal stack adds it, and a bias moves the argmax where
  the cap and the multiplier cannot. The rest of the graph was already
  served: the shared-norm parallel residual, the biased LayerNorm, Q/K/V
  biases split or fused, the required `attn_output` / FFN biases with
  the ungated GELU, a partial NEOX rotary. `tests/phi2_graphs.rs`: KL
  2.95e-7 (max delta 2.5e-3, the f16 GELU-table class), the fused and
  split spellings on one golden (libllama byte-identical); dropping
  the bias moves the logits by more than 1. 67 audited.
- **`falcon` runs: Falcon-7B, 40B and 180B, both shapes.** Every
  Falcon layer is the parallel residual
  (`frink_models::parallel_residual`); the OPTIONAL `attn_norm_2`
  (`falcon.cpp:35-36`) picks the arm, not whether. 7B: the shared norm
  over the biased LayerNorm, a fused multi-query `attn_qkv` with no
  bias, the ungated GELU (`capability::uses_gelu_ungated`), NEOX over
  the whole head. 40B / 180B: `:79-85` norm the layer input with
  `attn_norm_2` FOR ATTENTION and `:124` keeps the FFN on
  `attn_norm(x)`, the two-norm arm with the names crossed relative to
  `gptneox`; `norm_sites::ATTN_NORM_2_FEEDS_ATTENTION` and
  `NormSites::for_layer` cross the two pre-norm slots on the layers
  that carry the tensor (one graph of 140 on the generic path,
  measured). `ParallelResidual::second_norm` is the table column for
  it; the first row had said "parallel when `attn_norm_2` is present",
  and the 7B fixture refused to load before any code ran.
  `tests/falcon_graphs.rs`: KL 3.80e-8 (7B) and 1.94e-7 (40B) at the
  f16 GELU-table line; swapping the slots back diverges by more than
  1. 66 audited.
- **`command-r` runs: Command-R 35B and Aya-23.** The shared-norm
  parallel residual (`frink_models::parallel_residual`) over the
  weighted LayerNorm WITHOUT a bias (`command-r.cpp:68,127`,
  `capability::WEIGHTED_LAYER_NORM`'s second caller after `dbrx`), a
  `logit_scale` MULTIPLY read `required = false` and skipped at zero
  (`:4,137-138`; `LogitScaleUse::AsIsOptional`, a new variant beside
  the REQUIRED `AsIs` Grok and Talkie use, with a negative value
  refused), a tied lm_head, NORM RoPE. `tests/command_r_graphs.rs`: KL
  1.02e-15 with the key, 2.26e-13 without it (libllama's two goldens
  are in the ratio 0.0625, the key's value). Command-R+ (64 layers)
  carries the per-head LayerNorm QK norm `:28-31` REQUIRE at that depth
  and is refused by name (`frink_models::qk_layer_norm`) from a
  64-layer fixture libllama runs. 65 audited.
- **The parallel residual; `gptneox` (Pythia, GPT-NeoX-20B) and
  `plamo` (PLaMo-13B) run.** `x + attn(norm(x)) + ffn(norm(x))`,
  refused by name the PR before, is served by
  `frink_models::parallel_residual`: the FFN input on a parallel layer
  is a norm of the LAYER INPUT (`gptneox.cpp:149` its own `ffn_norm`;
  `plamo.cpp:97` and `stablelm.cpp:137` the vector attention read), so
  it is captured before attention beside the router's operand as
  `decoder::ffn_block::BranchInputs`, from one constructor
  (`Decoder::branch_inputs`) every host body calls at the top of every
  layer; `MoeWeights::parallel` is the per-layer fact, a shared-norm
  layer's pre-FFN slot is `NormOp::None`, and
  `ModelConfig::parallel_residual` fences every fused Metal launch.
  The table has all eight graphs that build the shape (a two-adds scan
  over the 140; `plamo` had been missed by a grep for `attn_out`), with
  the rule that decides each (`use_parallel_residual`, `ffn_norm`
  absent, `attn_norm_2` present, always). `gptneox` also needed the
  biased LayerNorm, the fused `attn_qkv` bias and the REQUIRED
  projection biases with the ungated GELU, all seams from the two days
  before. `tests/parallel_residual_graphs.rs`: `gptneox` under the key
  `true` and `false` (libllama's logits differ by 3.73; both matched
  at the f16 GELU-table line, 1.9e-3 / 2.1e-3, KL 5e-7 / 3e-7), `plamo`
  KL 1.64e-13; the `stablelm` parallel fixture matches, KL 1.95e-12
  (`tests/stablelm_graphs.rs`). Refusals for `command-r`, `cohere2`,
  `cohere2moe`, `falcon` and `phi2` now name what each needs on top of
  the residual. 64 audited.
- **`stablelm` runs: StableLM-2-1.6B and StableLM-3B-4E1T on the
  biased LayerNorm, with the graph's two other shapes refused by name.**
  `stablelm.cpp` decides three shapes by TENSOR PRESENCE and none by a
  key. A layer with `ffn_norm` is sequential and matches libllama, KL
  3.39e-13 (`tests/stablelm_graphs.rs`; the pre-FFN norm pair is
  `TENSOR_NOT_REQUIRED` upstream and `NormFunction::resolve` requires
  it as a pair). A layer without it is the PARALLEL residual
  (`x + attn(norm(x)) + ffn(norm(x))`, `:135-137`):
  `frink_models::parallel_residual` refuses it from a fixture whose
  libllama logits move by 8.85, and its table records the eight graphs
  that build the shape in two spellings (one shared norm: `stablelm`,
  `phi2`, `falcon`-7B, `command-r`, `cohere2`, `cohere2moe`, `plamo`;
  two norms under `gptneox`'s key or `falcon`'s `attn_norm_2`). A layer with
  `attn_q_norm` (`{n_embd_head_k, n_head}`, `LLM_NORM` per head, a
  distinct weight per head) is refused by `frink_models::qk_layer_norm`
  from a fixture whose logits move by 8.73, because the loader's length
  rule and the fused Metal attention would both have read that weight
  as one RMS over the whole projection; `command-r` and `chameleon`
  build the same op. `use_parallel_residual`, written by every export
  and read by nothing in the graph, is pinned ignored (libllama
  byte-identical). `frink_models::test_source::StubSource` is the one
  names-and-keys `TensorSource` for unit tests, replacing the first of
  five copies. 62 audited.
- **Projection biases on the dense path; `starcoder2`, `codeshell`
  and `jais2` run, and a `llama` file with biases loads.**
  `frink_models::proj_bias` reads `attn_output.bias` into
  `AttnWeights::o_bias` (added after `wo` and `o_scale`, `build_attn`'s
  order; gpt-oss's bias moved here from its side table) and the dense
  FFN's `ffn_{up,gate,down}.bias` into `MoeWeights::dense_bias`
  (`frink_moe::DenseBias`; `run_expert_biased` adds `up_b` / `gate_b`
  before the activation and `down_b` after `down`, on the same
  gate/up projections the unbiased body uses), for exactly the
  architectures whose graph creates the tensors: two tables measured
  over all 140 graphs (33 create `wo_b`, 27 the FFN biases, most
  OPTIONAL) with a `Required` / `Optional` column, so a bias on an
  architecture whose graph never creates it stays refused as unread.
  `FfnActivation::GeluUngated` (`LLM_FFN_GELU` under `LLM_FFN_SEQ`,
  eleven graphs) is aliased like `ReluSqr`. Every fused Metal dense
  launch and the attention view fence on the two fields. Goldens
  (`tests/proj_bias_graphs.rs`): `jais2` KL 6.6e-13; a `llama` with
  all four biases 2.7e-13 (the gate bias's only exercise); `starcoder2`
  4.7e-7 and `codeshell` 4.7e-6 at a documented 1e-2 line, because
  their 3e-3 / 8e-3 max deltas are entirely llama.cpp's f16 GELU table
  (emulated: 2e-13 / 1e-12); Nemotron's optional biases, refused the
  PR before, 2.1e-13. 61 audited.
- **`orion` and `nemotron` run: the LayerNorm with a bias.**
  `NormOp::LayerNormBias` is `build_norm(x, w, b, LLM_NORM, il)` --
  multiply, then add -- the variant `capability::WEIGHTED_LAYER_NORM`
  had named as having no caller. Read row by row, two of the eight
  "LayerNorm-with-bias group" rows need nothing else: Orion-14B
  (`orion.cpp:63-66,104-107,127-130`; a Llama with NEOX RoPE at the
  defaults) and Nemotron-4 / Minitron (`nemotron.cpp:71-74,111-114,
  136-139`; the ungated ReLU-squared FFN, partial NEOX RoPE).
  `NormFunction::resolve` asks the file for each PART the function has
  (`NormParam::{Weight, Bias}`), so the biased form cannot be built with
  the bias forgotten and a function with no bias never asks for one.
  `tests/biased_layer_norm_graphs.rs`: KL 2.33e-11 (orion; its 1.6e-5
  max delta is f32 noise on a SwiGLU fed a non-zero-mean norm, measured
  at unit weight scale too) and 5.13e-13 (nemotron); zeroing a bias,
  dropping it (dbrx's form) and RMS-norming the final norm each diverge;
  Nemotron's optional projection biases are refused as unread from a
  fixture whose libllama logits differ by 8.07. `tests/attn_bias.rs`
  now asks `BIASED_LAYER_NORM` which rows apply the three norm biases,
  and the six remaining rows keep their refusals with the bias named.
  58 audited.
- **`glm4` runs: GLM-4-0414 (9B, 32B), GLM-Z1 and GLM-OCR on the
  generic path, audited against libllama.** The row had been sent to
  the GLM-5.2 MLA loader for `q_lora_rank` and three more keys
  `src/models/glm4.cpp:3-9` never read -- the `glm4moe` defect on the
  family's dense members -- so a real GLM-4-9B-0414 failed on a key it
  is not supposed to have. Plain GQA with Q/K/V biases, NORM RoPE over
  half the head, Gemma-2's `post_attention_norm` / `post_ffw_norm` in
  Gemma-2's slots and a fused SwiGLU `ffn_up` (the Phi-3 split): no code
  changed for the row, its profile moved to `gqa_norm` and the fixture
  matched at KL 9.67e-15 (`tests/glm4_graphs.rs`). `frink_models::mrope`
  is new: a vision export's text tower declares `rope.dimension_sections`,
  under which llama.cpp rotates M-RoPE; on text positions that is NEOX
  band for band, so `glm4moe` serves it (byte-identical, measured) and
  `glm4` -- NORM, with weights the converter permuted to NEOX order,
  libllama's logits 0.72 apart -- refuses it by name. `glm-dsa` is the
  only architecture the GLM-5.2 loader accepts now. 56 audited.
- **`glm4moe` runs: GLM-4.5, GLM-4.5-Air and GLM-4.6 on the generic
  path, audited against libllama.** The refusal had named the pre-FFN
  norm stored as `blk.N.post_attention_norm` (`glm4-moe.cpp:75,215`,
  gpt-oss's slot) for a year; it is one row in
  `norm_sites::PRE_FFN_NORM_IS_POST_ATTENTION_NORM`, and with it the
  existing fixture matched at KL 1.51e-15 on the first run. `glm4moe`
  joins `EXPERT_WEIGHTS_SCALE_READERS` / `_NORM_READERS`
  (`glm4-moe.cpp:13-14`), leaves the GLM-5.2 dispatch everywhere, and
  three fixtures pin it (`tests/glm4moe_graphs.rs`): the 355B shape with
  per-head Q/K norms, the Air shape without (KL 2.41e-15), and a
  GLM-4.5V text tower's `rope.dimension_sections`, under which llama.cpp
  switches to M-RoPE and its logits on text positions are byte-identical
  to NEOX (measured), so frink rotates NEOX and pins the identity. 55
  audited.
- **`frink parity` reaches the MLA engine, and the first real MLA
  checkpoint went through it.** `prefill_logits` dispatches `deepseek2`
  / `mistral4` / `plm` to `MlaEngine` where it used to refuse them as
  `DedicatedOnly`. PLM-1.8B-Instruct Q8_0: tokenizer MATCH; the Q8_0
  logits read `WRONG` at KL 3.54e-2, and the arbiter -- the same file
  dequantized to f32 (`scripts/dequantize_gguf.py`, new) through both
  engines -- shows the graph at 4.5e-5, frink's Q8_0 at 4.5e-5 from
  llama.cpp's f32 (2.8e-9 from its own) and llama.cpp's Q8_0 at 3.7e-2
  from its own f32: the verdict is the reference's 8-bit activation
  quantization on the MLA latent, not frink (gap inventory §10.1). The
  reference dumper gained `LLAMA_LOGITS_FLASH_ATTN=0`, because llama.cpp
  aborts on this file with flash attention on. Twelve greedy tokens from
  the same prompt are identical between the two engines. The same
  libllama (1269cb1) has `gemma4.cpp` now, so the real Gemma-4-E2B went
  through parity too: tokenizer MATCH, logits MATCH at KL 5.1e-4 on
  Q4_K_M, the `Gemma4Engine`'s first cross-engine evidence.
- **YaRN on the MLA engine, as every real DeepSeek-V2 / V3 export
  declares it.** `frink_models::mla_yarn` resolves the three pieces
  llama.cpp computes across `deepseek2.cpp:34-37` (the key divided by
  0.1), `llama-context.cpp:194-231` (the `attn_factor`, with
  `LLM_ARCH_DEEPSEEK2`'s `mscale == mscale_all_dim` rule, which
  `mistral4` does not take -- a column in `mla_arch`) and
  `deepseek2.cpp:438-448` (`mscale^2` folded into `kq_scale`): per-band
  divisors on the `pe` slice, one magnitude, one softmax scale, handed
  to `mla_forward_token` as an argument. Three fixtures against
  libllama -- V2's `0.707`, V3's `1.0`, V2 on the legacy form -- KL
  2.98e-15 / 4.42e-15 / 2.94e-15 (`tests/deepseek2_graphs.rs`); YaRN
  moves the plain golden by 3.6e-3, the generations differ by 1.5e-3.
  The refusal that had stopped every real DeepSeek on this engine is
  gone; `yarn` without `original_context_length` and any other scaling
  type stay refused by name.
- **The MLA engine serves the split `attn_k_b` / `attn_v_b` every real
  DeepSeek export carries, and `deepseek2` has libllama goldens in both
  tensor forms.** `frink_models::mla::MlaKvB::{Combined, Split}`: the
  combined `attn_kv_b` expands per head and attends with per-head caches
  (`deepseek2.cpp:600-635`); the split pair, refused until now, absorbs
  the query through `wk_b`, attends as MQA over the latent `concat(c,
  k_pe)` and pulls through `wv_b` (`:563-598`; `frink_core::
  mla_absorbed`, unit-pinned equal to the naive form), with a cache
  `kv_lora_rank + qk_rope` wide instead of `n_heads * (qk_nope + qk_rope
  + v)`. `kq_scale` stays `1/sqrt(qk_nope + qk_rope)` for both. KL
  2.35e-15 (split) and 3.57e-15 (legacy), `tests/deepseek2_graphs.rs`.
  The fixture had never produced a golden: `scripts/
  make_deepseek2_fixture.py` wrote `head_count_kv = n_head` where
  `conversion/deepseek.py:307-308` writes 1 for every MLA export, and
  had blamed llama.cpp for the resulting `ggml.c:3942` abort. Fixed,
  with a `--legacy-kv-b` variant derived from the same draw the way the
  converter splits it (libllama's two branches agree on the pair to
  1.79e-7).
- **`arctic` runs, and Grok-2's refusal by name lifts with it.**
  `src/models/arctic.cpp:118-154` runs a dense SiLU FFN sized
  `{n_embd, n_embd}` on the post-attention residual and its router AND
  experts on `ffn_norm_exps(inpSA)`, the layer input under a second
  norm, and sums the two. Reach measured over every `build_moe_ffn`
  graph that also reads a dense `ffn_up`: two of 140 SUM the dense FFN
  with the routed output, `grok.cpp:171-184` (Grok-2, `sqrt(2)/2` on
  the sum) and `arctic.cpp`, so `frink_models::parallel_dense_ffn` is
  a two-row table (`DensePresence`, `sum_scale`) served through the
  shared-expert slot, and `MoeWeights::parallel_sum_scale` is applied
  beside `down_scale` at every site. `grep -l FFN_NORM_EXPS` is
  `arctic.cpp` alone: `RouterInput::NormedLayerInput`, the third
  variant, carries that the experts read the operand too, and the
  combine bodies take the routed and the dense operand as two
  arguments. KL 6.23e-14 (arctic; the same golden for a file declaring
  `expert_weights_scale`, which `arctic.cpp` never reads, libllama
  byte-identical), 3.28e-10 (Grok-2, GELU table)
  (`tests/parallel_dense_ffn_graphs.rs`). Measured rather than
  sabotaged: Grok-2's `sqrt(2)/2` sits before an RMSNorm and moves
  llama.cpp's own logits by 2.5e-4. 54 audited, 2 refusing (1 NEW CODE,
  1 UNKNOWN).
- **`plm` runs, on the MLA engine, which has its first libllama golden
  with it.** `src/models/plm.cpp:84-166` is `deepseek2.cpp`'s naive MLA
  branch on a dense model, and the three ways it differs are one table,
  `frink_models::mla_arch`: a DIRECT `attn_q` (`frink_models::
  mla_q_proj`, an enum the forward pass cannot reach without the file
  having answered low-rank or direct; `deepseek2.cpp:8,11-13` decide
  the same for the LITE layer counts before reading `q_lora_rank`, so
  every DeepSeek-V2-Lite / GigaChat3 / Kanana-2 export, which the loader
  had refused for that key, loads direct now), an ungated ReLU-squared
  dense FFN (`GluAct::ReluSqr` carried ON `MlaDenseFfn`, gate aliased as
  for `arcee`), and a tied lm_head (a decoy `output.weight` is refused
  as libllama refuses it, `wrong number of tensors`, measured). Head
  widths from `attention.key_length` / `value_length` when the `_mla`
  keys are absent, as `llama-hparams.cpp:259-265`. KL 1.87e-13
  (`tests/plm_graphs.rs`). The engine now REFUSES any `rope.scaling.type`
  but `none`, naming `deepseek2.cpp:312-328`: it has neither the
  frequency rewrite nor YaRN's mscale in `kq_scale`, and every real
  DeepSeek-V2 / V3 export declares YaRN. 53 audited on the generic path,
  3 refusing.
- **`talkie` runs, on four seams at once.** `src/models/talkie.cpp`
  norms without a weight at every site (`NormOp::RmsNoParams`, the RMS
  twin of OLMo-1's parameterless LayerNorm, through the same
  `NormFunction` table), applies one learned scalar per head after a
  per-head RMS on Q and a weightless per-head RMS on K, after RoPE
  (`QkNormStyle::PerHeadScalar`, decided by architecture because the
  weight's length is ambiguous with `head_dim`), adds the normed
  embedding into every layer's output times `layer_output_scale`
  (`frink_models::skip_stream`: the norm at the one embedding site,
  the add at the end of both FFN bodies), and its converter writes
  `attn_output.scale` / `ffn_down.scale`, which `build_lora_mm`
  multiplies in -- `frink_models::weight_scales` now serves exactly
  those two companions for any architecture (`AttnWeights::o_scale`,
  `MoeWeights::down_scale`) and still refuses the rest by name.
  `logit_scale` is required and multiplied (`MultiplierSupport::TALKIE`).
  Each was one graph of 140, measured. Two libllama-golden fixtures,
  KL 6.43e-14 (the converter's shape) and 1.47e-14 (without the gains;
  dropping them from the first file lands on the second's golden). The
  fused Metal launches refuse the model. 53 audited, 4 refusing.
- **`nanbeige` runs, on the layer-loop seam.** `src/models/nanbeige.cpp:
  6-31` read `num_loops` and make the logical layer count `n_phys *
  n_loops`, every logical layer with its own KV cache over shared
  weights (`:69-73` alias `layers[i + j * n_phys] = layers[i]`), and
  `:167-175` norm the residual with `output_norm` after every pass but
  the last unless `skip_loop_final_norm`; frink's decoder walked its
  layer vector once and the row refused as unaudited. One graph of 140
  reads either key (measured). `frink_models::layer_loops` says what
  the graph says -- the weights are shared and the KV is not:
  `Decoder::layers` stays physical, `ModelConfig::n_layers` is the
  logical count every KV cache and per-layer table is sized by,
  `Decoder::layer_for` / `physical_index` are the one mapping the three
  host bodies, the gpt-oss side table and the residency plan go
  through, `LayerShapes::replicated` is the copy `nanbeige.cpp:24-26`
  makes of the per-layer arrays, and the loop norm sits at the end of
  both FFN bodies. Every fused Metal launch refuses a looped model.
  Three libllama-golden fixtures (two passes over two layers, the same
  with the loop norm skipped, `num_loops = 1`), KL 3.06e-13 to
  8.70e-13; sabotaging the mapping turns four tests red. 52 audited, 5
  refusing.
- **`mimo2` (MiMo-V2-Flash, every export) runs on the host paths, on
  the split K/V head-width seam.** `conversion/mimo.py:154` writes
  `attention.value_length` from `v_head_dim` apart from the
  `attention.key_length` the base converter writes from `head_dim`
  (`192` / `128`), and `src/models/mimo2.cpp:47-48,132-140,152-154`
  size and view K and V separately with `wo` at `n_embd_head_v *
  n_head`; every KV cache, attention kernel and projection check here
  took ONE head width and the loader refused the file.
  `frink_models::kv_head_dims` admits the pair for the one generic-path
  architecture whose converter writes them apart (fourteen write
  `value_length`, three apart, two of those on the MLA engine) and keeps
  refusing it, naming llama.cpp's assert, for everyone else.
  `ModelConfig::v_head_dim` is `Some` only when the widths differ, so a
  `head_dim` set alone cannot leave V behind; `KvCache` / `PagedKvStore`
  size V by it (`new_split`), the three contiguous single-query kernels
  collapsed onto one `causal_gqa_attention_row` that accumulates over
  it, the paged kernel reads it off the store, the batched prefill
  kernel's PV tile takes its own offset and stride, and the projection
  check, the fused-QKV cut, both batched host bodies and the KV budget
  read it. Every fused Metal launch, the CUDA resident hook, the slot
  file and the KV block file refuse a split model. Its second half,
  `attention.value_scale` (`:180-183`, `0.707` on every export), is
  `frink_models::attn_value_scale`: one reader of 140, applied after
  `wo` in the one attention tail. Three libllama-golden fixtures, each
  carrying the per-layer `head_count_kv` array, the per-layer window
  array with `rope.freq_base_swa`, sinks, sigmoid routing with
  `exp_probs_b`, partial NEOX RoPE and MoE on every layer: KL 5.42e-15
  (fused `attn_qkv`), 5.42e-15 (split), 3.49e-15 (no value scale).
  51 audited, 6 refusing.

- **`bitnet` runs, on the two norms INSIDE the blocks.**
  `src/models/bitnet.cpp:24,36` require `attn_sub_norm` (on the
  attention output BEFORE `wo`, `:101-106`) and `ffn_sub_norm` (on
  `silu(gate) * up` BEFORE `down`, `:127-141`), two sites the generic
  decoder's four norm slots did not have; a file carrying them died on
  the unread-tensor gate. One graph of 140 creates either tensor
  (measured), so `frink_models::sub_norms` is one `bool` on
  `ModelConfig` read by the loader (the pair is REQUIRED; refused on a
  routed layer, where `build_moe_ffn` has no such site) and by the
  Metal predicate (every fused launch refuses; the per-layer fused
  attention loses its view of the layer through the exhaustive
  destructure), and two tensors on the layer applied in the one
  attention tail and the one dense FFN row body --
  `frink_moe::run_expert_sub_normed`, which shares its gate/up half
  with `run_expert` and cannot reach the fused on-device SwiGLU. One
  libllama-golden fixture, KL 1.88e-14, with the norm weights drawn
  away from one so that skipping either norm, applying either with
  unit weights, or reading the attention one as Gemma's post-norm each
  diverges by orders of magnitude (measured). The tied lm_head
  (`:164`, no `output` tensor) and `rope.scaling.type = linear` at 1.0
  (`conversion/bitnet.py:19-20`) were already served. A real
  BitNet-b1.58-2B-4T still needs `TQ1_0` / `TQ2_0` kernels; a Q8_0 or
  F16 re-export runs. 50 audited, 7 refusing.
- **Per-tensor weight scales are refused by name.** `build_lora_mm`
  multiplies a projection's output by an optional `<tensor>.scale`
  companion (`llama-graph.cpp:1492-1494`), and since
  `llama-model.cpp:1355-1440` a generic pass creates `.scale` and
  `.input_scale` beside EVERY architecture's projections; the NVFP4
  converter writes them, and older BitNet exports carry the seven
  `bitnet.cpp:27-43` created. frink does not apply them, and such a
  file used to die on the unread-tensor gate -- the right outcome with
  a message `FRINK_ALLOW_UNKNOWN_TENSORS=1` could talk past into every
  projection running at the wrong magnitude. `frink_models::
  weight_scales` refuses either suffix before that gate, from a fixture
  whose libllama logits differ from the unscaled file's (measured).
- **`smallthinker` (every SmallThinker export) runs, on the MoE
  router-operand seam.** `src/models/smallthinker.cpp:111` computes
  the router logits from `inpL` -- the residual stream as it ENTERS
  the layer, before `attn_norm` and before attention -- and `:151-161`
  hands them to `build_moe_ffn` as a precomputed `probs`; every frink
  MoE body routed on the normed FFN input, which is `build_moe_ffn`'s
  own default and what the experts read. `frink_models::router_input`
  is one two-variant enum and one table row; `Decoder::router_operand`
  is the one constructor, called where each host body applies
  `attn_norm`, and it carries logits rather than the operand so the
  post-attention residual (the same `Vec`, mutated in place) cannot
  reach the router by mistake; the GPU router paths refuse the row
  through `gpu_router_matches_host_routing`, the predicate they
  already shared. The reach was measured before it was written: all
  fifty-nine `build_moe_ffn` call sites in the 140 graphs, four pass a
  precomputed `probs_in`, and only this one on the generic path routes
  on something other than the normed FFN input (`grovemoe` shares the
  mechanism and not the cause; `gemma4` and `nemotron-h` are on other
  engines). Its `LLM_FFN_RELU` experts have a REAL gate, so
  `FfnActivation::Reglu` (`relu(gate) * up`) is split from `arcee`'s
  ungated `ReluSqr` -- the one `GluAct` variant that existed served
  `arcee` by aliasing gate to up, and a SmallThinker loaded through it
  would have dropped its gate tensors and computed `relu(up) * up`.
  `smallthinker.cpp:8` pins `n_swa` to 4096 over whatever the file
  declares (`capability::swa_window_override`, a third answer beside
  honour and drop; libllama's logits for a fixture declaring 3 and the
  same fixture declaring 4096 are byte-identical, measured). Three
  libllama-golden fixtures: window declared with sigmoid gating and
  NoPE on layers 0 and 4, no window with softmax gating, and a keyed
  `sliding_window_pattern = 2` beside the literal NoPE step. KL
  6.56e-15 to 1.27e-14. The three batched FFN tails (the Metal-prefill
  and host arms of the prefill body, and the multi-sequence body)
  collapsed onto one `Decoder::ffn_block_batch` on the way, because
  the seam needed an eighth fact in all of them. 49 audited, 8
  refusing.
- **`mistral3` (every Ministral-3 export) runs, on the per-position
  attention temperature seam.** `src/models/mistral3.cpp:5,14-17,153-156`
  reads `attention.temperature_scale`, floors it on `n_ctx_orig_yarn`
  and multiplies Q after RoPE by `log(floor(pos / floor) + 1) * scale +
  1` per token (`llama-graph.cpp:163-167`); frink had no per-position
  Q scale and no gate on the key, so a real Ministral-3 loaded and ran
  at the wrong temperature with no error. `frink_models::attn_temperature`
  is one value, one accessor and one helper on the three host bodies,
  with the fused Metal launches fenced off through the predicate the
  other host-only facts already share, and the census measured before
  it was written: three graphs of 140 build the input, and the other
  two (`llama4`, `deepseek2` / `mistral4`) are on other engines. Five
  libllama-golden fixtures: plain, the temperature stepping twice
  inside the prompt, the floor from `context_length` when the YaRN key
  is absent (byte-identical upstream, measured), and two YaRN files.
  KL 5.14e-15 to 9.16e-15. 48 audited, 9 refusing.

### Fixed

- **`expert_weights_scale` and `expert_weights_norm` were honoured for
  EVERY architecture; llama.cpp reads the two keys in twenty
  per-architecture loaders and nowhere else.** Everywhere else the
  graph passes a literal into `build_moe_ffn` and the key is dead
  metadata. Found by `mimo2`'s fixture, which declares
  `expert_weights_scale = 2.5` that `mimo2.cpp` never reads: libllama's
  golden is unscaled and frink was 2e-3 of KL away until the loader's
  `EXPERT_WEIGHTS_SCALE_READERS` / `EXPERT_WEIGHTS_NORM_READERS` (eight
  and seven on the generic path, measured) gated the keys on their
  readers. No real export of a non-reader writes either key, so no
  published checkpoint changes; a hand-written file would have. In the
  same measurement, `GATING_LITERAL_ARCHITECTURES` records the one
  generic-path graph that passes a SIGMOID literal (`mimo2.cpp:227`),
  so a file's `expert_gating_func` cannot turn it.
- **YaRN's magnitude term was not applied for ANY architecture on the
  generic path.** `llama-context.cpp:196-231` multiplies
  `rope.scaling.attn_factor` by `get_mscale(factor, 1) /
  get_mscale(factor, yarn_log_multiplier)` -- `1 + 0.1 ln factor` with
  no multiplier -- and frink's `rope_attn_factor` carried the key
  alone, so every YaRN checkpoint (`*-128K` Qwen3 exports, every
  Ministral-3) was roped at the right frequencies and the wrong
  magnitude, with attention logits low by `(1 + 0.1 ln factor)^2`:
  1.30x at factor 4. Found reading `mistral3.cpp:9` for
  `rope.scaling.yarn_log_multiplier`, whose only job is to adjust a
  term frink turned out not to have. `frink_models::yarn_magnitude`
  folds it into the field the CPU helper and the Metal `mscale`
  uniform already read; two fixtures at factor 4 match libllama with
  and without the multiplier (KL 9.14e-15, 3.45e-15). Only `mistral3`
  reads the multiplier on the generic path (measured), so it stays
  dead metadata everywhere else, as upstream.
- **The MLA engine (`deepseek2` / `mistral4`) silently dropped
  Mistral-Large-3's attention temperature.** `deepseek2.cpp:46-47`
  reads `attention.temperature_scale` and `attention.temperature_length`
  and the graph applies them; the MLA loader read neither. It refuses
  a nonzero scale by name now, from a fixture that carries both keys,
  rather than implementing a multiply that engine has no golden to
  check.
- **LoRA adapters, llama.cpp's `--lora` / `--lora-scaled`,
  `GET`/`POST /lora-adapters` and the per-request `lora` field.** An
  adapter GGUF (the file `convert_lora_to_gguf.py` writes) is applied
  as `build_lora_mm` applies it, `W x + scale * alpha / rank * B (A x)`
  on every projection it names, `token_embd` and `output` included,
  with two adapters on one weight summing. The delta is a decoration
  on `WeightMatrix` itself (`WeightMatrix::Adapted`,
  `frink-core/src/weight_matrix/lora.rs`), so the CPU row body, the
  batched host bodies and the per-matrix Metal and CUDA launches serve
  it through the one method they already call; the fused Metal stacks
  cannot see it and are fenced off for the whole model through the
  predicate they share (`metal_can_serve_model`), so an adapted model
  on Metal runs on the per-matrix path with the same tokens as CPU.
  Checked against libllama loaded with the same adapter: KL at or
  under 5.0e-13 on a fixture whose adapters were converted by
  upstream's own script, and 5.2e-4 on Llama-3.2-1B-Instruct Q8_0
  with a rank-8 adapter (base floor 1.9e-4; the adapter moves the
  distribution by 1.5e-1); scale 0 is byte-identical to no adapter,
  as it is upstream. Refused by name rather than approximated: an
  adapter for another architecture, a tensor the base lacks or does
  not fit (`llama-adapter.cpp`'s three checks), a routed-expert
  target (`build_lora_mm_id`), an activated LoRA, the embedding pair
  on a tied output head (libllama aborts in `ggml_mul_mat` on it,
  measured), and the dedicated engines. The server's scales are one
  atomic per adapter read at apply time; a `POST` or a per-request
  override runs exclusively against the generations in flight and the
  response cache keys on the scales a generation ran under.

## [0.21.0] - 2026-09-11

### Fixed

- **The Metal benchmark ledger booked the lm_head's GPU time as host
  time.** #149's "host = wall minus GPU" used the GPU time of ONE
  command buffer per token; a sampled decode token has two, and the
  lm_head's had no timing tag, so 1.2 ms (Llama-3.2-1B) to 2.8 ms
  (Gemma-2-2B) of GPU work read as a 26-29% host share that did not
  exist. `FRINK_METAL_GPU_TIMING=1` clocks encode, GPU and submit
  latency for every submission from one clock, and a submission cannot
  be timed with a phase left out. Measured encode is 2-3% of wall, so
  the argument-packing lever the issue named was retired unbuilt;
  the correction moved the worst Metal row from "host" to "kernel".
- **`/v1/rerank` scores from the published `ms-marco-MiniLM-L6-v2` GGUF
  were fifty times too small.** llama.cpp's converter drops
  `pooler.dense.*` for every BERT, so the file scores `classifier(cls)`
  where HuggingFace scores `classifier(tanh(pooler(cls)))`; orderings
  were right and the scale was not, which is why a threshold copied from
  another engine never fired. `frink splice-pooler -m in.gguf
  --safetensors model.safetensors -o out.gguf` writes the pooler back
  under llama.cpp's own tensor names (llama-server loads the result and
  applies it too), tied to the checkpoint by the classifier both files
  hold rather than by a name the published file gets wrong. Seventeen
  pairs within 0.051 of HuggingFace afterwards, orderings identical.
- **The GLM, MLA and hybrid dedicated loaders would have run a
  checkpoint's MTP block as one more decoder layer.** `glm4moe`,
  `glm-dsa`, `glm4`, `deepseek2`, `qwen3next`, `qwen35` and `qwen35moe`
  all subtract `nextn_predict_layers` from `block_count` in llama.cpp,
  and their converters append the block INSIDE `block_count`
  (`conversion/glm.py:99`, `deepseek.py:457`); the three loaders read
  `block_count` verbatim and no gate stood in front of them, so a real
  GLM-4.5 or DeepSeek-V3 export would have loaded, run its NextN block
  as a 47th or 62nd layer, and left the `nextn.*` tensors silently
  unread. All four dedicated loaders take their layer count from
  `frink_models::mtp_blocks::trunk_layers` now; the MLA loader has a
  trunk-only fixture that fails without it. Not observed on a
  checkpoint: found by reading the seventeen graphs that read the key
  against the loaders that own them.
- **A reasoning model that ran out of `max_tokens` inside its thought
  showed thinking and then nothing.** Studio sent `max_tokens: 512` on
  every request, DeepSeek-R1-Distill spends about 900 tokens thinking
  on an ordinary question, the server correctly returned `finish_reason:
  "length"` with empty content, and the UI rendered that as silence.
  The default is now no cap (the context is the limit, llama.cpp's
  `n_predict: -1`), a saved 512 from the old default is migrated away,
  a `length` finish is shown as a cut-off with the token count, and a
  **Continue** button carries on from where it stopped.
- **The conversation store dropped a reasoning model's chain of
  thought.** An answer that was entirely `reasoning_content` persisted
  as `content: ""` and reloaded as an empty turn. The store keeps
  `reasoning_content` beside `content`; records written before the
  field read back unchanged.
- **A leading-dense MoE file that omits `expert_shared_count` loaded
  with its shared experts unread.** The inference probed `blk.0` for a
  `_shexp` tensor, and layer 0 of such a model is dense. `laguna.cpp:20`
  assigns the count before reading a key its converter never writes,
  so a real Laguna export would have run without its REQUIRED shared
  expert on every MoE layer. The probe is the first MoE layer now.
- `afmoe` scales its embeddings by `sqrt(n_embd)` from arithmetic, the
  only non-Gemma graph that does (measured over all 140); the Gemma
  family match in the loader is a table with two rows now.

### Added

- **Every one of the 16 comparable Metal `tg128` rows is now faster
  than llama.cpp** (gap 0.60x-0.96x; prefill 0.99x-1.09x), re-measured
  on a quiet host with `frink 0.20.0` receipts. Gemma-2-2B decode went
  from 1.11x to 0.94x on three kernel rewrites none of which is
  Gemma-specific: RoPE (one thread per rotary pair, one templated
  NORM/NEOX kernel where there were four sources; 61.6 to 3.4 us),
  RMSNorm (float4 loads, at most two loop trips; 14.2 to 4.7 us),
  FA-vec decode at d=128/256 (compile-time tile loops so K and V loads
  overlap; 38.9 to 17.6 us with the softcap), and the final logit
  softcap moved off the host into the lm_head's command buffer as an
  epilogue (0.65 ms of scalar `tanh` per Gemma-2 token, gone).
  `FRINK_METAL_KERNEL_TIMING=1` attributes a token's GPU time per
  dispatch kind, which is how the three were found: the matvecs, 80% of
  the stack, were already at parity. `frink verify` token-identical and
  `frink parity` unchanged on four models. Closes #149.
- **The Studio Thinking block times the thought.** A live `Thinking for
  12 seconds` while the model reasons, collapsing to `Thought for 15
  seconds` (`1 min 20 s`, `1 h 5 min`) once the answer begins, one click
  away; a thought cut off with no answer stays open under the Continue
  banner and a continuation keeps counting. The time is the wall-clock
  between the first `reasoning_content` delta and the first `content`
  delta, persisted as `reasoning_ms` beside `reasoning_content`; older
  records load and show their thought with no time.
- **`grok` and `dbrx` are audited**, 37 to 39, each by extending a seam
  from the day before by one column. Grok-1 on the MiniCPM defaults
  hook: `grok.cpp:5-12` seeds seven hyper-parameters before the file
  may override them, `logit_scale` is a multiply there and
  `attention.output_scale` is "pre-scale Q, then softcap", so it
  resolves into slots that existed; two of the seven are read by
  llama.cpp and applied nowhere (measured), and frink neither applies
  nor refuses them. DBRX on `NormOp::LayerNorm` (weight, no bias, the
  variant OLMo-1 deliberately left unwritten until a caller arrived),
  a REQUIRED `attention.clamp_kqv` (the three QKV-bias loops collapsed
  onto `decoder/qkv_bias.rs` first, so the clamp is one line rather
  than three), and `norm_sites.rs`, one table for which tensor feeds
  which norm site, because `blk.N.attn_output_norm` is DBRX's pre-FFN
  norm and Grok's post-attention norm. The clamp closed OLMo-1's
  `clip_qkv` checkpoints with it. KL 4.7e-10 / 1.6e-10 (Grok, at the
  GeGLU f16-table tolerance, measured), 3.4e-12 (DBRX). Grok-2's
  parallel dense FFN stays refused by name from a fixture that has it.
- **`arcee`, `deci` and `openelm` are audited**, 39 to 42. `arcee` is
  the ungated ReLU-squared FFN, spelled as `GluAct::Reglu` with the gate
  aliased to the up matrix rather than a fourth expert shape; on the
  way, six Metal launch sites that derived `gelu = !is_swiglu()` (a
  third activation would have run as GELU on all of them) now refuse on
  `None` from one function. `deci` and `openelm` are one seam,
  `layer_shapes.rs`: llama.cpp reads `head_count`, `head_count_kv` and
  `feed_forward_length` as scalar-or-array for every architecture and
  frink carried scalars; all 140 graphs were scanned for which honour
  a per-layer value before a line was written, and the table records
  each. `AttnShape::{Gqa, Linear, Absent}` is an enum so an
  attention-less layer cannot be a zero count a loop accepts;
  `ModelConfig::new_kv_caches` replaced ninety hand-written
  `KvCache::new(config.n_kv_heads, ..)` sites and `KvCache::push`
  asserts the width. KL 2.27e-14, 1.44e-13, 7.29e-13, 1.28e-13. `plm`
  did NOT close with `arcee`: its verdict had been read from one file,
  and the diff is 150 lines of MLA attention.
- **`apertus` and `step35` are audited, on ONE seam with TWO
  activation bodies.** An FFN activation whose parameters vary by
  layer had no home: `FfnActivation` was a unit enum and every FFN
  body converted the model-wide value to a `GluAct` without knowing
  which layer it ran. `apertus.cpp:6-9` reads xIELU's `xielu.alpha_n` /
  `.alpha_p` / `.beta` / `.eps` as `n_layer`-long arrays (or a scalar
  broadcast) and `:132-138` hands layer `il`'s four to `ggml_xielu`;
  `step35.cpp:28-29` reads `swiglu_clamp_exp` / `_shexp` the same way
  and llama.cpp's generic `build_moe_ffn` / `build_ffn` clamp SwiGLU by
  layer `il`'s entry, the routed experts from one array and the shared
  experts AND the dense layers from the other. `frink_models::
  act_layers` reads both families as `get_key_or_arr` does,
  `FfnActivation::Xielu` / `::SwigluClamped` carry their tables,
  `frink_moe::GluAct` gained the two bodies (and lost `gate_fn`, a
  gate-only signature xIELU cannot fit), and `ModelConfig::
  layer_ffn_acts(il)` answers a `routed` / `dense` pair at every FFN
  body; no fused Metal kernel spells either, and both refuse through
  the predicate the launches share. Step-3.5's other blocker, a rotary
  width halved on the full layers with no key (`step35.cpp:9`), is
  `ModelConfig::rope_dim_swa` -- the two-valued `n_rot(il)` llama.cpp
  already had -- handed out per layer by `layer_rope`, which also
  SERVES a `rope.dimension_count_swa` differing from the full width
  (Laguna-XS.2's shape) where it used to refuse it; the two `_swa`
  head-width keys stay refused. Five libllama-golden fixtures: apertus
  KL 4.91e-14 (arrays) and 5.06e-14 (the scalar spelling llama.cpp
  broadcasts); step35 KL 1.59e-13 (clamped), 2.59e-13 (neither key),
  1.59e-13 (a NextN block inside `block_count`); the Laguna-XS.2
  rotary-width fixture 7.29e-14. Building them found `apertus.cpp:93,96`
  pass `NULL` for the QK-norm biases `:50,52` create, so a file
  carrying them is served with them ignored, as libllama serves it
  (measured byte-identical; `frink_models::unread_tensors`). 47
  architectures audited, 10 refuse, 9 of them NEW CODE.
- **`mellum` is audited, and every real EXAONE-4 32B, EXAONE-MoE and
  Olmo-3 export loads.** `{arch}.attention.sliding_window_pattern` as a
  per-layer bool ARRAY was refused for every architecture; llama.cpp
  reads it through `get_key_or_arr`, which for fifteen graphs
  (`exaone4`, `exaone-moe`, `olmo2`, ...) IGNORES the array and keeps
  the seeded period, for `mimo2` / `step35` / `gemma4` honours it and
  broadcasts a scalar as a bool, and for `mellum` / `cohere2moe` tries
  the scalar then the array. `frink_models::swa_layers` carries the
  three modes as one table and one enum (`All`, `Period`, `PerLayer`)
  behind `ModelConfig::layer_sliding_window(il)`, which every backend
  already asked per layer. Fixtures with the EXAONE array agreeing with
  and INVERTED against the period measure that libllama's logits do not
  move (KL 1.43e-14 both), and a Mellum whose array disagrees with the
  seed on two layers is honoured (KL 1.02e-14); a Mellum with a window
  AND a RoPE scaling, which every real Mellum2 is, stays refused by
  name. 45 architectures audited, 12 refuse, 11 of them NEW CODE.
- **NextN / MTP blocks are skipped, as llama.cpp skips them.**
  `nextn_predict_layers` was refused for every architecture on any
  nonzero value. `frink_models::mtp_blocks` subtracts the blocks from
  `block_count` for the seventeen graphs that read the key (measured),
  keeps refusing it elsewhere, marks the skipped tensors deliberately
  unread so the consumption gate still sees a missing term, and hands
  `block_count` rather than the trunk to the two things llama.cpp
  decides before reading the key (`exaone4.cpp:4`'s 64-layer gate, the
  per-layer shape array lengths). K-EXAONE's shape, one block after the
  trunk, has a fixture (KL 1.09e-14). `mimo2` and `step35` verdicts
  now lead with what is left: a V head width differing from K's, and
  per-layer SwiGLU clamp arrays with a half-width rotary.
- `continue_final_message` on `/v1/chat/completions`, llama.cpp's field
  and value set (`true`, `"reasoning_content"`, `"content"`): the
  trailing assistant message renders as a turn still being written, its
  thought re-opened in the family's own markers, so the model continues
  rather than starting over. **Default on, as llama.cpp's server**: a
  trailing assistant message is continued unless the request says
  `false` or the server was started with `--no-prefill-assistant`
  (llama.cpp's flag). `/v1/messages` and `/v1/responses` render through
  the same function, so the three routes hold one default rather than
  three literals. The channel-grammar families and a content
  continuation for an always-open family are 501 by name.
- `reasoning_budget_tokens` / `thinking_budget_tokens`, llama.cpp's
  sampler-level thinking budget (`common/reasoning-budget.cpp`), on
  `/v1/chat/completions`, `/v1/responses` and, as `thinking.budget_tokens`,
  `/v1/messages`; `--reasoning-budget N` is the server default. After N
  tokens of thought the closing tag is forced one token per step, so the
  answer still arrives with `finish_reason: "stop"`; `0` closes the
  block the moment it opens; `-1` is unrestricted. Measured
  token-for-token against llama.cpp master on DeepSeek-R1-Distill at
  temperature 0. Studio's sampling panel carries the field. Replaces the
  501 that refused the field by name.
  rather than starting over. Default off; the channel-grammar families
  and a content continuation for an always-open family are 501 by name.
- `reasoning_budget_tokens` / `thinking_budget_tokens` are refused by
  name (501) except `-1`, rather than silently dropped. llama.cpp
  enforces the budget in its sampler; frink has no such sampler yet.
- **`afmoe` and `laguna` run with evidence**, 42 to 44, on one seam:
  the learned attention output gate (`attn_gate.rs`). llama.cpp's three
  gating graphs were read side by side and are one op with two free
  parameters, the activation (sigmoid / softplus) and the width (per
  channel / per head, read off the tensor), so the type has two axes
  and a table. Three libllama-golden fixtures, KL 7.03e-13, 1.51e-13,
  9.57e-14. `step35`, the third graph, keeps its clamp arrays and window
  array and says the gate is done.
- Attention sinks are a tensor-presence fact (`AttnWeights::sinks`)
  rather than a gpt-oss name check: four llama.cpp graphs pass the
  tensor into the one `build_attn_mha`. gpt-oss still requires it and
  the fused Metal launches still refuse a layer that has one, by the
  tensor. `mimo2` does not close on it, because every real export
  carries MTP blocks and a per-layer window array.
- Every fused Metal attention launch takes its view of a layer's
  weights from ONE exhaustive destructure of `AttnWeights`, so a field
  added there does not compile until the Metal side says whether the
  kernels serve it.
- A sliding-window geometry the full-attention layers do not share --
  `rope.dimension_count_swa`, `attention.key_length_swa`,
  `attention.value_length_swa` -- is refused by name for every
  architecture (`swa_geometry.rs`); it loaded and was ignored before.
  The Olmo-3 "window plus scaling" refusal is one table for `olmo2`,
  `mellum` and `laguna` rather than one `if`.

## [0.20.0] - 2026-09-11

### Fixed

- **The tokenizer disagreed with llama.cpp on text that mentions special
  tokens** (#198). Silent-wrong-answer class. Two bugs, not one. A
  heuristic added for one checkpoint's `<|im_end|>` promoted ANY
  angle-bracket-shaped vocabulary entry to special, sweeping in `<s>`,
  `</s>`, `<unk>` and `<?>`; in Qwen2.5's file `<s>` is a plain token
  and llama.cpp never treats it as special under any setting. Removed.
  And llama.cpp's common tokenize path defaults to NOT parsing special
  tokens in text, while frink behaved as if it always did. Every
  `encode` now takes an explicit `SpecialTokens::{AsText, Parse}`, so a
  caller cannot avoid choosing, and each caller sits where llama.cpp's
  own source puts it, cited line by line. The oracle was strengthened
  first: the golden dumper writes every case under both settings, two
  new cases contain markers as prose, and they were confirmed RED on
  main before the fix. After it, all 20 local checkpoints match a
  current libllama across 21 cases and both settings.

- **Metal decode was thread-affine, and the pool gate is now relaxed
  for it** (#184, closing #166). A thread-local configuration flag was
  read on whichever thread ran the step. It is carried across the pool
  explicitly now: `Carry` destructures itself exhaustively with no `..`,
  so a setting added later and not carried fails to compile. CUDA and
  Vulkan still decline, stated as a verdict with the reason.

- **Evicting behind a sliding window leaked pool blocks** (#189). The
  shrink made `push` compare rows against a capacity that had moved, so
  a pool-backed cache drew a fresh block every `slack + 1` tokens forever
  and exhausted a pool where `push` is documented infallible. Reachable
  with `FRINK_KV_POOL_BLOCKS` and `FRINK_KV_WINDOW` together; the
  existing pool test never reached the shrink.

- **Gemma 4 tool calls fell through to the Llama 3 parser** (#190).
  Format inference tested the literal string `gemma4` while the
  checkpoint calls itself `Gemma-4-E2B-It`, so the arm had NEVER fired.
  Widened, with the earlier Gemmas asserted still not to match.

- **The Gemma family was exempted from four scaling keys it does not
  read** (#186), so a hand-written `gemma3.residual_scale` would have
  loaded and been ignored. Found by deriving the refusal list from the
  implementation table instead of restating it beside it.

- **`nextn_predict_layers` was refused nowhere** (#193). MTP blocks sit
  inside `block_count` and llama.cpp skips them, so a real EXAONE-MoE
  export with an MTP head would have run its speculative head as two
  more decoder layers. Gated on the value, since the converter writes
  `0` for sizes with no head.

- **Swapping the model through `POST /admin/models/load` could leave
  generation fluent and wrong** (#180). Silent-wrong-answer class, so
  read it twice. Two Metal caches map `(host pointer, host length)` to
  an uploaded `MTLBuffer`, and an address is not an identity: the
  outgoing model's allocations are freed with it and the incoming
  model's land on the same addresses. `resident_weight_buffer` knew
  that and checked; `resident_f32_buffer`, seventy lines below it in the
  same file, did not, and what it caches are the RMSNorm gammas a
  `Decoder` owns. Measured, not assumed: instrumenting every hit to
  compare it against the host bytes reported **49 stale hits in a
  24-token decode** of Llama-3.2-1B-Q6_K loaded after
  Llama-3.2-1B-Q4_K_M. The MoE stack had a third cache in front of both,
  keyed on a bare pointer with no length at all. Every resident cache
  now goes through one lookup that will not serve an entry unless the
  entry can still prove it holds the caller's bytes, and the Studio
  model selector is that endpoint. The repack cache named as the likely
  cause in the issue was NOT it: the same swap sequence under
  `FRINK_METAL=0` answers identically on every pair.

### Added

- **Ten more architectures run with evidence**, 26 to 37, each with a
  libllama-golden fixture, and the two cheap triage classes are now
  EMPTY: nothing still refusing is one fixture or one match arm away.
  - `olmo2` and `exaone4`, one post-norm residual topology (#183)
  - `chatglm` and `qwen`, sharing the fused `attn_qkv.bias` arm; `qwen`
    also needed its FFN width halved, which the verdict never named
    (#185)
  - `granite`, `granitemoe` and `granite-moe`, on one implementation of
    the four scalar multipliers, with 18 hand-written residual adds in
    `decoder.rs` collapsed onto one function (#186)
  - `minicpm`, on the defaults hook that Granite made a field of the
    same table (#191)
  - `olmo`, on a non-parametric LayerNorm; its `clamp_kqv` stays a
    refusal because llama.cpp's logits move when the key is present
    (#191)
  - `exaone-moe`, `smollm3` and EXAONE-4 32B, on one per-layer RoPE
    gate that llama.cpp applies in six architectures from literals and
    never from a key (#193)
- **Three rows turned out not to be architectures**: `mistral`,
  `mixtral` and `yi`. libllama refuses all three strings and every real
  checkpoint declares `llama`. They now refuse by name and say to
  re-convert (#185).
- **Forced `tool_choice` on 10 of 11 wire formats**, up from 8. Two
  recorded refusals were wrong: MiniMax-M3's ambiguity is real when
  reading and absent when writing from a schema, and Gemma 4 needed a
  new shape rather than a new branch. Muse-Glimmer still refuses for a
  stated reason (#190, closing #29).
- **`frink imatrix`**, a port of `llama-imatrix`, and `frink quantize
  --imatrix`. Importance-weighted quantization is byte-identical to
  `llama-quantize --imatrix` on 311 of 311 tensors across five targets.
  The matrix file matches llama.cpp's in format, names and shapes, with
  values agreeing to the forward pass's precision (#195).
- **`frink batched-bench`**, a port of `llama-batched-bench`, on the
  same guards as `frink bench` rather than beside them; the host
  preflight and receipt envelope were extracted so both tools call one
  function. Four llama.cpp flags refuse by name through an exhaustive
  destructure (#197).
- **Server slot save and restore**, with an identity check llama.cpp's
  own slot file lacks: a digest over the checkpoint's sorted metadata,
  tensor directory and a sample of every tensor, plus layer geometry.
  Restore under a different quantisation refuses naming the checkpoint;
  under a different model, naming the model (#192).
- **`-b` and `-ub` batch flags** on the server, which exposed one number
  spelled as two independent environment variables at two readers; both
  names now live in one array both readers index. `-np` was already
  wired and the audit was stale, so `/metrics` now reports the scheduler
  configuration the worker was spawned with (#192).
- **Windowed models are priced by their window** (#189, on #61). The
  store already evicted behind `FRINK_KV_WINDOW`, including per-layer
  for alternating models; what was missing was that nothing spent the
  saving. `KvBudget` now carries a residency, so `--ctx-size auto`, the
  pre-load check and `inspect-plan` see it. Gemma-2-2B at 32k context:
  6.50 GiB to 4.06 GiB.
- **`/v1/tokenize` honours `parse_special`**, which was a 501 by name
  (#198).

### Changed

- **Frink Studio has a mark instead of two letters, and a palette with
  no brand hue** (#187). The logo is alpha-iron's body-centred cubic
  cell seen down its body diagonal, one outline, three spokes, one
  node; it reads at 16 pixels. The neutral ramp is zero-chroma and the
  only coloured tokens are the semantic states; a test fails if a
  non-semantic token gains colour or the two theme blocks drift. The
  send button is a circle, and the send glyph is one whose horizontal
  centre is verifiable.

### Documented

- `docs/MODELS.md`'s list of audited architectures had been severed in
  half by an inserted section and was three rows stale; it is rejoined
  and now verified against `AUDITED_GENERIC_GQA` in both directions
  (#188).
- Studio screenshots in the README and `ui/README.md`, captured from a
  real server, in the dark theme (#194, #196).
- `frink quantize --help` claimed only Q8_0 and Q4_K write; a test now
  walks every target against the long help (#195).

## [0.19.1] - 2026-09-10

Frink Studio only. No crate in the workspace changed, so an engine
built from 0.19.0 behaves identically.

### Changed

- **One model selector instead of three surfaces claiming the model.**
  The chat header switcher and a `Load` button on every row of the
  Models table posted the *same* request for the same server-wide
  effect: the server holds one checkpoint and `/v1/chat/completions` has
  no per-request override, so these were not different scopes. Models is
  now facts plus the one verb a picker cannot express, `Unload`, and the
  header keeps the picker. The rule comes from how this class of UI
  splits generally: a management screen installs, removes and reports,
  while a picker beside the conversation selects.
- **The status control at the bottom of the sidebar reads as status.**
  It was rendering the loaded model id under a chevron, which made a
  health indicator look like a fourth model selector. It now shows the
  server state and version, with the model id, capabilities and
  last-request age one click into its popover.
- **The base URL is a new chat, and a conversation has its own URL.**
  Entry previously ended in an unconditional "reopen the newest
  conversation", on every path and after any elapsed time. `/ui/chat` is
  now empty and `/ui/chat/<id>` is that conversation, with the id
  stamped in by the first message.
- **Returning after 30 minutes away starts fresh** and offers the
  previous conversation back. "Away" is measured only while the document
  is visible and is stored per tab, so a reload of an old tab reads as
  away while a deep link into a new tab is simply honoured. It never
  fires over a running generation or a half-typed message.

  Worth recording that the research contradicted this one: no product
  surveyed implements a staleness rule, and the advice was not to invent
  one. It is here because it was asked for, which is why it is narrow,
  announced, and undoable rather than silent.

### Fixed

- The staleness rule **could never fire**: the visibility heartbeat
  stamped the tab awake from its mount effect while the entry check read
  that stamp from inside a promise, so the evidence was always already
  overwritten.
- Correcting the URL **re-loaded the conversation the rule had just
  declined**, because a route correction is a state update and the
  effect ran again on the old id, printing its banner over a resurrected
  transcript.

### Known issue

`POST /admin/models/load` can leave generation producing garbage until
the server is restarted, and the Studio model selector is that endpoint
([#180](https://github.com/antonellof/frink/issues/180)). Present in
0.19.0 and not fixed here. Deterministic, and a fresh start on the same
checkpoint is correct, so restarting the server clears it.

## [0.19.0] - 2026-09-10

### Fixed

- **Metal greedy decoding returned the wrong token in the default
  configuration.** This is the silent-wrong-answer kind, so read it
  twice. Metal has two paths for the final `lm_head` step: a fused GPU
  fold that argmaxes on device, and a host path that samples on the CPU.
  **The fold argmaxes the RAW logits; the host path applies the
  repetition penalties first**, and `--repeat-penalty` defaults to 1.1
  here rather than llama.cpp's 1.0. So the fast path answered a question
  nobody asked. Proven by checksum: at 1.1 the folded completion was
  bit-identical to the completion at 1.0, which is what "the penalty
  never ran" looks like, while the unfolded completion matched the CPU
  reference exactly.

  The cause was **one predicate answering two questions**.
  `greedy_equals_argmax` was called both after the penalties had been
  applied, where excluding them is correct, and by Metal before anything
  had been applied, where it is not. It is now split in two, each
  derived from one exhaustive match over the sampler chain and one
  exhaustive destructure of the parameters with no `..`, and the old
  name is deleted so every call site had to choose. A sampler field
  added later fails to compile until it is classified.

  The fold is refused when a penalty is live rather than the penalties
  being reimplemented in a Metal kernel, deliberately: a second
  implementation that must agree with the host about sign convention,
  the once-per-candidate rule and the window is this repo's dominant
  defect shape, and putting it inside the fix for that shape is how it
  recurs (#170, #172).

- **Two same-length activations could alias on Metal, across requests.**
  The residency cache matched on length alone. `cpu-cuda-parity.md`
  recorded that as safe because "exactly one site sets it": one site
  sets it and **three** consume it, each routinely handed a same-length
  activation that is not the published one. The published value was a
  raw buffer pointer that escaped the mutex protecting it, so two
  concurrent `frink-server` requests could have one answer the other's
  `lm_head`. Live, not latent. Publication now lives inside the guarded
  scratch, keyed on the host address and length of the exact buffer
  returned, and drops on any borrow (#171).

- **Metal decode was thread-affine and did not say so.** Running a
  forward on a different thread changed its output. The stated cause was
  the resident activation cache and that was wrong: disabling the reuse
  changes nothing, because the dense stack downloads with an exact copy.
  The cause was `GREEDY_ARGMAX`, a thread-local *configuration* flag set
  on the main thread and read on whichever thread ran the step, so a
  worker read the default and silently took the other `lm_head` path. It
  is now a three-state type with capture and adopt, since the two
  spellings of "false" are not the same claim (#166, #171).

### Changed

- **CPU decode enters the thread pool once per forward instead of about
  150 times.** Rayon's `join` has two arms: from a rayon worker it runs
  inline on a spin latch with no syscall, and from any other thread it
  injects the job and blocks on a mutex and condvar. Every forward was
  driven from a thread rayon did not own, at roughly five regions per
  layer. A profile put **74% of the token** in `__psynch_cvwait` on the
  driving thread, against 6.6% of samples in the actual matvec kernel.
  Interleaved within-process ratios: **135M +29%, 3B +9%, 8B +3%**,
  prefill flat (#128, #167).

  Note what this corrects. #128 had computed scheduling at 6.7% of a
  token and ruled it out. The arithmetic was right and **the denominator
  was stale**, taken against a 17 ms token before the repack fix in
  0.18.0 shrank it to about 5 ms. The ruled-out cause was the real one.

- **Every engine gets that, not just the generic decoder.** The five
  dedicated engines share the `Engine` trait, so the wrapper is written
  once as that trait's provided body and an engine supplies only its
  inner worker. Cold regions per decode step: Gemma-4 **100 to 1**, the
  BERT encoder **72 to 1**, Kimi and GLM-5.2 30 to 1, DeepSeek-V4 16 to
  1. A structural test refuses any engine that overrides the promoted
  entry point, so the seam cannot be bypassed silently (#169).

### Documented

- `benchmarks/RESULTS.md` is now the generated table and nothing else,
  291 lines to 81, with each model's prefill and decode on **one row**
  rather than ten rows apart. Nothing was re-benchmarked: `--render`
  reads the committed receipts, and the sorted multiset of every gap
  cell is identical before and after. The prose moved to
  `benchmarks/HISTORY.md` rather than being deleted, because the
  aarch64 rows and the before/after studies are measurements a generator
  cannot reproduce (#175).
- The 8.2x SmolLM2-135M row is marked **stale** rather than edited. It
  predates both the 0.18.0 repack fix and #167; on an M2 Pro it now
  reads about 1.9x. That is a different machine, so it is recorded as
  evidence the gap shrank rather than as a replacement number, and the
  row still needs a quiet Cortex-A725 (#168).
- `frink bench` does not use the greedy fold, so #172 cannot move the
  published Metal rows. Written down because it looks like it should
  (#173), along with a correction: the fold has **two** callers, not
  one, and the conclusion rests on neither being on the bench path
  (#174).
- **An inverted quant claim is corrected.** `body_quant`'s doc used
  `Llama-3.2-1B-Instruct-IQ4_XS.gguf` to teach that a filename is not a
  quantization, and said the file holds 96 `IQ4_NL` tensors and no
  IQ4_XS. The count was right and the type was backwards: it holds 96
  IQ4_XS, 16 Q5_K, one Q6_K and zero IQ4_NL. That inversion crosses the
  line that decides a verdict, because ggml declares `vec_dot_type =
  Q8_K` for IQ4_XS and `Q8_0` for IQ4_NL, so the comment described a
  file whose DRIFT would be unexplained while the real file's DRIFT is
  the expected case. The tool's verdict was always right; only the
  explanation was wrong. A new test pins the two look-alike neighbours
  to their ggml facts, because the existing one walked whatever the
  lists happened to contain and stayed green under the inversion (#176).
- The file-size table in `CLAUDE.md` was re-measured. One of five files
  shrank, so the rule still lost on balance, and the real wins left the
  table entirely: `repack.rs` 6446 lines to a directory of ten,
  `sampling.rs` 1569 to 894, `mul_mm.rs` to 865. Every one of those
  splits happened because somebody was about to add to the file and
  split it first (#165).

## [0.18.0] - 2026-09-09

### Added

- **The four missing samplers**, so the chain is llama.cpp's full nine
  steps in upstream's order: `dry`, `xtc`, `typ_p` and `top_n_sigma`
  join `penalties`, `top_k`, `top_p`, `min_p` and `temperature`. Golden
  values were read out of `libllama` rather than reasoned about. Every
  new step is a no-op at its neutral value, so a default run is
  unchanged. `mirostat` and `infill` remain refused by name, and the
  reason mirostat is refused is written down: upstream *replaces* the
  chain with it and it carries per-sequence state frink has nowhere to
  put (#160).
- **`frink gguf-split`**, a port of `llama-gguf-split`: split by tensor
  count or by size, merge, `--dry-run`, the same shard names and the
  same `split.*` metadata keys. Cross-checked against the real tool,
  which produced 6 of 6 shards of identical size, and each tool merges
  the other's output (#154).
- **Three more architectures run with evidence**: `gemma`,
  `hunyuan-dense` and `ernie4_5-moe` at step 1, each with a
  libllama-golden fixture. Audited 23 to 26, unaudited refusals 34 to
  31, and **the fixture-away class is now empty**: every row that needed
  only evidence has it, so everything left needs code (#161).
- **`frink quantize` writes Q5_K_M and Q6_K**, byte-identically (#162).
- **CUDA gains Q2_K, Q3_K, IQ4_NL, IQ4_XS and MXFP4**, each with both a
  matvec and a GEMM, since landing half of a kind is forbidden. Verified
  on the host across 11 kinds, 33 shapes and 75,042 positions with zero
  mismatches. **None has run on a GPU** (#157).
- **AVX2 GEMMs for all five interleaved repack kinds on x86**, with one
  per-workload dispatch rule shared by ten call sites. Verified by
  execution on real AVX2 silicon, not emulation, and **not yet
  benchmarked** (#159).

### Fixed

- **Q4_K quantization was not byte-identical, and the documented reason
  it "could never be" was wrong.** llama.cpp's `sumlx += w*x[i]*l` is
  contracted by its compiler into a single fused multiply-add; Rust does
  not contract, so the strict transcription that shipped was the defect.
  One unit in the last place flips a comparison and rewrites a whole
  super-block, which is why 1.15% of super-blocks differed rather than a
  rounding-sized fraction. Spelling the fusion as `mul_add` takes Q4_K,
  Q5_K and Q6_K to **zero differing super-blocks across all 147 tensors**
  of a real model (#162).
- **The int-dot matvec repacked every weight matrix on every call.** It
  passed a hand-written "uncacheable" identity where every other matvec
  passed a real cache key, so the interleaved layout was rebuilt and
  copied per token. It was 89% to 90% of decode work on Q8_0 and Q4_0
  models. This also corrects the premise of #128: the cost is
  proportional to gate and up projection bytes rather than fixed, and
  fires only on those two formats (#155).
- **`FRINK_CUDA=0` did not mean CPU.** The matvec launcher never
  honoured the disable flag, on the strength of a comment claiming CUDA
  needed no guard because launchers return an error with no device.
  That is true of Metal and false of CUDA, where the binding panics, so
  any quantized matvec on a CUDA build without a driver aborted the
  process (#157).
- **The CUDA host-check harness had silently stopped compiling** when
  the `float4` inner loop landed, so it verified nothing while still
  exiting green-adjacent. It now iterates the kind table rather than a
  hand-kept list (#157).
- **`frink gguf-split` was unreachable**: the CLI module existed but
  was never registered, so the subcommand would have fallen through to
  an implicit `frink run` and started generating text (#154).
- Three pre-existing test races on a process-global override, which
  passed only because the two halves of the int-dot tier used to move
  together (#159).

### Changed

- **The CPU scheduler is chosen by work size**, not by
  `FRINK_CPU_POOL`. One predicate decides per operation and the
  environment variable is now an A/B override. The crossover constant is
  **bracketed by the published measurements rather than swept**, and no
  quiet-host before-and-after has been run, so this is not yet a
  performance claim (#155).
- **The repack cache has a derived byte budget with eviction.** The fix
  above retains the packed copy, which cost +527 MB at 1.1B on Q8_0 and
  would scale per expert on a mixture-of-experts model. The budget is
  available memory minus a shared headroom constant minus committed
  expert bytes, then a quarter share, so it spends the same pool as
  `expert_store` rather than opening a second one. Zero disables the
  cache and reproduces the previous behaviour exactly (#158).
- **Metal encodes less per token**: 13% fewer dispatches and 9% fewer
  barriers, by fusing RoPE for Q and K, folding the K and V cache append
  into one grid, and deleting a barrier that guarded zero work on models
  without QKV bias or QK-norm (#156).

### Measured

- **The Metal suite was re-measured on a quiet M2 Pro** and the stale
  0.13.3 rows retired. Prefill spans **1.01× to 1.10×** and decode
  **0.64× to 1.11×**, with **12 of 14 comparable decode rows faster than
  llama.cpp**. MoE decode on OLMoE moved from ~1.41× to **0.96×**.
  Gemma-2-2B at 1.11× is the worst row, confirming the 1.12× that the
  concurrent-encode work predicted. Mistral-7B leaves the table because
  `--fit-host` refuses it at ~10 GiB needed against 11.2 GiB free, and a
  run from swap measures the swap (#163).

### Retired hypotheses

Both of these were the stated cause of an open issue, and both are now
disproven by measurement rather than argument.

- **Metal host cost is not dispatch count.** Removing 11.5% of encode
  operations bought **2.3%** of host time, and barriers were already
  hazard-driven rather than per-operation. What remains is per-dispatch
  argument binding: roughly 2400 encoder calls per token against 418
  dispatches and barriers combined (#149, #156).
- **The CPU per-token cost is not fixed.** See the repack fix above
  (#128, #155).

## [0.17.1] - 2026-09-04 

### Fixed

- **A character split across two tokens was destroyed.** The decode
  loop resolved UTF-8 one token at a time, so any character whose
  encoding spanned a token boundary became two U+FFFD before a caller
  saw the bytes. A DeepSeek answer ending in an emoji rendered as
  `today? \u{fffd}\u{fffd}`; the same applies to CJK text and every
  byte-fallback token. Both the streamed and the buffered paths, and
  each batched row now buffers its own tail (#124).

### Documented

- `FRINK_CPU_POOL` is measured rather than "unmeasured": on a quiet
  20-core aarch64 host the persistent pool is **+123% at 3B and +87% at
  8B**, which takes decode past llama.cpp (23.14 vs 17.86, 12.41 vs
  9.06). It stays opt-in because at 135M it is 37% slower,
  reproducibly and on quiet hosts (#27).
- `benchmarks/RESULTS.md` gains a second host and three caveats the
  single-laptop table could not show: prefill on server aarch64 is
  ~3x FASTER than llama.cpp; the published `1.41x to 5.06x` CPU range
  is aarch64-only and x86 looks far worse (#127); and the `cpu` rows
  may include Metal, because `--n-gpu-layers 0` does not force CPU
  (#126).
- `benchmarks/README.md` records the discipline lesson that cost this
  session a day of numbers: **a load average cannot see one busy
  core.**

### Added

- The batch GEMM tests now run through the int-dot kernels they gate,
  under both `ForceIntDot` settings, plus sub-tile shapes that bypass
  the repack path entirely.

## [0.17.0] - 2026-09-04

### Added

- **Seven more architectures run with evidence**, each with a
  libllama-golden fixture: `internlm2`, `xverse`, `ernie4_5`,
  `baichuan`, `exaone`, `bailingmoe2` (Ling-2.0) and `plamo3`. The
  audited count moves 16 to 23; unaudited refusals 41 to 34.
- `frink perplexity` — the quality axis nothing in the repo measured.
- `frink quantize` writes **Q4_K_S and Q4_K_M**, with a sub-block
  probe for llama.cpp encoder parity. Byte-identity is documented as
  the wrong bar for a K-quant; perplexity is the bar it is held to.
- A caller-supplied **sampler order** is honoured, and the samplers
  frink lacks are refused by name rather than ignored.
- **Sliding-window KV eviction** (`FRINK_KV_WINDOW`, off by default):
  a windowed layer's CPU cache drops rows behind its window. Gemma-3-4B
  at 32k context falls from 9.13 GiB to 1.69 GiB.
- `frink parity` gained a repeatable `--dumper` and `--dump-logits`,
  so a verdict can be taken against more than one reference build.
- The rerank route runs the GGUF's **pooler** when the file carries one,
  and reports which scale a score is on.

### Fixed

- **The parity oracle's WRONG line was a property of the reference
  build, not of frink.** It is now measured per checkpoint as
  `max(KL_WRONG, spread)` against the *nearest* reference. Three tuned
  constants were deleted and none added; no threshold moved. With one
  reference, a Q8_K-dotted checkpoint gets no WRONG line at all rather
  than a guessed one.
- Phi-4 applied LongRoPE context **after** the decode load, so the
  parity run measured a model configured differently from the one that
  answered.
- Metal Q5_0 MoE prefill on Qwen1.5-MoE mixed quant planes.
- `chatglm` was triaged FIXTURE-AWAY; it needs the fused QKV bias, and
  the triage now says so.
- The KV budget takes residency from the store instead of restating the
  rule — the repo's dominant bug shape, removed at one more site.
- **A batched row reported token counts and no rates at all.**
  Continuous batching built its `Usage` without timings, so every rate
  and duration came back null — and batching is the default on Metal,
  so Frink Studio's tok/s and duration columns were blank for every
  answer a Mac produced (#116).
- **Frink Studio dropped `reasoning_content`.** The server streams a
  reasoning model's thinking correctly; the client read only `content`,
  so an R1 distill looked like a dead stream and an answer that spent
  its whole budget thinking rendered as an empty message under a stat
  line reporting 512 decoded tokens. Thinking is now shown collapsed
  above the answer, and still never replayed as context (#118).

## [0.16.0] - 2026-09-02

### Added

- `-hf user/repo:QUANT`, llama.cpp's one-command model fetch, and `-d`
  to load a draft model so speculative decoding runs on a real
  checkpoint pair.
- `frink quantize` writes Q8_0 (byte-identical to llama.cpp, 272/272
  tensors) and refuses every other target by name.
- A forced tool call in eight of eleven wire formats.
- A persistent CPU worker pool behind `FRINK_CPU_POOL`.
- The llama.cpp server flags a copied command line actually carries.

### Fixed

- **`/v1/rerank` shipped broken**: it could not load a reranker at all,
  and ranked the answering document last.
- **Sampling penalties never saw the prompt.** Five call sites gave four
  different answers about the penalty window; one type decides it now,
  and the HTTP API penalises what llama-server penalises.
- `max_tokens` from an HTTP body could reach
  `Vec::with_capacity(usize::MAX)` from a single unauthenticated POST.
- A partial answer had a spelling that reached the response cache.
- The cache key is built from an **exhaustive destructure** of
  `GenerationParams`, so a new field cannot be silently dropped.
- Gemma-3 sliding-window layers roped unscaled; Gemma 27B takes
  llama.cpp's attention scale; the KV budget priced a window cap no
  store implements.
- A short `tokenizer.ggml.scores` array loaded and then panicked once
  per request.

## [0.15.3] - 2026-09-02

- Five one-match-arm architectures now run, and a dead gate is no longer
  mistaken for coverage.
- The batched prefill re-ran the whole prompt on top of the prefix it
  had just adopted.
- Host K/V stays authoritative for Metal continuous-batching prefill.

## [0.15.2] - 2026-09-02

- A second GGUF can be the drafter, which is what makes speculation
  worth running.
- Incremental token streaming under continuous batching.
- The reranker head gets a route; `response_format: json_schema` is
  served rather than refused.

## [0.15.1] - 2026-09-02

- **Release plumbing.** `frink-vulkan` must be publishable, because
  `frink-core` depends on it — this is what left 0.15.0 half-published.
  The dry run's "blocked by ordering" detector matched only one of the
  two ways Cargo says it, which is the same defect twice.
- The startup banner promised a KV dtype the run does not keep.

## [0.15.0] - 2026-09-02

### Added

- **BGE embeddings end to end**, checked against llama.cpp with a
  calibrated threshold; an encoder-only checkpoint can be the loaded
  model; the reranker classification head, ahead of its route.
- **Vulkan is a third backend**, and the registry list is generated
  rather than hand-kept.

### Fixed

- GGUF bounds allocations sized by untrusted header counts, and bounds
  array length and nesting depth (#25).
- The repack cache served a dead mapping's bytes, because a bool cannot
  say "still alive".

## [0.14.0] - 2026-09-01

### Added

- **Grammar-constrained decoding**: a GBNF engine ported from llama.cpp
  (parser and stack machine), JSON Schema to GBNF with everything
  unported refused by name, lazy grammars, `--grammar`,
  `--grammar-file`, `--json-schema`, and `tool_choice` required/named.
- **WordPiece tokenizer**, byte-exact against llama.cpp on a real BGE
  checkpoint — and putting it in the oracle showed the reference was the
  thing that was wrong.
- llama.cpp's native `POST /completion`, with four copies of the decode
  setup collapsed to one.
- A batched quantized CUDA GEMM, reached from batched prefill, closing
  the last silent CPU fallback.
- All 47 unaudited architectures **triaged**: the refusal now says which
  of three things is missing.

### Fixed

- `logit_bias` and JSON mode were both silently dropped, in four places
  between them; `/v1/completions` dropped four sampler fields.
- The attention-softcap refusal gated on a GGUF key no converter writes,
  so it could never fire — a gate that cannot fire reads as coverage.
- Metal Q5_0 decode ran on the CPU while its prefill ran on the GPU.
- The cross-target gate could not see Linux, which is what broke two
  releases.

## [0.13.3] - 2026-08-28

Chat-template and tokenizer correctness: Yi leaked its turn marker as
text, R1 distills lost their reasoning, a base model answered correctly
and then talked to itself for 512 tokens, one hardcoded regex was
pre-tokenizing every BPE checkpoint, and olmo's pre-tokenizer ends where
gpt2's does not. **The generic architecture path became opt-in**, because
a guess that loads is worse than a refusal.

## [0.13.0] - 2026-08-28

- **Four checkpoints loaded clean and computed the wrong thing.**
- The Metal MoE decode stack ignored four features its dense twin
  implements; Metal paged prefill kept the KV and handed back zeros, so
  two caches read a prompt the model never saw.
- The radix cache never gave a page back, so the pool drained until
  admission refused.
- `frink download`, so fetching a model needs no Python.
- CPU: swiglu/geglu spent a libm call per element; the i8mm feature
  probe ran 131k times per GEMM.

## [0.12.0] - 2026-08-27

Serving and bench hardening: two routes never matched, the paged-KV
guard missed the common way to ask for a GPU, the decode guard refused
every GPU run, and a suite that measured nothing could still republish
the ledger.

## [0.11.0] - 2026-08-24

- `frink serve` behind an optional, default-off `serve` feature.
- A compile-time assertion that backend features reach the server.
- 0.11.1 fixed the publish order and built the shipped binary with
  `serve`.

## [0.10.0] - 2026-08-24

- **Speculative decoding**: lossless verification, a `Drafter` trait,
  warm-cache resume, and acceptance metrics on `usage` and
  `/admin/stats`.
- Opt-in **resumable SSE streams** with a replay buffer and a JSON
  polling fallback, consumed by the UI.
- The bench asserts the engine *answered* the same, not just that it was
  asked the same.

## [0.9.0] - 2026-08-21

- **24 architectures rotated the wrong RoPE pairs**, plus MoE routing
  bias — the largest single correctness fix in the project.
- Stop sequences in two layers shared by both decode paths; stopping on
  the whole EOG set rather than `eos_token_id` alone.
- Batched requests admitted on an integer KV block budget and cancellable
  at a step boundary.
- Frink Studio rebuilt on React, Tailwind and assistant-ui.

## [0.8.0] - 2026-08-20

- **Published to crates.io** for the first time.
- A disk tier for KV prefix-cache blocks, read asynchronously and ahead
  of the request.
- The GGUF's own Jinja chat template is evaluated instead of sniffed.
- Two-tier cancellation for streamed generations.

## [0.7.0] - 2026-08-20

- `frink parity` — first-token distribution against llama.cpp. The
  oracle this project is now held to.
- The gpt-oss CPU graph, checked against llama.cpp.
- Frink Studio, three-state `/health`, `/admin`, request ids and
  per-phase usage timings, resumable chunked prefill.
- Exact pre-load KV budget arithmetic and a real per-backend device
  memory budget.

## [0.6.0] - 2026-08-18

- IQ2_XS / IQ2_S / IQ3_S / IQ1_M decode and mmap-resident load.
- Refuse checkpoints whose tensors this build never reads.
- Swappable active model and the `/admin` control surface.
- MoE layers run inside the fused Metal prefill stack.

## [0.5.0] - 2026-08-13

- F16 tensor loading; `frink verify --prompt` reaches prefill kernels;
  the `clippy -D warnings` gate restored on both feature sets.

## [0.4.0] - 2026-08-11

CPU quantization throughput: i8mm SMMLA tiers for Q8_0/Q4_0,
interleave-8 NEON kernels for Q4_K/Q5_K/Q6_K, NEON DotProd GEMV/GEMM for
Q6_K, one activation-quant pass shared across q/k/v and gate/up. 0.4.1
added simdgroup-MMA flash attention at d=128 and d=64, `frink verify`,
and a sealed kernel-lookup registry so a missing kernel is loud.

## [0.3.0] - 2026-08-10

Metal prefill rewritten around llama.cpp's `mul_mm`: a real simdgroup
GEMM for Q4_K extended to every quant kind, FA-vec prefill at d=64/96,
a batched dense FFN (4x on Metal pp512), and pooled scratch buffers.
`frink bench -m`, a `llama-bench` work-alike, landed here. Two silent
fallbacks were closed: batched prefill never touched the GPU, and
`--features cuda` never enabled CUDA in `frink-core`.

## [0.2.0] - 2026-08-06

Q8_0 KV cache in the Metal backend, a CUDA backend, and the first
benchmark ledger.

## [0.1.0] - 2026-08-05

First tag. GGUF mmap loader, quantized CPU kernels, Metal backend,
`frink` CLI and `frink-server`.

[0.22.0]: https://github.com/antonellof/frink/compare/v0.21.0...v0.22.0
[0.21.0]: https://github.com/antonellof/frink/compare/v0.20.0...v0.21.0
[0.20.0]: https://github.com/antonellof/frink/compare/v0.19.1...v0.20.0
[0.19.1]: https://github.com/antonellof/frink/compare/v0.19.0...v0.19.1
[0.19.0]: https://github.com/antonellof/frink/compare/v0.18.0...v0.19.0
[0.18.0]: https://github.com/antonellof/frink/compare/v0.17.1...v0.18.0
[0.17.1]: https://github.com/antonellof/frink/compare/v0.17.0...v0.17.1
[0.17.0]: https://github.com/antonellof/frink/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/antonellof/frink/compare/v0.15.3...v0.16.0
[0.15.3]: https://github.com/antonellof/frink/compare/v0.15.2...v0.15.3
[0.15.2]: https://github.com/antonellof/frink/compare/v0.15.1...v0.15.2
[0.15.1]: https://github.com/antonellof/frink/compare/v0.15.0...v0.15.1
[0.15.0]: https://github.com/antonellof/frink/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/antonellof/frink/compare/v0.13.3...v0.14.0
[0.13.3]: https://github.com/antonellof/frink/compare/v0.13.0...v0.13.3
[0.13.0]: https://github.com/antonellof/frink/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/antonellof/frink/compare/v0.11.1...v0.12.0
[0.11.0]: https://github.com/antonellof/frink/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/antonellof/frink/compare/v0.9.1...v0.10.0
[0.9.0]: https://github.com/antonellof/frink/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/antonellof/frink/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/antonellof/frink/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/antonellof/frink/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/antonellof/frink/compare/v0.4.1...v0.5.0
[0.4.0]: https://github.com/antonellof/frink/compare/v0.3.1...v0.4.0
[0.3.0]: https://github.com/antonellof/frink/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/antonellof/frink/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/antonellof/frink/releases/tag/v0.1.0
