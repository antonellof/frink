# A recurrent layer that does not come back to the host

Status, 2026-10-28: Bonsai-2-27B decode on the M2 Pro is **11.17
tok/s against the reference fork's 11.46 (97.5%)**, flat with context
(11.17 / 11.19 / 11.16 at 32 / 300 / 600 tokens), at 20 waits a token
with the attention on the device. Prefill is 43.1 against 66.8, and
its host side is finished. The remaining decode gap is about 3% in the
PTQ1_0 matvec's inner loop; "The limit" below prices it.

The chunked rule landed for prefill, and then took the device with
it. `frink_core::gdn_chunk` is 2.1x on the step alone, and
`frink_metal::gdn_chunk` is 3.0x on top of THAT at prefill lengths, so
a batch's recurrence now runs on the GPU after three attempts that lost.
The row-at-a-time kernels in `frink-metal/src/gdn.rs` are still not
wired, and the three measurements below still say why: the thing that
changed is not the kernel, it is what the kernel is given to do.

## The measurement this plan exists for

Bonsai-2-27B on an M2 Pro, `frink bench -p 128 -n 32`, decode at 7.7
tok/s against PrismML's llama.cpp fork at 11.5, measured back to back on
a quiet box. Every Metal submission
is timed (`FRINK_METAL_GPU_TIMING=1`), so the token accounts for
itself:

| per decode token | |
|---|---|
| command buffers | 159 |
| GPU, summed | 66 ms |
| submission overhead beyond that GPU time | 34 ms |
| host compute | 41 ms |
| wall | 141 ms (130 ms on a quiet box) |

A `sample` of the decode thread agrees from the other side: **83% of it
sits in `waitUntilCompleted`**. The fork's token is 87 ms and it encodes
ONE graph. So the gap is not kernel speed -- 66 ms of GPU for 5.95 GB of
weights is ~90 GB/s of a 200 GB/s machine, and the trit decode is
arithmetic-bound in both engines -- it is that a Frink layer returns to
the host about three times.

## What a recurrent layer costs today

Per gated delta-net layer, at one token:

1. `qkv` + `z` projections, one submission, result downloaded.
2. conv step, SiLU, per-head l2 norms, **the delta rule**, the gated
   output norm: all host.
3. `ssm_out`, one submission, result downloaded.

Plus the FFN's one submission. 48 recurrent layers, 16 attention layers
and 64 FFNs is the 159.

## What landed

`frink-metal/src/gdn.rs`:

- `gdn_delta_step`: one threadgroup per value head, one thread per state
  row; decay in place, predict `k`, scale the error by beta, rank-one
  update, read out with the scaled query. Pinned against
  `frink_core::gdn::delta_step` on three shapes including Bonsai's
  (48 heads, 128 wide) by
  `gdn::tests::the_device_delta_step_matches_this_one`; swapping the
  head map in the kernel turns it red.
- `gdn_gated_norm`: `rms_norm(o, weight) * silu(z)` per head.
- `launch_gdn_tail`: delta step, gated norm, the folded rotation and
  `ssm_out` in ONE command buffer.

## Why the decode step is not the kernel, measured at last

The three losses below were only ever measured END TO END, where they
read as "the kernel is not better than six cores" without saying why.
Measured on its own, at one row, Bonsai's shape, state wrapped in place
and nothing copied (`gdn_chunk::tests::device_chunk_against_host_throughput`):

| rows | host | device | |
|---|---|---|---|
| 1 | 0.26 ms | 1.01 ms | 0.25x |
| 2 | 0.58 ms | 0.95 ms | 0.61x |
| 8 | 0.81 ms | 1.08 ms | 0.75x |
| 32 | 3.47 ms | 1.45 ms | 2.40x |

One row and eight rows cost the DEVICE the same, so what it is paying
is not the recurrence: it is about 0.9 ms of fixed submission and
buffer setup, and the kernel is the small part. That is why no faster
recurrence kernel can win a decode token, and it is the measurement
that turns the three rows below from "the GPU is not better at this"
into "a decode token cannot afford a submission per layer". The fix is
the one the arithmetic at the end of this file already names -- fewer
command buffers, not a better kernel -- and `DEVICE_ROWS = 32` in
`frink_core::gdn_chunk` is where the two sides cross.

The access pattern was still wrong, and fixing it is what made the
prefill numbers above: `gdn_delta_step` gave each thread a whole state
ROW and walked it, so adjacent threads touched addresses `head_dim`
floats apart and every 128-byte transaction carried 4 useful bytes.
`GDN_DELTA_STEP_COALESCED_SRC` replaces it -- one threadgroup per
`(head, row)`, one thread per COLUMN, the dot products as threadgroup
reductions -- and is what `encode_delta_step` now encodes, so there is
one kernel rather than two that would have to agree. A head width that
is not a power of two is refused there, because the reduction halves
its stride from it.

## Why it is not wired: three measurements, not one

Wired into `Gdn::forward_rows`, the fused tail is slower every way it
has been built, against a host baseline of **7.3 to 7.4 tok/s**:

| how the state travels | tg32 |
|---|---|
| copied to the device and back each token | 6.0 |
| wrapped in place, `newBufferWithBytesNoCopy` | 6.6 |
| wrapped in place, the wrapper cached by address | 6.9 |

The first says what everyone expects: Bonsai's state is
`48 heads x 128 x 128 x 4 B = 3.1 MB` per layer, so copying it both ways
is 300 MB a token, more traffic than the whole 5.95 GB weight read.

The second and third are the interesting ones. `RecurrentState::ssm` is
page-aligned (`frink_core::recurrent_state::AlignedF32`) precisely so
that Metal can wrap the host's own bytes with no copy at all, and
wrapping them still costs: mapping host pages for the GPU is not free,
which the cached wrapper then removes. With BOTH of those gone the fused
tail is STILL 6% behind the host recurrence.

So the remaining difference is the kernel, not the plumbing: one
threadgroup per head streaming a 128x128 state through registers is
simply not better than six CPU cores doing the same reduction out of
cache, when the layer's own work is only 3 MFLOP.

## What would close it

The answer is not "move this loop to the GPU", and it is not a faster
kernel either. `delta_step_throughput_probe` measures the host step at
**36 GB/s of state**, which is the floor for reading and writing 3.1 MB
once per row: rewriting its two passes as one read pass plus a
streaming update -- algebraically exact, because `S_new . q` expands to
`decay (S_old . q) + d (k . q)` -- measured 197 us against 175, since
both passes already hit the same 512-byte row while it is hot in L1.

The prize is therefore sized, and it is the largest single one left.
A 128-token prefill moves `128 rows x 48 layers x 6.2 MB = 38 GB` of
state at that 36 GB/s, which is **1.05 s of a 3.9 s `pp128` run**, and
the profile agrees (23% of samples). The CHUNKED delta rule llama.cpp
uses (`delta-net-base.cpp`) processes a block of C rows together with
matrix products, so the state is read and written once per CHUNK rather
than once per row: at C = 16 that 1.05 s becomes about 0.07 s, which is
`pp128` 32.9 -> roughly 44 tok/s on arithmetic alone. That is a different
algorithm with its own correctness story (a WY-style representation of
the rank-one updates), and it is the honest next step. Everything below
is what the plumbing around it should look like.

## The decode token, priced

Measured on the real checkpoint, `FRINK_METAL_GPU_TIMING=1`, 24 decode
tokens after a short prompt. The submissions are two `matvec-fused` and
one `dense-ffn` per layer, 64 layers:

| per decode token | count | GPU each | latency each | GPU | latency |
|---|---|---|---|---|---|
| `matvec-fused` | 128 | 0.185 ms | 0.152 ms | 23.7 ms | 19.5 ms |
| `dense-ffn` | 64 | 0.748 ms | 0.162 ms | 47.9 ms | 10.4 ms |
| | **192** | | | **71.6 ms** | **29.9 ms** |

Host work is the remaining ~35 ms of a 137 ms token, and a `sample`
puts two thirds of it in `delta_step`.

Three things follow, and the third is why no increment closes this.

**The GPU work is already ahead.** 71.6 ms for a 5.95 GB weight read is
83 GB/s; the reference's WHOLE token is 87 ms, so its GPU cannot be
under about 68 GB/s. Nothing in the kernels is the gap.

**The latency per submission is irreducible.** 0.15 ms is the OS
wake-up from `waitUntilCompleted`, and every way around it has been
measured and lost: spin-then-block (3.7 against 7.1, polling takes the
core the host work needs) and
`commandBufferWithUnretainedReferences` (7.00 against 7.1). So the only
lever on 29.9 ms is the COUNT.

**And the count only falls to one per layer if the host has nothing to
do inside a layer.** Fusing the tail -- output projection, residual,
FFN norm, FFN, residual -- into the FFN's own command buffer removes
64 submissions, which is 64 x 0.152 = 9.7 ms, or 137 ms to 127: **7.3
to 7.9 tok/s**. Making the host recurrence hit 60 GB/s instead of the
25.8 it measures would be about 7 ms more, so **8.3**. Both together do
not reach 11.5, and each one adds a branch to the hottest, most
decorated code in the repo -- `ffn_block_row` alone applies a parallel
sum scale, a down scale, a post-FFN norm, a residual scale, a skip
stream and two FFN-input shapes, and this file's own history is eight
model features lost one at a time to exactly that kind of branch.

So the honest ceiling on increments is ~8.3 tok/s, and parity needs the
whole layer in one command buffer with NO host step inside it: the QKV
projection, the conv and gates, the recurrence, the output projection,
the residual, the norm and the FFN, with the hidden state resident
across layers. That is `frink-metal/src/decode_dense.rs` -- which
already does exactly this for dense models -- extended to PTQ1_0
weights, to the folded Hadamard rotation between matmuls, and to the
gated delta-net block. At one submission per layer the arithmetic is
64 x 0.15 = 9.6 ms of latency, 71.6 ms of GPU plus about 5 ms for the
recurrence on device, and no host time: **about 86 ms, which is the
reference's 87.**

That is the project. It is three pieces, each with its own correctness
story, and none of them is a faster kernel.

## The host recurrence is gone, and what that bought

`frink-metal/src/gdn_branch.rs` runs a recurrent layer's WHOLE branch
in the submission `ssm_out` already cost: the two gates, the causal
convolution with its SiLU, the per-head l2 norms
(`frink-metal/src/gdn_head.rs`), the delta rule, the gated norm, the
folded rotation and the output projection. A `sample` of a decode run
no longer has `delta_step` in it at all, where it had been two thirds
of the host time.

Interleaved A/B on one build, Bonsai, 220 decode tokens:

    device branch  7.27 / 7.27 / 7.30 / 7.26 tok/s
    host branch    7.13 / 7.10

**+2.2%, and that is the honest surprise**: removing what the profile
said was two thirds of the host time is worth 3 ms of a 137 ms token,
because a decode thread spends 83% of itself in `waitUntilCompleted`
and most of the host work was already inside somebody else's wait. The
profile named the biggest host cost correctly and the ledger says host
cost was never the gap.

Getting there took three findings, each worth more than the 2.2%:

- **`Q5_0` and `PTQ1_0` were unreachable from every fused Metal path in
  `frink-models`.** `frink_metal::gpu::MATVEC_KINDS` served both and
  a hand-written match in the decoder listed six kinds and neither.
  `QuantKind::metal_kind_name` is exhaustive with no `_` arm now, and
  `crate::metal_launch` asks the backend's own table by that name, so
  the two cannot drift again. This is the repo's dominant bug shape,
  found for the fifth time.
- **A launch that allocates its own scratch pays for it per call, and
  the GPU ledger cannot see that.** The first version of this branch
  ran 7.22 to 6.68 -- SLOWER -- while the ledger accounted for only 10
  ms of the 19 it had lost, because it made eleven fresh Metal buffers
  per layer per token, some 500 allocations a token. Resident
  constants, the convolution window wrapped in place, and
  `frink-metal/src/scratch_pool.rs` took it 6.68 to 6.98 to 7.11 to
  7.27.
- **`RecurrentState::conv` was not page-aligned** while `ssm` was, so
  the one state small enough to seem harmless was the one being copied
  both ways per layer per token.

## What the decode token looks like now

GPU 79 ms, submission latency 35 ms, host about 20 ms -- and the host
that is left is the SIXTEEN attention layers' row kernel, not the 48
recurrent ones. The token is submission-bound, and the count is
unchanged at 192 because this branch rides in a submission that already
existed.

So the next step is the one the arithmetic named before any of this was
built, and it is now the ONLY step: merge a layer's three submissions
into one. The branch ends with `ssm_out` and the FFN begins with a
residual add and a norm, with nothing host-side between them; the
pieces are all `encode_` functions already. What stands in the way is
not a kernel, it is that the FFN's weights live in the decoder's layer
loop while the branch's live in `Gdn`, so the fusion has to happen at a
call site that owns both. At one submission per layer the arithmetic is
64 x 0.15 = 9.6 ms of latency against today's 35, no host step, and
about 86 ms a token: the reference's 87.

## A recurrent layer is ONE submission

`frink-metal/src/gdn_branch.rs` now encodes a whole recurrent layer in
one command buffer: `attn_norm`, the four projections, the two gates,
the causal convolution, the l2 norms, the delta rule, the gated norm,
the folded rotation, `ssm_out`, the residual add, `ffn_norm`, the
SwiGLU FFN and the second residual add. `crate::fused_layer` says what
the layer's weights have to be and `decoder::fused_recurrent` what the
model has to be; anything else takes the host bodies.

Bonsai, 220 decode tokens, one box:

| | tok/s | submissions/token | latency |
|---|---|---|---|
| host branch | 7.10 | 192 | 35 ms |
| device branch | 7.27 | 192 | 35 ms |
| + FFN fused | 7.95 | 144 | 24.7 ms |
| + head fused | **8.93** | **97** | **16.7 ms** |

Parity MATCH at every step (KL 2.203e-5 on the 5-token path, and the
256-token GEMM path unchanged).

The ordering that matters and is easy to get wrong: the two gate
projections are stored UNFOLDED and `attn_qkv` / `attn_gate` share one
fold, so the gates must read `attn_norm(x)` BEFORE the rotation
rewrites that buffer in place. Moving them after it left the fused-layer
test green until the test was given a real fold -- with unfolded
weights the order is unobservable, and Bonsai's are folded. That is the
second time in this file's history that a sabotage passing meant the
TEST was wrong rather than the code right.

## A RUN of layers, one wait

Nothing required the host to wait per layer. Consecutive recurrent
layers hand each other a residual stream the host never looks at, and
one Metal queue is ordered, so `frink_metal::gdn_branch::GdnRun`
commits them back to back against ONE device buffer and waits for the
last. Qwen3.5 puts a full-attention layer every fourth, so the runs are
three layers long and three waits become one.

What it costs is care with the pool: a command buffer that has not been
waited for is still going to read its scratch, so every layer's scratch
is held by the run until `finish` returns.

Bonsai `tg32`, the reference's own shape, interleaved on one box:

| | tok/s | waits/token |
|---|---|---|
| session start | 9.49 | 101 |
| whole layer fused | 9.49 | 101 |
| + runs of three | **10.12** | **69** |
| PrismML fork (`llama-bench`) | 11.45 - 11.54 | (one graph) |

And the DIAGNOSIS has moved with it. The ledger now reads GPU 88.8 ms
against a 98.8 ms token, so submission overhead is about 10 ms of it
and the reference's WHOLE token is 86.7 ms. Our GPU work is no longer
faster than the reference's token; it IS the gap. Every remaining
submission could vanish and this engine would sit at 88.8 ms, or 11.3
tok/s, which is the first time that number has been below the
reference's.

So the plan's whole premise -- "the gap is the number of submissions"
-- was right for three rounds and is now spent. What is left is kernel
throughput: 88.8 ms for a 5.95 GB weight read is 67 GB/s of a 200 GB/s
part, and a recurrent layer is 1.31 ms of it. Inside that layer are
some twelve small dispatches (two gates, the convolution, two l2 norms,
the delta step, the gated norm, two folds, two adds, two norms) whose
per-dispatch cost measures at about 8 microseconds, or 0.1 ms a layer
and 5 ms a token; the rest is the PTQ1_0 matvecs, which
`gemm_throughput_probe` and the six measured non-results below say are
close to what this kernel shape gives.

## The attention layers' tail, fused

The sixteen attention layers cost three submissions each while the
forty-eight recurrent ones cost one, and two of those three are `wo`
and the FFN with nothing but a vector add and a norm between them. They
are one now (`frink_metal::gdn_branch::launch_attn_tail`,
`crate::decoder::fused_attention`), which is 51 waits a token instead
of 69: **10.29 to 10.59 tok/s**, interleaved on one build, parity
MATCH. The attention itself still runs on the host, because the KV
lives there.

Two things made it cheap. `encode_ffn_tail` was extracted from the
recurrent layer's encoder first, so the two layer kinds cannot drift
about what a layer tail is; and `attn_block` gained an `AttnTail`
parameter rather than a second copy of itself, so the deferring path
gets the QKV projections, the biases, the two QK norms, RoPE, the
scale, the temperature, the KV push, the attend and the gates
identically -- a second copy of that body is how this file lost eight
model features.

It also cost a bug worth recording: `fused_attention_tail_eligible`
asked `fused_attention_refusals(..).is_none()` where the function
returns `Some(())` to mean "nothing refuses". The predicate was
inverted, every layer declined, and the ledger showed no `attn-tail`
label at all -- which is the only reason it was caught, because the
fallback is correct and the model ran fine.

## The decode attention was serial

`causal_gqa_attention_row` looped `for h in 0..n_heads` on one core.
On a hybrid that loop is the WHOLE of what a decode token still does on
the host -- sixteen attention layers of twenty-four heads each -- and
the heads share nothing: each reads its own slice of `q`, its own KV
group, and writes its own slice of `out`. It is `crate::par::chunks_mut`
over `out` in `v_head_dim` runs now.

Interleaved at a 300-token context, which is where a decode actually
lives:

    parallel  10.19 / 10.19 tok/s
    serial     9.44

**+7.9%**, and the gap grows with the context, because the loop's work
is linear in `seq_len` while the fork is not. At `tg32` -- 33 keys a
head, the smallest context anyone measures -- it reads 10.33 against
10.59, which is inside this box's spread and the wrong shape to tune
for: a tg32 token spends more time forking than attending, and a real
one does not.

## The attention tail rides in the next run

A hybrid alternates three recurrent layers and one attention layer, and
the attention layer's tail FEEDS the next three: its output is the
residual stream they read, which the host never looks at. So it belongs
in their command buffer, not one of its own
(`GdnRun::attn_tail`). Fifteen of sixteen tails a token move that way;
the last has no recurrent layer behind it and keeps the standalone
launch.

Waits a token: 51 to 36. Interleaved at a 300-token context:

    10.61 / 10.50 tok/s   with
    10.29                 without

`encode_attn_tail` is the one encoder both paths share, and
`Decoder::attn_tail_launch` the one builder, so the standalone launch
and the run cannot disagree about the fold width or the refusals.

## The attention head rides in the previous run

The mirror of the tail. An attention layer reads `attn_norm(hidden)`
and projects Q, K and V from it, and `hidden` is what the recurrent run
before it just finished writing. So the run encodes that norm and those
three matvecs at its END and `finish_with_head` returns all four to the
host for its one wait (`GdnRun::attn_head`). The host still does the
attention itself -- the KV lives there -- but it starts with q, k and v
already in hand.

Waits a token: 36 to **20**. Interleaved:

    tg32   11.11 / 11.06 tok/s   with       10.69  without
    n300   10.95

`Decoder::project_qkv` is the projection split out of `attn_block`, so
the precomputed path and the ordinary one cannot compute it two ways;
`attn_block_tail` takes the three vectors when the run made them and
projects when it did not.

## The attention on the device, and decode stops sloping

The last host step in a decode token was the attention itself, because
the KV lived on the host. `KvCache::metal_attn` is the sequence's own
device mirror -- per-sequence, because two requests in flight have two
histories and a mirror hung off a shared `Decoder` would hand one the
other's keys. The host `k`/`v` stay authoritative, so truncation, the
prefix cache, slot files and every host reader are untouched; the
mirror is trusted only while its `seq_len` equals the cache's `rows()`
and re-uploads from the authority whenever it does not.

`launch_attn_layer` then does the whole layer in one command buffer --
the KV append, the attention, the sigmoid gate, `wo`, the residual, the
FFN norm, the FFN, the residual -- and `GdnRun::attn_layer` puts that
at the head of a run, so a four-layer group (one attention, three
recurrent) is ONE wait with nothing in it returning to the host.

Waits a token: 20, and the shape of the curve changed:

| tokens | before | after | reference |
|---|---|---|---|
| 32 | 11.06 - 11.17 | **11.17** | 11.46 |
| 300 | 10.95 | **11.19** | 11.50 |
| 600 | -- | **11.16** | -- |

It is FLAT now, which is the point: the reference was flat and this
engine sloped, and the slope was exactly a host attention whose work
grows with `seq_len`.

**The bug worth recording cost an hour and would have shipped.** The
first version encoded this on a CONCURRENT encoder, and
`encode_attn_tail` was written for one that orders its own dispatches.
The model generated fluent nonsense -- and `frink parity` stayed
MATCH, because parity reads the FIRST token, which is prefill, and this
path is decode. Nothing in the test suite covers a decode step against
a reference. What caught it was reading the generated text, and what
proves the fix is a greedy 100-token generation compared against the
same build with the device path off: byte-identical. That comparison
should be a test, and it is the gap this plan leaves behind.

## The limit, which is not where this plan assumed

Our GPU time for a decode token is **88.8 ms**. The reference's WHOLE
token is 86.7 to 87.3 ms. So:

    every submission removed, every host microsecond removed  ->  88.8 ms = 11.26 tok/s
    the reference                                                 86.7-87.3 ms = 11.45-11.54

**Scheduling cannot reach parity.** Not the fused branch, not the fused
layer, not the runs, not the attention layers, not one command buffer
for the whole token: the limit of all of it together is about 2% SHORT,
because the kernels themselves are about 2% slower than the reference's
and everything else is already overhead that can only go to zero.

That is worth stating plainly because this plan spent four rounds on
the premise that the gap was the submission count. It WAS, for 7.10 to
10.12 tok/s. It is not any more, and the table below prices what is
left of it precisely so nobody spends a fifth round finding that out.

| | waits | latency | token | tok/s |
|---|---|---|---|---|
| before the tail was fused | 69 | 11.7 ms | 100.5 ms | 10.29 |
| the attention TAIL fused (`wo` + residual + norm + FFN) | 51 | 8.7 ms | 97.5 ms | 10.59 |
| the tail riding in the next run | 36 | 6.1 ms | 94.9 ms | 10.6 - 10.8 |
| the head riding in the previous one | 20 | 3.4 ms | ~90 ms | 11.06 - 11.17 |
| **the attention on the device too, today** | **20** | **3.4 ms** | **~89 ms** | **11.16 - 11.19, flat** |
| the attention LAYER fused | 37 | 6.3 ms | 95.1 ms | 10.52 |
| the whole token as ONE run | 2 | 0.3 ms | 89.1 ms | 11.22 |
| **the floor, at today's kernels** | 0 | 0 | **88.8 ms** | **11.26** |

So the work that closes this is in the PTQ1_0 matvec's inner loop and
nowhere else, and it needs about 3% -- which is small, but three
hypotheses against it (a four-row prefetch, wider loads, eight rows a
threadgroup) measured NEUTRAL, and both concurrency schemes measured
flat or worse because one matvec already saturates the part. The next
idea has to be a different kernel, not a different schedule: the decode
is 42% of it, and the only untried shape is a lookup of the five trits
of a byte out of threadgroup memory rather than five `floor`s on the
float pipe. That has a recorded failure behind it (the FIRST version of
this kernel decoded with integer ops and reached 2.4 tok/s), so it would
have to be the table WITHOUT the rest of that version's shape.

## What the scheduling steps are worth

Waits per token are 69: sixteen `gdn-run` (three recurrent layers each),
thirty-five `matvec-fused` and eighteen `dense-ffn`, and forty-eight of
those sixty-nine belong to the SIXTEEN attention layers, which still
cost three submissions each. GPU is 88.8 ms of a 98 ms token and the
reference's whole token is 86.7 to 87.3.

So the arithmetic for the last step, and it is the last one:

| | waits | latency | token | tok/s |
|---|---|---|---|---|
| today | 69 | 11.7 ms | 98 ms | 10.1 |
| attention layers fused, one submission each | 37 | 6.3 ms | 93 ms | 10.8 |
| the whole token as ONE run | ~2 | 0.3 ms | 89 ms | 11.2 |

That last row is within noise of the reference, and it needs the
attention layer on the device end to end: a gated Q split from a
double-width `wq`, the per-head QK norm, RoPE, the attention itself
against a Metal-resident KV, the sigmoid gate, `wo`. `launch_decode_attn_block`
already does the shape of that for un-gated models and
`Decoder::metal_attn_view` answers `None` for a gate, which is the fence
that has to move. Nothing smaller closes the gap: every scheduling lever
is now measured, and the kernel's own inner loop has three measured
neutral results against it.

## What is left, and it is one thing

The token is now GPU 82.7 ms, latency 16.7 ms, host about 13 ms. The
host that remains is the SIXTEEN attention layers, which still cost
three submissions each (two `matvec-fused`, one `dense-ffn`) and run
their attention row on the CPU, because the fused Metal attention block
refuses them: Qwen3.5's Q is gated, and `Decoder::metal_attn_view`
answers `None` for a gate.

Fusing those the way the recurrent ones are now fused is 48 submissions
and about 13 ms of host: 97 to 65 submissions, 16.7 to 11 ms of
latency, and roughly 94 ms a token, or **10.6 tok/s**. Against the
reference's 87 ms the remainder would then be the GPU time itself,
82.7 ms for a 5.95 GB weight read, which is 72 GB/s of a 200 GB/s part
and the only number left that is not overhead.

## The plumbing, once the algorithm is right

The state has to live on the device across tokens, with the host copy
updated only when a host consumer actually reads it. The consumers are
the ones that make a recurrent cache different from an attention cache
(`frink_core::recurrent_state`): `KvCache::truncate` refuses a middle
position, the prefix cache does not store such a cache, `--model-draft`
refuses such a model, and a slot file writes it out. So the shape is:

1. `RecurrentState` gains a device mirror and a `dirty` side: written by
   the GPU path, read back by `as_host_slice()`.
2. Every host reader goes through that accessor, so a reader that
   forgets it does not compile rather than reading a stale state. This
   is the seam the repo's dominant bug shape demands: two copies of one
   fact, with something enforcing agreement.
3. Only then does `launch_gdn_tail` pay, and the conv step and the l2
   norms should join it, which makes a recurrent layer ONE submission
   from projection to `ssm_out`.

Arithmetic for the prize, at today's numbers: 48 layers x (one
submission at ~0.2 ms + the host recurrence at ~0.15 ms) is about 17 ms
of a 141 ms token, and fusing the FFN in behind a device-side residual
and norm is about 10 ms more. That is 8.8 tok/s, not 11.5: the rest is
the remaining submissions and the 66 ms of GPU itself, which is where
the fused decode stack (`frink-metal/src/decode_dense.rs`, already
serving dense models) would have to take over for PTQ1_0 and for folded
weights.

## The prefill surprise

The recurrence is **23% of a prefill step** (a `sample` of `pp512`-shaped
work), and a batch amortises the state copy over all its rows: upload
once, `rows` dispatches in one command buffer against one state buffer,
read back once. `launch_delta_step_rows` does exactly that and is pinned
against the host stepping the same rows in order
(`the_device_delta_rows_step_in_order`).

It measured `pp128` **32.7 -> 28.5 tok/s**. The state's working set is
why: 3.1 MB per layer stays in the CPU's shared cache across a batch's
rows, while the GPU re-reads and re-writes it from memory for every row,
which is 38 GB of traffic for a 128-token prefill. So the batched entry
is also kernels-without-a-caller until the state is resident AND the
rows are chunked so a block of them shares one pass over it.


## What reversed it

The three losses above share one cause, and it is not "the GPU is bad at
this": every one of them moved the state once per ROW. Bonsai's state is
3.1 MB a layer, so a row-at-a-time kernel reads and writes 38 GB for a
128-token prefill whatever it computes, and 3 MFLOP of work cannot hide
that. Chunking is what removes it -- the state is touched once per CHUNK
-- and only then is there anything for a GPU to be good AT.

`frink-metal/src/gdn_chunk.rs` is the same algebra as the host chunk:
one threadgroup per value head, one thread per state ROW, the `t` loop
kept sequential behind threadgroup barriers because that is the
recurrence, and the `j` loop spread across threads because it is not.
`CHUNK` is 16 there against the host's 32, since what bounds it is a
threadgroup's 32 KiB rather than cache, and the two agreeing about the
answer at different widths is the test
(`gdn_chunk::tests::the_device_chunk_matches_this_one`, both measured
against the sequential rule, which is the definition).

Measured on an M2 Pro at Bonsai's shape, device against host, both
warmed (`device_chunk_against_host_throughput`):

| rows | host chunk | device chunk | |
|---|---|---|---|
| 8 | 1.24 ms | 1.13 ms | 1.10x |
| 32 | 3.43 ms | 2.49 ms | 1.37x |
| 64 | 5.48 ms | 2.48 ms | 2.21x |
| 128 | 10.43 ms | 5.41 ms | 1.93x |
| 512 | 43.38 ms | 13.86 ms | 3.13x |
| 1201 | 96.10 ms | 32.15 ms | 2.99x |

The first version of that kernel read **1.05x at 512 rows**, and what
fixed it is worth more than the kernel: `m[t]` and `n[t]` are per-thread
arrays indexed by a loop whose trip count was the RUNTIME `c`, so the
compiler could not unroll it and spilled both out of registers into
device memory. Padding the chunk's tiles to the compile-time `CHUNK` and
running every hot loop to that constant is 50 GFLOP/s to 160, on a part
that peaks near 6.8 TFLOP/s -- so there is more there, and the next step
is the simdgroup-matrix form, since the two `S x S x C` products are
exactly a matmul.

End to end on the real checkpoint, a 2420-token prompt, interleaved
A/B/A on the same box, `--max-load 0` with `suggestd` held down:

    device  39.78 / 41.27 / 41.66 t/s
    host             36.49 / 36.60

so +13% of prefill, and parity stays MATCH through it (KL 2.03e-6
against the fork's libllama on a 256-token prompt, which is the same
path). A decode token still steps one row on the host, where there is
no traffic to amortise and the three losses below still hold.

## Where the PTQ1_0 matvec actually stands

With the submissions down to 69 a token the gap is kernel time, so the
kernel was taken apart. Removing the trit decode from it entirely and
timing the loads alone splits it: on Bonsai's `17408x5120` the full
matvec is 0.94 ms and the loads alone 0.57 ms, so the decode is 42% and
the memory side 58%.

Three hypotheses about that memory side were tested and all three came
back NEUTRAL:

- **Four rows' bytes requested before any is decoded**, so a lane has
  four outstanding requests instead of one. 1.004 to 0.980 ms.
- **One aligned 16-bit load for the two adjacent bytes a lane owns**,
  five loads to four. 0.980 to 0.943 ms.
- **The two l2 norms as ONE dispatch.** Q and K are the first
  `2 * n_k_heads` heads of the convolution's `[q | k | v]` output,
  contiguous, so one call does both: one fewer dispatch a layer, and
  one fewer false dependency for any barrier scheme that is per-buffer.
  This one is kept.
- **Eight rows per threadgroup instead of four**, halving the activation
  re-reads, which the arithmetic said were 4.6x the weight traffic
  (every threadgroup reads the whole 20 KB activation, 4352 times for
  `ffn_gate`). 0.943 to 0.941 ms raw -- and REVERTED later, see below.
  Sixteen and thirty-two rows are worse still (0.822 and 0.911 net),
  which is register spill.

### The per-stage breakdown, from PRODUCTION

The probe does not predict production, but three production
configurations measured across this work do, because each differs from
the next by exactly one stage. Their `FRINK_METAL_GPU_TIMING` averages
subtract:

| stage of a recurrent layer | GPU | bytes | achieved |
|---|---|---|---|
| head: `attn_norm` + the four projections | 0.286 ms | 19.4 MB | 67.8 GB/s |
| branch: twelve small dispatches + `ssm_out` | 0.251 ms | 13.1 MB | 52.2 GB/s |
| FFN: three matvecs | 0.762 ms | 58.5 MB | 76.8 GB/s |
| **layer** | **1.299 ms** | 90.9 MB | 70.0 GB/s |

`gdn-branch` is the first row measured alone (0.251), `gdn-layer` the
branch plus the FFN (1.013), `gdn-layer-full` all three (1.299).

Times 48 recurrent layers that is 62.4 ms, plus 23.9 for the sixteen
attention layers: 86.2 ms of GPU a token, which is the 88.8 the ledger
reports, within the noise.

So the FFN is the fastest of the three per byte and the branch the
slowest -- and the branch's 13.1 MB includes 6.2 MB of recurrent state
and twelve dispatches whose fixed cost is about 8 microseconds each,
some 0.1 ms of its 0.251. That is ~4.8 ms a token of pure dispatch
overhead, the largest single identified inefficiency left, and
halving it is worth about 2.4 ms: 10.12 to roughly 10.4 tok/s. Still
not parity, and it is the biggest item on the list.

Whoever picks this up starts here rather than at the probe. The first
one of those twelve is already merged: `h += branch` and `rms_norm(h)`
are now the ONE `encode_add_rms_norm` the dense decode stack uses, so
the layer is thirteen dispatches and not fourteen. **It is below
measurement noise** -- 10.06 to 10.09 tok/s against 10.14 to 10.16,
inside a spread that has read 10.06 to 10.20 for nominally identical
builds all day -- which is the honest size of ONE dispatch: about 0.4
ms of a 98 ms token, 0.4%. It is kept because it is strictly fewer
dispatches and the standard fused form, not because it measured.

That is also the shape of everything left on this list. The remaining
merges are worth a fraction of a percent EACH and cannot be told apart
from noise individually; only doing most of them would show up, and
each one is a chance to break a layer that four oracle tests currently
hold. That is the cost side of the 2.4 ms the twelve dispatches are
worth in total.

### The probe, fixed, and still not predictive

Those three were measured against a raw wall-clock number that includes
a command buffer's worth of host cost. The probe now measures that cost
on a shape whose kernel is negligible (`64x512`, 0.20 ms) and reports
every other row NET of it, which changes what it says:

| shape | PTQ1_0 net | Q4_0 net |
|---|---|---|
| `17408x5120` (`ffn_gate`, `ffn_up`) | 0.785 ms | 0.707 |
| `5120x17408` (`ffn_down`) | 0.371 | 0.742 |
| `248320x5120` (the head) | 3.417 | 3.943 |

So PTQ1_0 BEATS Q4_0 on two shapes of three and loses only on the tall
one, where it reads 2.6x fewer bytes and still takes longer. And at
four rows a threadgroup the FFN's three matrices sum to 1.941 ms
against eight rows' 2.093 -- **7% better**, the opposite of what the
raw number said, which is why the row count is back to four (the
reference's own geometry, and fewer registers).

**And end to end that 7% is worth nothing: 10.14 to 10.16 tok/s either
way.** Even net, one matvec in isolation does not predict a matvec
inside a command buffer with resident weights and neighbours. Eight
kernel and schedule experiments have now been run against this probe
and it has failed to predict production every time. It is good for
correctness and for nothing else; the next kernel attempt needs a GPU
capture or per-dispatch timestamps, not this.

Together they are +2% end to end (9.99 to 10.20 tok/s, measured
interleaved), which is worth keeping and is not what the probe implied.

**The probe is why they looked bigger than they are.** It reports 20
GB/s where the production path reaches 77 on the same matrices, and the
difference is not the kernel: the probe times ONE matvec in isolation,
while a real layer has several in flight in one command buffer. A
kernel that is latency-bound per launch and fine under concurrency
reads as catastrophic there. Its numbers are only good for comparing
one variant of the kernel against another, which is how they are used
above.

The obvious follow-on, letting those matvecs overlap EXPLICITLY, is a
measured non-result below.

## Measured non-results, so they are not tried again

- The fused recurrent layer on a concurrent encoder with RESOURCE-scoped
  barriers (`MemRanges::begin_op`, every dispatch declaring what it
  reads and writes, a barrier only on a real conflict), so `beta` runs
  beside `alpha`, `qkv` beside `z` and `gate` beside `up` -- the two
  biggest matvecs in the layer. Correct, and **10.11 to 10.20 tok/s
  against 10.12 to 10.17: flat.** Reverted, because it buys nothing and
  costs a concurrent encoder plus a dependency declaration at every
  dispatch.

  This one is worth reading as a MODEL and not just a number. The
  argument for it was that the kernel splits 42% ALU and 58% memory
  (measured: removing the trit decode takes the isolated matvec from
  0.94 ms to 0.57), so perfect overlap should have had 1.7x in it. It
  does not, because a single PTQ1_0 matvec at Bonsai's shape already
  dispatches 4352 threadgroups and fills the part: the GPU is ALREADY
  interleaving that kernel's own ALU and memory across its concurrent
  threads, and a second dispatch has no idle capacity to use. Overlap
  pays when a dispatch UNDER-occupies, and in this layer the only
  dispatches that do are the small ones that are not where the time is.
  So the 42/58 split is not headroom; it is the kernel sitting at its
  balance point.

  It was also built TWICE. The first version paired `begin_op` and
  `end_op` by hand at each of 21 sites, got one pair's declaration
  wrong, and produced wrong answers that the oracle tests caught. The
  second routes every dispatch through ONE `stage(reads, writes)` helper
  and was correct on the first run. The difference is the repo's own
  rule: 21 hand-written pairs that must agree, against one helper they
  all go through.
- The fused recurrent layer on a CONCURRENT encoder
  (`computeCommandEncoderWithDispatchType`), with scope-Buffers
  barriers only between stages that depend on each other, so `qkv` runs
  beside `z` and `gate` beside `up` -- the two biggest matvecs in the
  layer: 10.10 against 10.20 tok/s. Eighteen barriers a layer cost more
  than the overlap buys, which is what `memory_barrier_buffers`' own
  comment already said about scope-Buffers. A resource-scoped version
  (`MemRanges::begin_op`, as the dense decode stack uses) is the form
  that might pay, and it needs the read and write set of every dispatch
  named.
  Building it found three real missing barriers by turning the oracle
  tests red, and a fourth that a patch had SILENTLY not applied -- the
  race it left was 1%, the shape a loose tolerance would have passed.
- The Hadamard rotation on the device for a prefill BATCH: `pp128` 33.96
  to 32.04 tok/s. `transform_rows` is already parallel across cores;
  the kernel serialises into the GEMM's own command buffer.
- Spin-then-block instead of `waitUntilCompleted`, aimed at the 0.166 ms
  of wake-up latency: 3.7 tok/s against 7.1. Polling the status through
  objc takes the core the host work needs.
- `commandBufferWithUnretainedReferences` on the matvec path: 7.00
  against 7.1, for an `unsafe` invariant somebody has to keep.
- 8 rows per simdgroup in the PTQ1_0 matvec instead of 4: 6.9 against
  7.1 end to end.
- The batched recurrence on the device, with the state copied once per
  BATCH rather than once per token: `pp128` 32.7 to 28.5 (above). This
  is the one entry that was later REVERSED, and only by changing what
  the kernel was asked to do: chunked, the same idea is 3x ahead.
- The PTQ1_0 GEMM's dequant moved to the float pipe, the rewrite that
  bought the matvec 1.8x: `pp128` 32.8 to 31.9. The GEMM dequantizes a
  tile ONCE into threadgroup memory and the simdgroup matrix ops hide
  it, so the trit decode is not what that kernel waits on.

## What prefill spends itself on now

With the recurrence on the device the profile moved twice in one day,
and the second move is the one that matters. A `sample` of a 2420-token
Bonsai prefill, by top-of-stack:

| | samples |
|---|---|
| `attention::dot_f32` | 5640 |
| `attention::pv_tile` | 3295 |
| `attention::qk_tile` | 2243 |
| `hadamard::fwht_normalized` | 1504 |
| `mamba2::conv_step` | 769 |

`delta_chunk` is not in it. The attention rows are Bonsai's 16 softmax
layers, which the FUSED Metal attention block refuses because their Q
is gated, so they fell back to the Rayon host kernel while
`frink-metal`'s own `launch_gqa_prefill_host_ex` sat there with no
caller. Wiring it (`Decoder::prefill_attention_blocked`) is 41.5 to
43.1 tok/s.

That is much less than 11178 of ~18000 samples suggests, and the
`FRINK_METAL_GPU_TIMING` ledger says why: those samples are on worker
threads that were already overlapping the GPU. With both host costs
gone, prefill is the GEMM and nothing else -- 82 ms a submission over
some 500 submissions of a 56 s run, about 3.2 TFLOP/s against the
fork's 3.6 overall. So the prefill gap is now entirely the kernel
below, and the host side of it is finished.

## Where the prefill gap actually is

Arithmetic, not a profile: 128 tokens through 26.9B parameters is
6.9 TFLOP of matmul. The fork's `pp128` of 66.8 tok/s is 1.92 s, so
3.6 TFLOP/s; ours of 32.9 is 3.89 s, so 1.8 TFLOP/s. An M2 Pro's GPU
peaks near 6.8 TFLOP/s in f32, so the fork runs its PTQ1_0 GEMM at 53%
of peak and frink at 26%. Both use llama.cpp's simdgroup-matrix
`mul_mm` shape with the tile dequantized once into threadgroup memory,
so the difference is in the tiling constants and the dequant's cost per
tile, not in the algorithm. That is a bounded kernel project and it is
the prefill half of this gap; the decode half is the chunked recurrence
above plus the submissions the fused stack would remove.
