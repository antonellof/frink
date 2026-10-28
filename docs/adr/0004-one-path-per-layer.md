# 0004. One code path per layer, parameterised, never copied

Status: Accepted
Date: 2026-10-28

## Context

The same decoder layer was once written out about eleven times across
`decoder.rs` and the Metal launch code. Copies lost eight model
features one at a time (`attention_scale`, `post_attn_norm`,
`post_ffn_norm`, gpt-oss `o_bias`, and others), and the paged decode
copy lost five, with nothing failing.

## Decision

- A variant is a parameter, not a copy. Precedent: `forward_multi_seq`
  takes a `MultiSeqKv`, and every KV layout goes through one
  `attn_block` taking a `KvStep`.
- Where two structures must agree, make disagreement fail to compile or
  turn a test red: an exhaustive destructure with no `..`, one predicate
  shared by every call site, a table derived from another rather than
  restated.

## Consequences

Some functions take more parameters than a copy would. That cost is
accepted; the alternative has a measured failure rate.
