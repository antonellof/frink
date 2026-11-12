---
name: "how work lands: a branch and a PR per feature, an issue per defect"
overview: "The delivery rule for every plan item. A COMPLETED FEATURE lands as its own branch and its own pull request, one feature per PR, never several. A DEFECT OR OPEN QUESTION becomes a GitHub issue instead, even when it is found in passing and even when it is small, so that nothing depends on a person remembering it. The two are different artifacts on purpose: a PR is a thing to review, an issue is a thing to track, and mixing them means the review argues about scope while the tracking silently disappears. This file also carries the parallel-agent rules, because the failure mode of running several agents at once is not bad code, it is two branches editing one file."
---

# How work lands

## The two artifacts

| What you have | What you make |
|---|---|
| A feature that is complete, builds, and has tests | A branch and a pull request |
| A defect, a gap, an open question, a thing that smells | A GitHub issue |
| A defect you fixed as part of the feature you were already writing | Stays in the PR, described in the PR body |
| A defect in code nobody asked you to touch | An issue, not a drive-by commit |

The last row is the one that gets broken. A fix that arrives inside an
unrelated PR is a fix nobody reviewed, in a diff nobody expected, and it
is how a branch reverts three others.

## Branch and PR rules

- **One feature per branch.** `feat/<short-name>` or `fix/<short-name>`.
- **The branch is cut from `main` at the moment work starts**, not from
  another feature branch. A stack is a merge conflict with extra steps.
- **The PR body says what changed, why, and what would have to be true
  for it to be wrong.** The commit messages in this repo already do
  this; the PR body is allowed to be shorter than the commits, never
  longer than the diff deserves.
- **A PR is opened only when the gates pass**: `cargo build
  --workspace`, `cargo test --workspace`, `cargo clippy --workspace
  --all-targets -- -D warnings` (and `--release`, because
  `debug_assert!` type-checks its argument there), `cargo fmt --all`.
- **A PR that cannot state its evidence is not finished.** "It compiles"
  is not evidence. A test that goes red when the change is reverted is.

## Issue rules

Open an issue the moment a defect is identified, before deciding whether
to fix it. The issue is cheap and forgetting is not.

An issue states:

1. What is wrong, in one sentence.
2. The failure it produces, concretely: inputs, and the wrong output or
   the crash. Not "this could be a problem".
3. Where: `path/to/file.rs:line`.
4. Whether it is reachable from untrusted input, because that changes
   the priority and nothing else in the description will say so.

Label it, and if it is an instance of the repo's dominant bug shape (two
structures that must agree about one thing, with nothing enforcing it),
say so in the title. That phrase is searchable and the class matters more
than the instance.

## Running several agents at once

The rules in [`README.md`](README.md) apply, and these are the ones that
break first when the fleet grows:

- **One agent owns a file.** Partition the work by file before
  launching, not after a conflict. If two items need the same file, they
  are one item or they are sequential.
- **An agent may not run benchmarks.** Measurement needs a quiet host,
  and a loaded machine reads 25-45% low. An agent that wants a number
  opens an issue asking for the number.
- **An agent may not tag, publish, or force-push.** Releases are the
  parent session's, because a half-published workspace cannot be undone:
  crates.io versions are immutable.
- **An agent reports what it did not do.** A task delivered at 80% with
  the missing 20% named is worth more than one that quietly narrowed its
  own scope, because the second is indistinguishable from success until
  someone depends on it.

## Releasing

Only when the user asks for a release.

1. Open a `chore(release): X.Y.Z - title` PR that bumps
   `version = "…"` in the root `Cargo.toml` (the workspace and every
   internal pin), `Cargo.lock` with `cargo update --workspace
   --offline`, and `FRINK_VERSION=vX.Y.Z` and `frink-inference = "X.Y"`
   in `README.md` and `crates/frink-inference/README.md`. It also dates
   the CHANGELOG's `[Unreleased]` section, keeps an empty
   `[Unreleased]` above it, and merges duplicate `###` headings.
2. After the merge, tag the merge commit:
   `git tag -a vX.Y.Z <merge-sha>`, with the message `vX.Y.Z - title`
   and the release PR's body, then push the tag. `release.yml` builds
   the tarballs, and `publish-crates.yml` publishes every crate. A tag
   push publishes for real. Only a manual dispatch defaults to a dry
   run.
3. Check `gh release view vX.Y.Z` for both tarballs and their
   checksums, and that the publish run finished green.

## Why this is written down

The alternative, which this project used, was to land several features
in one push to `main` and describe them in the commit messages. That
works exactly until someone needs to revert one of them, ask when a
behaviour changed, or review a change without reviewing four others at
the same time.
