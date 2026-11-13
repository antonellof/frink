# 0007. Issues track work, ADRs record decisions, plans are short-lived

Status: Accepted
Date: 2026-10-28

## Context

By 2026-10 the repo held about 10,000 lines of plans, audits and
roadmaps. Status fields in plans went stale and were read as evidence
(a merged PR marked a plan complete while it returned wrong tokens on
Metal). The same lesson was restated in `CLAUDE.md`, the north star,
the plans index and the plan itself, and the copies disagreed.

## Decision

- **Work** is a GitHub issue. Its status is the issue's state.
- **Decisions** are ADRs in `docs/adr/`. Lessons that justify a rule go
  in the ADR's Context, once.
- **Plans** in `docs/plans/` are design docs for work in flight. When
  the work lands, its lasting decisions become ADRs and the plan is
  deleted. Git history is the archive: there is no `archive/`, `done/`
  or `on-hold/` directory.
- **Goal and ranking** is [`docs/plans/north-star.md`](../plans/north-star.md),
  the one roadmap: one page, no status fields. Other plans are the
  design of one item on it, never a second ranking.
- `CLAUDE.md` holds commands and rules, each rule pointing at the ADR
  that explains it. No war stories.
- Counts are stated in at most one place and pinned by a test, or are
  not stated.

## Consequences

Fewer, shorter documents. History stays in git and `CHANGELOG.md`.
A finished plan is cited, when it is worth citing, by a link pinned to
a commit where it still existed.
