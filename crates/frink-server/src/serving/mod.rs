//! Running a request: admitting it, prefilling it, decoding it, and
//! deciding where its answer ends.

/// Admission and chunked prefill for the WINDOW and RECURRENT memory
/// models, which [`batch`] excludes.
///
/// Wired: nothing yet -- `PrefillPass`, `SlotTable`, `Capacity`,
/// `Geometry`, `PromptAdmission`, `NotAdmitted` and `FinishReason` are
/// the largest unwired block left in this crate, and they are one
/// roadmap item: chunk-duration scheduling (#632), whose
/// quantum is chunk DURATION because a GPU cannot preempt a running
/// kernel.
#[allow(dead_code)]
pub(crate) mod admission;

pub(crate) mod batch;
