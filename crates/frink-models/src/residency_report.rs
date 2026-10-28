//! Dry-run residency planning: what a checkpoint would cost to run,
//! computed from its GGUF header alone -- no weights loaded, no
//! allocation. This is the "plan before you allocate" half of the
//! residency workstream; the enforcement half at decode time is the
//! bounded expert store (`frink_core::expert_store`) plus the global
//! device plan (`frink_moe::ResidencyPlan`).
//!
//! Accounting model (deliberately explicit about what it does NOT yet
//! cover): dense/always-resident weight bytes, routed-expert bytes
//! (resident, or capped by the streaming budget), per-request KV-cache
//! bytes at a stated context length, request concurrency, and a safety
//! headroom fraction. Not yet accounted: activation/scratch buffers,
//! tokenizer/runtime overhead, GPU placement (VRAM budgets are planned
//! separately by `ResidencyPlan`), and Kimi's recurrent state (this
//! report reads GGUF headers; the Kimi safetensors path has no
//! header-only reporter yet).
//!
//! The KV line and the fits/does-not-fit verdict are the same
//! arithmetic [`crate::kv_budget`] runs -- this module is the
//! whole-checkpoint view of it (it is the one that knows what the
//! weights cost), and [`ResidencyReport::kv_budget`] hands back the
//! priced inequality so `--ctx auto` and the server's admission check
//! cannot drift away from what `inspect-plan` prints.

use frink_gguf::ShardedGguf;

use crate::config::ModelConfig;
use crate::decoder::KvWindowPolicy;
use crate::device_budget::human;
use crate::kv_budget::{ContextFit, KvBudget, KvElem, KvResidency, KvShape, CTX_AUTO_GRANULARITY};
use crate::loader::LoadError;

/// The inputs a plan is computed against. `expert_cache_bytes: None`
/// means routed experts load resident (mmap); `Some(budget)` means
/// they stream through the bounded store and cost at most `budget`
/// resident bytes.
#[derive(Debug, Clone, Copy)]
pub struct ResidencyAssumptions {
    pub context_tokens: usize,
    pub concurrent_requests: usize,
    pub expert_cache_bytes: Option<u64>,
    /// Fraction of the budget held back for everything this report does
    /// not model (activations, allocator slack, OS). 0.2 is the
    /// default the CLI uses.
    pub headroom_fraction: f64,
    /// Width of the KV store the run will actually keep -- f32 for the
    /// host cache, f16 for Metal's device KV, and so on. Only this
    /// module's caller knows which backend is selected.
    pub kv_elem: KvElem,
    /// Whether the run this plan describes will evict KV rows behind a
    /// layer's sliding window (#61).
    ///
    /// The plan has to be priced against the run that will happen, so
    /// the default reads `FRINK_KV_WINDOW` exactly as the decoder does
    /// -- one policy value, so the report and the stores cannot be
    /// describing different runs. Pass [`KvWindowPolicy::off`]
    /// explicitly for a what-if plan of the non-evicting engine.
    pub kv_window: KvWindowPolicy,
}

impl Default for ResidencyAssumptions {
    /// Single request, 4096 tokens, resident experts, host f32 KV --
    /// the CLI's own defaults.
    fn default() -> Self {
        ResidencyAssumptions {
            context_tokens: 4096,
            concurrent_requests: 1,
            expert_cache_bytes: None,
            headroom_fraction: 0.2,
            kv_elem: KvElem::F32,
            kv_window: KvWindowPolicy::from_env(),
        }
    }
}

/// One line of the plan, with the reason it costs what it costs.
#[derive(Debug, Clone)]
pub struct ResidencyLine {
    pub label: String,
    pub bytes: u64,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct ResidencyReport {
    pub lines: Vec<ResidencyLine>,
    pub required_bytes: u64,
    /// The ceiling this plan was priced against: a probed
    /// [`crate::device_budget::DeviceBudget::usable_bytes`], or a
    /// hypothetical machine's size for what-if planning.
    pub budget_bytes: u64,
    /// `budget_bytes` minus the headroom fraction.
    pub usable_bytes: u64,
    pub assumptions: ResidencyAssumptions,
    /// Weight bytes the plan charged (dense plus whatever the expert
    /// line resolved to), kept separately from the KV term so
    /// [`Self::kv_budget`] can rebuild the inequality exactly.
    pub weights_bytes: u64,
    /// The KV geometry this checkpoint runs on.
    pub kv_shape: KvShape,
    /// How many positions each layer's store will really keep, under
    /// [`ResidencyAssumptions::kv_window`]. Kept beside the shape so
    /// [`Self::kv_budget`] rebuilds the inequality with the same
    /// residency the KV line was priced with, rather than deriving a
    /// second one.
    pub kv_residency: KvResidency,
}

impl ResidencyReport {
    /// Computes the plan for the checkpoint at `path` (any shard of a
    /// split set, or a single file) against `budget_bytes` (pass a
    /// probed [`crate::device_budget::DeviceBudget::usable_bytes`], or
    /// a hypothetical machine's size for what-if planning).
    pub fn from_gguf(
        path: impl AsRef<std::path::Path>,
        assumptions: ResidencyAssumptions,
        budget_bytes: u64,
    ) -> Result<Self, LoadError> {
        let file = ShardedGguf::open(path)?;
        let config = ModelConfig::from_gguf(&file)?;

        let mut dense_bytes: u64 = 0;
        let mut routed_bytes: u64 = 0;
        let mut routed_tensors = 0usize;
        // A tensor whose dtype this build cannot size contributes
        // nothing here, which is what the old `byte_len() -> 0` did
        // implicitly. It is explicit now, and counted, so a footprint
        // that silently omits tensors says so rather than reading as a
        // smaller model.
        let mut unsized_tensors = 0usize;
        for (_, t) in file.tensors() {
            let Some(bytes) = t.byte_len() else {
                unsized_tensors += 1;
                continue;
            };
            if t.name.contains("_exps.weight") {
                routed_bytes += bytes as u64;
                routed_tensors += 1;
            } else {
                dense_bytes += bytes as u64;
            }
        }
        if unsized_tensors > 0 {
            eprintln!(
                "frink: {unsized_tensors} tensor(s) have a dtype this build cannot size; \
                 the footprint below EXCLUDES them and is therefore a lower bound"
            );
        }

        let mut lines = Vec::new();
        lines.push(ResidencyLine {
            label: "dense weights".to_string(),
            bytes: dense_bytes,
            reason: "attention/norms/router/shared-expert/embedding/output tensors, \
                     always resident (quantized in place, mmap or owned)"
                .to_string(),
        });

        let expert_line = match assumptions.expert_cache_bytes {
            Some(budget) => {
                let capped = budget.min(routed_bytes);
                ResidencyLine {
                    label: "routed experts (streamed)".to_string(),
                    bytes: capped,
                    reason: format!(
                        "{routed_tensors} packed expert tensors totalling {routed_bytes} \
                         bytes on disk, streamed through a bounded cache of {budget} bytes \
                         (resident cost = min(budget, total))"
                    ),
                }
            }
            None => ResidencyLine {
                label: "routed experts (resident)".to_string(),
                bytes: routed_bytes,
                reason: format!(
                    "{routed_tensors} packed expert tensors, loaded as zero-copy mmap views \
                     -- resident under memory pressure only via OS page cache eviction; \
                     enable expert streaming to bound this explicitly"
                ),
            },
        };
        lines.push(expert_line);
        // Everything charged so far is weights; the KV line follows.
        let weights_bytes = lines.iter().map(|l| l.bytes).sum();

        // KV cache at the stated context, per concurrent request --
        // `kv_budget`'s arithmetic, so an MLA checkpoint is priced the
        // way it will really run rather than as a GQA one, and a
        // windowed checkpoint is priced at what the store keeps rather
        // than at what attention reads.
        let kv_shape = KvShape::from_config(&config, assumptions.kv_elem);
        let kv_residency = KvResidency::from_config(&config, assumptions.kv_window);
        // The PEAK, not the resting number: a windowed layer holds the
        // whole prompt while it is being prefilled and hands the rows
        // back only afterwards, so a plan priced at rest would approve
        // a run whose high-water mark it never modelled.
        let kv_per_request =
            kv_shape.peak_kv_bytes_for_tokens(assumptions.context_tokens, &kv_residency);
        let evicting = kv_residency.evicting_layers();
        lines.push(ResidencyLine {
            label: "KV caches".to_string(),
            bytes: kv_per_request * assumptions.concurrent_requests as u64,
            reason: format!(
                "{} at {} context tokens = {kv_per_request} bytes/request, x {} concurrent \
                 requests{}",
                kv_shape.describe(),
                assumptions.context_tokens,
                assumptions.concurrent_requests,
                match evicting {
                    0 => String::new(),
                    n => format!(
                        "; {n} of {} layers evict behind their sliding window \
                         (FRINK_KV_WINDOW) and stop growing, so this is below the \
                         per-token figure times the context",
                        kv_shape.n_layers
                    ),
                }
            ),
        });

        let required_bytes = lines.iter().map(|l| l.bytes).sum();
        let usable_bytes = (budget_bytes as f64 * (1.0 - assumptions.headroom_fraction)) as u64;
        Ok(ResidencyReport {
            lines,
            required_bytes,
            budget_bytes,
            usable_bytes,
            assumptions,
            weights_bytes,
            kv_shape,
            kv_residency,
        })
    }

    pub fn fits(&self) -> bool {
        self.required_bytes <= self.usable_bytes
    }

    /// The same plan expressed as the priced inequality, so
    /// `--ctx auto` and the server's admission check reuse this
    /// report's terms instead of recomputing their own.
    ///
    /// The headroom fraction becomes the `activation_headroom_bytes`
    /// term rather than shrinking the budget, which is what makes
    /// `kv_budget().check(context_tokens).is_ok()` agree with
    /// [`Self::fits`] exactly.
    pub fn kv_budget(&self) -> KvBudget {
        KvBudget {
            weights_bytes: self.weights_bytes,
            activation_headroom_bytes: self.budget_bytes - self.usable_bytes,
            device_budget_bytes: self.budget_bytes,
            shape: self.kv_shape,
            residency: self.kv_residency.clone(),
            concurrent_requests: self.assumptions.concurrent_requests,
        }
    }

    /// Largest context that fits this plan, capped at `cap` (the
    /// model's own trained context length).
    pub fn auto_context(&self, cap: usize) -> ContextFit {
        self.kv_budget().max_context(cap, CTX_AUTO_GRANULARITY)
    }

    /// Strict mode: an `Err` with the full plan text when the plan
    /// overcommits the usable budget. Callers refuse to load on `Err`.
    pub fn check_strict(&self) -> Result<(), String> {
        if self.fits() {
            Ok(())
        } else {
            Err(format!(
                "residency plan overcommits: requires {} bytes but only {} usable \
                 ({} budget minus {:.0}% headroom)\n{self}",
                self.required_bytes,
                self.usable_bytes,
                self.budget_bytes,
                self.assumptions.headroom_fraction * 100.0
            ))
        }
    }
}

impl std::fmt::Display for ResidencyReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "residency plan (context={}, concurrency={}, kv={}, headroom={:.0}%):",
            self.assumptions.context_tokens,
            self.assumptions.concurrent_requests,
            self.assumptions.kv_elem.as_str(),
            self.assumptions.headroom_fraction * 100.0
        )?;
        for line in &self.lines {
            writeln!(
                f,
                "  {:<28} {:>12}  {}",
                line.label,
                human(line.bytes),
                line.reason
            )?;
        }
        writeln!(
            f,
            "  {:<28} {:>12}",
            "TOTAL required",
            human(self.required_bytes)
        )?;
        writeln!(
            f,
            "  {:<28} {:>12}  ({} device budget minus headroom)",
            "usable budget",
            human(self.usable_bytes),
            human(self.budget_bytes)
        )?;
        write!(
            f,
            "  verdict: {}",
            if self.fits() {
                "FITS"
            } else {
                "DOES NOT FIT (strict mode refuses to load)"
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> String {
        format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
    }

    /// Pinned to the non-evicting engine ON PURPOSE.
    ///
    /// `ResidencyAssumptions::default()` reads `FRINK_KV_WINDOW`,
    /// because the plan has to describe the run that will happen. That
    /// makes every test built on the default env-sensitive, and a test
    /// whose expected numbers depend on the developer's shell is a test
    /// that goes red for the wrong reason. Everything below asserts the
    /// non-evicting engine's arithmetic, so it says so.
    fn assumptions(cache: Option<u64>) -> ResidencyAssumptions {
        ResidencyAssumptions {
            context_tokens: 128,
            concurrent_requests: 2,
            expert_cache_bytes: cache,
            kv_window: KvWindowPolicy::off(),
            ..ResidencyAssumptions::default()
        }
    }

    #[test]
    fn moe_fixture_plan_accounts_experts_kv_and_streaming_cap() {
        let path = fixture("frink_real_moe_test.gguf");

        let resident = ResidencyReport::from_gguf(&path, assumptions(None), 1 << 30).expect("plan");
        let experts_resident = resident.lines[1].bytes;
        assert!(experts_resident > 0, "MoE fixture has routed expert bytes");

        // Streaming with a tiny budget caps the expert line at the
        // budget; everything else is identical.
        let streamed =
            ResidencyReport::from_gguf(&path, assumptions(Some(100)), 1 << 30).expect("plan");
        assert_eq!(streamed.lines[1].bytes, 100);
        assert_eq!(streamed.lines[0].bytes, resident.lines[0].bytes);
        assert_eq!(streamed.lines[2].bytes, resident.lines[2].bytes);
        assert_eq!(
            resident.required_bytes - streamed.required_bytes,
            experts_resident - 100
        );

        // A budget larger than the experts costs only the experts.
        let big =
            ResidencyReport::from_gguf(&path, assumptions(Some(u64::MAX)), 1 << 30).expect("plan");
        assert_eq!(big.lines[1].bytes, experts_resident);

        // KV arithmetic is exactly the documented formula.
        let file = ShardedGguf::open(&path).unwrap();
        let cfg = ModelConfig::from_gguf(&file).unwrap();
        let expected_kv = (cfg.n_layers * 2 * cfg.n_kv_heads * cfg.head_dim * 4 * 128 * 2) as u64;
        assert_eq!(resident.lines[2].bytes, expected_kv);
    }

    #[test]
    fn strict_mode_refuses_overcommit_and_accepts_a_fitting_plan() {
        let path = fixture("frink_real_moe_test.gguf");
        let fits = ResidencyReport::from_gguf(&path, assumptions(None), 1 << 30).unwrap();
        assert!(fits.check_strict().is_ok());

        // A "machine" with 1 byte of RAM cannot fit anything.
        let no_fit = ResidencyReport::from_gguf(&path, assumptions(None), 1).unwrap();
        let err = no_fit.check_strict().expect_err("must refuse");
        assert!(err.contains("overcommits"), "{err}");
        assert!(err.contains("DOES NOT FIT"), "{err}");
    }

    /// The report's verdict and the priced inequality must be the same
    /// answer, or `inspect-plan` would print one thing and admission
    /// would enforce another.
    #[test]
    fn kv_budget_view_agrees_with_the_reports_own_verdict() {
        let path = fixture("frink_real_moe_test.gguf");
        for budget in [1u64, 1 << 20, 1 << 30, u64::MAX / 4] {
            let report = ResidencyReport::from_gguf(&path, assumptions(None), budget).unwrap();
            let priced = report.kv_budget();
            assert_eq!(
                priced.check(report.assumptions.context_tokens).is_ok(),
                report.fits(),
                "budget {budget}: verdict and priced inequality disagree"
            );
            assert_eq!(
                priced.estimated_bytes(report.assumptions.context_tokens),
                report.required_bytes + (report.budget_bytes - report.usable_bytes),
                "budget {budget}: the priced total must be the plan's total plus headroom"
            );
        }
    }

    /// `--ctx auto` on a real header: the chosen context must actually
    /// pass the same check, and one granularity step further must not.
    #[test]
    fn auto_context_picks_a_context_that_really_fits() {
        let path = fixture("frink_real_moe_test.gguf");
        let report =
            ResidencyReport::from_gguf(&path, assumptions(None), 64 * 1024 * 1024).unwrap();
        let fit = report.auto_context(131_072);
        let priced = report.kv_budget();
        assert!(fit.tokens > 0, "64 MiB must fit some context: {fit}");
        assert!(priced.check(fit.tokens).is_ok(), "{fit}");
        if fit.capped_by == crate::kv_budget::ContextCap::DeviceBudget {
            assert!(
                priced.check(fit.tokens + fit.granularity).is_err(),
                "auto context left a whole granularity step on the table: {fit}"
            );
        }
    }

    /// A checkpoint whose header declares a sliding window is priced
    /// exactly like one that does not, because no KV store this engine
    /// allocates ever evicts a position (#33). This report used to make
    /// a windowed model look ~60x cheaper at 32k, which is a plan an
    /// operator can act on and a machine cannot honour.
    #[test]
    fn a_sliding_window_config_is_priced_like_a_full_attention_one() {
        let mut cfg = crate::config::test_dense_fixture();
        cfg.n_layers = 8;
        cfg.n_kv_heads = 4;
        cfg.head_dim = 64;
        cfg.sliding_window = None;
        cfg.swa_layers = crate::swa_layers::SwaLayers::All;
        let full = KvShape::from_config(&cfg, KvElem::F32);

        cfg.sliding_window = Some(512);
        cfg.swa_layers = crate::swa_layers::SwaLayers::period(6, false);
        let windowed = KvShape::from_config(&cfg, KvElem::F32);

        assert_eq!(
            full.kv_bytes_for_tokens(512),
            windowed.kv_bytes_for_tokens(512)
        );
        assert_eq!(
            full.kv_bytes_for_tokens(32_768),
            windowed.kv_bytes_for_tokens(32_768)
        );
    }
}
