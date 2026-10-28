//! Whether a windowed layer's host KV cache may drop rows behind its
//! window, and which window it drops behind.
//!
//! [`frink_core::kv_swa::KvWindow`] is the arithmetic -- how many rows
//! survive N positions. This module is the *decision*: which layers get
//! one at all, and whether this particular run is one where dropping a
//! row is safe. Those are different questions with different owners, and
//! keeping them apart is what stops the budget from pricing a saving the
//! store did not take (#33) or the store from taking one the budget did
//! not price.
//!
//! # On by default
//!
//! A windowed layer keeps `window + slack` rows instead of the whole
//! context, which is what llama.cpp's `llama_kv_cache_iswa` does with
//! its default `swa_full = false`. `FRINK_KV_WINDOW=0` keeps every row,
//! so the before and the after are one word apart and reverting costs
//! nothing. It shipped behind an opt-in first (#113, #189) and went on
//! by default once every interaction below had either been handled or
//! been given a named fallback (#61):
//!
//! - **Speculative rollback.** A drain keeps a reserve inside its slack
//!   (`KvWindow::max_rollback`), and every speculative caller bounds its
//!   draft block by `KvCache::rollback_limit` before it writes.
//! - **The prefix cache** stores windowed caches and reuses one only for
//!   a match the cache can still truncate to (`KvCache::can_truncate_to`);
//!   a deeper match is a miss, never a truncated history.
//! - **`n > 1` forks** clone the post-prefill caches, and a clone carries
//!   its window and its rows, so each fork drains on its own.
//! - **Slot files** hold a whole history, so their prefill keeps every
//!   row (`KvCache::forbid_eviction`).
//! - **The paged prefill's scratch copy** is read back by offset, so it
//!   never evicts either.
//! - **Context shift** is refused before any of this is reached.
//!
//! # What the switch refuses to do
//!
//! Eviction is the contiguous host store, and only that (the GPU-resident
//! stores and the paged one are #61's remaining steps). Two things it
//! therefore turns itself off for:
//!
//! - **Metal attention.** `Decoder`'s Metal arms compare
//!   `MetalKvBuffers::seq_len` against the host cache's `rows()` and its
//!   `positions()` in five places, and take the two as interchangeable.
//!   They are, until a host cache evicts. So a run with
//!   `FRINK_METAL_ATTN` on does not evict, and says so here rather than
//!   in five separate conditions that would drift apart.
//! - **Full-attention layers**, obviously, and that is the interesting
//!   half of the saving rather than a caveat: an alternating-SWA model
//!   keeps every position in its dense layers no matter what this
//!   switch says. The Gemma-3 figure in #61 is a per-layer number, not
//!   a whole-model one.
//!
//! CUDA needs no exclusion: the resident-KV decode hook in
//! `Decoder::gqa_attention` is reachable only from the `window == None`
//! arm of `push_and_attend_row`, so it never sees a cache that evicts.

use frink_core::cache::KvCache;
use frink_core::kv_swa::KvWindow;

use super::Decoder;

/// Whether this run may evict, decided once at load time.
///
/// A `Copy` value on the `Decoder` rather than a cached global, so a
/// test can build a decoder that evicts and one that does not in the
/// same process and compare their tokens. A global read once per
/// process would need a subprocess per arm to say the same thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvWindowPolicy {
    enabled: bool,
}

/// The one spelling of the switch.
pub const KV_WINDOW_ENV: &str = "FRINK_KV_WINDOW";

impl KvWindowPolicy {
    /// Never evicts. What every cache in this engine did before #61, and
    /// what a run gets when [`KV_WINDOW_ENV`] is `0`.
    pub const fn off() -> Self {
        KvWindowPolicy { enabled: false }
    }

    /// Always evicts, where the layer has a window. For tests and for a
    /// caller that has already made the decision itself.
    pub const fn on() -> Self {
        KvWindowPolicy { enabled: true }
    }

    /// Reads [`KV_WINDOW_ENV`], then subtracts the runs that must not
    /// evict. See the module doc for which and why.
    pub fn from_env() -> Self {
        let on = Self::parse(std::env::var(KV_WINDOW_ENV).ok().as_deref());
        #[cfg(feature = "metal")]
        let on = on && !frink_metal::attn::metal_attn_enabled();
        KvWindowPolicy { enabled: on }
    }

    /// The switch's spelling: on unless it says off. Anything that is
    /// not one of the four "off" spellings, including a typo, leaves the
    /// default, which is on.
    fn parse(value: Option<&str>) -> bool {
        !matches!(
            value.map(str::trim),
            Some("0") | Some("off") | Some("false") | Some("no")
        )
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// The window a layer's cache evicts behind, given the window that
    /// layer's attention actually reads.
    ///
    /// **These two must be the same number.** The kernel reads the last
    /// `window` rows; keeping fewer answers out of a truncated history,
    /// and this is the only place that says so, so there is nowhere for
    /// a second opinion to live.
    pub fn window_for(&self, attention_window: Option<usize>) -> Option<KvWindow> {
        if !self.enabled {
            return None;
        }
        KvWindow::with_default_slack(attention_window?)
    }

    /// The window layer `layer_idx` of `config` evicts behind.
    ///
    /// The ONE expression that answers it, for the decoder
    /// (`Decoder::kv_window_for_layer`) and for the budget
    /// (`crate::kv_budget::KvResidency::from_config`) alike. Those two
    /// have to agree about which layers keep how much or the budget is
    /// pricing a cache that does not exist, which is #33.
    pub fn layer_window(
        &self,
        config: &crate::config::ModelConfig,
        layer_idx: usize,
    ) -> Option<KvWindow> {
        self.window_for(config.layer_sliding_window(layer_idx))
    }
}

/// Never evicts, as a value. NOT what a run gets, which is
/// [`KvWindowPolicy::from_env`]: a `Default` that read the environment
/// would make every struct literal built with `..Default::default()` in
/// a test depend on the shell it ran in.
impl Default for KvWindowPolicy {
    fn default() -> Self {
        Self::off()
    }
}

impl Decoder {
    /// The window layer `layer_idx`'s host cache may evict behind, or
    /// `None` when it must keep everything.
    ///
    /// One predicate, shared by the decode path, the prefill path and
    /// `kv_budget`'s residency, because four spellings of one
    /// eligibility check is exactly how the GPU-router gate drifted four
    /// ways.
    pub fn kv_window_for_layer(&self, layer_idx: usize) -> Option<KvWindow> {
        self.kv_window.layer_window(&self.config, layer_idx)
    }

    /// Arms `cache` for layer `layer_idx` if this run evicts, then drops
    /// whatever has fallen behind the window. A no-op otherwise.
    ///
    /// Called after the layer's attention has been computed, never
    /// before: the rows this drops are rows the kernel has finished
    /// with, and `forward_batch` reads a whole prefill batch back
    /// against an offset captured before its pushes.
    pub(crate) fn evict_layer_kv(&self, layer_idx: usize, cache: &mut KvCache) {
        let Some(window) = self.kv_window_for_layer(layer_idx) else {
            return;
        };
        if cache.window() != Some(window) {
            cache.arm_window(window);
        }
        cache.evict_behind_window();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The switch is on unless it says off, and every spelling of off is
    /// off. A typo is not one of them.
    #[test]
    fn the_switch_is_on_unless_it_says_off() {
        assert!(KvWindowPolicy::parse(None));
        assert!(KvWindowPolicy::parse(Some("1")));
        assert!(KvWindowPolicy::parse(Some("on")));
        assert!(KvWindowPolicy::parse(Some("of")));
        for off in ["0", "off", "false", "no", " 0 "] {
            assert!(!KvWindowPolicy::parse(Some(off)), "{off:?}");
        }
    }

    /// "Off" means no window for any layer however windowed the model is.
    #[test]
    fn the_off_policy_evicts_nothing() {
        let off = KvWindowPolicy::off();
        assert!(!off.enabled());
        assert_eq!(off.window_for(Some(1024)), None);
        assert_eq!(KvWindowPolicy::default(), off);
    }

    /// The eviction window IS the attention window. A policy that
    /// narrowed it to save more memory would be answering out of a
    /// shorter history than the model asks for, which is a different
    /// model.
    #[test]
    fn the_eviction_window_equals_the_window_attention_reads() {
        let on = KvWindowPolicy::on();
        let w = on.window_for(Some(1024)).expect("a windowed layer");
        assert_eq!(w.window(), 1024);
        // Slack is headroom above the window, never below it.
        assert!(w.max_rows() >= 1024);
    }

    /// A full-attention layer has no window to evict behind, switch on
    /// or off. This is the half of an alternating-SWA model that keeps
    /// costing what it always did.
    #[test]
    fn a_full_attention_layer_never_evicts() {
        assert_eq!(KvWindowPolicy::on().window_for(None), None);
        assert_eq!(KvWindowPolicy::off().window_for(None), None);
    }

    /// A zero window is not a window; it must not become one here by
    /// arithmetic.
    #[test]
    fn a_zero_window_does_not_become_an_evicting_window() {
        assert_eq!(KvWindowPolicy::on().window_for(Some(0)), None);
    }
}
