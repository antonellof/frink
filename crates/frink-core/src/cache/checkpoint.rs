//! Checkpoint and restore of a cache's recurrent state, so a rejected
//! speculative block can be undone on a layer whose state is a reduction
//! over the whole prefix.
//!
//! [`KvCache::truncate`] cannot roll a recurrent state back to a middle
//! position: there is no "one token ago" in it. llama.cpp's server
//! speculates on such a target all the same, by saving the state before
//! the verify batch and, on a partial accept, restoring it and replaying
//! the accepted tokens (`tools/server/server-context.cpp:3938-3949` and
//! `:4960-4993` at the pin, the `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` path).
//! This is that save and restore. The replay is the caller's: it is a
//! forward pass.
//!
//! The attention rows need no copy: they are a history and truncate as
//! they always have. Only the recurrent state is copied, and it is put
//! back IN PLACE, into the same page-aligned allocation, because a Metal
//! kernel wraps those bytes without copying them and caches the wrap by
//! address (`frink_metal::gdn::buffer_no_copy`).

use super::KvCache;
use crate::recurrent_state::RecurrentState;

/// What [`KvCache::restore`] needs to put a cache back where it was.
#[derive(Debug, Clone, PartialEq)]
pub struct KvCheckpoint {
    positions: usize,
    /// `None` when the layer had not run a token: its state was the
    /// zeros a fresh sequence starts from.
    recurrent: Option<RecurrentState>,
}

impl KvCheckpoint {
    /// The positions the cache held when the checkpoint was taken.
    pub fn positions(&self) -> usize {
        self.positions
    }
}

impl KvCache {
    /// Saves what [`Self::restore`] needs: the position, and a copy of
    /// the recurrent state when the layer has one.
    pub fn checkpoint(&self) -> KvCheckpoint {
        KvCheckpoint {
            positions: self.positions,
            recurrent: self.recurrent.clone(),
        }
    }

    /// How many positions written after a [`Self::checkpoint`] a
    /// [`Self::restore`] can undo: [`Self::rollback_limit`] without the
    /// recurrent state's zero, since the checkpoint holds that state.
    pub fn rollback_limit_with_checkpoint(&self) -> usize {
        match self.window {
            None => usize::MAX,
            Some(w) => w.max_rollback(),
        }
    }

    /// Puts the cache back to `checkpoint`: the rows written since are
    /// dropped, and the recurrent state is the one saved.
    ///
    /// Panics when the cache is shorter than the checkpoint, when the
    /// rows cannot be dropped (a window drained past them; the caller
    /// bounds the block by [`Self::rollback_limit_with_checkpoint`]), or
    /// when the state's shape changed, which no forward pass does.
    pub fn restore(&mut self, checkpoint: &KvCheckpoint) {
        let target = checkpoint.positions;
        assert!(
            target <= self.positions,
            "restore to {target} positions, but the cache holds only {}",
            self.positions
        );
        // The rows, as `truncate` drops them, with the recurrent state
        // set aside: `truncate` refuses a middle position on a cache
        // that holds one, and this one is about to be replaced.
        let state = self.recurrent.take();
        if target < self.positions {
            self.truncate(target);
        }
        self.recurrent = match (state, &checkpoint.recurrent) {
            (Some(mut live), Some(saved)) => {
                assert!(
                    live.conv.len() == saved.conv.len() && live.ssm.len() == saved.ssm.len(),
                    "a recurrent state changed shape between checkpoint and restore"
                );
                live.conv.copy_from_slice(&saved.conv);
                live.ssm.copy_from_slice(&saved.ssm);
                Some(live)
            }
            // Created after the checkpoint: the saved state is a fresh
            // sequence's zeros, written into the live allocation.
            (Some(mut live), None) => {
                live.conv.fill(0.0);
                live.ssm.fill(0.0);
                Some(live)
            }
            (None, saved) => saved.clone(),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_store::KvDtype;

    fn cache_with_state(positions: usize) -> KvCache {
        let mut c = KvCache::new(1, 2).with_dtype(KvDtype::F32);
        for p in 0..positions {
            let row = [p as f32; 2];
            c.push(&row, &row).unwrap();
        }
        let mut s = RecurrentState::zeros(3, 4);
        s.conv.copy_from_slice(&[1.0, 2.0, 3.0]);
        s.ssm.copy_from_slice(&[4.0, 5.0, 6.0, 7.0]);
        c.recurrent = Some(s);
        c
    }

    /// A block written after the checkpoint is undone exactly: rows and
    /// position back, and the state equal to the saved one, in the SAME
    /// allocation a Metal wrap may hold.
    #[test]
    fn restore_undoes_a_block_in_place() {
        let mut c = cache_with_state(3);
        let cp = c.checkpoint();
        let before = c.clone();
        let ptr = c.recurrent.as_ref().unwrap().ssm.as_ptr();

        for p in 3..6 {
            c.push(&[p as f32; 2], &[p as f32; 2]).unwrap();
        }
        let s = c.recurrent.as_mut().unwrap();
        s.conv.fill(9.0);
        s.ssm.fill(9.0);
        assert!(
            !c.can_truncate_to(3),
            "a recurrent cache refuses a plain truncate"
        );

        c.restore(&cp);
        assert_eq!(c.positions(), 3);
        assert_eq!(c.rows(), before.rows());
        assert_eq!(c.k.to_f32_vec(), before.k.to_f32_vec());
        assert_eq!(c.recurrent, before.recurrent);
        assert_eq!(c.recurrent.as_ref().unwrap().ssm.as_ptr(), ptr);
    }

    /// A state created after the checkpoint is put back to a fresh
    /// sequence's zeros.
    #[test]
    fn a_state_created_after_the_checkpoint_restores_to_zeros() {
        let mut c = KvCache::new(1, 2).with_dtype(KvDtype::F32);
        c.push(&[1.0; 2], &[1.0; 2]).unwrap();
        let cp = c.checkpoint();
        c.push(&[2.0; 2], &[2.0; 2]).unwrap();
        let mut s = RecurrentState::zeros(3, 4);
        s.ssm.fill(5.0);
        c.recurrent = Some(s);

        c.restore(&cp);
        assert_eq!(c.positions(), 1);
        assert_eq!(c.recurrent, Some(RecurrentState::zeros(3, 4)));
    }
}
