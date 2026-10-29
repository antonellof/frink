//! The region helpers: every flat "run this over items in parallel" shape
//! the crate uses, each with a rayon arm and a persistent-pool arm.
//!
//! # One shape per helper, and why the spin arm takes a pool
//!
//! Each public helper is the same three lines: ask [`super::backend`],
//! run the spin arm on the process pool if it said so, otherwise fork
//! with rayon. The spin arm is its own function (`*_on`) that takes the
//! [`CpuPool`] as an argument rather than reaching for the process-wide
//! one. That is what lets a test drive the exact spin-arm code on a
//! private pool and compare it against the rayon arm in one process: the
//! process pool can be busy with another test's region (it refuses a
//! second submitter, by design), and a test that silently fell back to
//! rayon would compare rayon with rayon.

use rayon::prelude::*;

use super::{backend, note_rayon_region, note_spin_region, pool, split_for, Backend};
use crate::cpu_pool::CpuPool;

/// A raw pointer that may cross into worker threads.
///
/// Only ever used to hand each task a *disjoint* sub-slice of one
/// allocation the submitter borrows mutably for the whole region.
struct SendPtr<T>(*mut T);

// Hand-written rather than derived: `#[derive(Copy)]` would add a
// `T: Copy` bound, and the element types here are `f32` today but a
// `Q8Activations` tomorrow.
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}

impl<T> SendPtr<T> {
    /// The element pointer at `offset`.
    ///
    /// A method rather than a field read at the call sites, because
    /// closure capture is per *field*: reading `base.0` inside a task
    /// captures the bare `*mut T`, which is not `Sync`, and the whole
    /// point of this wrapper is the `unsafe impl` below.
    ///
    /// # Safety
    /// `offset` must be within the allocation this was built from.
    unsafe fn at(self, offset: usize) -> *mut T {
        // SAFETY: the caller's invariant.
        unsafe { self.0.add(offset) }
    }
}

// SAFETY: the pointer comes from a `&mut [T]` the submitter holds for
// the duration of the region, and each task derives a sub-slice from a
// half-open index range that no other task's range overlaps. `T: Send`
// is required at every call site, which is what makes moving those
// sub-slices onto worker threads sound.
unsafe impl<T: Send> Send for SendPtr<T> {}
unsafe impl<T: Send> Sync for SendPtr<T> {}

/// The half-open item range task `t` of a `(per, n)` split covers.
fn task_range(t: usize, per: usize, n: usize) -> std::ops::Range<usize> {
    let lo = t * per;
    lo..((t + 1) * per).min(n)
}

/// Run `f(index)` for every `index` in `0..n`.
pub fn indices<F>(n: usize, min_len: usize, f: F)
where
    F: Fn(usize) + Send + Sync,
{
    if n == 0 {
        return;
    }
    if backend() == Backend::Spin && indices_on(pool(), n, &f) {
        note_spin_region();
        return;
    }
    note_rayon_region();
    (0..n)
        .into_par_iter()
        .with_min_len(min_len.max(1))
        .for_each(&f);
}

/// [`indices`]'s spin arm on `pool`; `false` when the pool was busy.
fn indices_on<F>(pool: &CpuPool, n: usize, f: &F) -> bool
where
    F: Fn(usize) + Send + Sync,
{
    let (per, n_tasks) = split_for(pool.num_threads(), n);
    pool.run(n_tasks, &|t: usize| {
        for i in task_range(t, per, n) {
            f(i);
        }
    })
}

/// [`indices`] with a per-task scratch value, the shape rayon spells
/// `for_each_init`. One `S` is created per task, not per index.
pub fn indices_init<S, I, F>(n: usize, min_len: usize, init: I, f: F)
where
    S: Send,
    I: Fn() -> S + Send + Sync,
    F: Fn(&mut S, usize) + Send + Sync,
{
    if n == 0 {
        return;
    }
    if backend() == Backend::Spin && indices_init_on(pool(), n, &init, &f) {
        note_spin_region();
        return;
    }
    note_rayon_region();
    (0..n)
        .into_par_iter()
        .with_min_len(min_len.max(1))
        .for_each_init(&init, |state, i| f(state, i));
}

/// [`indices_init`]'s spin arm on `pool`; `false` when the pool was busy.
fn indices_init_on<S, I, F>(pool: &CpuPool, n: usize, init: &I, f: &F) -> bool
where
    S: Send,
    I: Fn() -> S + Send + Sync,
    F: Fn(&mut S, usize) + Send + Sync,
{
    let (per, n_tasks) = split_for(pool.num_threads(), n);
    pool.run(n_tasks, &|t: usize| {
        let range = task_range(t, per, n);
        if range.is_empty() {
            return;
        }
        let mut state = init();
        for i in range {
            f(&mut state, i);
        }
    })
}

/// Run `f(index, &mut item)` over `data`, the shape rayon spells
/// `par_iter_mut().with_min_len(..).enumerate()`.
pub fn items_mut<T, F>(data: &mut [T], min_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut T) + Send + Sync,
{
    if data.is_empty() {
        return;
    }
    if backend() == Backend::Spin && items_mut_on(pool(), data, &f) {
        note_spin_region();
        return;
    }
    note_rayon_region();
    items_mut_rayon(data, min_len, &f);
}

/// [`items_mut`]'s rayon arm.
fn items_mut_rayon<T, F>(data: &mut [T], min_len: usize, f: &F)
where
    T: Send,
    F: Fn(usize, &mut T) + Send + Sync,
{
    data.par_iter_mut()
        .with_min_len(min_len.max(1))
        .enumerate()
        .for_each(|(i, slot)| f(i, slot));
}

/// [`items_mut`]'s spin arm on `pool`; `false` when the pool was busy.
fn items_mut_on<T, F>(pool: &CpuPool, data: &mut [T], f: &F) -> bool
where
    T: Send,
    F: Fn(usize, &mut T) + Send + Sync,
{
    let n = data.len();
    let base = SendPtr(data.as_mut_ptr());
    let (per, n_tasks) = split_for(pool.num_threads(), n);
    pool.run(n_tasks, &|t: usize| {
        for i in task_range(t, per, n) {
            // SAFETY: `base` points at `data`, borrowed mutably for the
            // whole call and outliving the region (`CpuPool::run` does
            // not return until every worker has stopped touching the
            // closure). Index `i` is inside `0..n` and belongs to exactly
            // one task, so no two of these `&mut T` overlap.
            f(i, unsafe { &mut *base.at(i) });
        }
    })
}

/// [`chunks_mut`] over two slices of the same length at once, the shape
/// rayon spells `a.par_chunks_mut(k).zip(b.par_chunks_mut(k))`.
///
/// Exists because the MoE decode path computes a gate row and an up row
/// from one shared activation: splitting that into two regions would
/// double the region count, which is the thing this whole module is
/// trying to reduce.
pub fn chunks_mut2<T, U, F>(a: &mut [T], b: &mut [U], chunk_len: usize, min_len: usize, f: F)
where
    T: Send,
    U: Send,
    F: Fn(usize, &mut [T], &mut [U]) + Send + Sync,
{
    assert_eq!(a.len(), b.len(), "zipped slices must be the same length");
    chunks_mut2_by(a, b, chunk_len, chunk_len, min_len, f);
}

/// [`chunks_mut2`] with a chunk length per slice: `a` in runs of
/// `len_a`, `b` in runs of `len_b`, the SAME number of chunks (a gated
/// delta-net's per-head `S x S` state beside its per-head `S` output).
pub fn chunks_mut2_by<T, U, F>(
    a: &mut [T],
    b: &mut [U],
    len_a: usize,
    len_b: usize,
    min_len: usize,
    f: F,
) where
    T: Send,
    U: Send,
    F: Fn(usize, &mut [T], &mut [U]) + Send + Sync,
{
    assert!(len_a > 0 && len_b > 0, "chunk lengths must be positive");
    let n_chunks = a.len().div_ceil(len_a);
    assert_eq!(
        n_chunks,
        b.len().div_ceil(len_b),
        "zipped slices must split into the same number of chunks"
    );
    if n_chunks == 0 {
        return;
    }
    if backend() == Backend::Spin && chunks_mut2_by_on(pool(), a, b, len_a, len_b, &f) {
        note_spin_region();
        return;
    }
    note_rayon_region();
    chunks_mut2_by_rayon(a, b, len_a, len_b, min_len, &f);
}

/// [`chunks_mut2_by`]'s rayon arm.
fn chunks_mut2_by_rayon<T, U, F>(
    a: &mut [T],
    b: &mut [U],
    len_a: usize,
    len_b: usize,
    min_len: usize,
    f: &F,
) where
    T: Send,
    U: Send,
    F: Fn(usize, &mut [T], &mut [U]) + Send + Sync,
{
    a.par_chunks_mut(len_a)
        .zip(b.par_chunks_mut(len_b))
        .with_min_len(min_len.max(1))
        .enumerate()
        .for_each(|(c, (ca, cb))| f(c, ca, cb));
}

/// [`chunks_mut2_by`]'s spin arm on `pool`; `false` when the pool was
/// busy. The caller has checked that both slices split into the same
/// non-zero number of chunks.
fn chunks_mut2_by_on<T, U, F>(
    pool: &CpuPool,
    a: &mut [T],
    b: &mut [U],
    len_a: usize,
    len_b: usize,
    f: &F,
) -> bool
where
    T: Send,
    U: Send,
    F: Fn(usize, &mut [T], &mut [U]) + Send + Sync,
{
    let n_chunks = a.len().div_ceil(len_a);
    let (total_a, total_b) = (a.len(), b.len());
    let base_a = SendPtr(a.as_mut_ptr());
    let base_b = SendPtr(b.as_mut_ptr());
    let (per, n_tasks) = split_for(pool.num_threads(), n_chunks);
    pool.run(n_tasks, &|t: usize| {
        for c in task_range(t, per, n_chunks) {
            // SAFETY: both pointers come from slices the caller borrows
            // mutably for the whole call, splitting into the same number
            // of chunks, and chunk `c` of each is visited by exactly one
            // task.
            unsafe {
                f(
                    c,
                    chunk_of(base_a, total_a, len_a, c),
                    chunk_of(base_b, total_b, len_b, c),
                );
            }
        }
    })
}

/// Run `f(chunk_index, &mut chunk)` over `data` split into runs of
/// `chunk_len`, the shape rayon spells `par_chunks_mut(chunk_len)`.
///
/// A trailing partial chunk is delivered short, exactly as
/// `par_chunks_mut` does.
pub fn chunks_mut<T, F>(data: &mut [T], chunk_len: usize, min_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    assert!(chunk_len > 0, "chunk length must be positive");
    if data.is_empty() {
        return;
    }
    if backend() == Backend::Spin && chunks_mut_on(pool(), data, chunk_len, &f) {
        note_spin_region();
        return;
    }
    note_rayon_region();
    data.par_chunks_mut(chunk_len)
        .with_min_len(min_len.max(1))
        .enumerate()
        .for_each(|(c, chunk)| f(c, chunk));
}

/// [`chunks_mut`]'s spin arm on `pool`; `false` when the pool was busy.
fn chunks_mut_on<T, F>(pool: &CpuPool, data: &mut [T], chunk_len: usize, f: &F) -> bool
where
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    let len = data.len();
    let n_chunks = len.div_ceil(chunk_len);
    let base = SendPtr(data.as_mut_ptr());
    let (per, n_tasks) = split_for(pool.num_threads(), n_chunks);
    pool.run(n_tasks, &|t: usize| {
        for c in task_range(t, per, n_chunks) {
            // SAFETY: see `chunks_mut_init_on`; the ranges are the same
            // disjoint half-open chunks of one live borrow.
            f(c, unsafe { chunk_of(base, len, chunk_len, c) });
        }
    })
}

/// The `c`-th `chunk_len`-sized chunk of the `len`-element allocation at
/// `base`, delivered short when it is the trailing one.
///
/// # Safety
/// `base` must point at a live allocation of at least `len` elements
/// that outlives the returned slice, and the caller must guarantee that
/// no other live slice covers chunk `c` -- which the callers do by
/// visiting each chunk index from exactly one task.
unsafe fn chunk_of<'a, T>(base: SendPtr<T>, len: usize, chunk_len: usize, c: usize) -> &'a mut [T] {
    let start = c * chunk_len;
    let end = ((c + 1) * chunk_len).min(len);
    debug_assert!(start < end && end <= len);
    // SAFETY: the caller's invariants, plus `start..end` being inside
    // `0..len` by construction of `c < len.div_ceil(chunk_len)`.
    unsafe { std::slice::from_raw_parts_mut(base.at(start), end - start) }
}

/// [`chunks_mut`] with a per-task scratch value.
pub fn chunks_mut_init<T, S, I, F>(data: &mut [T], chunk_len: usize, min_len: usize, init: I, f: F)
where
    T: Send,
    S: Send,
    I: Fn() -> S + Send + Sync,
    F: Fn(&mut S, usize, &mut [T]) + Send + Sync,
{
    assert!(chunk_len > 0, "chunk length must be positive");
    if data.is_empty() {
        return;
    }
    if backend() == Backend::Spin && chunks_mut_init_on(pool(), data, chunk_len, &init, &f) {
        note_spin_region();
        return;
    }
    note_rayon_region();
    chunks_mut_init_rayon(data, chunk_len, min_len, &init, &f);
}

/// [`chunks_mut_init`]'s rayon arm.
fn chunks_mut_init_rayon<T, S, I, F>(
    data: &mut [T],
    chunk_len: usize,
    min_len: usize,
    init: &I,
    f: &F,
) where
    T: Send,
    S: Send,
    I: Fn() -> S + Send + Sync,
    F: Fn(&mut S, usize, &mut [T]) + Send + Sync,
{
    data.par_chunks_mut(chunk_len)
        .with_min_len(min_len.max(1))
        .enumerate()
        .for_each_init(init, |state, (c, chunk)| f(state, c, chunk));
}

/// [`chunks_mut_init`]'s spin arm on `pool`; `false` when the pool was
/// busy.
fn chunks_mut_init_on<T, S, I, F>(
    pool: &CpuPool,
    data: &mut [T],
    chunk_len: usize,
    init: &I,
    f: &F,
) -> bool
where
    T: Send,
    S: Send,
    I: Fn() -> S + Send + Sync,
    F: Fn(&mut S, usize, &mut [T]) + Send + Sync,
{
    let len = data.len();
    let n_chunks = len.div_ceil(chunk_len);
    let base = SendPtr(data.as_mut_ptr());
    let (per, n_tasks) = split_for(pool.num_threads(), n_chunks);
    pool.run(n_tasks, &|t: usize| {
        let range = task_range(t, per, n_chunks);
        if range.is_empty() {
            return;
        }
        let mut state = init();
        for c in range {
            // SAFETY: `base` points at `data`, which the caller borrows
            // mutably for this whole call and which outlives the region
            // (`CpuPool::run` does not return until every worker has
            // stopped touching the closure). Chunk index `c` is visited
            // by exactly one task, so no two of these slices overlap.
            f(&mut state, c, unsafe { chunk_of(base, len, chunk_len, c) });
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Mutex;

    /// Both arms must visit every index exactly once and produce the
    /// same answer, whichever one the env var picked -- that is the
    /// property the whole switch rests on.
    #[test]
    fn indices_visits_every_index_exactly_once() {
        for n in [0usize, 1, 7, 64, 5000] {
            let hits: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
            indices(n, 8, |i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "n={n}");
        }
    }

    #[test]
    fn items_mut_writes_every_slot_with_its_own_index() {
        for n in [0usize, 1, 9, 257] {
            let mut data = vec![0u32; n];
            items_mut(&mut data, 4, |i, slot| *slot = i as u32 + 1);
            assert_eq!(data, (1..=n as u32).collect::<Vec<_>>(), "n={n}");
        }
    }

    /// The trailing partial chunk is the easy thing to lose, and losing
    /// it silently drops the last rows of a matvec.
    #[test]
    fn chunks_mut_delivers_a_short_trailing_chunk() {
        let mut data = vec![0u32; 10];
        let seen: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
        chunks_mut(&mut data, 4, 1, |c, chunk| {
            seen.lock().unwrap().push((c, chunk.len()));
            for (i, slot) in chunk.iter_mut().enumerate() {
                *slot = (c * 4 + i) as u32;
            }
        });
        let mut seen = seen.into_inner().unwrap();
        seen.sort_unstable();
        assert_eq!(seen, vec![(0, 4), (1, 4), (2, 2)]);
        assert_eq!(data, (0..10).collect::<Vec<u32>>());
    }

    /// Per-task scratch is created per task, never shared between two
    /// tasks that might run at the same time.
    #[test]
    fn chunks_mut_init_gives_each_task_its_own_scratch() {
        let mut data = vec![0u64; 512];
        chunks_mut_init(
            &mut data,
            8,
            1,
            || Vec::<u64>::with_capacity(8),
            |scratch: &mut Vec<u64>, c, chunk| {
                scratch.clear();
                scratch.extend(chunk.iter().map(|_| c as u64));
                chunk.copy_from_slice(scratch);
            },
        );
        for (c, chunk) in data.chunks(8).enumerate() {
            assert!(chunk.iter().all(|&v| v == c as u64));
        }
    }

    // ---------------------------------------------------------------
    // The spin arms, driven on a private pool and held against their
    // rayon arms. `the_two_cpu_schedulers_produce_token_identical_output`
    // (frink-models) covers `indices`, `indices_init` and `chunks_mut`
    // end to end, and states in its own docs that a dense llama decode
    // does not reach these three, so a row-skipping bug in any of them
    // passed every suite.
    // ---------------------------------------------------------------

    /// Widths that exercise a single-thread pool (everything on the
    /// submitter), an even split and an odd one.
    const POOL_WIDTHS: [usize; 3] = [1, 3, 4];

    /// Lengths that are empty-adjacent, prime, smaller than the pool's
    /// task budget, and far larger than it, so the last task is short
    /// in some and absent in others.
    const LENGTHS: [usize; 6] = [1, 2, 7, 31, 97, 1000];

    /// The value a slot or chunk element must hold after the closures
    /// below run. Index- and position-dependent, so a slot written by
    /// the wrong task, or twice, or not at all, reads differently.
    fn mark(index: usize, offset: usize) -> u32 {
        (index as u32).wrapping_mul(2_654_435_761) ^ (offset as u32 + 1)
    }

    /// `items_mut`'s spin arm writes the same slots as its rayon arm.
    ///
    /// Each slot also counts its visits, so a range that overlaps its
    /// neighbour fails on the count even where the value would agree.
    ///
    /// Sabotage, run: `task_range(t, per, n)` -> `task_range(t, per,
    /// n.saturating_sub(1))` in `items_mut_on` leaves the last slot zero
    /// and this goes red at every width.
    #[test]
    fn items_mut_spin_arm_matches_its_rayon_twin() {
        for width in POOL_WIDTHS {
            let pool = CpuPool::new(width);
            for n in LENGTHS {
                let f = |i: usize, slot: &mut (u32, u32)| {
                    slot.0 = mark(i, 0);
                    slot.1 += 1;
                };
                let mut spun = vec![(0u32, 0u32); n];
                assert!(items_mut_on(&pool, &mut spun, &f), "private pool is idle");
                let mut forked = vec![(0u32, 0u32); n];
                items_mut_rayon(&mut forked, 4, &f);
                assert_eq!(spun, forked, "width={width} n={n}");
                assert!(spun.iter().all(|s| s.1 == 1), "width={width} n={n}");
            }
        }
    }

    /// `chunks_mut2_by`'s spin arm hands chunk `c` of BOTH slices to the
    /// same call, with the right lengths, including the short trailing
    /// chunk of each.
    ///
    /// The chunk lengths differ (`3` and `5`) so a spin arm that used
    /// one slice's chunk length for the other would mis-slice; and the
    /// slice lengths are chosen so that only `a` has a short last chunk.
    ///
    /// Sabotage, run: passing `len_a` for `b` in `chunks_mut2_by_on`'s
    /// second `chunk_of` turns this red (or trips the debug assertion).
    #[test]
    fn chunks_mut2_by_spin_arm_matches_its_rayon_twin() {
        let (len_a, len_b) = (3usize, 5usize);
        for width in POOL_WIDTHS {
            let pool = CpuPool::new(width);
            for n_chunks in LENGTHS {
                let total_a = n_chunks * len_a - 1;
                let total_b = n_chunks * len_b;
                let f = |c: usize, ca: &mut [u32], cb: &mut [u32]| {
                    for (o, v) in ca.iter_mut().enumerate() {
                        *v = v.wrapping_add(mark(c, o));
                    }
                    for (o, v) in cb.iter_mut().enumerate() {
                        *v = v.wrapping_add(mark(c, o + 100) ^ ca.len() as u32);
                    }
                };
                let (mut sa, mut sb) = (vec![0u32; total_a], vec![0u32; total_b]);
                assert!(
                    chunks_mut2_by_on(&pool, &mut sa, &mut sb, len_a, len_b, &f),
                    "private pool is idle"
                );
                let (mut ra, mut rb) = (vec![0u32; total_a], vec![0u32; total_b]);
                chunks_mut2_by_rayon(&mut ra, &mut rb, len_a, len_b, 1, &f);
                assert_eq!(sa, ra, "a: width={width} chunks={n_chunks}");
                assert_eq!(sb, rb, "b: width={width} chunks={n_chunks}");
                assert!(sa.iter().all(|&v| v != 0), "a has an unvisited element");
            }
        }
    }

    /// `chunks_mut_init`'s spin arm visits every chunk once and builds
    /// one scratch per task, never one per process or per chunk.
    ///
    /// The scratch counts how many chunks it has served; the per-chunk
    /// value records that count, so the answer is order-dependent
    /// WITHIN a task and only the total is comparable with rayon (whose
    /// task boundaries differ). What must agree exactly is the chunk
    /// payload, and that every chunk ran once.
    ///
    /// Sabotage, run: `task_range(t, per, n_chunks)` -> `task_range(t,
    /// per, n_chunks.saturating_sub(1))` in `chunks_mut_init_on` leaves
    /// the final chunk untouched and this goes red.
    #[test]
    fn chunks_mut_init_spin_arm_matches_its_rayon_twin() {
        let chunk_len = 4usize;
        for width in POOL_WIDTHS {
            let pool = CpuPool::new(width);
            for n_chunks in LENGTHS {
                let total = n_chunks * chunk_len - 2;
                let inits = AtomicU32::new(0);
                let init = || {
                    inits.fetch_add(1, Ordering::Relaxed);
                    0u32
                };
                let f = |served: &mut u32, c: usize, chunk: &mut [u32]| {
                    *served += 1;
                    for (o, v) in chunk.iter_mut().enumerate() {
                        *v = v.wrapping_add(mark(c, o));
                    }
                };
                let mut spun = vec![0u32; total];
                assert!(
                    chunks_mut_init_on(&pool, &mut spun, chunk_len, &init, &f),
                    "private pool is idle"
                );
                let spin_inits = inits.swap(0, Ordering::Relaxed);
                let mut forked = vec![0u32; total];
                chunks_mut_init_rayon(&mut forked, chunk_len, 1, &init, &f);
                assert_eq!(spun, forked, "width={width} chunks={n_chunks}");
                let (per, _) = split_for(width, n_chunks);
                assert_eq!(
                    spin_inits as usize,
                    n_chunks.div_ceil(per),
                    "one scratch per non-empty task: width={width} chunks={n_chunks}"
                );
            }
        }
    }
}
