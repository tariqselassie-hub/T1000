//! Multithreaded statevector kernels.
//!
//! Work runs on a dedicated rayon pool whose size comes from `GUDDALM_QNN_THREADS` (read once,
//! at first use) or, if unset, the number of logical CPUs. States smaller than
//! [`PARALLEL_MIN_DIM`] amplitudes run single-threaded, where thread overhead would dominate.
//!
//! Every kernel here only partitions independent amplitude pairs or blocks across threads; no
//! value is reduced across threads, so results are deterministic and identical for any thread
//! count above one. The multithreaded high-qubit Ry path uses plain multiply-add instead of FMA,
//! so it can differ from the single-threaded kernel in the last bit (relative ~1e-16).

use rayon::prelude::*;
use std::sync::OnceLock;

/// Minimum number of qubits required to engage multithreading.
/// States with <= 16 qubits (dim <= 65,536 amplitudes, <= 1MB) fit in cache
/// and execute sequentially via AVX2 to eliminate threadpool scheduling overhead.
pub const PARALLEL_MIN_QUBITS: usize = 17;

/// Smallest statevector (in amplitudes) that is split across threads (2^17 = 131,072).
pub const PARALLEL_MIN_DIM: usize = 1 << PARALLEL_MIN_QUBITS;

/// Target number of work items per thread, for load balancing.
const TASKS_PER_THREAD: usize = 4;

fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::env::var("GUDDALM_QNN_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&t| t >= 1)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("guddalm-qnn-{i}"))
            .build()
            .expect("failed to build t1000 thread pool")
    })
}

/// Runs `f` inside the crate's thread pool (for rayon parallel iterators).
pub(crate) fn install<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    pool().install(f)
}

/// Number of threads used by the parallel kernels.
pub fn num_threads() -> usize {
    pool().current_num_threads()
}

fn use_parallel(dim: usize) -> bool {
    dim >= PARALLEL_MIN_DIM && num_threads() > 1
}

/// Largest power of two `<= dim / (threads * TASKS_PER_THREAD)`, but at least `min`.
fn chunk_len(dim: usize, min: usize) -> usize {
    let target = (dim / (num_threads() * TASKS_PER_THREAD)).max(1);
    let pow2 = 1usize << (usize::BITS - 1 - target.leading_zeros());
    pow2.max(min).min(dim)
}

/// Multithreaded single-qubit Ry. Same result as [`crate::simd::apply_ry_kernel`].
pub fn apply_ry(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let n_qubits = crate::simd::assert_statevector(real, imag);
    crate::simd::assert_qubit(qubit, n_qubits);
    let dim = real.len();
    if !use_parallel(dim) {
        crate::tiled_butterfly::tiled_apply_ry(real, imag, qubit, theta);
        return;
    }

    let block = 1usize << (qubit + 1);
    let chunk = chunk_len(dim, 1);
    pool().install(|| {
        if block <= chunk {
            // Butterfly blocks fit inside a chunk: chunks are independent sub-statevectors.
            real.par_chunks_mut(chunk)
                .zip(imag.par_chunks_mut(chunk))
                .for_each(|(re, im)| crate::simd::apply_ry_kernel(re, im, qubit, theta));
        } else {
            // High qubit: pair the |0> and |1> halves of each block chunk by chunk.
            let (s, c) = (theta * 0.5).sin_cos();
            let half = block / 2;
            let sub = chunk.min(half);
            for (re_block, im_block) in real.chunks_mut(block).zip(imag.chunks_mut(block)) {
                let (re0, re1) = re_block.split_at_mut(half);
                let (im0, im1) = im_block.split_at_mut(half);
                re0.par_chunks_mut(sub)
                    .zip(re1.par_chunks_mut(sub))
                    .zip(im0.par_chunks_mut(sub).zip(im1.par_chunks_mut(sub)))
                    .for_each(|((r0, r1), (i0, i1))| ry_pair(r0, r1, i0, i1, c, s));
            }
        }
    });
}

/// `[x0; x1] <- [c -s; s c] [x0; x1]` elementwise for paired slices.
fn ry_pair(r0: &mut [f64], r1: &mut [f64], i0: &mut [f64], i1: &mut [f64], c: f64, s: f64) {
    for k in 0..r0.len() {
        let (a, b) = (r0[k], r1[k]);
        r0[k] = c * a - s * b;
        r1[k] = s * a + c * b;
        let (a, b) = (i0[k], i1[k]);
        i0[k] = c * a - s * b;
        i1[k] = s * a + c * b;
    }
}

/// Fixed chunk length for deterministic parallel reductions (independent of the thread count).
const REDUCE_CHUNK: usize = 1 << 14;

/// Sums `f(range)` over `0..len` split into fixed chunks. Partial sums are combined in chunk
/// order, so the result is identical for any thread count. Each partial is a vector of `width`
/// values (one per output, e.g. one per qubit).
pub(crate) fn chunked_sum<F>(len: usize, width: usize, f: F) -> Vec<f64>
where
    F: Fn(std::ops::Range<usize>, &mut [f64]) + Sync,
{
    let n_chunks = len.div_ceil(REDUCE_CHUNK).max(1);
    let run = |c: usize| {
        let mut acc = vec![0.0; width];
        f(c * REDUCE_CHUNK..((c + 1) * REDUCE_CHUNK).min(len), &mut acc);
        acc
    };
    let partials: Vec<Vec<f64>> = if n_chunks > 1 && num_threads() > 1 {
        pool().install(|| (0..n_chunks).into_par_iter().map(run).collect())
    } else {
        (0..n_chunks).map(run).collect()
    };
    let mut total = vec![0.0; width];
    for p in partials {
        for (t, v) in total.iter_mut().zip(p) {
            *t += v;
        }
    }
    total
}

/// Runs `f(block_index, re_block, im_block)` over consecutive blocks of `block` amplitudes,
/// in parallel for large states. Used by the fused diagonal layer.
pub(crate) fn for_each_block<F>(real: &mut [f64], imag: &mut [f64], block: usize, f: F)
where
    F: Fn(usize, &mut [f64], &mut [f64]) + Sync,
{
    let dim = real.len();
    if !use_parallel(dim) {
        for (h, (re, im)) in real.chunks_mut(block).zip(imag.chunks_mut(block)).enumerate() {
            f(h, re, im);
        }
        return;
    }
    let chunk = chunk_len(dim, block);
    let blocks_per_chunk = chunk / block;
    pool().install(|| {
        real.par_chunks_mut(chunk)
            .zip(imag.par_chunks_mut(chunk))
            .enumerate()
            .for_each(|(t, (re, im))| {
                for (k, (rb, ib)) in re.chunks_mut(block).zip(im.chunks_mut(block)).enumerate() {
                    f(t * blocks_per_chunk + k, rb, ib);
                }
            });
    });
}
