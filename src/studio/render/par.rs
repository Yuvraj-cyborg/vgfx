//! Work shared out over threads.
//!
//! A pass is cut into pieces that each write only their own part of the
//! picture (as a rule a band of rows one tile high) and read nothing another
//! piece of the same pass writes. Each piece does what one thread would have
//! done for those rows, in the same order, so the picture is the same to the
//! bit whatever number of threads draws it, and however the pieces fall to
//! them.

use std::sync::Mutex;

/// The most threads one frame uses.
pub const MAX_THREADS: usize = 64;

/// Runs `f` on every item, on up to `threads` threads: the calling one and
/// as many more as there are items to share, each thread with its own
/// `scratch`. Items are handed out in order as threads come free. With one
/// thread, and always in WebAssembly, the items are done in order here.
pub fn each<I, T, S, F>(threads: usize, items: I, scratch: &mut [S], f: F)
where
    I: Iterator<Item = T> + Send,
    T: Send,
    S: Send,
    F: Fn(&mut S, T) + Sync,
{
    let mut n = threads.clamp(1, scratch.len().max(1));
    n = n.min(items.size_hint().0.max(1));
    if n == 1 || cfg!(target_arch = "wasm32") {
        let s = &mut scratch[0];
        for it in items {
            f(s, it);
        }
        return;
    }
    let queue = Mutex::new(items);
    let next = || queue.lock().unwrap_or_else(|e| e.into_inner()).next();
    std::thread::scope(|sc| {
        let (mine, rest) = scratch[..n].split_first_mut().unwrap();
        for s in rest {
            let (next, f) = (&next, &f);
            sc.spawn(move || {
                while let Some(it) = next() {
                    f(s, it);
                }
            });
        }
        while let Some(it) = next() {
            f(mine, it);
        }
    });
}

/// `each` for work that needs no scratch of its own.
pub fn each0<I, T, F>(threads: usize, items: I, f: F)
where
    I: Iterator<Item = T> + Send,
    T: Send,
    F: Fn(T) + Sync,
{
    let mut none = [(); MAX_THREADS];
    each(threads, items, &mut none, |_, it| f(it));
}
