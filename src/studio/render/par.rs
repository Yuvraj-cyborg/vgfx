//! Work shared out over threads.
//!
//! A pass is cut into pieces that each write only their own part of the
//! picture (as a rule a band of rows one tile high) and read nothing another
//! piece of the same pass writes. Each piece does what one thread would have
//! done for those rows, in the same order, so the picture is the same to the
//! bit whatever number of threads draws it, and however the pieces fall to
//! them.
//!
//! The threads that help are kept in one pool for the whole process (`pool`):
//! a frame has a score of passes, and starting threads for each of them would
//! cost more than some of the passes do. They are started the first time a
//! pass asks for them, and wait for the next pass in between.

use std::sync::Mutex;

/// The most threads one frame uses.
pub const MAX_THREADS: usize = 64;

/// Whether everything is done on the calling thread: WebAssembly with no
/// system under it (`wasm32-unknown-unknown`) cannot start a thread. Under
/// WASI the pool starts as many as the host gives it, which is threads with
/// `wasm32-wasip1-threads` and a host that has wasi-threads, and none
/// otherwise; with none, the caller does everything.
pub const SERIAL: bool = cfg!(all(target_family = "wasm", target_os = "unknown"));

/// Runs `f` on every item, on up to `threads` threads: the calling one and
/// as many from the pool as there are items to share, each thread with its
/// own `scratch`. Items are handed out in order as threads come free. With
/// one thread, and in WebAssembly that cannot start threads, the items are
/// done in order here.
pub fn each<I, T, S, F>(threads: usize, items: I, scratch: &mut [S], f: F)
where
    I: Iterator<Item = T> + Send,
    T: Send,
    S: Send,
    F: Fn(&mut S, T) + Sync,
{
    let mut n = threads.clamp(1, scratch.len().max(1));
    n = n.min(items.size_hint().0.max(1));
    if n == 1 || SERIAL {
        let s = &mut scratch[0];
        for it in items {
            f(s, it);
        }
        return;
    }
    let queue = Mutex::new(items);
    let next = || queue.lock().unwrap_or_else(|e| e.into_inner()).next();
    let (mine, rest) = scratch[..n].split_first_mut().unwrap();
    // Each helper takes the scratch of its own number; the locks are never
    // contended, they only hand a `&mut` across.
    let rest: Vec<Mutex<&mut S>> = rest.iter_mut().map(Mutex::new).collect();
    pool::run(
        rest.len(),
        &|k| {
            let mut s = rest[k].lock().unwrap_or_else(|e| e.into_inner());
            while let Some(it) = next() {
                f(&mut s, it);
            }
        },
        || {
            while let Some(it) = next() {
                f(mine, it);
            }
        },
    );
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

/// `each`, with the items handed out dearest first by `cost` when there is
/// more than one thread: the long pieces start first and the short ones fill
/// in behind them. The order never changes a result.
pub fn each_by_cost<I, T, S, C, F>(threads: usize, items: I, scratch: &mut [S], cost: C, f: F)
where
    I: Iterator<Item = T> + Send,
    T: Send,
    S: Send,
    C: Fn(&T) -> usize,
    F: Fn(&mut S, T) + Sync,
{
    if threads <= 1 || SERIAL {
        return each(1, items, scratch, f);
    }
    let mut all: Vec<T> = items.collect();
    all.sort_by_cached_key(|it| core::cmp::Reverse(cost(it)));
    each(threads, all.into_iter(), scratch, f);
}

/// `each_by_cost` for work that needs no scratch of its own.
pub fn each0_by_cost<I, T, C, F>(threads: usize, items: I, cost: C, f: F)
where
    I: Iterator<Item = T> + Send,
    T: Send,
    C: Fn(&T) -> usize,
    F: Fn(T) + Sync,
{
    let mut none = [(); MAX_THREADS];
    each_by_cost(threads, items, &mut none, cost, |_, it| f(it));
}

/// Threads kept between passes.
///
/// A caller puts one task per helper it wants into a queue, does its own
/// share, then takes back the tasks no thread has started and waits for the
/// ones that have. Helpers that find the queue empty look again a few times
/// before they sleep. Several callers may share the pool at once; each waits
/// only for its own tasks.
///
/// A thread that cannot be started is not asked for again: the pool works
/// with the threads it has, and with none the caller does everything. So a
/// host that gives fewer threads than asked for (a browser page with a few
/// workers) gets the same pictures, only later.
mod pool {
    use std::collections::VecDeque;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Condvar, Mutex, MutexGuard};

    /// What helpers run: `help(k)` for helper number `k`.
    type Help<'a> = dyn Fn(usize) + Sync + 'a;

    /// One caller's work, on the caller's stack.
    struct Job {
        /// Borrowed from the caller. The caller does not return before every
        /// task that names this job is either back out of the queue or done.
        help: *const Help<'static>,
        /// Tasks of this job that a helper has started and not finished;
        /// only changed with the pool's lock held.
        busy: AtomicUsize,
        panicked: AtomicBool,
    }

    struct Task {
        job: *const Job,
        k: usize,
    }

    // A task only travels to a helper inside the pool's lock, and the job it
    // points to outlives it (see `Job::help`).
    unsafe impl Send for Task {}

    struct State {
        /// Helper threads started.
        threads: usize,
        /// A thread could not be started: no more are tried.
        full: bool,
        tasks: VecDeque<Task>,
    }

    struct Pool {
        state: Mutex<State>,
        /// Helpers sleep here while there is nothing to do.
        work: Condvar,
        /// Callers wait here for their helpers to finish.
        done: Condvar,
        /// How many tasks are queued, for helpers looking before they sleep.
        queued: AtomicUsize,
    }

    static POOL: Pool = Pool {
        state: Mutex::new(State {
            threads: 0,
            full: false,
            tasks: VecDeque::new(),
        }),
        work: Condvar::new(),
        done: Condvar::new(),
        queued: AtomicUsize::new(0),
    };

    /// How many times an idle helper looks at the queue before it sleeps.
    /// Looking much longer than this only takes cores from the threads still
    /// working (measured natively and in WebAssembly, eight threads).
    const LOOKS: u32 = 1 << 6;

    impl Pool {
        fn lock(&self) -> MutexGuard<'_, State> {
            self.state.lock().unwrap_or_else(|e| e.into_inner())
        }
    }

    /// Runs `own` here while up to `helpers` threads of the pool run
    /// `help(0)`, `help(1)`, ... and returns when all of them are done.
    /// `help` must do nothing harmful when there is nothing left for it to
    /// do: the helpers that start late find the work gone.
    pub fn run(helpers: usize, help: &Help<'_>, own: impl FnOnce()) {
        // SAFETY: only the lifetime is erased. `Wait` below does not let this
        // function return or unwind until no task points at `job`.
        let help: *const Help<'static> =
            unsafe { core::mem::transmute::<&Help<'_>, &'static Help<'static>>(help) };
        let job = Job {
            help,
            busy: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
        };
        {
            let mut st = POOL.lock();
            grow(&mut st, helpers);
            let n = helpers.min(st.threads);
            for k in 0..n {
                st.tasks.push_back(Task { job: &job, k });
            }
            POOL.queued.fetch_add(n, Ordering::Release);
            for _ in 0..n {
                POOL.work.notify_one();
            }
        }
        let wait = Wait(&job);
        own();
        drop(wait);
        if job.panicked.load(Ordering::Relaxed) {
            panic!("a thread drawing a frame panicked");
        }
    }

    /// Takes a job's unstarted tasks back and waits for its started ones,
    /// when the caller is done or unwinding.
    struct Wait<'a>(&'a Job);

    impl Drop for Wait<'_> {
        fn drop(&mut self) {
            let job: *const Job = self.0;
            let mut st = POOL.lock();
            let before = st.tasks.len();
            st.tasks.retain(|t| !core::ptr::eq(t.job, job));
            POOL.queued
                .fetch_sub(before - st.tasks.len(), Ordering::Relaxed);
            while self.0.busy.load(Ordering::Relaxed) > 0 {
                st = POOL.done.wait(st).unwrap_or_else(|e| e.into_inner());
            }
        }
    }

    /// Starts helpers until there are `want`, unless one has failed to start.
    fn grow(st: &mut State, want: usize) {
        let want = want.min(super::MAX_THREADS - 1);
        while st.threads < want && !st.full {
            let started = std::thread::Builder::new()
                .name(format!("vgfx-{}", st.threads + 1))
                .spawn(serve);
            match started {
                Ok(_) => st.threads += 1,
                Err(_) => st.full = true,
            }
        }
    }

    /// A helper's life: take a task, run it, look for the next.
    fn serve() {
        let mut st = POOL.lock();
        loop {
            if let Some(task) = st.tasks.pop_front() {
                POOL.queued.fetch_sub(1, Ordering::Relaxed);
                // SAFETY: the job's caller is waiting in `Wait` until `busy`
                // is back to 0, which it can only see after this thread has
                // stopped using the job.
                let job = unsafe { &*task.job };
                job.busy.fetch_add(1, Ordering::Relaxed);
                drop(st);
                // SAFETY: as for the job, which borrows it.
                let help = unsafe { &*job.help };
                let ok = catch_unwind(AssertUnwindSafe(|| help(task.k))).is_ok();
                st = POOL.lock();
                if !ok {
                    job.panicked.store(true, Ordering::Relaxed);
                }
                if job.busy.fetch_sub(1, Ordering::Relaxed) == 1 {
                    POOL.done.notify_all();
                }
                continue;
            }
            drop(st);
            let mut looks = 0;
            while POOL.queued.load(Ordering::Acquire) == 0 && looks < LOOKS {
                core::hint::spin_loop();
                looks += 1;
            }
            st = POOL.lock();
            if st.tasks.is_empty() {
                st = POOL.work.wait(st).unwrap_or_else(|e| e.into_inner());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn every_item_is_done_once_on_any_number_of_threads() {
        for threads in [1, 2, 3, 8, MAX_THREADS] {
            let seen: Vec<AtomicUsize> = (0..1000).map(|_| AtomicUsize::new(0)).collect();
            each0(threads, 0..1000usize, |i| {
                seen[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(
                seen.iter().all(|s| s.load(Ordering::Relaxed) == 1),
                "{threads} threads"
            );
        }
    }

    #[test]
    fn each_thread_keeps_its_own_scratch() {
        let mut scratch = vec![0usize; 8];
        each(8, 0..10_000usize, &mut scratch, |s, i| *s += i);
        assert_eq!(scratch.iter().sum::<usize>(), (0..10_000).sum());
    }

    #[test]
    fn callers_on_several_threads_share_the_pool() {
        std::thread::scope(|sc| {
            for _ in 0..4 {
                sc.spawn(|| {
                    for _ in 0..50 {
                        let total = AtomicUsize::new(0);
                        each0(6, 0..200usize, |i| {
                            total.fetch_add(i, Ordering::Relaxed);
                        });
                        assert_eq!(total.load(Ordering::Relaxed), (0..200).sum());
                    }
                });
            }
        });
    }

    #[test]
    fn a_panic_on_a_helper_reaches_the_caller() {
        let caught = std::panic::catch_unwind(|| {
            each0(4, 0..64usize, |i| {
                if i == 63 {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    panic!("item {i}");
                }
            });
        });
        assert!(caught.is_err());
        // The pool still works afterwards.
        let total = AtomicUsize::new(0);
        each0(4, 0..64usize, |i| {
            total.fetch_add(i, Ordering::Relaxed);
        });
        assert_eq!(total.load(Ordering::Relaxed), (0..64).sum());
    }
}
