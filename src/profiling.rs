
#[cfg(feature = "record_timings")]
use std::sync::Mutex;
use std::time::Instant;

pub trait TimeTracker {
    fn reset(&mut self);
    fn print(&self);
}

#[cfg(feature = "record_timings")]
pub static PRINT_TIMINGS: Mutex<Vec<Box<dyn 'static + Send + TimeTracker>>> = Mutex::new(Vec::new());

#[cfg(feature = "record_timings")]
pub fn print_all_timings() {
    let locked = PRINT_TIMINGS.lock().unwrap();
    for tracker in locked.iter() {
        tracker.print();
    }
}

#[cfg(not(feature = "record_timings"))]
pub fn print_all_timings() {}

#[cfg(feature = "record_timings")]
pub fn clear_all_timings() {
    let mut locked = PRINT_TIMINGS.lock().unwrap();
    locked.iter_mut().for_each(|tracker| tracker.reset());
}

#[cfg(not(feature = "record_timings"))]
pub fn clear_all_timings() {}

macro_rules! record_time {
    ($name:literal, $fn:expr) => {
        {
            #[cfg(feature = "record_timings")] {
                use std::sync::atomic::{AtomicBool, Ordering, AtomicU64};
                use std::time::Instant;
                use $crate::profiling::*;

                static COUNTER: AtomicU64 = AtomicU64::new(0);
                static REGISTERED: AtomicBool = AtomicBool::new(false);

                struct LocalTimeTracker;

                impl TimeTracker for LocalTimeTracker {
                    
                    fn reset(&mut self) {
                        COUNTER.store(0, Ordering::SeqCst);
                    }

                    fn print(&self) {
                        println!("{}: {} ms", $name, COUNTER.load(Ordering::SeqCst) / 1000000);
                    }
                }

                if !REGISTERED.swap(true, Ordering::SeqCst) {
                    let mut locked = PRINT_TIMINGS.lock().unwrap();
                    locked.push(Box::new(LocalTimeTracker) as Box<dyn 'static + Send + TimeTracker>);
                }

                #[inline(never)]
                fn prevent_inline<T, F: FnOnce() -> T>(f: F) -> T {
                    f()
                }

                let start = Instant::now();
                let result = prevent_inline($fn);
                let end = Instant::now();
                COUNTER.fetch_add((end - start).as_nanos() as u64 + 1, Ordering::SeqCst);
                result
            }
            #[cfg(not(feature = "record_timings"))] {
                ($fn)()
            }
        }
    };
}

pub fn log_time<F, T, const LOG: bool, const COUNTER_VAR_COUNT: usize>(description: &str, step_fn: F) -> T
    where F: FnOnce(&mut [usize; COUNTER_VAR_COUNT]) -> T
{
    if LOG {
        println!("{}", description);
    }
    let mut counters = [0; COUNTER_VAR_COUNT];
    let start = Instant::now();
    let result = step_fn(&mut counters);
    let end = Instant::now();
    if LOG {
        println!("done in {} ms, {:?}", (end - start).as_millis(), counters);
    }
    return result;
}