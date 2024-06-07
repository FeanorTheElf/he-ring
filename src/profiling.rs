
#[cfg(feature = "record_timings")]
use std::sync::Mutex;

#[cfg(feature = "record_timings")]
pub static PRINT_TIMINGS: Mutex<Vec<(&'static std::sync::atomic::AtomicBool, Box<dyn 'static + Send + Fn()>)>> = Mutex::new(Vec::new());

#[cfg(feature = "record_timings")]
pub fn print_all_timings() {
    let locked = PRINT_TIMINGS.lock().unwrap();
    for (_, f) in locked.iter() {
        f();
    }
}

#[cfg(not(feature = "record_timings"))]
pub fn print_all_timings() {}

#[cfg(feature = "record_timings")]
pub fn clear_all_timings() {
    let mut locked = PRINT_TIMINGS.lock().unwrap();
    locked.drain(..).for_each(|(registered, _)| registered.store(false, std::sync::atomic::Ordering::SeqCst));
}

#[cfg(not(feature = "record_timings"))]
pub fn clear_all_timings() {}

macro_rules! timed {
    ($name:literal, $fn:expr) => {
        {
            #[cfg(feature = "record_timings")] {
                use std::sync::atomic::{AtomicBool, Ordering, AtomicU64};
                use std::time::Instant;
                use crate::profiling::PRINT_TIMINGS;

                static COUNTER: AtomicU64 = AtomicU64::new(0);
                static REGISTERED: AtomicBool = AtomicBool::new(false);
                if !REGISTERED.swap(true, Ordering::SeqCst) {
                    let mut locked = PRINT_TIMINGS.lock().unwrap();
                    locked.push((&REGISTERED, Box::new(|| {
                        println!("{}: {} ms", $name, COUNTER.load(Ordering::SeqCst) / 1000000);
                    })));
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
