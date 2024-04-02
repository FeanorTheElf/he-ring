
#[cfg(feature = "record_timings")]
use std::sync::Mutex;

#[cfg(feature = "record_timings")]
pub static PRINT_TIMINGS: Mutex<Vec<Box<dyn 'static + Send + Fn()>>> = Mutex::new(Vec::new());

#[cfg(feature = "record_timings")]
pub fn print_all_timings() {
    let locked = PRINT_TIMINGS.lock().unwrap();
    for f in locked.iter() {
        f();
    }
}

#[cfg(not(feature = "record_timings"))]
pub fn print_all_timings() {}

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
                    locked.push(Box::new(|| {
                        println!("{}: {} ms", $name, COUNTER.load(Ordering::SeqCst) / 1000);
                    }));
                }
                let start = Instant::now();
                let result = ($fn)();
                let end = Instant::now();
                COUNTER.fetch_add((end - start).as_micros() as u64 + 1, Ordering::SeqCst);
                result
            }
            #[cfg(not(feature = "record_timings"))] {
                ($fn)()
            }
        }
    };
}
