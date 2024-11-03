use std::borrow::Borrow;
use std::fmt::Display;
use std::ops::Deref;
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;
use std::{cell::RefCell, collections::BTreeMap, sync::{atomic::{AtomicU64, AtomicUsize}, RwLock}, time::Instant};

use thread_local::ThreadLocal;

pub static GLOBAL_TIME_RECORDER: TimeRecorder = TimeRecorder::new("");

pub static TIME_RECORDER_KEY_IDS: AtomicUsize = AtomicUsize::new(1);

#[derive(Clone, Copy)]
pub struct TimeRecorderKey {
    description: &'static str,
    id: usize
}

impl TimeRecorderKey {

    pub const fn new(desc: &'static str, id: usize) -> Self {
        Self {
            description: desc,
            id: id
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct TimeRecorderKeyPath {
    data: Vec<TimeRecorderKey>
}

impl TimeRecorderKeyPath {
    fn str_len(&self) -> usize {
        self.data.iter().map(|key| key.description.chars().count()).sum::<usize>() + (self.data.len() - 1) * 3
    }
}

impl Display for TimeRecorderKeyPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for key in &self.data[..(self.data.len() - 1)] {
            write!(f, "{} / ", key.description)?;
        }
        write!(f, "{}", self.data[self.data.len() - 1].description)?;
        if let Some(width) = f.width() {
            let padding = width.saturating_sub(self.str_len());
            for _ in 0..padding {
                write!(f, " ")?;
            }
        }
        return Ok(());
    }
}

impl Borrow<[TimeRecorderKey]> for TimeRecorderKeyPath {

    fn borrow(&self) -> &[TimeRecorderKey] {
        &self.data
    }
}

#[derive(Default)]
pub struct TimeRecorderEntry {
    calls: AtomicU64,
    spent_nanos: AtomicU64
}

impl TimeRecorderEntry {
    fn count(&self) -> u64 {
        self.calls.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn spent_nanos(&self) -> u64 {
        self.spent_nanos.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl PartialEq for TimeRecorderKey {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for TimeRecorderKey {}

impl PartialOrd for TimeRecorderKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TimeRecorderKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

pub struct TimeRecorder {
    desc: &'static str,
    registered: AtomicBool,
    stack: ThreadLocal<RefCell<Vec<TimeRecorderKey>>>,
    counters: RwLock<BTreeMap<TimeRecorderKeyPath, TimeRecorderEntry>>
}

impl TimeRecorder {

    pub const fn new(desc: &'static str) -> Self {
        Self {
            desc: desc,
            registered: AtomicBool::new(false),
            stack: ThreadLocal::new(),
            counters: RwLock::new(BTreeMap::new())
        }
    }

    pub fn measure_call<F, T>(&'static self, key: TimeRecorderKey, fun: F) -> T
        where F: FnOnce() -> T
    {
        #[inline(never)]
        fn prevent_inline<T, F: FnOnce() -> T>(f: F) -> T {
            f()
        }

        if !self.registered.swap(true, std::sync::atomic::Ordering::SeqCst) {
            let mut locked = REGISTERED_RECORDERS.lock().unwrap();
            locked.push(self);
        }
        self.enter(key);
        let start = Instant::now();
        let result = prevent_inline(fun);
        let end = Instant::now();
        self.register_call((end - start).as_nanos() as u64);
        self.leave(key);
        return result;
    }

    fn enter(&self, key: TimeRecorderKey) {
        self.stack.get_or_default().borrow_mut().push(key);
        // print!("enter ");
        // let stack = self.stack.get_or_default().borrow();
        // for i in 0..stack.len() {
        //     print!("::{}", stack[i].description);
        // }
        // println!();
    }

    fn leave(&self, key: TimeRecorderKey) {
        // print!("exit ");
        // let stack = self.stack.get_or_default().borrow();
        // for i in 0..stack.len() {
        //     print!("::{}", stack[i].description);
        // }
        // println!();
        // drop(stack);
        assert!(self.stack.get_or_default().borrow_mut().pop().unwrap() == key);
    }

    fn register_call(&self, spent_nanos: u64) {
        let path = self.stack.get_or_default().borrow();

        assert!(path.len() >= 1);

        let read_counters = self.counters.read().unwrap();
        if !read_counters.contains_key(&path[..]) {
            drop(read_counters);
            let mut write_counters = self.counters.write().unwrap();
            write_counters.insert(TimeRecorderKeyPath { data: path[..].iter().copied().collect() }, TimeRecorderEntry::default());
            assert!(write_counters.contains_key(&path[..]));
            drop(write_counters);
            self.register_call(spent_nanos);
        } else {
            let entry = read_counters.get(&path[..]).unwrap();
            entry.calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            entry.spent_nanos.fetch_add(spent_nanos, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn print_timings(&self) {
        let counters = self.counters.write().unwrap();
        let counters_iter = || counters.iter().filter(|(_, entry)| entry.count() > 0);

        let longest_desc_len = counters_iter().map(|(k, _)| k.str_len()).max().unwrap_or(0);
        let longest_calls_len = counters_iter().map(|(_, v)| (v.count() as f64).log(10.).ceil() as usize).max().unwrap_or(0);
        let longest_time_len = counters_iter().map(|(_, v)| ((v.spent_nanos() / 1000000) as f64).log(10.).ceil() as usize).max().unwrap_or(0);
        let longest_time_per_call_len = counters_iter().map(|(_, v)| ((v.spent_nanos() / (1000 * v.count())) as f64).log(10.).ceil() as usize).max().unwrap_or(0);
        for (k, v) in counters_iter() {
            println!("{} / {:<ldesc$}    {:>lcalls$} calls,  {:>ltime$} ms,  {:>ltime_per_call$} us/call", 
                self.desc,
                k, 
                v.count(), 
                v.spent_nanos() / 1000000, 
                v.spent_nanos() / (v.count() * 1000),
                ldesc = longest_desc_len, 
                lcalls = longest_calls_len, 
                ltime = longest_time_len, 
                ltime_per_call = longest_time_per_call_len
            );
        }
    }

    fn clear(&self) {
        let mut counters = self.counters.write().unwrap();
        for (_, v) in counters.iter_mut() {
            v.calls.store(0, std::sync::atomic::Ordering::Relaxed);
            v.spent_nanos.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

pub static REGISTERED_RECORDERS: Mutex<Vec<&'static TimeRecorder>> = Mutex::new(Vec::new());

pub fn print_all_timings() {
    let locked = REGISTERED_RECORDERS.lock().unwrap();
    for recorder in locked.iter() {
        recorder.print_timings();
    }
}

// #[cfg(not(feature = "record_timings"))]
// pub fn print_all_timings() {}

pub fn clear_all_timings() {
    let locked = REGISTERED_RECORDERS.lock().unwrap();
    for recorder in locked.iter() {
        recorder.clear();
    }
}

// #[cfg(not(feature = "record_timings"))]
// pub fn clear_all_timings() {}

macro_rules! record_time {
    ($recorder:ident, $name:literal, $fn:expr) => {
        {
            {
                use std::sync::*;
                use std::sync::atomic::*;
                use $crate::profiling::*;

                static KEY_ID: AtomicUsize = AtomicUsize::new(0);

                let id = if KEY_ID.load(Ordering::SeqCst) == 0 {
                    let new_id = TIME_RECORDER_KEY_IDS.fetch_add(1, Ordering::SeqCst);
                    match KEY_ID.compare_exchange(0, new_id, Ordering::SeqCst, Ordering::SeqCst) {
                        Ok(_) => new_id,
                        Err(actual_id) => actual_id
                    }
                } else {
                    KEY_ID.load(Ordering::SeqCst)
                };

                (&$recorder).measure_call(TimeRecorderKey::new($name, id), $fn)
            }
            // #[cfg(not(feature = "record_timings"))] {
            //     ($fn)()
            // }
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