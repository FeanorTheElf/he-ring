use std::time::Instant;

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
