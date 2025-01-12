use polars_core::prelude::*;
use polars_core::series::IsSorted;

use crate::series::ClosedInterval;
use crate::series::ClosedInterval::*;

pub fn new_linear_space(
    start: f64,
    end: f64,
    n: u64,
    closed: ClosedInterval,
    name: PlSmallStr,
) -> PolarsResult<Series> {
    let mut ca = match n {
        0 => Float64Chunked::full_null(name, 0),
        1 => match closed {
            None => Float64Chunked::from_slice(name, &[(end + start) * 0.5]),
            Left | Both => Float64Chunked::from_slice(name, &[start]),
            Right => Float64Chunked::from_slice(name, &[end]),
        },
        _ => Float64Chunked::from_iter_values(name, {
            let span = end - start;

            let (start, d, end) = match closed {
                None => {
                    let d = span / (n + 1) as f64;
                    (start + d, d, end - d)
                },
                Left => (start, span / n as f64, end - span / n as f64),
                Right => (start + span / n as f64, span / n as f64, end),
                Both => (start, span / (n - 1) as f64, end),
            };
            (0..n - 1)
                .map(move |v| (v as f64 * d) + start)
                .chain(std::iter::once(end)) // ensures floating point accuracy of final value
        }),
    };

    let is_sorted = if end < start {
        IsSorted::Descending
    } else {
        IsSorted::Ascending
    };
    ca.set_sorted_flag(is_sorted);

    Ok(ca.into_series())
}
