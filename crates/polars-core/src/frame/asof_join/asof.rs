use std::fmt::Debug;
use std::ops::Sub;

use num_traits::Bounded;
use polars_arrow::index::IdxSize;

pub(super) fn join_asof_forward_with_tolerance<T: PartialOrd + Copy + Debug + Sub<Output = T>>(
    left: &[T],
    right: &[T],
    tolerance: T,
) -> Vec<Option<IdxSize>> {
    if right.is_empty() {
        return vec![None; left.len()];
    }
    if left.is_empty() {
        return vec![];
    }

    let mut out = Vec::with_capacity(left.len());
    let mut offset = 0 as IdxSize;

    for &val_l in left {
        loop {
            match right.get(offset as usize) {
                Some(&val_r) => {
                    if val_r >= val_l {
                        let dist = val_r - val_l;
                        let value = if dist > tolerance { None } else { Some(offset) };

                        out.push(value);
                        break;
                    }
                    offset += 1;
                },
                None => {
                    out.extend(std::iter::repeat(None).take(left.len() - out.len()));
                    return out;
                },
            }
        }
    }
    out
}

pub(super) fn join_asof_backward_with_tolerance<T>(
    left: &[T],
    right: &[T],
    tolerance: T,
) -> Vec<Option<IdxSize>>
where
    T: PartialOrd + Copy + Debug + Sub<Output = T>,
{
    if right.is_empty() {
        return vec![None; left.len()];
    }
    if left.is_empty() {
        return vec![];
    }
    let mut out = Vec::with_capacity(left.len());

    let mut offset = 0 as IdxSize;
    // left array could start lower than right;
    // left: [-1, 0, 1, 2],
    // right: [1, 2, 3]
    // first values should be None, until left has caught up
    let mut left_caught_up = false;

    // init with left so that the distance starts at 0
    let mut previous_right = left[0];
    let mut dist;

    for &val_l in left {
        loop {
            dist = val_l - previous_right;

            match right.get(offset as usize) {
                Some(&val_r) => {
                    // we fill nulls until left value is larger than right
                    if !left_caught_up {
                        if val_l < val_r {
                            out.push(None);
                            break;
                        } else {
                            left_caught_up = true;
                        }
                    }

                    // right is larger than left.
                    // we take the last value before that
                    if val_r > val_l {
                        let value = if dist > tolerance {
                            None
                        } else {
                            Some(offset - 1)
                        };

                        out.push(value);
                        break;
                    }
                    // right still smaller or equal to left
                    // continue looping the right side
                    else {
                        previous_right = val_r;
                        offset += 1;
                    }
                },
                // we depleted the right array
                // we cannot fill the remainder of the value, because we need to check tolerances
                None => {
                    // if we have previous value, continue with that one
                    let val = if left_caught_up && dist <= tolerance {
                        Some(offset - 1)
                    }
                    // else null
                    else {
                        None
                    };
                    out.push(val);
                    break;
                },
            }
        }
    }
    out
}

pub(super) fn join_asof_backward<T: PartialOrd + Copy + Debug>(
    left: &[T],
    right: &[T],
) -> Vec<Option<IdxSize>> {
    let mut out = Vec::with_capacity(left.len());

    let mut offset = 0 as IdxSize;
    // left array could start lower than right;
    // left: [-1, 0, 1, 2],
    // right: [1, 2, 3]
    // first values should be None, until left has caught up
    let mut left_caught_up = false;

    for &val_l in left {
        loop {
            match right.get(offset as usize) {
                Some(&val_r) => {
                    // we fill nulls until left value is larger than right
                    if !left_caught_up {
                        if val_l < val_r {
                            out.push(None);
                            break;
                        } else {
                            left_caught_up = true;
                        }
                    }

                    // right is larger than left.
                    // we take the last value before that
                    if val_r > val_l {
                        out.push(Some(offset - 1));
                        break;
                    }
                    // right still smaller or equal to left
                    // continue looping the right side
                    else {
                        offset += 1;
                    }
                },
                // we depleted the right array
                None => {
                    // if we have previous value, continue with that one
                    let val = if left_caught_up {
                        Some(offset - 1)
                    }
                    // else all null
                    else {
                        None
                    };
                    out.extend(std::iter::repeat(val).take(left.len() - out.len()));
                    return out;
                },
            }
        }
    }
    out
}

pub(super) fn join_asof_nearest_with_tolerance<T: PartialOrd + Copy + Debug + Sub<Output = T>>(
    left: &[T],
    right: &[T],
    tolerance: T,
) -> Vec<Option<IdxSize>> {
    let mut out = Vec::with_capacity(left.len());
    let mut offset = 0 as IdxSize;
    let max_value = tolerance;
    let mut dist: T = max_value;

    for &val_l in left {
        loop {
            match right.get(offset as usize) {
                Some(&val_r) => {
                    // This is (val_r - val_l).abs(), but works on strings/dates
                    let dist_curr = if val_r > val_l {
                        val_r - val_l
                    } else {
                        val_l - val_r
                    };
                    if dist_curr <= dist {
                        // candidate for match
                        dist = dist_curr;
                        offset += 1;
                    } else {
                        // distance has increased, we're now farther away, so previous element was closest
                        out.push(Some(offset - 1));

                        // reset distance
                        dist = max_value;

                        // The next left-item may match on the same item, so we need to rewind the offset
                        offset -= 1;
                        break;
                    }
                },

                None => {
                    if offset > 1 {
                        // we've reached the end with no matches, so the last item is the nearest for all remaining
                        out.extend(
                            std::iter::repeat(Some(offset - 1)).take(left.len() - out.len()),
                        );
                    } else {
                        // this is only hit when the right frame is empty
                        out.extend(std::iter::repeat(None).take(left.len() - out.len()));
                    }
                    return out;
                },
            }
        }
    }

    out
}

pub(super) fn join_asof_nearest<T: PartialOrd + Copy + Debug + Sub<Output = T> + Bounded>(
    left: &[T],
    right: &[T],
) -> Vec<Option<IdxSize>> {
    let mut out = Vec::with_capacity(left.len());
    let mut offset = 0 as IdxSize;
    let max_value = <T as num_traits::Bounded>::max_value();
    let mut dist: T = max_value;

    for &val_l in left {
        loop {
            match right.get(offset as usize) {
                Some(&val_r) => {
                    // This is (val_r - val_l).abs(), but works on strings/dates
                    let dist_curr = if val_r > val_l {
                        val_r - val_l
                    } else {
                        val_l - val_r
                    };
                    if dist_curr <= dist {
                        // candidate for match
                        dist = dist_curr;
                        offset += 1;
                    } else {
                        // distance has increased, we're now farther away, so previous element was closest
                        out.push(Some(offset - 1));

                        // reset distance
                        dist = max_value;

                        // The next left-item may match on the same item, so we need to rewind the offset
                        offset -= 1;
                        break;
                    }
                },

                None => {
                    if offset > 1 {
                        // we've reached the end with no matches, so the last item is the nearest for all remaining
                        out.extend(
                            std::iter::repeat(Some(offset - 1)).take(left.len() - out.len()),
                        );
                    } else {
                        // this is only hit when the right frame is empty
                        out.extend(std::iter::repeat(None).take(left.len() - out.len()));
                    }
                    return out;
                },
            }
        }
    }

    out
}

pub(super) fn join_asof_forward<T: PartialOrd + Copy + Debug>(
    left: &[T],
    right: &[T],
) -> Vec<Option<IdxSize>> {
    let mut out = Vec::with_capacity(left.len());
    let mut offset = 0 as IdxSize;

    for &val_l in left {
        loop {
            match right.get(offset as usize) {
                Some(&val_r) => {
                    if val_r >= val_l {
                        out.push(Some(offset));
                        break;
                    }
                    offset += 1;
                },
                None => {
                    out.extend(std::iter::repeat(None).take(left.len() - out.len()));
                    return out;
                },
            }
        }
    }
    out
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_asof_backward() {
        let a = [-1, 2, 3, 3, 3, 4];
        let b = [1, 2, 3, 3];

        let tuples = join_asof_backward(&a, &b);
        assert_eq!(tuples.len(), a.len());
        assert_eq!(tuples, &[None, Some(1), Some(3), Some(3), Some(3), Some(3)]);

        let b = [1, 2, 4, 5];
        let tuples = join_asof_backward(&a, &b);
        assert_eq!(tuples, &[None, Some(1), Some(1), Some(1), Some(1), Some(2)]);

        let a = [2, 4, 4, 4];
        let b = [1, 2, 3, 3];
        let tuples = join_asof_backward(&a, &b);
        assert_eq!(tuples, &[Some(1), Some(3), Some(3), Some(3)]);
    }

    #[test]
    fn test_asof_backward_tolerance() {
        let a = [-1, 20, 25, 30, 30, 40];
        let b = [10, 20, 30, 30];
        let tuples = join_asof_backward_with_tolerance(&a, &b, 4);
        assert_eq!(tuples, &[None, Some(1), None, Some(3), Some(3), None]);
    }

    #[test]
    fn test_asof_forward_tolerance() {
        let a = [-1, 20, 25, 30, 30, 40, 52];
        let b = [10, 20, 33, 55];
        let tuples = join_asof_forward_with_tolerance(&a, &b, 4);
        assert_eq!(
            tuples,
            &[None, Some(1), None, Some(2), Some(2), None, Some(3)]
        );
    }

    #[test]
    fn test_asof_forward() {
        let a = [-1, 1, 2, 4, 6];
        let b = [1, 2, 4, 5];

        let tuples = join_asof_forward(&a, &b);
        assert_eq!(tuples.len(), a.len());
        assert_eq!(tuples, &[Some(0), Some(0), Some(1), Some(2), None]);
    }
}
