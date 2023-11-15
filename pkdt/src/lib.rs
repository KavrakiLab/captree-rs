#![feature(portable_simd)]
#![feature(slice_swap_unchecked)]
#![warn(clippy::pedantic)]

use std::{
    hint::unreachable_unchecked,
    simd::{LaneCount, Mask, Simd, SimdConstPtr, SimdPartialOrd, SupportedLaneCount},
};

mod ball;
mod forest;

pub use ball::BallTree;
pub use forest::PkdForest;
use rand::{thread_rng, Rng};

#[derive(Clone, Debug, PartialEq)]
/// A power-of-two KD-tree.
///
/// # Generic parameters
///
/// - `D`: The dimension of the space.
pub struct PkdTree<const D: usize> {
    /// The test values for determining which part of the tree to enter.
    ///
    /// The first element of `tests` should be the first value to test against.
    /// If we are less than `tests[0]`, we move on to `tests[1]`; if not, we move on to `tests[2]`.
    /// At the `i`-th test performed in sequence of the traversal, if we are less than `tests[idx]`,
    /// we advance to `2 * idx + 1`; otherwise, we go to `2 * idx + 2`.
    ///
    /// The length of `tests` must be `N`, rounded up to the next power of 2, minus one.
    tests: Box<[f32]>,
    /// The relevant points at the center of each volume divided by `tests`.
    points: Box<[[f32; D]]>,
}

impl<const D: usize> PkdTree<D> {
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    /// Construct a new `PkdTree` containing all the points in `points`.
    /// For performance, this function changes the ordering of `points`, but does not affect the
    /// set of points inside it.
    ///
    /// # Panics
    ///
    /// This function will panic if `D` is greater than or equal to 255.
    ///
    /// TODO: do all our sorting on the allocation that we return?
    pub fn new(points: &[[f32; D]]) -> Self {
        /// Recursive helper function to sort the points for the KD tree and generate the tests.
        fn recur_sort_points<const D: usize>(
            points: &mut [[f32; D]],
            tests: &mut [f32],
            d: u8,
            i: usize,
        ) {
            // TODO make this algorithm O(n log n) instead of O(n log^2 n)
            if points.len() > 1 {
                tests[i] = median_partition(points, d as usize, &mut thread_rng());
                let next_dim = (d + 1) % D as u8;
                let (lhs, rhs) = points.split_at_mut(points.len() / 2);
                recur_sort_points(lhs, tests, next_dim, 2 * i + 1);
                recur_sort_points(rhs, tests, next_dim, 2 * i + 2);
            }
        }

        assert!(D < u8::MAX as usize);

        let n2 = points.len().next_power_of_two();

        let mut tests = vec![f32::INFINITY; n2 - 1].into_boxed_slice();

        // hack: just pad with infinity to make it a power of 2
        let mut new_points = vec![[f32::INFINITY; D]; n2].into_boxed_slice();
        new_points[..points.len()].copy_from_slice(points);
        recur_sort_points(new_points.as_mut(), tests.as_mut(), 0, 0);

        PkdTree {
            tests,
            points: new_points,
        }
    }

    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    /// Get the indices of points which are closest to `needles`.
    ///
    /// TODO: refactor this to use `needles` as an out parameter as well, and shove the nearest
    /// points in there?
    pub fn query<const L: usize>(&self, needles: &[Simd<f32, L>; D]) -> Simd<usize, L>
    where
        LaneCount<L>: SupportedLaneCount,
    {
        let mut test_idxs: Simd<usize, L> = Simd::splat(0);
        let n2 = self.tests.len() + 1;
        debug_assert!(n2.is_power_of_two());

        // in release mode, tell the compiler about this invariant
        if !n2.is_power_of_two() {
            unsafe { unreachable_unchecked() };
        }

        // Advance the tests forward
        for i in 0..n2.trailing_zeros() as usize {
            let test_ptrs = Simd::splat((self.tests.as_ref() as *const [f32]).cast::<f32>())
                .wrapping_add(test_idxs);
            let relevant_tests: Simd<f32, L> = unsafe { Simd::gather_ptr(test_ptrs) };
            let cmp_results: Mask<isize, L> = needles[i % D].simd_lt(relevant_tests).into();

            // TODO is there a faster way than using a conditional select?
            test_idxs <<= Simd::splat(1);
            test_idxs += cmp_results.select(Simd::splat(1), Simd::splat(2));
        }

        test_idxs - Simd::splat(self.tests.len())
    }

    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    /// Get the access index of the point closest to `needle`
    pub fn query1(&self, needle: [f32; D]) -> usize {
        let n2 = self.tests.len() + 1;
        assert!(n2.is_power_of_two());

        let mut test_idx = 0;
        for i in 0..n2.trailing_zeros() as usize {
            // println!("current idx: {test_idx}");
            let add = if needle[i % D] < (self.tests[test_idx]) {
                1
            } else {
                2
            };
            test_idx <<= 1;
            test_idx += add;
        }

        test_idx - self.tests.len()
    }

    #[must_use]
    /// Determine whether a ball centered at `needle` with radius `r_squared` could collide with a
    /// point in this tree.
    pub fn might_collide(&self, needle: [f32; D], r_squared: f32) -> bool {
        distsq(self.get_point(self.query1(needle)), needle) <= r_squared
    }

    #[must_use]
    pub fn might_collide_simd<const L: usize>(
        &self,
        needles: &[Simd<f32, L>; D],
        radii_squared: Simd<f32, L>,
    ) -> bool
    where
        LaneCount<L>: SupportedLaneCount,
    {
        let indices = self.query(needles);
        let mut dists_squared = Simd::splat(0.0);
        let mut ptrs = Simd::splat((self.points.as_ref() as *const [[f32; D]]).cast::<f32>())
            .wrapping_add(indices * Simd::splat(D));
        for needle_values in needles {
            let deltas = unsafe { Simd::gather_ptr(ptrs) } - needle_values;
            dists_squared += deltas * deltas;
            ptrs = ptrs.wrapping_add(Simd::splat(1));
        }
        dists_squared.simd_lt(radii_squared).any()
    }

    #[must_use]
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    /// Query for one point in this tree, returning an exact answer.
    pub fn query1_exact(&self, needle: [f32; D]) -> usize {
        let mut id = usize::MAX;
        let mut best_distsq = f32::INFINITY;
        self.exact_help(
            0,
            0,
            &[[-f32::INFINITY, f32::INFINITY]; D],
            needle,
            &mut id,
            &mut best_distsq,
        );
        id
    }

    #[allow(clippy::cast_possible_truncation)]
    fn exact_help(
        &self,
        test_idx: usize,
        d: u8,
        bounding_box: &[[f32; 2]; D],
        point: [f32; D],
        best_id: &mut usize,
        best_distsq: &mut f32,
    ) {
        if bb_distsq(point, bounding_box) > *best_distsq {
            return;
        }

        if self.tests.len() <= test_idx {
            let id = test_idx - self.tests.len();
            let new_distsq = distsq(point, self.get_point(id));
            if new_distsq < *best_distsq {
                *best_id = id;
                *best_distsq = new_distsq;
            }

            return;
        }

        let test = self.tests[test_idx];

        let mut bb_below = *bounding_box;
        bb_below[d as usize][1] = test;
        let mut bb_above = *bounding_box;
        bb_above[d as usize][0] = test;

        let next_d = (d + 1) % D as u8;
        if point[d as usize] < test {
            self.exact_help(
                2 * test_idx + 1,
                next_d,
                &bb_below,
                point,
                best_id,
                best_distsq,
            );
            self.exact_help(
                2 * test_idx + 2,
                next_d,
                &bb_above,
                point,
                best_id,
                best_distsq,
            );
        } else {
            self.exact_help(
                2 * test_idx + 2,
                next_d,
                &bb_above,
                point,
                best_id,
                best_distsq,
            );
            self.exact_help(
                2 * test_idx + 1,
                next_d,
                &bb_below,
                point,
                best_id,
                best_distsq,
            );
        }
    }

    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn get_point(&self, id: usize) -> [f32; D] {
        self.points[id]
    }
}

/// Calculate the "true" median (halfway between two midpoints) and partition `points` about said
/// median along axis `d`.
fn median_partition<const D: usize>(points: &mut [[f32; D]], d: usize, rng: &mut impl Rng) -> f32 {
    let median_hi = quick_median(points, d, rng);
    let (median_lo_idx, median_lo) = points[..points.len() / 2]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a[d].partial_cmp(&b[d]).unwrap())
        .unwrap();
    let ret = (median_lo[d] + median_hi) / 2.0;
    points.swap(median_lo_idx, points.len() / 2 - 1);
    ret
}

/// Partition `points[left..right]` about the value at index `pivot_idx` on dimension `d`.
/// Returns the resultant index of the pivot in the array.
unsafe fn partition<const D: usize>(
    points: &mut [[f32; D]],
    left: usize,
    right: usize,
    pivot_idx: usize,
    d: usize,
) -> usize {
    let pivot_val = points[pivot_idx][d];
    let mut i = left.overflowing_sub(1).0;
    let mut j = right;
    loop {
        i = i.overflowing_add(1).0;
        while *points.get_unchecked(i).get_unchecked(d) < pivot_val {
            i += 1;
        }
        j -= 1;
        while *points.get_unchecked(j).get_unchecked(d) > pivot_val {
            j -= 1;
        }

        if i < j {
            points.swap_unchecked(i, j);
        } else {
            return j + 1;
        }
    }
}

/// Calculate the median of `points` by dimension `d` and partition `points` so that all points
/// below the median come before it in the buffer.
fn quick_median<const D: usize>(points: &mut [[f32; D]], d: usize, rng: &mut impl Rng) -> f32 {
    let k = points.len() / 2;
    let mut left = 0;
    let mut right = points.len();

    loop {
        if right - left < 2 {
            return points[left][d];
        }

        // index of the first element greater than or equal to the pivot
        let pivot_idx = unsafe { partition(points, left, right, rng.gen_range(left..right), d) };
        // match k.cmp(&pivot_idx) {
        //     Ordering::Equal => return points[k][d],
        //     Ordering::Less => right = pivot_idx,
        //     Ordering::Greater => left = pivot_idx + 1,
        // };
        if k < pivot_idx {
            right = pivot_idx;
        } else {
            left = pivot_idx;
        }
    }
}

fn bb_distsq<const D: usize>(point: [f32; D], bb: &[[f32; 2]; D]) -> f32 {
    point
        .into_iter()
        .zip(bb.iter())
        .map(|(x, [lower, upper])| {
            (if x < *lower {
                *lower - x
            } else if *upper < x {
                x - *upper
            } else {
                0.0
            })
            .powi(2)
        })
        .sum()
}

fn distsq<const D: usize>(a: [f32; D], b: [f32; D]) -> f32 {
    let mut total = 0.0f32;
    for i in 0..D {
        total += (a[i] - b[i]).powi(2);
    }
    total
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::*;

    #[test]
    fn single_query() {
        let points = vec![
            [0.1, 0.1],
            [0.1, 0.2],
            [0.5, 0.0],
            [0.3, 0.9],
            [1.0, 1.0],
            [0.35, 0.75],
            [0.6, 0.2],
            [0.7, 0.8],
        ];
        let kdt = PkdTree::new(&points);

        println!("testing for correctness...");

        let neg1 = [-1.0, -1.0];
        let neg1_idx = kdt.query1(neg1);
        assert_eq!(neg1_idx, 0);

        let pos1 = [1.0, 1.0];
        let pos1_idx = kdt.query1(pos1);
        assert_eq!(pos1_idx, points.len() - 1);
    }

    #[test]
    fn multi_query() {
        let points = vec![
            [0.1, 0.1],
            [0.1, 0.2],
            [0.5, 0.0],
            [0.3, 0.9],
            [1.0, 1.0],
            [0.35, 0.75],
            [0.6, 0.2],
            [0.7, 0.8],
        ];
        let kdt = PkdTree::new(&points);

        let needles = [Simd::from_array([-1.0, 2.0]), Simd::from_array([-1.0, 2.0])];
        assert_eq!(kdt.query(&needles), Simd::from_array([0, points.len() - 1]));
    }

    #[test]
    fn not_a_power_of_two() {
        let points = vec![[0.0], [2.0], [4.0]];
        let kdt = PkdTree::new(&points);

        println!("{kdt:?}");

        assert_eq!(kdt.query1([-1.0]), 0);
        assert_eq!(kdt.query1([0.5]), 0);
        assert_eq!(kdt.query1([1.5]), 1);
        assert_eq!(kdt.query1([2.5]), 1);
        assert_eq!(kdt.query1([3.5]), 2);
        assert_eq!(kdt.query1([4.5]), 2);
    }

    #[test]
    fn a_power_of_two() {
        let points = vec![[0.0], [2.0], [4.0], [6.0]];
        let kdt = PkdTree::new(&points);

        println!("{kdt:?}");

        assert_eq!(kdt.query1([-1.0]), 0);
        assert_eq!(kdt.query1([0.5]), 0);
        assert_eq!(kdt.query1([1.5]), 1);
        assert_eq!(kdt.query1([2.5]), 1);
        assert_eq!(kdt.query1([3.5]), 2);
        assert_eq!(kdt.query1([4.5]), 2);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn medians() {
        let points = vec![[1.0], [2.0], [1.5]];

        let mut points1 = points.clone();
        let median = quick_median(&mut points1, 0, &mut thread_rng());
        println!("{points1:?}");
        assert_eq!(median, 1.5);
        assert_eq!(points1, vec![[1.0], [1.5], [2.0]]);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn does_it_partition() {
        let points = vec![[1.0], [2.0], [1.5], [2.1], [-0.5]];

        let mut points1 = points.clone();
        let median = median_partition(&mut points1, 0, &mut thread_rng());
        assert_eq!(median, 1.25);
        for p0 in &points1[..points1.len() / 2] {
            assert!(p0[0] <= median);
        }

        for p0 in &points1[points1.len() / 2..] {
            assert!(p0[0] >= median);
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn just_test_partition() {
        let mut points = vec![[1.5], [2.0], [1.0]];
        let pivot_idx = unsafe { partition(&mut points, 0, 3, 1, 0) };
        assert_eq!(pivot_idx, 2);
        assert_eq!(points[2], [2.0]);
        println!("{points:?}");
    }
}
