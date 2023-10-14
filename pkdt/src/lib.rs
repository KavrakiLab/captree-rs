#![feature(portable_simd)]
#![feature(is_sorted)]
#![warn(clippy::pedantic)]

use std::{
    hint::unreachable_unchecked,
    simd::{LaneCount, Mask, Simd, SimdConstPtr, SimdPartialOrd, SupportedLaneCount},
};

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
    ///
    /// If there are `N` points in the tree, let `N2` be `N` rounded up to the next power of 2.
    /// Then `points` has length `N2 * D`.
    points: Box<[f32]>,
}

impl<const D: usize> PkdTree<D> {
    #[must_use]
    /// Construct a new `PkdTree` containing all the points in `points`.
    /// For performance, this function changes the ordering of `points`, but does not affect the
    /// set of points inside it.
    ///
    /// TODO: do all our sorting on the allocation that we return?
    ///
    /// # Panics
    ///
    /// This function may panic if any of the points has a non-finite value.
    pub fn new(points: &[[f32; D]]) -> Self {
        // hack to copy empty vectors
        const EMPTY_VEC: Vec<usize> = Vec::new();

        /// Recursive helper function to sort the points for the KD tree and generate the tests.
        fn recur_partition<'a, const D: usize>(
            points: &impl Fn(usize) -> &'a [f32; D],
            tests: &mut [f32],
            argsorts: &mut [Vec<usize>],
            d: usize,
            i: usize,
            l: usize,
            r: usize,
        ) {
            for (d2, argsort) in argsorts.iter().enumerate() {
                debug_assert!(
                    argsort[l..r].is_sorted_by(|&a, &b| points(a)[d2].partial_cmp(&points(b)[d2]))
                );
            }

            if r - l < 2 {
                return;
            }

            // indices for finding the ID of the points closest to the test plane
            let med_hi_idx = (l + r) / 2;
            let med_lo_idx = med_hi_idx - 1;

            debug_assert_eq!(med_hi_idx - l, r - med_hi_idx);

            // IDs of points closest to the test plane
            let med_lo_id = argsorts[d][med_lo_idx];
            let med_hi_id = argsorts[d][med_hi_idx];

            tests[i] = (points(med_lo_id)[d] + points(med_hi_id)[d]) / 2.0;

            // partition argsorts by this test
            for (d2, argsort) in argsorts.iter_mut().enumerate() {
                if d2 == d {
                    continue;
                }
                let mut more_buf = Vec::with_capacity((r - l) / 2);

                let mut less_buf_idx = l;
                for j in l..r {
                    let arg = argsort[j];
                    debug_assert_ne!(points(arg)[d], tests[i]);
                    if points(arg)[d] < tests[i] {
                        argsort[less_buf_idx] = arg;
                        less_buf_idx += 1;
                    } else {
                        more_buf.push(arg);
                    }
                }

                assert_eq!(less_buf_idx - l, more_buf.len());
                argsort[med_hi_idx..r].copy_from_slice(&more_buf);
            }

            let next_d = (d + 1) % D;
            recur_partition(points, tests, argsorts, next_d, 2 * i + 1, l, med_hi_idx);
            recur_partition(points, tests, argsorts, next_d, 2 * i + 2, med_hi_idx, r);
        }

        let n2 = points.len().next_power_of_two();

        let mut tests = vec![f32::INFINITY; n2 - 1].into_boxed_slice();

        let mut argsorts = [EMPTY_VEC; D];

        let inf = [f32::INFINITY; D];
        let points_fn = &|idx| points.get(idx).unwrap_or(&inf);
        for (d, argsort) in argsorts.iter_mut().enumerate() {
            *argsort = (0..n2).collect();
            argsort
                .sort_unstable_by(|&a, &b| points_fn(a)[d].partial_cmp(&points_fn(b)[d]).unwrap());
        }

        recur_partition(
            &|idx| points.get(idx).unwrap_or(&inf),
            tests.as_mut(),
            &mut argsorts,
            0,
            0,
            0,
            n2,
        );

        let mut my_points = vec![f32::NAN; n2 * D].into_boxed_slice();
        for pt in points {
            let mut test_idx = 0;
            for i in 0..n2.ilog2() as usize {
                // println!("current idx: {test_idx}");
                let add = if pt[i % D] < (tests[test_idx]) { 1 } else { 2 };
                test_idx <<= 1;
                test_idx += add;
            }

            let pt_idx = test_idx - tests.len();
            for d in 0..D {
                my_points[d * n2 + pt_idx] = pt[d];
            }
        }

        PkdTree {
            tests,
            points: my_points,
        }
    }

    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    /// Get the indices of points which are closest to `needles`.
    ///
    /// TODO: refactor this to use `needles` as an out parameter as well, and shove the nearest
    /// points in there?
    pub fn query<const L: usize>(&self, needles: &[[f32; L]; D]) -> [usize; L]
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
        for i in 0..n2.ilog2() as usize {
            let test_ptrs = Simd::splat((self.tests.as_ref() as *const [f32]).cast::<f32>())
                .wrapping_add(test_idxs);
            let relevant_tests: Simd<f32, L> = unsafe { Simd::gather_ptr(test_ptrs) };
            let needle_values = Simd::from_array(needles[i % D]);
            let cmp_results: Mask<isize, L> = needle_values.simd_lt(relevant_tests).into();

            // TODO is there a faster way than using a conditional select?
            test_idxs <<= Simd::splat(1);
            test_idxs += cmp_results.select(Simd::splat(1), Simd::splat(2));
        }

        (test_idxs - Simd::splat(self.tests.len())).into()
    }

    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    /// Get the access index of the point closest to `needle`
    pub fn query1(&self, needle: [f32; D]) -> usize {
        let n2 = self.tests.len() + 1;
        assert!(n2.is_power_of_two());

        let mut test_idx = 0;
        for i in 0..n2.ilog2() as usize {
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
    #[allow(clippy::missing_panics_doc)]
    pub fn get_point(&self, id: usize) -> [f32; D] {
        let mut point = [0.0; D];
        let n2 = self.tests.len() + 1;
        assert!(n2.is_power_of_two());
        for (d, value) in point.iter_mut().enumerate() {
            *value = self.points[d * n2 + id];
        }

        point
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_query() {
        let points = vec![
            [0.1, 0.1],
            [0.101, 0.2],
            [0.5, 0.0],
            [0.3, 0.9],
            [1.0, 1.0],
            [0.35, 0.75],
            [0.6, 0.201],
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
            [0.101, 0.2],
            [0.5, 0.0],
            [0.3, 0.9],
            [1.0, 1.0],
            [0.35, 0.75],
            [0.6, 0.201],
            [0.7, 0.8],
        ];
        let kdt = PkdTree::new(&points);

        let needles = [[-1.0, 2.0], [-1.0, 2.0]];
        assert_eq!(kdt.query(&needles), [0, points.len() - 1]);
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
}
