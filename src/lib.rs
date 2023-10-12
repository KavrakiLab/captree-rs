#![feature(portable_simd)]
#![feature(new_uninit)]

use std::{
    ops::ShlAssign,
    simd::{LaneCount, Mask, Simd, SimdPartialOrd, SupportedLaneCount},
};

/// A power-of-two KD-tree.
///
/// # Generic parameters
///
/// - `D`: The dimension of the space.
/// - `N`: The number of points in the tree. Must be a power of two.
pub struct PkdTree<const D: usize, const N: usize> {
    /// The test values for determining which part of the tree to enter.
    ///
    /// The first element of `tests` should be the first value to test against.
    /// If we are less than `tests[0]`, we move on to `tests[1]`; if not, we move on to `tests[2]`.
    /// At the `i`-th test performed in sequence of the traversal, if we are less than `tests[idx]`,
    /// we advance to `2 * i + 1`; otherwise, we go to `2 * i + 2`.
    tests: Box<[f32; N]>,
    /// The relevant points at the center of each volume divided by `tests`.
    points: Box<[[f32; N]; D]>,
}

impl<const D: usize, const N: usize> PkdTree<D, N> {
    /// Construct a new `PkdTree` containing all the points in `points`.
    /// For performance, this function changes the ordering of `points`, but does not affect the
    /// set of points inside it.
    ///
    /// TODO: do all our sorting on the allocation that we return?
    pub fn new(points: &mut [[f32; D]; N]) -> Self {
        /// Recursive helper function to sort the points for the KD tree and generate the tests.
        fn recur_sort_points<const D: usize>(
            points: &mut [[f32; D]],
            tests: &mut [f32],
            d: usize,
            i: usize,
        ) {
            if points.len() > 1 {
                points.sort_by(|a, b| a[d].partial_cmp(&b[d]).unwrap());
                let median = (points[points.len() / 2 - 1][d] + points[points.len() / 2][d]) / 2.0;
                tests[i] = median;
                let next_dim = (d + 1) % D;
                let (lhs, rhs) = points.split_at_mut(points.len() / 2);
                recur_sort_points(lhs, tests, next_dim, 2 * i + 1);
                recur_sort_points(rhs, tests, next_dim, 2 * i + 2);
            }
        }

        assert!(N.is_power_of_two());

        let mut tests = Box::new([0.0; N]);
        recur_sort_points(points, tests.as_mut(), 0, 0);

        let mut my_points = Box::new([[0.0; N]; D]);
        for (i, pt) in points.iter().enumerate() {
            for (d, value) in (*pt).into_iter().enumerate() {
                my_points[d][i] = value;
            }
        }

        PkdTree {
            tests,
            points: my_points,
        }
    }

    /// Get the indices of points which are closest to `needles`.
    ///
    /// TODO: refactor this to use `needles` as an out parameter as well, and shove the nearest
    /// points in there?
    pub fn query<const L: usize>(&self, needles: &[[f32; L]; D]) -> [usize; L]
    where
        LaneCount<L>: SupportedLaneCount,
    {
        assert!(N.is_power_of_two());

        let mut test_idxs: Simd<usize, L> = Simd::splat(0);

        // Advance the tests forward
        for i in 0..N.ilog2() as usize {
            // TODO do not bounds check on this gather
            let relevant_tests: Simd<f32, L> =
                Simd::gather_or_default(self.tests.as_ref(), test_idxs);
            // TODO do not bounds check on this either
            let needle_values = Simd::from_slice(&needles[i % D]);
            let cmp_results: Mask<isize, L> = needle_values.simd_lt(relevant_tests).into();

            // TODO is there a faster way than using a conditional select?
            test_idxs.shl_assign(Simd::splat(1));
            test_idxs += cmp_results.select(Simd::splat(1), Simd::splat(2));
        }

        let lookup_idxs = test_idxs - Simd::splat(N - 1);
        lookup_idxs.into()
    }

    /// Get the access index of the point closest to
    pub fn query1(&self, needle: [f32; D]) -> usize {
        assert!(N.is_power_of_two());

        let mut test_idx = 0;
        for i in 0..N.ilog2() as usize {
            let add = if needle[i % D] < self.tests[test_idx] {
                1
            } else {
                2
            };
            test_idx <<= 1;
            test_idx += add;
        }

        let lookup_idx = test_idx + 1 - N;
        lookup_idx
    }

    pub fn get_point(&self, id: usize) -> [f32; D] {
        let mut point = [0.0; D];
        for d in 0..D {
            point[d] = self.points[d][id];
        }

        point
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_query() {
        let mut points = [
            [0.1, 0.1],
            [0.1, 0.2],
            [0.5, 0.0],
            [0.3, 0.9],
            [1.0, 1.0],
            [0.35, 0.75],
            [0.6, 0.2],
            [0.7, 0.8],
        ];
        let kdt = PkdTree::new(&mut points);

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
        let mut points = [
            [0.1, 0.1],
            [0.1, 0.2],
            [0.5, 0.0],
            [0.3, 0.9],
            [1.0, 1.0],
            [0.35, 0.75],
            [0.6, 0.2],
            [0.7, 0.8],
        ];
        let kdt = PkdTree::new(&mut points);

        let needles = [[-1.0, 2.0], [-1.0, 2.0]];
        assert_eq!(kdt.query(&needles), [0, points.len() - 1]);
    }
}
