//! Power-of-two k-d forests.

use std::{
    hint::unreachable_unchecked,
    mem::MaybeUninit,
    simd::{
        LaneCount, Mask, Simd, SimdConstPtr, SimdPartialEq, SimdPartialOrd,
        SupportedLaneCount,
    },
};

use rand::Rng;

use crate::distsq;

#[derive(Clone, Debug)]
struct RandomizedTree<const D: usize> {
    test_dims: Box<[u8]>,
    tests: Box<[f32]>,
    points: Box<[[f32; D]]>,
}

#[derive(Clone, Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct PkdForest<const D: usize, const T: usize> {
    test_seqs: [RandomizedTree<D>; T],
}

impl<const D: usize, const T: usize> PkdForest<D, T> {
    pub fn new(points: &[[f32; D]], rng: &mut impl Rng) -> Self {
        unsafe {
            let mut buf: [MaybeUninit<RandomizedTree<D>>; T] = MaybeUninit::uninit().assume_init();

            for tree in &mut buf {
                tree.write(RandomizedTree::new(points, rng));
            }

            PkdForest {
                test_seqs: buf.map(|x| x.assume_init()),
            }
        }
    }

    #[must_use]
    /// # Panics
    ///
    /// This function will panic if `T` is 0.
    pub fn query1(&self, needle: [f32; D]) -> ([f32; D], f32) {
        self.test_seqs
            .iter()
            .map(|t| t.query1(needle))
            .map(|point| (point, distsq(needle, point)))
            .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap())
            .unwrap()
    }

    #[must_use]
    pub fn might_collide(&self, needle: [f32; D], r_squared: f32) -> bool {
        self.test_seqs
            .iter()
            .any(|t| distsq(t.query1(needle), needle) < r_squared)
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
        let mut not_yet_collided = Mask::splat(true);

        for tree in &self.test_seqs {
            let indices = tree.mask_query(needles, not_yet_collided);
            let mut dists_sq = Simd::splat(0.0);
            let mut ptrs = Simd::splat((tree.points.as_ref() as *const [[f32; D]]).cast::<f32>())
                .wrapping_add(indices);
            for needle_set in needles {
                let diffs =
                    unsafe { Simd::gather_select_ptr(ptrs, not_yet_collided, Simd::splat(0.0)) }
                        - needle_set;
                dists_sq += diffs * diffs;
                ptrs = ptrs.wrapping_add(Simd::splat(1));
            }

            not_yet_collided &= radii_squared.simd_lt(dists_sq).cast();

            if !not_yet_collided.all() {
                // all have collided - can return quickly
                return true;
            }
        }

        false
    }
}

impl<const D: usize> RandomizedTree<D> {
    pub fn new(points: &[[f32; D]], rng: &mut impl Rng) -> Self {
        /// Recursive helper function to sort the points for the KD tree and generate the tests.
        fn recur_sort_points<const D: usize>(
            points: &mut [[f32; D]],
            tests: &mut [f32],
            test_dims: &mut [u8],
            i: usize,
            rng: &mut impl Rng,
        ) {
            // TODO make this algorithm O(n log n) instead of O(n log^2 n)
            if points.len() > 1 {
                let d = rng.gen_range(0..D);
                points.sort_unstable_by(|a, b| a[d].partial_cmp(&b[d]).unwrap());
                let median = (points[points.len() / 2 - 1][d] + points[points.len() / 2][d]) / 2.0;
                tests[i] = median;
                test_dims[i] = u8::try_from(d).unwrap();
                let (lhs, rhs) = points.split_at_mut(points.len() / 2);
                recur_sort_points(lhs, tests, test_dims, 2 * i + 1, rng);
                recur_sort_points(rhs, tests, test_dims, 2 * i + 2, rng);
            }
        }

        assert!(D < u8::MAX as usize);

        let n2 = points.len().next_power_of_two();

        let mut tests = vec![f32::INFINITY; n2 - 1].into_boxed_slice();

        // hack: just pad with infinity to make it a power of 2
        let mut new_points = vec![[f32::INFINITY; D]; n2];
        new_points[..points.len()].copy_from_slice(points);
        let mut test_dims = vec![0; n2 - 1].into_boxed_slice();
        recur_sort_points(
            new_points.as_mut(),
            tests.as_mut(),
            test_dims.as_mut(),
            0,
            rng,
        );

        Self {
            tests,
            points: new_points.into_boxed_slice(),
            test_dims,
        }
    }

    pub fn query1(&self, needle: [f32; D]) -> [f32; D] {
        let n2 = self.tests.len() + 1;
        assert!(n2.is_power_of_two());
        assert_eq!(self.tests.len(), self.test_dims.len());

        let mut test_idx = 0;
        for _ in 0..n2.ilog2() as usize {
            // println!("current idx: {test_idx}");
            let add = if needle[self.test_dims[test_idx] as usize] < (self.tests[test_idx]) {
                1
            } else {
                2
            };
            test_idx <<= 1;
            test_idx += add;
        }

        self.points[test_idx - self.tests.len()]
    }

    #[allow(clippy::cast_possible_truncation)]
    /// Perform a masked SIMD query of this tree, only determining the location of the nearest
    /// neighbors for points in `mask`.
    fn mask_query<const L: usize>(
        &self,
        needles: &[Simd<f32, L>; D],
        mask: Mask<isize, L>,
    ) -> Simd<usize, L>
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
        for _ in 0..n2.ilog2() as usize {
            let relevant_tests: Simd<f32, L> = unsafe {
                Simd::gather_select_unchecked(&self.tests, mask, test_idxs, Simd::splat(f32::NAN))
            };
            let dim_nrs = unsafe {
                Simd::gather_select_unchecked(&self.test_dims, mask, test_idxs, Simd::splat(0))
            };
            let mut cmp_results = Mask::splat(false);
            for (d, &these_needles) in needles.iter().enumerate() {
                let loading_this_dim = dim_nrs.simd_eq(Simd::splat(d as u8)).cast() & mask;
                cmp_results |= loading_this_dim & these_needles.simd_lt(relevant_tests).cast();
            }

            // TODO is there a faster way than using a conditional select?
            test_idxs <<= Simd::splat(1);
            test_idxs += cmp_results.select(Simd::splat(1), Simd::splat(2));
        }

        test_idxs - Simd::splat(self.tests.len())
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::*;

    #[test]
    fn build_a_forest() {
        let points = [[0.0, 0.0], [0.2, 1.0], [-1.0, 0.4]];

        let forest = PkdForest::<2, 2>::new(&points, &mut thread_rng());
        println!("{forest:#?}");
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn find_the_closest() {
        let points = [[0.0, 0.0], [0.2, 1.0], [-1.0, 0.4]];

        let forest = PkdForest::<2, 2>::new(&points, &mut thread_rng());
        // assert_eq!(forest.query1([0.01, 0.02]), ([]))
        let (nearest, ndsq) = forest.query1([0.01, 0.02]);
        assert_eq!(nearest, [0.0, 0.0]);
        assert!((ndsq - 0.0005) < 1e-6);
        println!("{:?}", forest.query1([0.01, 0.02]));
    }
}
