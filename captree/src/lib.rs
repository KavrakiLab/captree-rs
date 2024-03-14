//! Efficient, branchless nearest-neighbor trees for robot collision checking.

#![cfg_attr(feature = "simd", feature(portable_simd))]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

#[cfg(feature = "simd")]
use std::simd::{prelude::*, LaneCount, SimdElement, SupportedLaneCount};

use std::{
    fmt::Debug,
    mem::size_of,
    ops::{Add, Sub},
};

mod affordance;
mod forest;

use affordance::Aabb;
pub use affordance::AffordanceTree;
pub use forest::PkdForest;

pub trait Axis: PartialOrd + Copy + Sub<Output = Self> + Add<Output = Self> {
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    const ZERO: Self;
    const SIZE: usize;

    #[must_use]
    fn is_finite(self) -> bool;

    #[must_use]
    fn abs(self) -> Self;

    #[must_use]
    fn in_between(self, rhs: Self) -> Self;
}

#[cfg(feature = "simd")]
pub trait AxisSimd<M>: SimdElement + Default {
    #[must_use]
    fn any(mask: M) -> bool;
}

pub trait Index: TryFrom<usize> + TryInto<usize> + Copy {
    const ZERO: Self;
}

#[cfg(feature = "simd")]
pub trait IndexSimd: SimdElement + Default {
    #[must_use]
    /// Convert a SIMD array of `Self` to a SIMD array of `usize`, without checking that each
    /// element is valid.
    ///
    /// # Safety
    ///
    /// This function is only safe if all values of `x` are valid when converted to a `usize`.
    unsafe fn to_simd_usize_unchecked<const L: usize>(x: Simd<Self, L>) -> Simd<usize, L>
    where
        LaneCount<L>: SupportedLaneCount;
}

pub trait Distance<A, const K: usize> {
    type Output: PartialOrd + Copy;
    const ZERO: Self::Output;

    #[must_use]
    fn distance(x1: &[A; K], x2: &[A; K]) -> Self::Output;

    #[must_use]
    fn closest_distance_to_volume(v: &Aabb<A, K>, x: &[A; K]) -> Self::Output;

    #[must_use]
    fn ball_contains_aabb(aabb: &Aabb<A, K>, center: &[A; K], r: Self::Output) -> bool;

    #[must_use]
    fn as_l1(r: Self::Output) -> A;
}

macro_rules! impl_axis {
    ($t: ty, $tm: ty) => {
        impl Axis for $t {
            const INFINITY: Self = <$t>::INFINITY;
            const NEG_INFINITY: Self = <$t>::NEG_INFINITY;
            const ZERO: Self = 0.0;
            const SIZE: usize = size_of::<Self>();

            fn is_finite(self) -> bool {
                <$t>::is_finite(self)
            }

            fn abs(self) -> Self {
                <$t>::abs(self)
            }

            fn in_between(self, rhs: Self) -> Self {
                (self + rhs) / 2.0
            }
        }

        impl<const K: usize> Distance<$t, K> for SquaredEuclidean {
            type Output = $t;
            const ZERO: Self::Output = 0.0;

            fn distance(x1: &[$t; K], x2: &[$t; K]) -> Self::Output {
                let mut total = 0.0;
                for i in 0..K {
                    total += (x1[i] - x2[i]).powi(2);
                }
                total
            }
            fn closest_distance_to_volume(v: &Aabb<$t, K>, x: &[$t; K]) -> Self::Output {
                let mut dist = 0.0;

                for d in 0..K {
                    let clamped = clamp(x[d], v.lo[d], v.hi[d]);
                    dist += (x[d] - clamped).powi(2);
                }

                dist
            }

            fn ball_contains_aabb(aabb: &Aabb<$t, K>, x: &[$t; K], r: Self::Output) -> bool {
                let mut dist = 0.0;

                for k in 0..K {
                    let lo_diff = (aabb.lo[k] - x[k]).powi(2);
                    let hi_diff = (aabb.hi[k] - x[k]).powi(2);

                    dist += if lo_diff < hi_diff { hi_diff } else { lo_diff };
                }

                dist <= r
            }

            fn as_l1(r: $t) -> $t {
                r.sqrt()
            }
        }

        #[cfg(feature = "simd")]
        impl<const L: usize> AxisSimd<Mask<$tm, L>> for $t
        where
            LaneCount<L>: SupportedLaneCount,
        {
            fn any(mask: Mask<$tm, L>) -> bool {
                Mask::<$tm, L>::any(mask)
            }
        }
    };
}

macro_rules! impl_idx {
    ($t: ty) => {
        impl Index for $t {
            const ZERO: Self = 0;
        }

        #[cfg(feature = "simd")]
        impl IndexSimd for $t {
            #[must_use]
            unsafe fn to_simd_usize_unchecked<const L: usize>(x: Simd<Self, L>) -> Simd<usize, L>
            where
                LaneCount<L>: SupportedLaneCount,
            {
                x.to_array().map(|a| a.try_into().unwrap_unchecked()).into()
            }
        }
    };
}

impl_axis!(f32, i32);
impl_axis!(f64, i64);

impl_idx!(u8);
impl_idx!(u16);
impl_idx!(u32);
impl_idx!(u64);
impl_idx!(usize);

#[derive(Debug)]
pub struct SquaredEuclidean;

/// Clamp a floating-point number.
fn clamp<A: PartialOrd>(x: A, min: A, max: A) -> A {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

#[derive(Clone, Debug, PartialEq)]
/// A power-of-two KD-tree.
///
/// # Generic parameters
///
/// - `D`: The dimension of the space.
pub struct PkdTree<const K: usize> {
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
    points: Box<[[f32; K]]>,
}

impl<const K: usize> PkdTree<K> {
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
    pub fn new(points: &[[f32; K]]) -> Self {
        /// Recursive helper function to sort the points for the KD tree and generate the tests.
        /// Runs in O(n log n)
        fn build_tree<const K: usize>(points: &mut [[f32; K]], tests: &mut [f32], k: u8, i: usize) {
            if points.len() > 1 {
                tests[i] = median_partition(points, k as usize);
                let next_k = (k + 1) % K as u8;
                let (lhs, rhs) = points.split_at_mut(points.len() / 2);
                build_tree(lhs, tests, next_k, 2 * i + 1);
                build_tree(rhs, tests, next_k, 2 * i + 2);
            }
        }

        assert!(K < u8::MAX as usize);

        let n2 = points.len().next_power_of_two();

        let mut tests = vec![f32::INFINITY; n2 - 1].into_boxed_slice();

        // hack: just pad with infinity to make it a power of 2
        let mut new_points = vec![[f32::INFINITY; K]; n2].into_boxed_slice();
        new_points[..points.len()].copy_from_slice(points);
        build_tree(new_points.as_mut(), tests.as_mut(), 0, 0);

        Self {
            tests,
            points: new_points,
        }
    }

    #[must_use]
    pub fn approx_nearest(&self, needle: [f32; K]) -> [f32; K] {
        self.get_point(forward_pass(&self.tests, &needle))
    }

    #[must_use]
    /// Determine whether a ball centered at `needle` with radius `r_squared` could collide with a
    /// point in this tree.
    pub fn might_collide(&self, needle: [f32; K], r_squared: f32) -> bool {
        distsq(self.approx_nearest(needle), needle) <= r_squared
    }

    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    #[cfg(feature = "simd")]
    pub fn might_collide_simd<const L: usize>(
        &self,
        needles: &[Simd<f32, L>; K],
        radii_squared: Simd<f32, L>,
    ) -> bool
    where
        LaneCount<L>: SupportedLaneCount,
    {
        let indices = forward_pass_simd(&self.tests, needles);
        let mut dists_squared = Simd::splat(0.0);
        let mut ptrs = Simd::splat(self.points.as_ptr().cast())
            .wrapping_offset(indices * Simd::splat(K as isize));
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
    pub fn query1_exact(&self, needle: [f32; K]) -> usize {
        let mut id = usize::MAX;
        let mut best_distsq = f32::INFINITY;
        self.exact_help(0, 0, &Aabb::ALL, needle, &mut id, &mut best_distsq);
        id
    }

    #[allow(clippy::cast_possible_truncation)]
    fn exact_help(
        &self,
        test_idx: usize,
        k: u8,
        bounding_box: &Aabb<f32, K>,
        point: [f32; K],
        best_id: &mut usize,
        best_distsq: &mut f32,
    ) {
        if SquaredEuclidean::closest_distance_to_volume(bounding_box, &point) > *best_distsq {
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
        bb_below.hi[k as usize] = test;
        let mut bb_above = *bounding_box;
        bb_above.lo[k as usize] = test;

        let next_k = (k + 1) % K as u8;
        if point[k as usize] < test {
            self.exact_help(
                2 * test_idx + 1,
                next_k,
                &bb_below,
                point,
                best_id,
                best_distsq,
            );
            self.exact_help(
                2 * test_idx + 2,
                next_k,
                &bb_above,
                point,
                best_id,
                best_distsq,
            );
        } else {
            self.exact_help(
                2 * test_idx + 2,
                next_k,
                &bb_above,
                point,
                best_id,
                best_distsq,
            );
            self.exact_help(
                2 * test_idx + 1,
                next_k,
                &bb_below,
                point,
                best_id,
                best_distsq,
            );
        }
    }

    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub const fn get_point(&self, id: usize) -> [f32; K] {
        self.points[id]
    }

    #[must_use]
    /// Return the total memory used (stack + heap) by this structure.
    pub const fn memory_used(&self) -> usize {
        size_of::<Self>() + (self.points.len() * K + self.tests.len()) * size_of::<f32>()
    }
}

#[inline]
fn forward_pass<A: Axis, const K: usize>(tests: &[A], point: &[A; K]) -> usize {
    // forward pass through the tree
    let mut test_idx = 0;
    let mut k = 0;
    for _ in 0..tests.len().trailing_ones() {
        test_idx =
            2 * test_idx + 1 + usize::from(unsafe { *tests.get_unchecked(test_idx) } <= point[k]);
        k = (k + 1) % K;
    }

    // retrieve affordance buffer location
    test_idx - tests.len()
}

#[inline]
#[allow(clippy::cast_possible_wrap)]
#[cfg(feature = "simd")]
fn forward_pass_simd<A, const K: usize, const L: usize>(
    tests: &[A],
    centers: &[Simd<A, L>; K],
) -> Simd<isize, L>
where
    Simd<A, L>: SimdPartialOrd,
    Mask<isize, L>: From<<Simd<A, L> as SimdPartialEq>::Mask>,
    A: Axis + AxisSimd<<Simd<A, L> as SimdPartialEq>::Mask>,
    LaneCount<L>: SupportedLaneCount,
{
    let mut test_idxs: Simd<isize, L> = Simd::splat(0);
    let mut k = 0;
    for _ in 0..tests.len().trailing_ones() {
        let test_ptrs = Simd::splat(tests.as_ptr()).wrapping_offset(test_idxs);
        let relevant_tests: Simd<A, L> = unsafe { Simd::gather_ptr(test_ptrs) };
        let cmp_results: Mask<isize, L> = centers[k % K].simd_ge(relevant_tests).into();

        let one = Simd::splat(1);
        test_idxs = (test_idxs << one) + one + (cmp_results.to_int() & Simd::splat(1));
        k = (k + 1) % K;
    }

    test_idxs - Simd::splat(tests.len() as isize)
}

#[inline]
/// Calculate the "true" median (halfway between two midpoints) and partition `points` about said
/// median along axis `d`.
fn median_partition<A: Axis, const K: usize>(points: &mut [[A; K]], k: usize) -> A {
    let (lh, med_hi, _) =
        points.select_nth_unstable_by(points.len() / 2, |a, b| a[k].partial_cmp(&b[k]).unwrap());
    let med_lo = lh
        .iter_mut()
        .map(|p| p[k])
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    A::in_between(med_lo, med_hi[k])
}

fn distsq<const K: usize>(a: [f32; K], b: [f32; K]) -> f32 {
    let mut total = 0.0f32;
    for i in 0..K {
        total += (a[i] - b[i]).powi(2);
    }
    total
}

#[cfg(test)]
mod tests {
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
        let neg1_idx = forward_pass(&kdt.tests, &neg1);
        assert_eq!(neg1_idx, 0);

        let pos1 = [1.0, 1.0];
        let pos1_idx = forward_pass(&kdt.tests, &pos1);
        assert_eq!(pos1_idx, points.len() - 1);
    }

    #[test]
    #[allow(clippy::cast_possible_wrap)]
    #[cfg(feature = "simd")]
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
        assert_eq!(
            forward_pass_simd(&kdt.tests, &needles),
            Simd::from_array([0, points.len() as isize - 1])
        );
    }

    #[test]
    fn not_a_power_of_two() {
        let points = vec![[0.0], [2.0], [4.0]];
        let kdt = PkdTree::new(&points);

        println!("{kdt:?}");

        assert_eq!(forward_pass(&kdt.tests, &[-0.1]), 0);
        assert_eq!(forward_pass(&kdt.tests, &[0.5]), 0);
        assert_eq!(forward_pass(&kdt.tests, &[1.5]), 1);
        assert_eq!(forward_pass(&kdt.tests, &[2.5]), 1);
        assert_eq!(forward_pass(&kdt.tests, &[3.5]), 2);
        assert_eq!(forward_pass(&kdt.tests, &[4.5]), 2);
    }

    #[test]
    fn a_power_of_two() {
        let points = vec![[0.0], [2.0], [4.0], [6.0]];
        let kdt = PkdTree::new(&points);

        println!("{kdt:?}");

        assert_eq!(forward_pass(&kdt.tests, &[-0.1]), 0);
        assert_eq!(forward_pass(&kdt.tests, &[0.5]), 0);
        assert_eq!(forward_pass(&kdt.tests, &[1.5]), 1);
        assert_eq!(forward_pass(&kdt.tests, &[2.5]), 1);
        assert_eq!(forward_pass(&kdt.tests, &[3.5]), 2);
        assert_eq!(forward_pass(&kdt.tests, &[4.5]), 2);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn does_it_partition() {
        let mut points = vec![[1.0], [2.0], [1.5], [2.1], [-0.5]];
        let median = median_partition(&mut points, 0);
        assert_eq!(median, 1.25);
        for p0 in &points[..points.len() / 2] {
            assert!(p0[0] <= median);
        }

        for p0 in &points[points.len() / 2..] {
            assert!(p0[0] >= median);
        }
    }
}
