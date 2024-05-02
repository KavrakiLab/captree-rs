//! Efficient, branchless nearest-neighbor trees for robot collision checking.

#![cfg_attr(feature = "simd", feature(portable_simd))]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

#[cfg(feature = "simd")]
use std::simd::{LaneCount, SimdElement, SupportedLaneCount};

use std::{
    fmt::Debug,
    mem::size_of,
    ops::{Add, Sub},
};

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
use std::{array, marker::PhantomData};

#[cfg(feature = "simd")]
use std::{
    ops::{AddAssign, Mul},
    simd::{
        cmp::{SimdPartialEq, SimdPartialOrd},
        ptr::SimdConstPtr,
        Mask, Simd,
    },
};

#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(clippy::module_name_repetitions)]
/// An affordance tree, which allows for efficient nearest-neighbor-within-a-radius queries.
///
/// # Generic parameters
///
/// - `K`: The dimension of the space.
/// - `A`: The value of the axes of each point.
///        This should typically be `f32` or `f64`.
/// - `I`: The index integer.
///        This should generally be an unsigned integer, such as `usize` or `u32`.
/// - `D`: The distance metric.
/// - `R`: The output value of the distance metric.
///        This should typically be `f32` or `f64`.
pub struct AffordanceTree<const K: usize, A = f32, I = usize, D = SquaredEuclidean, R = f32> {
    /// The test values for determining which part of the tree to enter.
    ///
    /// The first element of `tests` should be the first value to test against.
    /// If we are less than `tests[0]`, we move on to `tests[1]`; if not, we move on to `tests[2]`.
    /// At the `i`-th test performed in sequence of the traversal, if we are less than `tests[idx]`,
    /// we advance to `2 * idx + 1`; otherwise, we go to `2 * idx + 2`.
    ///
    /// The length of `tests` must be `N`, rounded up to the next power of 2, minus one.
    tests: Box<[A]>,
    /// The range of radii which are legal for queries on this tree.
    /// The first element is the minimum and the second element is the maximum.
    rsq_range: (R, R),
    aabbs: Box<[Aabb<A, K>]>,
    /// Indexes for the starts of the affordance buffer subsequence of `points` corresponding to
    /// each leaf cell in the tree.
    /// This buffer is padded with one extra `usize` at the end with the maximum length of `points`
    /// for the sake of branchless computation.
    starts: Box<[I]>,
    afforded: [Box<[A]>; K],
    _phantom: PhantomData<D>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// A prismatic bounding volume.
pub struct Aabb<A, const K: usize> {
    pub lo: [A; K],
    pub hi: [A; K],
}

impl<A, I, D, R, const K: usize> AffordanceTree<K, A, I, D, R>
where
    A: Axis,
    I: Index,
    D: Distance<A, K, Output = R>,
    R: PartialOrd + Copy,
{
    #[must_use]
    /// Construct a new affordance tree containing all the points in `points`.
    /// `r_range` is a `(minimum, maximum)` pair containing the lower and upper bound on the
    /// radius of the balls which will be queried against the tree.
    /// `rng` is a random number generator.
    /// Although the results of the tree are deterministic after construction, the construction
    /// process for the tree is probabilistic.
    /// The output of construction will be the same independent of the RNG, but the process to
    /// construct the tree may vary with the provided RNG.
    ///
    /// This function will return `None` if there are too many points to be indexed by `I`, or if
    /// `K` is greater than `255`.
    pub fn new(points: &[[A; K]], r_range: (R, R)) -> Option<Self> {
        if K >= u8::MAX as usize {
            return None;
        }

        let n2 = points.len().next_power_of_two();

        let mut tests = vec![A::INFINITY; n2 - 1].into_boxed_slice();

        // hack: just pad with infinity to make it a power of 2
        let mut points2 = vec![[A::INFINITY; K]; n2].into_boxed_slice();
        points2[..points.len()].copy_from_slice(points);
        // hack - reduce number of reallocations by allocating a lot of points from the start
        let mut afforded = array::from_fn(|_| Vec::with_capacity(n2 * 1000));
        let mut starts = vec![I::ZERO; n2 + 1].into_boxed_slice();

        let mut aabbs = vec![
            Aabb {
                lo: [A::ZERO; K],
                hi: [A::ZERO; K],
            };
            n2
        ]
        .into_boxed_slice();

        let r_max_l1 = D::as_l1(r_range.1);

        Self::new_help(
            &mut points2,
            &mut tests,
            &mut aabbs,
            &mut afforded,
            &mut starts,
            0,
            0,
            r_range,
            r_max_l1,
            Vec::new(),
            Aabb::ALL,
        )
        .ok()?;

        Some(Self {
            tests,
            rsq_range: r_range,
            starts,
            afforded: afforded.map(Vec::into_boxed_slice),
            aabbs,
            _phantom: PhantomData,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn new_help(
        points: &mut [[A; K]],
        tests: &mut [A],
        aabbs: &mut [Aabb<A, K>],
        afforded: &mut [Vec<A>; K],
        starts: &mut [I],
        k: usize,
        i: usize,
        r_range: (R, R),
        r_max_l1: A,
        in_range: Vec<[A; K]>,
        cell: Aabb<A, K>,
    ) -> Result<(), ()> {
        if let [rep] = *points {
            let z = i - tests.len();
            let aabb = &mut aabbs[z];
            *aabb = Aabb { lo: rep, hi: rep };
            if rep[0].is_finite() {
                for k in 0..K {
                    afforded[k].push(rep[k]);
                }

                if !D::ball_contains_aabb(&cell, &rep, r_range.0) {
                    for ak in afforded.iter_mut() {
                        ak.reserve(ak.len() + in_range.len());
                    }
                    for p in in_range {
                        aabb.insert(&p);
                        for k in 0..K {
                            afforded[k].push(p[k]);
                        }
                    }
                }
            }

            starts[z + 1] = afforded[0].len().try_into().map_err(|_| ())?;
            return Ok(());
        }

        let test = median_partition(points, k);
        tests[i] = test;

        let (lhs, rhs) = points.split_at_mut(points.len() / 2);
        let (lo_vol, hi_vol) = cell.split(test, k);

        let lo_too_small = D::distance(&lo_vol.lo, &lo_vol.hi) <= r_range.0;
        let hi_too_small = D::distance(&hi_vol.lo, &hi_vol.hi) <= r_range.0;

        // retain only points which might be in the affordance buffer for the split-out cells
        let (lo_afford, hi_afford) = match (lo_too_small, hi_too_small) {
            (false, false) => {
                let mut lo_afford = in_range;
                let mut hi_afford = lo_afford.clone();
                lo_afford.retain(|pt| lo_vol.affords::<D>(pt, &r_range));
                lo_afford.extend(rhs.iter().filter(|pt| pt[k] <= test + r_max_l1));
                hi_afford.retain(|pt| hi_vol.affords::<D>(pt, &r_range));
                hi_afford.extend(
                    lhs.iter()
                        .filter(|pt| pt[k].is_finite() && test - r_max_l1 <= pt[k]),
                );

                (lo_afford, hi_afford)
            }
            (false, true) => {
                let mut lo_afford = in_range;
                lo_afford.retain(|pt| lo_vol.affords::<D>(pt, &r_range));
                lo_afford.extend(rhs.iter().filter(|pt| pt[k] <= test + r_max_l1));

                (lo_afford, Vec::new())
            }
            (true, false) => {
                let mut hi_afford = in_range;
                hi_afford.retain(|pt| hi_vol.affords::<D>(pt, &r_range));
                hi_afford.extend(
                    lhs.iter()
                        .filter(|pt| pt[k].is_finite() && test - r_max_l1 <= pt[k]),
                );

                (Vec::new(), hi_afford)
            }
            (true, true) => (Vec::new(), Vec::new()),
        };

        let next_k = (k + 1) % K;
        Self::new_help(
            lhs,
            tests,
            aabbs,
            afforded,
            starts,
            next_k,
            2 * i + 1,
            r_range,
            r_max_l1,
            lo_afford,
            lo_vol,
        )?;
        Self::new_help(
            rhs,
            tests,
            aabbs,
            afforded,
            starts,
            next_k,
            2 * i + 2,
            r_range,
            r_max_l1,
            hi_afford,
            hi_vol,
        )?;

        Ok(())
    }

    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    /// Determine whether a point in this tree is within a distance of `radius` to `center`.
    ///
    /// # Panics
    ///
    /// This function will panic if `r_squared` is outside the range of radii passed to the
    /// construction of the tree.
    /// TODO: implement real error handling.
    pub fn collides(&self, center: &[A; K], radius: R) -> bool {
        // ball mus be in the rsq range
        debug_assert!(self.rsq_range.0 <= radius);
        debug_assert!(radius <= self.rsq_range.1);

        let i = forward_pass(&self.tests, center);
        let aabb = unsafe { self.aabbs.get_unchecked(i) };
        if D::closest_distance_to_volume(aabb, center) > radius {
            return false;
        }

        let range = unsafe {
            // SAFETY: The conversion worked the first way.
            self.starts[i].try_into().unwrap_unchecked()
                ..self.starts[i + 1].try_into().unwrap_unchecked()
        };

        // check affordance buffer
        for i in range {
            let mut aff_pt = [A::INFINITY; K];
            #[allow(clippy::needless_range_loop)]
            for (ak, sk) in aff_pt.iter_mut().zip(&self.afforded) {
                *ak = sk[i];
            }
            if D::distance(&aff_pt, center) <= radius {
                return true;
            }
        }

        false
    }

    #[must_use]
    /// Get the total memory used (stack + heap) by this structure, measured in bytes.
    pub const fn memory_used(&self) -> usize {
        size_of::<Self>()
            + K * self.afforded[0].len() * size_of::<A>()
            + self.starts.len() * size_of::<I>()
            + self.tests.len() * size_of::<I>()
            + self.aabbs.len() * size_of::<Aabb<A, K>>()
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    /// Get the average number of affordances per point.
    pub fn affordance_size(&self) -> f64 {
        self.afforded.len() as f64 / (self.tests.len() + 1) as f64
    }
}

#[allow(clippy::mismatching_type_param_order)]
#[cfg(feature = "simd")]
impl<A, I, const K: usize> AffordanceTree<K, A, I, SquaredEuclidean, A>
where
    I: IndexSimd,
    SquaredEuclidean: Distance<A, K, Output = A>,
    A: std::fmt::Debug,
{
    #[must_use]
    /// Determine whether any sphere in the list of provided spheres intersects a point in this
    /// tree.
    ///
    /// # Panics
    ///
    /// This function may panic if `L` is greater than the alignment of the underlying allocations
    /// backing the structure.
    pub fn collides_simd<const L: usize>(
        &self,
        centers: &[Simd<A, L>; K],
        radii: Simd<A, L>,
    ) -> bool
    where
        LaneCount<L>: SupportedLaneCount,
        Simd<A, L>:
            SimdPartialOrd + Sub<Output = Simd<A, L>> + Mul<Output = Simd<A, L>> + AddAssign,
        Mask<isize, L>: From<<Simd<A, L> as SimdPartialEq>::Mask>,
        A: Axis + AxisSimd<<Simd<A, L> as SimdPartialEq>::Mask>,
    {
        let zs = forward_pass_simd(&self.tests, centers);

        let mut inbounds = Mask::splat(true);

        let mut aabb_ptrs = Simd::splat(self.aabbs.as_ptr()).wrapping_offset(zs).cast();

        unsafe {
            for center in centers {
                inbounds &=
                    Simd::gather_select_ptr(aabb_ptrs, inbounds, Simd::splat(A::NEG_INFINITY))
                        - radii
                        <= *center;
                aabb_ptrs = aabb_ptrs.wrapping_add(Simd::splat(1));
            }
            for center in centers {
                inbounds &=
                    Simd::gather_select_ptr(aabb_ptrs, inbounds, Simd::splat(A::NEG_INFINITY))
                        >= *center - radii;
                aabb_ptrs = aabb_ptrs.wrapping_add(Simd::splat(1));
            }
        }
        if !inbounds.any() {
            return false;
        }

        // retrieve start/end pointers for the affordance buffer
        let start_ptrs = Simd::splat(self.starts.as_ptr()).wrapping_offset(zs);
        let starts = unsafe { I::to_simd_usize_unchecked(Simd::gather_ptr(start_ptrs)) }.to_array();
        let ends = unsafe {
            I::to_simd_usize_unchecked(Simd::gather_ptr(start_ptrs.wrapping_add(Simd::splat(1))))
        }
        .to_array();

        starts
            .into_iter()
            .zip(ends)
            .zip(inbounds.to_array())
            .filter_map(|(r, i)| i.then_some(r))
            .enumerate()
            .any(|(j, (start, end))| {
                let mut n_center = [Simd::splat(SquaredEuclidean::ZERO); K];
                for k in 0..K {
                    n_center[k] = Simd::splat(centers[k][j]);
                }
                let rs = Simd::splat(radii[j]);
                let rs_sq = rs * rs;
                for i in (start..end).step_by(L) {
                    let mut dists_sq = Simd::splat(SquaredEuclidean::ZERO);
                    #[allow(clippy::needless_range_loop)]
                    for k in 0..K {
                        let vals = Simd::from_slice(&self.afforded[k][i..]);
                        let diff = vals - centers[k];
                        dists_sq += diff * diff;
                    }
                    if A::any(dists_sq.simd_le(rs_sq)) {
                        return true;
                    }
                }

                false
            })
    }
}

impl<A, const K: usize> Aabb<A, K>
where
    A: Axis,
{
    pub(crate) const ALL: Self = Self {
        lo: [A::NEG_INFINITY; K],
        hi: [A::INFINITY; K],
    };

    /// Split this volume by a test plane with value `test` along `dim`.
    const fn split(mut self, test: A, dim: usize) -> (Self, Self) {
        let mut rhs = self;
        self.hi[dim] = test;
        rhs.lo[dim] = test;

        (self, rhs)
    }

    fn affords<D: Distance<A, K>>(&self, pt: &[A; K], rsq_range: &(D::Output, D::Output)) -> bool {
        D::closest_distance_to_volume(self, pt) <= rsq_range.1
    }

    fn insert(&mut self, point: &[A; K]) {
        self.lo
            .iter_mut()
            .zip(&mut self.hi)
            .zip(point)
            .for_each(|((l, h), &x)| {
                if *l > x {
                    *l = x;
                }
                if x > *h {
                    *h = x;
                }
            });
    }
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

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use super::*;

    #[test]
    fn build_simple() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = AffordanceTree::<2>::new(&points, (0.0, 0.04));
        println!("{t:?}");
    }

    #[test]
    fn exact_query_single() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = AffordanceTree::<2>::new(&points, (0.0, 0.2f32.powi(2))).unwrap();

        println!("{t:?}");

        let q0 = [0.0, -0.01];
        assert!(t.collides(&q0, (0.12f32).powi(2)));
    }

    #[test]
    fn another_one() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = AffordanceTree::<2>::new(&points, (0.0, 0.04)).unwrap();

        println!("{t:?}");

        let q0 = [0.003_265_380_9, 0.106_527_805];
        assert!(t.collides(&q0, 0.0004));
    }

    #[test]
    fn three_d() {
        let points = [
            [0.0; 3],
            [0.1, -1.1, 0.5],
            [-0.2, -0.3, 0.25],
            [0.1, -1.1, 0.5],
        ];

        let t = AffordanceTree::<3>::new(&points, (0.0, 0.04)).unwrap();

        println!("{t:?}");
        assert!(t.collides(&[0.0, 0.1, 0.0], 0.011));
        assert!(!t.collides(&[0.0, 0.1, 0.0], 0.05 * 0.05));
    }

    #[test]
    fn fuzz() {
        const R_SQ: f32 = 0.0004;
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let mut rng = thread_rng();
        let t = AffordanceTree::<2>::new(&points, (0.0, 0.0008)).unwrap();

        for _ in 0..10_000 {
            let p = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
            let collides = points
                .iter()
                .any(|a| SquaredEuclidean::distance(a, &p) < R_SQ);
            println!("{p:?}; {collides}");
            assert_eq!(collides, t.collides(&p, R_SQ));
        }
    }

    #[test]
    /// This test _should_ fail, but it doesn't somehow?
    fn weird_bounds() {
        const R_SQ: f32 = 1.0;
        let points = [
            [-1.0, 0.0],
            [0.001, 0.0],
            [0.0, 0.5],
            [-1.0, 10.0],
            [-2.0, 10.0],
            [-3.0, 10.0],
            [-0.5, 0.0],
            [-11.0, 1.0],
            [-1.0, -0.5],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
        ];
        let rsq_range = (R_SQ - f32::EPSILON, R_SQ + f32::EPSILON);
        let t = AffordanceTree::<2>::new(&points, rsq_range).unwrap();
        println!("{t:?}");

        assert!(t.collides(&[-0.001, -0.2], 1.0));
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
