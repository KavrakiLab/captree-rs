//! Efficient, branchless nearest-neighbor trees for robot collision checking.

#![cfg_attr(feature = "simd", feature(portable_simd))]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]

use std::{
    array,
    fmt::Debug,
    marker::PhantomData,
    mem::size_of,
    ops::{Add, Sub},
};

#[cfg(feature = "simd")]
use std::{
    cmp::max,
    ops::{AddAssign, Mul},
    simd::{
        cmp::{SimdPartialEq, SimdPartialOrd},
        ptr::SimdConstPtr,
        LaneCount, Mask, Simd, SimdElement, SupportedLaneCount,
    },
};

/// A generic trait representing values which may be used as an "axis;" that is, elements of a
/// vector representing a point.
///
/// An array of `Axis` values is a point which can be stored in a [`Capt`].
/// Accordingly, this trait specifies nearly all the requirements for points that [`Capt`]s require.
/// The only exception is that [`Axis`] values really ought to be [`Ord`] instead of [`PartialOrd`];
/// however, due to the disaster that is IEE 754 floating point numbers, `f32` and `f64` are not
/// totally ordered. As a compromise, we relax the `Ord` requirement so that you can use floats in a
/// `Capt`.
///
/// # Examples
///
/// ```
/// #[derive(Clone, Copy, PartialOrd, PartialEq)]
/// enum HyperInt {
///     MinusInf,
///     Real(i32),
///     PlusInf,
/// }
///
/// impl std::ops::Add for HyperInt {
/// // ...
/// #    type Output = Self;
/// #
/// #    fn add(self, rhs: Self) -> Self {
/// #        match (self, rhs) {
/// #            (Self::MinusInf, Self::PlusInf) => Self::Real(0), // evil, but who cares?
/// #            (Self::MinusInf, _) | (_, Self::MinusInf) => Self::MinusInf,
/// #            (Self::PlusInf, _) | (_, Self::PlusInf) => Self::PlusInf,
/// #            (Self::Real(x), Self::Real(y)) => Self::Real(x + y),
/// #        }
/// #    }
/// }
///
///
/// impl std::ops::Sub for HyperInt {
/// // ...
/// #    type Output = Self;
/// #
/// #    fn sub(self, rhs: Self) -> Self {
/// #        match (self, rhs) {
/// #            (Self::MinusInf, Self::MinusInf) | (Self::PlusInf, Self::PlusInf) => Self::Real(0), // evil, but who cares?
/// #            (Self::MinusInf, _) | (_, Self::PlusInf) => Self::MinusInf,
/// #            (Self::PlusInf, _) | (_, Self::MinusInf) => Self::PlusInf,
/// #            (Self::Real(x), Self::Real(y)) => Self::Real(x - y),
/// #        }
/// #    }
/// }
///
/// impl captree::Axis for HyperInt {
///     const INFINITY: Self = Self::PlusInf;
///     const NEG_INFINITY: Self = Self::MinusInf;
///     
///     fn is_finite(self) -> bool {
///         matches!(self, Self::Real(_))
///     }
///     
///     fn in_between(self, rhs: Self) -> Self {
///         match (self, rhs) {
///             (Self::PlusInf, Self::MinusInf) | (Self::MinusInf, Self::PlusInf) => Self::Real(0),
///             (Self::MinusInf, _) | (_, Self::MinusInf) => Self::MinusInf,
///             (Self::PlusInf, _) | (_, Self::PlusInf) => Self::PlusInf,
///             (Self::Real(a), Self::Real(b)) => Self::Real(a + (b - a) / 2)
///         }
///     }
/// }
/// ```
pub trait Axis: PartialOrd + Copy + Sub<Output = Self> + Add<Output = Self> {
    /// A value which is larger than any finite value.
    const INFINITY: Self;
    /// A value which is smaller than any finite value.
    const NEG_INFINITY: Self;

    #[must_use]
    /// Determine whether this value is finite or infinite.
    fn is_finite(self) -> bool;

    #[must_use]
    /// Compute a value of `Self` which is halfway between `self` and `rhs`.
    /// If there are no legal values between `self` and `rhs`, it is acceptable to return `self`
    /// instead.
    fn in_between(self, rhs: Self) -> Self;
}

#[cfg(feature = "simd")]
/// A trait used for masks over SIMD vectors, used for parallel querying on [`Capt`]s.
///
/// The interface for this trait should be considered unstable since the standard SIMD API may
/// change with Rust versions.
pub trait AxisSimd<M>: SimdElement + Default {
    #[must_use]
    /// Determine whether any element of this mask is set to `true`.
    fn any(mask: M) -> bool;
}

/// An index type used for lookups into and out of arrays.
///
/// This is implemented so that [`Capt`]s can use smaller index sizes (such as [`u32`] or [`u16`])
/// for improved memory performance.
pub trait Index: TryFrom<usize> + TryInto<usize> + Copy {
    /// The zero index. This must be equal to `(0usize).try_into().unwrap()`.
    const ZERO: Self;
}

#[cfg(feature = "simd")]
/// A SIMD parallel version of [`Index`].
///
/// This is used for implementing SIMD lookups in a [`Capt`].
/// The interface for this trait should be considered unstable since the standard SIMD API may
/// change with Rust versions.
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

/// A distance metric.
///
/// This distance metric is used for determining nearest-neighbor candidates for a [`Capt`].
/// For now, the only provided metric is [`SquaredEuclidean`].
///
/// # Examples
///
/// Below, a sample implementation of the taxicab (L1) distance metric.
///
/// ```
/// use captree::{Aabb, Distance};
///
/// struct Taxicab;
///
/// impl<const K: usize> Distance<f32, K> for Taxicab {
///     type Output = f32;
///     const ZERO: f32 = 0.0;
///
///     fn distance(x1: &[f32; K], x2: &[f32; K]) -> f32 {
///         x1.iter().zip(x2.iter()).map(|(&a, &b)| (a - b).abs()).sum()
///     }
///
///     fn closest_distance_to_aabb(v: &Aabb<f32, K>, x: &[f32; K]) -> f32 {
///         v.lo.iter()
///             .zip(v.hi.iter())
///             .zip(x.iter())
///             .map(|((&l, &h), &a)| {
///                 if l <= a && a <= h {
///                     0.0
///                 } else if a < l {
///                     l - a
///                 } else {
///                     a - h
///                 }
///             })
///             .sum()
///     }
///
///     fn ball_contains_aabb(aabb: &Aabb<f32, K>, center: &[f32; K], r: f32) -> bool {
///         aabb.lo
///             .iter()
///             .zip(aabb.hi.iter())
///             .zip(center.iter())
///             .all(|((&l, &h), &a)| a - r <= l && a + r >= h)
///     }
///
///     fn as_l1(r: f32) -> f32 {
///         r
///     }
/// }
/// ```
pub trait Distance<A, const K: usize> {
    /// The value returned by distance computations.
    /// In practice, this should be [`Ord`], but we cannot make that restriction and conveniently
    /// use floats due to IEEE 754 floats not being totally ordered.
    type Output: PartialOrd + Copy;
    /// The zero distance.
    ///
    /// If and only if two points have distance zero to one another, they are identical.
    const ZERO: Self::Output;

    #[must_use]
    /// Compute the distance between two points.
    fn distance(x1: &[A; K], x2: &[A; K]) -> Self::Output;

    #[must_use]
    /// Compute the minimum distance between a given point `x` and all points in the prismatic
    /// volume `v`.
    fn closest_distance_to_aabb(v: &Aabb<A, K>, x: &[A; K]) -> Self::Output;

    #[must_use]
    /// Determine whether the ball with center `center` and radius `r` (defined as the set of all
    /// points within distance `r` from `center`) contains all points in the volume of `aabb`.
    fn ball_contains_aabb(aabb: &Aabb<A, K>, center: &[A; K], r: Self::Output) -> bool;

    #[must_use]
    /// Conservatively convert a radius to an L1 distance value; that is, the minimum radius `r'`
    /// such that any two points further than `r'` apart by the L1 distance metric are also further
    /// than `r` apart by this distance metric.
    fn as_l1(r: Self::Output) -> A;
}

macro_rules! impl_axis {
    ($t: ty, $tm: ty) => {
        impl Axis for $t {
            const INFINITY: Self = <$t>::INFINITY;
            const NEG_INFINITY: Self = <$t>::NEG_INFINITY;
            fn is_finite(self) -> bool {
                <$t>::is_finite(self)
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
            fn closest_distance_to_aabb(v: &Aabb<$t, K>, x: &[$t; K]) -> Self::Output {
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
/// The distance metric for squared Euclidean distance.
///
/// For any two `k`-dimensional vectors `p` and `q`, the squared Euclidean distance is the sum from
/// `i = 0` to `k - 1` of `(p[i] - k[i]) ** 2`.
///
/// This structure is intended for use as a generic parameter for a distance metric with a [`Capt`].
/// For further detail, refer to the documentation for [`Distance`].
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

#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(clippy::module_name_repetitions)]
/// A collision-affording point tree (CAPT), which allows for efficient collision-checking in a
/// SIMD-parallel manner between spheres and point clouds.
///
/// # Generic parameters
///
/// - `K`: The dimension of the space.
/// - `A`: The value of the axes of each point. This should typically be `f32` or `f64`.
/// - `I`: The index integer. This should generally be an unsigned integer, such as `usize` or
///   `u32`.
/// - `D`: The distance metric. Note that the only distance metric implemented in this library is
///   [`SquaredEuclidean`].
/// - `R`: The output value of the distance metric. This should typically be `f32` or `f64`.
///
/// # Examples
///
/// ```
/// // list of points in cloud
/// let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
///
/// // query radii must be between 0.0 and 0.2
/// let t = captree::Capt::<2>::new(&points, (0.0, 0.04));
///
/// assert!(!t.collides(&[0.0, 0.3], 0.1 * 0.1));
/// assert!(t.collides(&[0.0, 0.2], 0.15 * 0.15));
/// ```
pub struct Capt<const K: usize, A = f32, I = usize, D = SquaredEuclidean, R = f32> {
    /// The test values for determining which part of the tree to enter.
    ///
    /// The first element of `tests` should be the first value to test against.
    /// If we are less than `tests[0]`, we move on to `tests[1]`; if not, we move on to `tests[2]`.
    /// At the `i`-th test performed in sequence of the traversal, if we are less than
    /// `tests[idx]`, we advance to `2 * idx + 1`; otherwise, we go to `2 * idx + 2`.
    ///
    /// The length of `tests` must be `N`, rounded up to the next power of 2, minus one.
    tests: Box<[A]>,
    /// Axis-aligned bounding boxes containing the set of afforded points for each cell.
    aabbs: Box<[Aabb<A, K>]>,
    /// Indexes for the starts of the affordance buffer subsequence of `points` corresponding to
    /// each leaf cell in the tree.
    /// This buffer is padded with one extra `usize` at the end with the maximum length of `points`
    /// for the sake of branchless computation.
    starts: Box<[I]>,
    /// The sets of afforded points for each cell.
    afforded: [Box<[A]>; K],
    _phantom: PhantomData<(D, R)>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// A prismatic bounding volume.
///
/// This structure is mostly used internally, and isn't very useful for library consumers.
/// It's only public so that downstream users can implement [`Distance`] on their own.
pub struct Aabb<A, const K: usize> {
    /// The lower bound on the volume.
    pub lo: [A; K],
    /// The upper bound on the volume.
    pub hi: [A; K],
}

impl<A, I, D, R, const K: usize> Capt<K, A, I, D, R>
where
    A: Axis,
    I: Index,
    D: Distance<A, K, Output = R>,
    R: PartialOrd + Copy,
{
    /// Construct a new CAPT containing all the points in `points`.
    ///
    /// `r_range` is a `(minimum, maximum)` pair containing the lower and upper bound on the
    /// radius of the balls which will be queried against the tree.
    /// `rng` is a random number generator.
    ///
    /// # Panics
    ///
    /// This function will panic if there are too many points in the tree to be addressed by `I`.
    /// This can even be the case if there are fewer points in `points` than can be addressed by `I`
    /// as the CAPT may duplicate points for efficiency.
    ///
    /// # Examples
    ///
    /// ```
    /// let points = [[0.0]];
    ///
    /// let capt = captree::Capt::<1>::new(&points, (0.0, f32::INFINITY));
    ///
    /// assert!(capt.collides(&[1.0], 1.5));
    /// assert!(!capt.collides(&[1.0], 0.5));
    /// ```
    ///
    /// If there are too many points in `points`, this could cause a panic!
    ///
    /// ```rust,should_panic
    /// let points = [[0.0]; 256];
    ///
    /// let capt = captree::Capt::<1, f32, u8, captree::SquaredEuclidean, f32>::new(
    ///     &points,
    ///     (0.0, f32::INFINITY),
    /// );
    /// ```
    pub fn new(points: &[[A; K]], r_range: (R, R)) -> Self {
        Self::try_new(points, r_range)
            .expect("index type I must be able to support all points in CAPT during construction")
    }
    #[must_use]
    /// Construct a new CAPT containing all the points in `points`, checking for index overflow.
    ///
    /// `r_range` is a `(minimum, maximum)` pair containing the lower and upper bound on the
    /// radius of the balls which will be queried against the tree.
    /// `rng` is a random number generator.
    ///
    /// This function will return `None` if there are too many points to be indexed by `I`.
    ///
    /// # Examples
    ///
    /// ```
    /// ```
    pub fn try_new(points: &[[A; K]], r_range: (R, R)) -> Option<Self> {
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
                lo: [A::NEG_INFINITY; K],
                hi: [A::INFINITY; K],
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
    /// Determine whether a point in this tree is within a distance of `radius` to `center`.
    ///
    /// Note that this function will accept query radii outside of the range `r_range` passed to the
    /// construction for this CAPT in [`Capt::new`] or [`Capt::try_new`]. However, if the query
    /// radius is outside this range, the tree may erroneously return `false` (that is, erroneously
    /// report non-collision).
    ///
    /// # Examples
    ///
    /// ```
    /// let points = [[0.0; 3], [1.0; 3], [0.1, 0.5, 1.0]];
    /// let capt = captree::Capt::<3>::new(&points, (0.0, 1.0));
    ///
    /// assert!(capt.collides(&[1.1; 3], 0.1));
    /// assert!(!capt.collides(&[2.0; 3], 1.0));
    ///
    /// // no guarantees about what this is, since the radius is greater than the construction range
    /// println!(
    ///     "collision check result is {:?}",
    ///     capt.collides(&[100.0; 3], 100.0)
    /// );
    /// ```
    pub fn collides(&self, center: &[A; K], radius: R) -> bool {
        // forward pass through the tree
        let mut test_idx = 0;
        let mut k = 0;
        for _ in 0..self.tests.len().trailing_ones() {
            test_idx = 2 * test_idx
                + 1
                + usize::from(unsafe { *self.tests.get_unchecked(test_idx) } <= center[k]);
            k = (k + 1) % K;
        }

        // retrieve affordance buffer location
        let i = test_idx - self.tests.len();
        let aabb = unsafe { self.aabbs.get_unchecked(i) };
        if D::closest_distance_to_aabb(aabb, center) > radius {
            return false;
        }

        let mut range = unsafe {
            // SAFETY: The conversion worked the first way.
            self.starts[i].try_into().unwrap_unchecked()
                ..self.starts[i + 1].try_into().unwrap_unchecked()
        };

        // check affordance buffer
        range.any(|i| {
            let mut aff_pt = [A::INFINITY; K];
            for (ak, sk) in aff_pt.iter_mut().zip(&self.afforded) {
                *ak = sk[i];
            }
            D::distance(&aff_pt, center) <= radius
        })
    }

    #[must_use]
    #[doc(hidden)]
    /// Get the total memory used (stack + heap) by this structure, measured in bytes.
    /// This function should not be considered stable; it is only used internally for benchmarks.
    pub const fn memory_used(&self) -> usize {
        size_of::<Self>()
            + K * self.afforded[0].len() * size_of::<A>()
            + self.starts.len() * size_of::<I>()
            + self.tests.len() * size_of::<I>()
            + self.aabbs.len() * size_of::<Aabb<A, K>>()
    }

    #[must_use]
    #[doc(hidden)]
    #[allow(clippy::cast_precision_loss)]
    /// Get the average number of affordances per point.
    /// This function should not be considered stable; it is only used internally for benchmarks.
    pub fn affordance_size(&self) -> f64 {
        self.afforded.len() as f64 / (self.tests.len() + 1) as f64
    }
}

#[allow(clippy::mismatching_type_param_order)]
#[cfg(feature = "simd")]
impl<A, I, const K: usize> Capt<K, A, I, SquaredEuclidean, A>
where
    I: IndexSimd,
    SquaredEuclidean: Distance<A, K, Output = A>,
    A: Mul<Output = A>,
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
                if self.afforded[0].len() < end + L {
                    // rare end case - we want a sequence of test points beyond the length of the
                    // buffer
                    let mut center = [A::NEG_INFINITY; K];
                    for k in 0..K {
                        center[k] = centers[k][j];
                    }
                    let rsq = radii[j] * radii[j];
                    (start..end - L).step_by(L).any(|i| {
                        let mut dists_sq = Simd::splat(SquaredEuclidean::ZERO);
                        #[allow(clippy::needless_range_loop)]
                        for k in 0..K {
                            let vals = Simd::from_slice(&self.afforded[k][i..]);
                            let diff = vals - centers[k];
                            dists_sq += diff * diff;
                        }
                        A::any(dists_sq.simd_le(rs_sq))
                    }) || (max(start, end - L)..end).any(|i| {
                        let mut pt = [A::NEG_INFINITY; K];
                        #[allow(clippy::needless_range_loop)]
                        for k in 0..K {
                            pt[k] = self.afforded[k][i];
                        }

                        SquaredEuclidean::distance(&center, &pt) <= rsq
                    })
                } else {
                    (start..end).step_by(L).any(|i| {
                        let mut dists_sq = Simd::splat(SquaredEuclidean::ZERO);
                        #[allow(clippy::needless_range_loop)]
                        for k in 0..K {
                            let vals = Simd::from_slice(&self.afforded[k][i..]);
                            let diff = vals - centers[k];
                            dists_sq += diff * diff;
                        }
                        A::any(dists_sq.simd_le(rs_sq))
                    })
                }
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
        D::closest_distance_to_aabb(self, pt) <= rsq_range.1
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
        let t = Capt::<2>::new(&points, (0.0, 0.04));
        println!("{t:?}");
    }

    #[test]
    fn exact_query_single() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = Capt::<2>::new(&points, (0.0, 0.2f32.powi(2)));

        println!("{t:?}");

        let q0 = [0.0, -0.01];
        assert!(t.collides(&q0, (0.12f32).powi(2)));
    }

    #[test]
    fn another_one() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = Capt::<2>::new(&points, (0.0, 0.04));

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

        let t = Capt::<3>::new(&points, (0.0, 0.04));

        println!("{t:?}");
        assert!(t.collides(&[0.0, 0.1, 0.0], 0.011));
        assert!(!t.collides(&[0.0, 0.1, 0.0], 0.05 * 0.05));
    }

    #[test]
    fn fuzz() {
        const R_SQ: f32 = 0.0004;
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let mut rng = thread_rng();
        let t = Capt::<2>::new(&points, (0.0, 0.0008));

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
        let t = Capt::<2>::new(&points, rsq_range);
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
