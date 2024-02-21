//! Affordance trees, a novel kind of collision tree with excellent performance, branchless queries,
//! and SIMD batch parallelism.

use crate::{forward_pass, median_partition, Axis, Distance, Index, SquaredEuclidean};
use std::{marker::PhantomData, mem::size_of};

#[cfg(feature = "simd")]
use std::{
    ops::{AddAssign, Mul, Sub},
    simd::{
        cmp::{SimdPartialEq, SimdPartialOrd},
        ptr::SimdConstPtr,
        LaneCount, Mask, Simd, SupportedLaneCount,
    },
};

#[cfg(feature = "simd")]
use crate::{forward_pass_simd, AxisSimd, IndexSimd};

#[derive(Clone, Debug, PartialEq)]
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
#[derive(Clone, Copy, Debug, PartialEq)]
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
        let mut afforded = [(); K].map(|()| Vec::with_capacity(n2 * 1000));
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

        Some(AffordanceTree {
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
    pub fn memory_used(&self) -> usize {
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
            .filter_map(|((s, e), i)| i.then_some((s..e).step_by(L)))
            .enumerate()
            .any(|(j, range)| {
                let mut n_center = [Simd::splat(SquaredEuclidean::ZERO); K];
                for k in 0..K {
                    n_center[k] = Simd::splat(centers[k][j]);
                }
                let rs = Simd::splat(radii[j]);
                let rs_sq = rs * rs;
                for i in range {
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
    pub(crate) const ALL: Self = Aabb {
        lo: [A::NEG_INFINITY; K],
        hi: [A::INFINITY; K],
    };

    /// Split this volume by a test plane with value `test` along `dim`.
    fn split(mut self, test: A, dim: usize) -> (Self, Self) {
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

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use crate::{distsq, AffordanceTree};

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
            let collides = points.iter().any(|a| distsq(*a, p) < R_SQ);
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
}
