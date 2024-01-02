//! Affordance trees, a novel kind of collision tree with excellent performance, branchless queries,
//! and SIMD batch parallelism.

use std::{
    hint::unreachable_unchecked,
    marker::PhantomData,
    mem::size_of,
    ops::{AddAssign, Mul, Sub},
    simd::{
        cmp::{SimdPartialEq, SimdPartialOrd},
        ptr::SimdConstPtr,
        LaneCount, Mask, Simd, SupportedLaneCount,
    },
};

use rand::Rng;

use crate::{median_partition, Axis, AxisSimd, Distance, Index, IndexSimd, SquaredEuclidean};

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
    /// Indexes for the starts of the affordance buffer subsequence of `points` corresponding to
    /// each leaf cell in the tree.
    /// This buffer is padded with one extra `usize` at the end with the maximum length of `points`
    /// for the sake of branchless computation.
    aff_starts: Box<[I]>,
    /// The relevant points which may collide with the outcome of some test.
    /// The affordance buffer for a point of index `i`
    points: Box<[[A; K]]>,
    _phantom: PhantomData<D>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// A prismatic bounding volume.
pub struct Volume<A, const K: usize> {
    pub lower: [A; K],
    pub upper: [A; K],
}

#[derive(Debug)]
/// Cursed evil structure used for unrolling a recursive function into an iterative one.
/// This is the contents of a stack frame as used during construction of the tree.
///
/// # Generic parameters
///
/// - `K`: The dimension of the space.
struct BuildStackFrame<'a, A, const K: usize> {
    /// A slice of the set of points belonging to the subtree currently being constructed.
    points: &'a mut [[A; K]],
    /// The current dimension to split on.
    d: u8,
    /// The current index in the test buffer.
    i: usize,
    /// The points which might collide with the contents of the current cell.
    possible_collisions: Vec<[A; K]>,
    /// The prism occupied by this subtree's cell.
    volume: Volume<A, K>,
}

impl<A, I, D, R, const K: usize> AffordanceTree<K, A, I, D, R>
where
    A: Axis,
    I: Index,
    D: Distance<A, K, Output = R>,
    R: PartialOrd,
{
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::float_cmp)]
    /// Construct a new affordance tree containing all the points in `points`.
    /// `rsq_range` is a `(minimum, maximum)` pair containing the lower and upper bound on the
    /// radius of the balls which will be queried against the tree.
    /// `rng` is a random number generator.
    /// Although the results of the tree are deterministic after construction, the construction
    /// process for the tree is probabilistic.
    /// The output of construction will be the same independent of the RNG, but the process to
    /// construct the tree may vary with the provided RNG.
    ///
    /// This function will return `None` if there are too many points to be indexed by `I`, or if `K` is greater than `255`
    pub fn new(points: &[[A; K]], rsq_range: (R, R), rng: &mut impl Rng) -> Option<Self> {
        if K >= u8::MAX as usize {
            return None;
        }

        let n2 = points.len().next_power_of_two();

        let mut tests = vec![A::INFINITY; n2 - 1].into_boxed_slice();

        // hack: just pad with infinity to make it a power of 2
        let mut points2 = vec![[A::INFINITY; K]; n2].into_boxed_slice();
        points2[..points.len()].copy_from_slice(points);
        let mut affordances = Vec::with_capacity(n2);
        let mut aff_starts = Vec::with_capacity(n2 + 1);

        let mut stack = Vec::with_capacity(n2.ilog2() as usize);

        // current frame - used as a CPS transformation to prevent excessive push/pop to stack
        let mut frame = BuildStackFrame {
            points: &mut points2,
            d: 0,
            i: 0,
            possible_collisions: points.to_vec(),
            volume: Volume {
                lower: [A::NEG_INFINITY; K],
                upper: [A::INFINITY; K],
            },
        };

        aff_starts.push(0.try_into().ok()?);
        // Iteratively-transformed construction procedure
        loop {
            if frame.points.len() <= 1 {
                let cell_center = frame.points[0];

                if cell_center[0].is_finite() {
                    affordances.push(cell_center);
                    let center_furthest_distsq =
                        D::furthest_distance_to_volume(&frame.volume, &cell_center);
                    if rsq_range.0 < center_furthest_distsq {
                        // check for contacting the volume is already covered
                        affordances.extend(
                            frame
                                .possible_collisions
                                .into_iter()
                                .filter(|pt| cell_center != *pt),
                        );
                    }
                }
                aff_starts.push((affordances.len() * K).try_into().ok()?);

                if let Some(f) = stack.pop() {
                    frame = f;
                } else {
                    break;
                }
            } else {
                // split the volume in half
                let test = median_partition(frame.points, frame.d as usize, rng);
                tests[frame.i] = test;
                let next_dim = (frame.d + 1) % K as u8;
                let (lhs, rhs) = frame.points.split_at_mut(frame.points.len() / 2);
                let (low_vol, hi_vol) = frame.volume.split(test, frame.d as usize);
                let mut lo_afford = frame.possible_collisions.clone();
                let mut hi_afford = Vec::with_capacity(lo_afford.len());

                // retain only points which might be in the affordance buffer for the split-out
                // cells
                lo_afford.retain(|pt| {
                    if D::closest_distance_to_volume(&hi_vol, pt) < rsq_range.1
                        && rsq_range.0 < D::furthest_distance_to_volume(&hi_vol, pt)
                    {
                        hi_afford.push(*pt);
                    }
                    D::closest_distance_to_volume(&low_vol, pt) < rsq_range.1
                        && rsq_range.0 < D::furthest_distance_to_volume(&low_vol, pt)
                });

                // because the stack is FIFO, we must put the left recursion last
                stack.push(BuildStackFrame {
                    points: rhs,
                    d: next_dim,
                    i: 2 * frame.i + 2,
                    possible_collisions: hi_afford,
                    volume: hi_vol,
                });

                // Save a push/pop operation by directly updating the current frame
                frame = BuildStackFrame {
                    points: lhs,
                    d: next_dim,
                    i: 2 * frame.i + 1,
                    possible_collisions: lo_afford,
                    volume: low_vol,
                };
            }
        }

        Some(AffordanceTree {
            tests,
            rsq_range,
            aff_starts: aff_starts.into_boxed_slice(),
            points: affordances.into_boxed_slice(),
            _phantom: PhantomData,
        })
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

        let n2 = self.tests.len() + 1;
        debug_assert!(n2.is_power_of_two());

        // forward pass through the tree
        let mut test_idx = 0;
        let mut k = 0;
        for _ in 0..n2.trailing_zeros() as usize {
            test_idx = 2 * test_idx
                + 1
                + usize::from(center[k] >= unsafe { *self.tests.get_unchecked(test_idx) });
            k = (k + 1) % K;
        }

        // retrieve affordance buffer location
        let i = test_idx - self.tests.len();
        let range = unsafe {
            // SAFETY: The conversion worked the first way.
            self.aff_starts[i].try_into().unwrap_unchecked() / K
                ..self.aff_starts[i + 1].try_into().unwrap_unchecked() / K
        };

        // check affordance buffer
        self.points[range]
            .iter()
            .any(|pt| D::distance(pt, center) <= radius)
    }

    #[must_use]
    /// Get the total memory used (stack + heap) by this structure, measured in bytes.
    pub fn memory_used(&self) -> usize {
        size_of::<Self>()
            + (self.points.len() * K + self.tests.len()) * size_of::<A>()
            + self.aff_starts.len() * size_of::<I>()
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    /// Get the average number of affordances per point.
    pub fn affordance_size(&self) -> f64 {
        self.points.len() as f64 / (self.tests.len() + 1) as f64
    }
}

#[allow(clippy::mismatching_type_param_order)]
impl<A, I, const K: usize> AffordanceTree<K, A, I, SquaredEuclidean, A>
where
    I: IndexSimd,
    SquaredEuclidean: Distance<A, K, Output = A>,
{
    #[must_use]
    /// Determine whether any sphere in the list of provided spheres intersects a point in this
    /// tree.
    pub fn collides_simd<const L: usize>(&self, centers: &[Simd<A, L>], radii: Simd<A, L>) -> bool
    where
        LaneCount<L>: SupportedLaneCount,
        Simd<A, L>:
            SimdPartialOrd + Sub<Output = Simd<A, L>> + Mul<Output = Simd<A, L>> + AddAssign,
        Mask<isize, L>: From<<Simd<A, L> as SimdPartialEq>::Mask>,
        A: Axis + AxisSimd<<Simd<A, L> as SimdPartialEq>::Mask>,
    {
        let mut test_idxs: Simd<isize, L> = Simd::splat(0);
        let n2 = self.tests.len() + 1;
        debug_assert!(n2.is_power_of_two());

        // in release mode, tell the compiler about this invariant
        if !n2.is_power_of_two() {
            unsafe { unreachable_unchecked() };
        }

        let mut k = 0;

        // Advance the tests forward
        for _ in 0..n2.trailing_zeros() as usize {
            let test_ptrs = Simd::splat((self.tests.as_ref() as *const [A]).cast::<A>())
                .wrapping_offset(test_idxs);
            let relevant_tests: Simd<A, L> = unsafe { Simd::gather_ptr(test_ptrs) };
            let cmp_results: Mask<isize, L> = centers[k % K].simd_ge(relevant_tests).into();

            // TODO is there a faster way than using a conditional select?
            test_idxs <<= Simd::splat(1);
            test_idxs += Simd::splat(1);
            test_idxs += cmp_results.to_int() & Simd::splat(1);

            k = (k + 1) % K;
        }

        // retrieve start/end pointers for the affordance buffer
        let start_ptrs = Simd::splat((self.aff_starts.as_ref() as *const [I]).cast::<I>())
            .wrapping_offset(test_idxs)
            .wrapping_sub(Simd::splat(self.tests.len()));
        let starts = unsafe { I::to_simd_usize_unchecked(Simd::gather_ptr(start_ptrs)) };
        let ends = unsafe {
            I::to_simd_usize_unchecked(Simd::gather_ptr(start_ptrs.wrapping_add(Simd::splat(1))))
        };

        let points_base = Simd::splat((self.points.as_ref() as *const [[A; K]]).cast::<A>());
        let mut aff_ptrs = points_base.wrapping_add(starts);
        let end_ptrs = points_base.wrapping_add(ends);

        // scan through affordance buffer, searching for a collision
        let mut inbounds = Mask::splat(true); // whether each of `aff_ptrs` is in a valid affordance buffer
        while inbounds.any() {
            let mut dists_sq = Simd::splat(SquaredEuclidean::ZERO);
            for center_set in centers {
                let vals = unsafe {
                    Simd::gather_select_ptr(aff_ptrs, inbounds, Simd::splat(A::INFINITY))
                };
                let diffs = *center_set - vals;
                dists_sq += diffs * diffs;
                aff_ptrs = aff_ptrs.wrapping_add(Simd::splat(1));
            }

            // is one ball in collision with a point?
            if A::any(dists_sq.simd_lt(radii)) {
                return true;
            }

            inbounds &= aff_ptrs.simd_lt(end_ptrs);
        }

        false
    }
}

impl<A, const K: usize> Volume<A, K>
where
    A: Axis,
{
    /// Split this volume by a test plane with value `test` along `dim`.
    fn split(mut self, test: A, dim: usize) -> (Self, Self) {
        let mut rhs = self;
        self.upper[dim] = test;
        rhs.lower[dim] = test;

        (self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use crate::{distsq, AffordanceTree};

    #[test]
    fn build_simple() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = AffordanceTree::<2>::new(&points, (0.0, 0.04), &mut thread_rng());
        println!("{t:?}");
    }

    #[test]
    fn exact_query_single() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t =
            AffordanceTree::<2>::new(&points, (0.0, 0.2f32.powi(2)), &mut thread_rng()).unwrap();

        println!("{t:?}");

        let q0 = [0.0, -0.01];
        assert!(t.collides(&q0, (0.12f32).powi(2)));
    }

    #[test]
    fn another_one() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = AffordanceTree::<2>::new(&points, (0.0, 0.04), &mut thread_rng()).unwrap();

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

        let t = AffordanceTree::<3>::new(&points, (0.0, 0.04), &mut thread_rng()).unwrap();

        println!("{t:?}");
        assert!(t.collides(&[0.0, 0.1, 0.0], 0.011));
        assert!(!t.collides(&[0.0, 0.1, 0.0], 0.05 * 0.05));
    }

    #[test]
    fn fuzz() {
        const R_SQ: f32 = 0.0004;
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let mut rng = thread_rng();
        let t = AffordanceTree::<2>::new(&points, (0.0, 0.0008), &mut rng).unwrap();

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
        let t = AffordanceTree::<2>::new(&points, rsq_range, &mut thread_rng()).unwrap();
        println!("{t:?}");

        assert!(t.collides(&[-0.001, -0.2], 1.0));
    }
}
