//! Affordance trees, a novel kind of collision tree with excellent performance, branchless queries,
//! and SIMD batch parallelism.

use std::{
    hint::unreachable_unchecked,
    mem::size_of,
    simd::{LaneCount, Mask, Simd, SimdConstPtr, SimdPartialOrd, SupportedLaneCount},
};

use rand::Rng;

use crate::{distsq, median_partition};

#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::module_name_repetitions)]
/// An affordance tree, which allows for efficient nearest-neighbor-within-a-radius queries.
///
/// # Generic parameters
///
/// - `D`: The dimension of the space.
pub struct AffordanceTree<const D: usize> {
    /// The test values for determining which part of the tree to enter.
    ///
    /// The first element of `tests` should be the first value to test against.
    /// If we are less than `tests[0]`, we move on to `tests[1]`; if not, we move on to `tests[2]`.
    /// At the `i`-th test performed in sequence of the traversal, if we are less than `tests[idx]`,
    /// we advance to `2 * idx + 1`; otherwise, we go to `2 * idx + 2`.
    ///
    /// The length of `tests` must be `N`, rounded up to the next power of 2, minus one.
    tests: Box<[f32]>,
    /// The range of radii which are legal for queries on this tree.
    /// The first element is the minimum and the second element is the maximum.
    rsq_range: (f32, f32),
    /// Indexes for the starts of the affordance buffer subsequence of `points` corresponding to
    /// each leaf cell in the tree.
    /// This buffer is padded with one extra `usize` at the end with the maximum length of `points`
    /// for the sake of branchless computation.
    aff_starts: Box<[usize]>,
    /// The relevant points which may collide with the outcome of some test.
    /// The affordance buffer for a point of index `i`
    points: Box<[[f32; D]]>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// A prismatic bounding volume.
struct Volume<const D: usize> {
    lower: [f32; D],
    upper: [f32; D],
}

#[derive(Debug)]
/// Cursed evil structure used for unrolling a recursive function into an iterative one.
/// This is the contents of a stack frame as used during construction of the tree.
///
/// # Generic parameters
///
/// - `D`: The dimension of the space.
struct BuildStackFrame<'a, const D: usize> {
    /// A slice of the set of points belonging to the subtree currently being constructed.
    points: &'a mut [[f32; D]],
    /// The current dimension to split on.
    d: u8,
    /// The current index in the test buffer.
    i: usize,
    /// The points which might collide with the contents of the current cell.
    possible_collisions: Vec<[f32; D]>,
    /// The prism occupied by this subtree's cell.
    volume: Volume<D>,
}

impl<const D: usize> AffordanceTree<D> {
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
    /// # Panics
    ///
    /// This function will panic if `D` is greater than or equal to 255.
    pub fn new(points: &[[f32; D]], rsq_range: (f32, f32), rng: &mut impl Rng) -> Self {
        assert!(D < u8::MAX as usize);

        let n2 = points.len().next_power_of_two();

        let mut tests = vec![f32::INFINITY; n2 - 1].into_boxed_slice();

        // hack: just pad with infinity to make it a power of 2
        let mut new_points = vec![[f32::INFINITY; D]; n2].into_boxed_slice();
        new_points[..points.len()].copy_from_slice(points);
        let mut affordances = Vec::with_capacity(n2);
        let mut aff_starts = Vec::with_capacity(n2 + 1);

        let mut stack = Vec::with_capacity(n2.ilog2() as usize);
        let points_clone = new_points.clone().to_vec();
        let mut frame = BuildStackFrame {
            points: &mut new_points,
            d: 0,
            i: 0,
            possible_collisions: points_clone,
            volume: Volume {
                lower: [-f32::INFINITY; D],
                upper: [f32::INFINITY; D],
            },
        };

        aff_starts.push(0);
        // Iteratively-transformed construction procedure
        loop {
            if frame.points.len() <= 1 {
                let cell_center = frame.points[0];

                if cell_center[0].is_finite() {
                    affordances.push(cell_center);
                    let center_furthest_distsq = frame.volume.furthest_distsq_to(&cell_center);
                    if rsq_range.0 < center_furthest_distsq {
                        // check for contacting the volume is already covered
                        affordances.extend(
                            frame
                                .possible_collisions
                                .into_iter()
                                .filter(|pt| cell_center != *pt),
                        );
                    }
                    aff_starts.push(affordances.len());
                }

                if let Some(f) = stack.pop() {
                    frame = f;
                } else {
                    break;
                }
            } else {
                // split the volume in half
                tests[frame.i] = median_partition(frame.points, frame.d as usize, rng);
                let next_dim = (frame.d + 1) % D as u8;
                let (lhs, rhs) = frame.points.split_at_mut(frame.points.len() / 2);
                let (low_vol, hi_vol) = frame.volume.split(tests[frame.i], frame.d as usize);
                let mut lo_afford = frame.possible_collisions.clone();
                let mut hi_afford = Vec::with_capacity(lo_afford.len());

                // retain only points which might be in the affordance buffer for the split-out
                // cells
                lo_afford.retain(|pt| {
                    if hi_vol.distsq_to(pt) < rsq_range.1
                        && rsq_range.0 < hi_vol.furthest_distsq_to(pt)
                    {
                        hi_afford.push(*pt);
                    }
                    low_vol.distsq_to(pt) < rsq_range.1
                        && rsq_range.0 < low_vol.furthest_distsq_to(pt)
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

        AffordanceTree {
            tests,
            rsq_range,
            aff_starts: aff_starts.into_boxed_slice(),
            points: affordances.into_boxed_slice(),
        }
    }

    #[must_use]
    /// Determine whether a point in this tree collides with a ball with radius squared `r_squared`.
    ///
    /// # Panics
    ///
    /// This function will panic if `r_squared` is outside the range of squared radii passed to the
    /// construction of the tree.
    /// TODO: implement real error handling.
    pub fn collides(&self, center: &[f32; D], r_squared: f32) -> bool {
        // ball mus be in the rsq range
        assert!(self.rsq_range.0 <= r_squared);
        assert!(r_squared <= self.rsq_range.1);

        let n2 = self.tests.len() + 1;
        assert!(n2.is_power_of_two());

        // forward pass through the tree
        let mut test_idx = 0;
        for i in 0..n2.trailing_zeros() as usize {
            // println!("current idx: {test_idx}");
            let add = if center[i % D] < (self.tests[test_idx]) {
                1
            } else {
                2
            };
            test_idx <<= 1;
            test_idx += add;
        }

        // retrieve affordance buffer location
        let i = test_idx - self.tests.len();
        let range = self.aff_starts[i]..self.aff_starts[i + 1];

        // check affordance buffer
        self.points[range]
            .iter()
            .any(|pt| distsq(*pt, *center) <= r_squared)
    }

    #[must_use]
    /// Determine whether any sphere in the list of provided spheres intersects a point in this
    /// tree.
    pub fn collides_simd<const L: usize>(
        &self,
        centers: &[Simd<f32, L>],
        radii_squared: Simd<f32, L>,
    ) -> bool
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
            let cmp_results: Mask<isize, L> = centers[i % D].simd_lt(relevant_tests).into();

            // TODO is there a faster way than using a conditional select?
            test_idxs <<= Simd::splat(1);
            test_idxs += cmp_results.select(Simd::splat(1), Simd::splat(2));
        }

        // retrieve start/end pointers for the affordance buffer
        let start_ptrs = Simd::splat((self.aff_starts.as_ref() as *const [usize]).cast::<usize>())
            .wrapping_add(test_idxs)
            .wrapping_sub(Simd::splat(self.tests.len()));
        let starts = unsafe { Simd::gather_ptr(start_ptrs) } * Simd::splat(D);
        let ends =
            unsafe { Simd::gather_ptr(start_ptrs.wrapping_add(Simd::splat(1))) } * Simd::splat(D);

        let points_base = Simd::splat((self.points.as_ref() as *const [[f32; D]]).cast::<f32>());
        let mut aff_ptrs = points_base.wrapping_add(starts);
        let end_ptrs = points_base.wrapping_add(ends);

        // scan through affordance buffer, searching for a collision
        let mut inbounds = Mask::from_int(Simd::splat(-1)); // whether each of `aff_ptrs` is in a valid affordance buffer
        while inbounds.any() {
            let mut dists_sq = Simd::splat(0.0);
            for center_set in centers {
                let vals = unsafe { Simd::gather_select_ptr(aff_ptrs, inbounds, *center_set) };
                let diffs = center_set - vals;
                dists_sq += diffs * diffs;
                aff_ptrs = aff_ptrs.wrapping_add(Simd::splat(1));
            }

            // is one ball in collision with a point?
            if dists_sq.simd_lt(radii_squared).any() {
                return true;
            }

            inbounds &= aff_ptrs.simd_lt(end_ptrs);
        }

        false
    }

    #[must_use]
    /// Get the total memory used (stack + heap) by this structure, measured in bytes.
    pub fn memory_used(&self) -> usize {
        size_of::<AffordanceTree<D>>()
            + (self.points.len() * D + self.tests.len()) * size_of::<f32>()
            + self.aff_starts.len() * size_of::<usize>()
    }

    #[must_use]
    /// Get the average number of affordances per point.
    pub fn affordance_size(&self) -> usize {
        self.points.len() / (self.tests.len() + 1)
    }
}

impl<const D: usize> Volume<D> {
    #[allow(clippy::needless_range_loop)]
    /// Get the minimum distance squared from all points in this volume to a test point.
    pub fn distsq_to(&self, point: &[f32; D]) -> f32 {
        let mut dist = 0.0;

        for d in 0..D {
            let clamped = clamp(point[d], self.lower[d], self.upper[d]);
            dist += (point[d] - clamped).powi(2);
        }

        dist
    }

    #[allow(clippy::needless_range_loop)]
    /// Get the furthest distance squared from all points in this volume to a test point.
    pub fn furthest_distsq_to(&self, point: &[f32; D]) -> f32 {
        let mut dist = 0.0;

        for d in 0..D {
            let lo_diff = (self.lower[d] - point[d]).abs();
            let hi_diff = (self.upper[d] - point[d]).abs();

            dist += if lo_diff < hi_diff { hi_diff } else { lo_diff }.powi(2);
        }

        dist
    }

    /// Split this volume by a test plane with value `test` along `dim`.
    pub fn split(mut self, test: f32, dim: usize) -> (Self, Self) {
        let mut rhs = self;
        self.upper[dim] = test;
        rhs.lower[dim] = test;

        (self, rhs)
    }

    /// Find the closest point in this volume to `query`.
    pub fn closest_point(&self, query: &[f32; D]) -> [f32; D] {
        let mut closest = [f32::NAN; D];
        for d in 0..D {
            closest[d] = clamp(query[d], self.lower[d], self.upper[d]);
        }
        closest
    }
}

/// Clamp a floating-point number.
fn clamp(x: f32, min: f32, max: f32) -> f32 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use crate::{distsq, AffordanceTree};

    #[test]
    fn build_simple() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = AffordanceTree::new(&points, (0.0, 0.04), &mut thread_rng());
        println!("{t:?}");
    }

    #[test]
    fn exact_query_single() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = AffordanceTree::new(&points, (0.0, 0.2f32.powi(2)), &mut thread_rng());

        println!("{t:?}");

        let q0 = [0.0, -0.01];
        assert!(t.collides(&q0, (0.12f32).powi(2)));
    }

    #[test]
    fn another_one() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = AffordanceTree::new(&points, (0.0, 0.04), &mut thread_rng());

        println!("{t:?}");

        let q0 = [0.003_265_380_9, 0.106_527_805];
        assert!(t.collides(&q0, 0.0004));
    }

    #[test]
    fn fuzz() {
        const R_SQ: f32 = 0.0004;
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let mut rng = thread_rng();
        let t = AffordanceTree::new(&points, (0.0, 0.0008), &mut rng);

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
        let t = AffordanceTree::new(&points, rsq_range, &mut thread_rng());
        println!("{t:?}");

        assert!(t.collides(&[-0.001, -0.2], 1.0));
    }
}
