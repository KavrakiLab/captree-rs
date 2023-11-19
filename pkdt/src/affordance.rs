//! Affordance trees, a novel kind of collision tree with excellent performance, branchless queries,
//! and SIMD batch parallelism.

use std::{
    hint::unreachable_unchecked,
    mem::{size_of, swap, take},
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
struct BuildNode<'a, const D: usize> {
    /// A slice of the set of points belonging to the subtree currently being constructed.
    points: &'a mut [[f32; D]],
    /// The current dimension to split on.
    d: u8,
    /// The current index in the test buffer.
    i: usize,
    /// Indexes into the volume set that this tree might collide with.
    possible_collisions: Vec<usize>,
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
        aff_starts.push(0);

        let mut last_row = Vec::with_capacity(n2);
        last_row.push(BuildNode {
            points: &mut new_points,
            d: 0,
            i: 0,
            possible_collisions: Vec::new(),
            volume: Volume {
                lower: [-f32::INFINITY; D],
                upper: [f32::INFINITY; D],
            },
        });
        let mut next_row = Vec::with_capacity(n2);

        loop {
            if last_row.len() >= n2 {
                // breakout case: populate affordance buffer
                for node in &last_row {
                    debug_assert_eq!(node.points.len(), 1);

                    let cell_center = node.points[0];
                    let center_furthest_distsq = node.volume.furthest_distsq_to(&cell_center);

                    affordances.push(node.points[0]);
                    if center_furthest_distsq > rsq_range.0 {
                        // center not so close to all its edges that there are possibly-free points
                        // in the cell
                        affordances.extend(node.possible_collisions.iter().filter_map(|&idx| {
                            let pt = last_row[idx].points[0];
                            let closest = node.volume.closest_point(&pt);
                            let pt_dist = distsq(closest, pt);
                            (pt_dist < rsq_range.1 && pt_dist < center_furthest_distsq)
                                .then_some(pt)
                        }));
                    }
                    aff_starts.push(affordances.len());
                }
                break;
            }

            // first pass: construct tests and volumes
            for prev in last_row.drain(..) {
                tests[prev.i] = median_partition(prev.points, prev.d as usize, rng);
                let (lhs, rhs) = prev.points.split_at_mut(prev.points.len() / 2);
                let (low_vol, hi_vol) = prev.volume.split(tests[prev.i], prev.d as usize);
                let next_dim = (prev.d + 1) % D as u8;

                // HACK: store the previous possible collisions in the left child.
                // will be fixed up on the second pass
                next_row.push(BuildNode {
                    points: lhs,
                    d: next_dim,
                    i: 2 * prev.i + 1,
                    possible_collisions: prev.possible_collisions,
                    volume: low_vol,
                });

                next_row.push(BuildNode {
                    points: rhs,
                    d: next_dim,
                    i: 2 * prev.i + 2,
                    possible_collisions: Vec::new(),
                    volume: hi_vol,
                });
            }
            // second pass: construct affordance volumes
            for x in (0..next_row.len()).step_by(2) {
                // must use indices to satisfy Mr. Borrow Checker
                // x is the index of a left child always
                let l = x;
                let r = x + 1;

                let parent_collisions = take(&mut next_row[l].possible_collisions);

                for parent_idx in parent_collisions {
                    // indices of nodes that the new nodes might collide with
                    let cl = 2 * parent_idx;
                    let cr = cl + 1;

                    // TODO: also prune when the max distance between two points in a volume is less
                    // than sphere radius

                    // check pairwise collisions across each index
                    if next_row[l].volume.affords(&next_row[cl].volume, rsq_range) {
                        next_row[l].possible_collisions.push(cl);
                    }

                    if next_row[r].volume.affords(&next_row[cl].volume, rsq_range) {
                        next_row[r].possible_collisions.push(cl);
                    }

                    if next_row[l].volume.affords(&next_row[cr].volume, rsq_range) {
                        next_row[l].possible_collisions.push(cr);
                    }

                    if next_row[r].volume.affords(&next_row[cr].volume, rsq_range) {
                        next_row[r].possible_collisions.push(cr);
                    }
                }

                // check if split volumes imply one another
                // we know they are adjacent, so don't check if they're too close
                if next_row[l].volume.max_separation(&next_row[r].volume) > rsq_range.1 {
                    next_row[l].possible_collisions.push(r);
                    next_row[r].possible_collisions.push(l);
                }
            }

            swap(&mut next_row, &mut last_row);
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

    pub fn affords(&self, other: &Volume<D>, rsq_range: (f32, f32)) -> bool {
        let mut min_sep = 0.0;
        let mut max_sep = 0.0;
        for d in 0..D {
            if self.lower[d] > other.upper[d] {
                min_sep += (self.lower[d] - other.upper[d]).powi(2);
            } else if self.upper[d] < other.lower[d] {
                min_sep += (other.lower[d] - self.upper[d]).powi(2);
            }
            // fall-through case: volumes intersect. just let the separation be 0

            let diff1 = other.upper[d] - self.lower[d];
            let diff2 = self.upper[d] - other.lower[d];

            max_sep += if diff1 < diff2 { diff2 } else { diff1 }.powi(2);
        }
        min_sep < rsq_range.1 && max_sep > rsq_range.0
    }

    /// Get the maximum L2 distance squared between any two points in this volume.
    pub fn max_separation(&self, other: &Volume<D>) -> f32 {
        let mut max_sep = 0.0;
        for d in 0..D {
            let diff1 = other.upper[d] - self.lower[d];
            let diff2 = self.upper[d] - other.lower[d];

            max_sep += if diff1 < diff2 { diff2 } else { diff1 }.powi(2);
        }
        max_sep
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
        const R_SQ: f32 = 0.002;
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let mut rng = thread_rng();
        let t = AffordanceTree::new(&points, (0.0, R_SQ * 2.0), &mut rng);

        for _ in 0..10_000 {
            let p = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
            let collides = points.iter().any(|a| distsq(*a, p) < R_SQ);
            println!("{p:?}; {collides}");
            assert_eq!(collides, t.collides(&p, R_SQ));
        }
    }
}
