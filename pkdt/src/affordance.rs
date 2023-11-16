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
    /// The size of each individual affordance buffer in `points`.
    affordance_size: usize,
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

impl<const D: usize> AffordanceTree<D> {
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
    pub fn new(points: &[[f32; D]], radius_range: (f32, f32), rng: &mut impl Rng) -> Self {
        #[allow(clippy::float_cmp)]
        #[allow(clippy::too_many_arguments)]
        /// Recursive helper function to sort the points for the KD tree and generate the tests.
        /// Runs in O(n log n)
        fn build_tree<const D: usize>(
            points: &mut [[f32; D]],
            tests: &mut [f32],
            d: u8,
            i: usize,
            mut possible_collisions: Vec<[f32; D]>,
            volume: Volume<D>,
            affordances: &mut Vec<Vec<[f32; D]>>,
            rsq_range: (f32, f32),
            rng: &mut impl Rng,
        ) {
            if points.len() <= 1 {
                let cell_center = points[0];

                let (rsq_min, rsq_max) = rsq_range;

                possible_collisions.retain(|pt| {
                    let closest = volume.closest_point(pt);
                    let center_dist = distsq(cell_center, closest);
                    let closest_dist = distsq(*pt, closest);
                    cell_center != *pt
                        && closest_dist < rsq_max
                        && closest_dist < center_dist
                        && rsq_min < center_dist
                });
                possible_collisions.push(cell_center);
                let l = possible_collisions.len();
                possible_collisions.swap(0, l - 1); // put the center at the front
                affordances.push(possible_collisions);
            } else {
                tests[i] = median_partition(points, d as usize, rng);
                let next_dim = (d + 1) % D as u8;
                let (lhs, rhs) = points.split_at_mut(points.len() / 2);
                let (low_vol, hi_vol) = volume.split(tests[i], d as usize);
                let mut lo_afford = possible_collisions.clone();
                let mut hi_afford = possible_collisions;
                lo_afford.retain(|pt| low_vol.distsq_to(pt) < rsq_range.1);
                hi_afford.retain(|pt| hi_vol.distsq_to(pt) < rsq_range.1);
                build_tree(
                    lhs,
                    tests,
                    next_dim,
                    2 * i + 1,
                    lo_afford,
                    low_vol,
                    affordances,
                    rsq_range,
                    rng,
                );
                build_tree(
                    rhs,
                    tests,
                    next_dim,
                    2 * i + 2,
                    hi_afford,
                    hi_vol,
                    affordances,
                    rsq_range,
                    rng,
                );
            }
        }

        assert!(D < u8::MAX as usize);

        let n2 = points.len().next_power_of_two();

        let mut tests = vec![f32::INFINITY; n2 - 1].into_boxed_slice();

        // hack: just pad with infinity to make it a power of 2
        let mut new_points = vec![[f32::INFINITY; D]; n2].into_boxed_slice();
        new_points[..points.len()].copy_from_slice(points);
        let mut affordance_vec = Vec::with_capacity(n2);
        let possible_collisions = new_points.clone().to_vec();
        build_tree(
            new_points.as_mut(),
            tests.as_mut(),
            0,
            0,
            possible_collisions,
            Volume {
                lower: [-f32::INFINITY; D],
                upper: [f32::INFINITY; D],
            },
            &mut affordance_vec,
            (radius_range.0.powi(2), radius_range.1.powi(2)),
            rng,
        );

        let affordance_size = affordance_vec.iter().map(Vec::len).max().unwrap();
        let mut points = vec![[f32::INFINITY; D]; affordance_size * n2].into_boxed_slice();
        for (i, v) in affordance_vec.into_iter().enumerate() {
            points[i * affordance_size..][..v.len()].copy_from_slice(&v);
        }

        AffordanceTree {
            tests,
            affordance_size,
            points,
        }
    }
}

impl<const D: usize> Volume<D> {
    pub fn distsq_to(&self, point: &[f32; D]) -> f32 {
        let mut p2 = [0.0; D];

        point
            .iter()
            .zip(self.lower)
            .zip(self.upper)
            .map(|((p, l), u)| clamp(*p, l, u))
            .zip(p2.iter_mut())
            .for_each(|(clamped, coord)| *coord = clamped);

        distsq(p2, *point)
    }

    pub fn split(mut self, test: f32, dim: usize) -> (Self, Self) {
        let mut rhs = self;
        self.upper[dim] = test;
        rhs.lower[dim] = test;

        (self, rhs)
    }

    pub fn closest_point(&self, query: &[f32; D]) -> [f32; D] {
        let mut closest = [f32::NAN; D];
        for d in 0..D {
            closest[d] = clamp(query[d], self.lower[d], self.upper[d]);
        }
        closest
    }
}

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
    use rand::thread_rng;

    use crate::AffordanceTree;

    #[test]
    fn build_simple() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = AffordanceTree::new(&points, (0.001, 0.2), &mut thread_rng());
        println!("{t:?}");
    }
}
