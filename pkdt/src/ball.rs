//! Cache-friendly ball-trees.

use rand::Rng;

use crate::distsq;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Ball<const D: usize> {
    center: [f32; D],
    radius: f32,
}

#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::module_name_repetitions)]
pub struct BallTree<const D: usize, const LW: usize> {
    balls: Box<[Ball<D>]>,
    leaf_start: usize,
    points: Box<[[f32; D]]>,
}

impl<const LW: usize> BallTree<3, LW> {
    /// I was too lazy to try to figure out how to compute the circumsphere of an N-dimensional set
    /// of points.
    ///
    /// # Panics
    ///
    /// This function will panic if `LW` is 0.
    pub fn new3(points: &[[f32; 3]], rng: &mut impl Rng) -> Self {
        fn ball_partition<const LW: usize>(
            points_to_partition: &mut [[f32; 3]],
            balls: &mut [Ball<3>],
            points_buf: &mut [[f32; 3]],
            i: usize,
            leaf_start: &mut Option<usize>,
            rng: &mut impl Rng,
        ) {
            if points_to_partition.len() <= LW {
                let ls = *leaf_start.get_or_insert(i);
                points_buf[(i - ls) * LW..][..points_to_partition.len()]
                    .copy_from_slice(points_to_partition);
                return;
            }
            // construct the enveloping ball of all the points
            let envelope = Ball::covering3(points_to_partition, &mut Vec::with_capacity(4), rng);
            balls[i] = envelope;

            // first, find the axis of greatest variance
            let pc = principal_component(points_to_partition, rng);

            // partition the points in half by their principal component
            points_to_partition
                .sort_unstable_by(|a, b| dot(a, &pc).partial_cmp(&dot(b, &pc)).unwrap());

            // anything below the midpoint goes off to one side
            let halflen = points_to_partition.len() / 2;
            ball_partition::<LW>(
                &mut points_to_partition[..halflen],
                balls,
                points_buf,
                2 * i + 1,
                leaf_start,
                rng,
            );
            ball_partition::<LW>(
                &mut points_to_partition[halflen..],
                balls,
                points_buf,
                2 * i + 2,
                leaf_start,
                rng,
            );
        }

        assert!(LW > 0);

        let mut points_to_partition = points.iter().copied().collect::<Box<_>>();
        let n2 = points_to_partition.len().next_power_of_two();
        let n_leaves = n2 / LW; // TODO: this could be wrong if `LW` is not a power of 2
        let mut balls = vec![Ball::<3>::bad_ball(); 2 * n_leaves - 1].into_boxed_slice();
        let mut points_buf = vec![[f32::INFINITY; 3]; n_leaves * LW * 2].into_boxed_slice();
        let mut leaf_start = None;

        ball_partition::<LW>(
            &mut points_to_partition,
            &mut balls,
            &mut points_buf,
            0,
            &mut leaf_start,
            rng,
        );

        BallTree {
            balls,
            leaf_start: leaf_start.unwrap(),
            points: points_buf,
        }
    }
}

impl<const D: usize, const LW: usize> BallTree<D, LW> {
    #[must_use]
    pub fn collides(&self, needle: [f32; D], radius: f32) -> bool {
        self.collides_help(0, needle, radius)
    }

    /// Assumes that we already know that the ball at `test_idx` is in collision.
    fn collides_help(&self, test_idx: usize, needle: [f32; D], radius: f32) -> bool {
        if test_idx >= self.balls.len() {
            // println!("{test_idx} is terminal");
            return self.points[((test_idx - 1) / 2 - self.leaf_start) * LW..][..LW]
                .iter()
                .any(|&p| distsq(p, needle) < radius.powi(2));
        }
        let ball = &self.balls[test_idx];
        if (radius + ball.radius).powi(2) < distsq(ball.center, needle) {
            // println!("{test_idx} bailed!");
            return false;
        }

        // println!("{test_idx} carries on");
        self.collides_help(2 * test_idx + 1, needle, radius)
            || self.collides_help(2 * test_idx + 2, needle, radius)
    }
}

impl<const D: usize> Ball<D> {
    const fn bad_ball() -> Self {
        Ball {
            center: [f32::NAN; D],
            radius: f32::NAN,
        }
    }

    fn contains(&self, point: &[f32; D]) -> bool {
        distsq(self.center, *point) <= self.radius.powi(2)
    }
}

impl Ball<3> {
    fn covering3(x: &mut [[f32; 3]], boundary: &mut Vec<[f32; 3]>, rng: &mut impl Rng) -> Self {
        // Fudge factor for oversizing ball radii
        const FUDGE_FACTOR: f32 = 1e-3;

        debug_assert!(boundary.len() <= 4);

        if boundary.len() == 4 {
            // Use Miroslav Fiedler's method
            // https://math.stackexchange.com/questions/2414640/circumsphere-of-a-tetrahedron
            // I don't have the patience to compute matrix inverses by hand
            let d12 = distsq(boundary[0], boundary[1]);
            let d13 = distsq(boundary[0], boundary[2]);
            let d14 = distsq(boundary[0], boundary[3]);
            let d23 = distsq(boundary[1], boundary[2]);
            let d24 = distsq(boundary[1], boundary[3]);
            let d34 = distsq(boundary[2], boundary[3]);

            // Cayley-Menger matrix
            let c_mat = nalgebra::matrix![
                0.0, 1.0, 1.0, 1.0, 1.0;
                1.0, 0.0, d12, d13, d14;
                1.0, d12, 0.0, d23, d24;
                1.0, d13, d23, 0.0, d34;
                1.0, d14, d24, d34, 0.0;
            ];
            let c_inv = c_mat.try_inverse().unwrap();
            let big_m = -2.0 * c_inv;
            let little_m = big_m.row(0);

            let mut center = [0.0; 3];
            let div = little_m.iter().skip(1).sum::<f32>();
            for d in 0..3 {
                center[d] += little_m
                    .iter()
                    .skip(1)
                    .zip(boundary.iter())
                    .map(|(weight, bound)| weight * bound[d])
                    .sum::<f32>();
                center[d] /= div;
            }

            let ball = Ball {
                center,
                radius: little_m[0].sqrt() / 2.0 + FUDGE_FACTOR,
            };

            for point in boundary {
                let dist = distsq(*point, ball.center);
                debug_assert!(dist <= ball.radius.powi(2));
                debug_assert!(ball.radius.powi(2) * 0.999 < dist);
            }
            return ball;
        }
        if x.is_empty() {
            let ball = match boundary.len() {
                0 => Ball {
                    center: [f32::NAN; 3],
                    radius: f32::NAN,
                },
                1 => Ball {
                    center: boundary[0],
                    radius: 0.0,
                },
                2 => {
                    let mut center = boundary[0];
                    for (c, b) in center.iter_mut().zip(boundary[1]) {
                        *c += b;
                    }
                    Ball {
                        center: center.map(|c| c / 2.0),
                        radius: distsq(boundary[0], boundary[1]).sqrt() / 2.0 + FUDGE_FACTOR,
                    }
                }
                3 => {
                    // https://en.wikipedia.org/wiki/Circumcircle#Higher_dimensions
                    let c = nalgebra::Vector3::from_row_slice(&boundary[0]);
                    let a = nalgebra::Vector3::from_row_slice(&boundary[1]) - c;
                    let b = nalgebra::Vector3::from_row_slice(&boundary[2]) - c;

                    let r_squared = (a.norm_squared() * b.norm_squared() * (a - b).norm_squared())
                        / (4.0 * a.cross(&b).norm_squared())
                        + FUDGE_FACTOR;
                    let center = (a.norm_squared() * b - b.norm_squared() * a).cross(&a.cross(&b))
                        / (2.0 * a.cross(&b).norm_squared())
                        + c;

                    Ball {
                        center: [center[0], center[1], center[2]],
                        radius: r_squared.sqrt(),
                    }
                }
                _ => unreachable!("too many points on boundary"),
            };
            for point in boundary {
                let dist = distsq(*point, ball.center);
                debug_assert!(dist <= ball.radius.powi(2));
                debug_assert!(
                    ball.radius.powi(2) * 0.99 <= dist
                        || ball.radius.powi(2) - 2.0 * FUDGE_FACTOR <= dist
                );
            }
            return ball;
        }

        x.swap(rng.gen_range(0..x.len()), 0);
        let old_len = boundary.len();
        let sub_ball = Ball::covering3(&mut x[1..], boundary, rng);
        if sub_ball.contains(&x[0]) {
            sub_ball
        } else {
            boundary.truncate(old_len);
            boundary.push(x[0]);
            Ball::covering3(&mut x[1..], boundary, rng)
        }
    }
}

#[allow(clippy::cast_precision_loss)]
/// Efficiently approximate the principal component of `x`.
fn principal_component<const D: usize>(x: &[[f32; D]], rng: &mut impl Rng) -> [f32; D] {
    const ITERS: usize = 5;
    let mut rand_vector = [0.0; D].map(|_| rng.gen_range(-1.0..1.0));
    let magnitude = rand_vector
        .into_iter()
        .map(|r: f32| r.powi(2))
        .sum::<f32>()
        .sqrt();
    rand_vector = rand_vector.map(|x| x / magnitude);

    for _ in 0..ITERS {
        let mut s = [0.0; D];
        for pt in x {
            let dot_product = dot(pt, &rand_vector);
            for d in 0..D {
                s[d] += pt[d] * dot_product;
            }
        }
        let s_norm = s.into_iter().map(|a| a.powi(2)).sum::<f32>().sqrt();
        rand_vector = s.map(|si| si / s_norm);
    }

    rand_vector
}

fn dot<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    a.iter().zip(b).map(|(aa, bb)| aa * bb).sum()
}

#[cfg(test)]
mod tests {

    use rand::thread_rng;

    use super::{principal_component, Ball, BallTree};

    #[test]
    fn make_pca() {
        let x = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]];

        let approx_pc = principal_component(&x, &mut thread_rng());
        println!("{approx_pc:?}");
        assert!(
            approx_pc
                .iter()
                .map(|a| (a - 1.0 / 3.0f32.sqrt()).abs())
                .sum::<f32>()
                < 1e-4
                || approx_pc
                    .iter()
                    .map(|a| (a + 1.0 / 3.0f32.sqrt()).abs())
                    .sum::<f32>()
                    < 1e-4
        );
    }

    #[test]
    fn build_tree() {
        let points = [
            [0.0, 1.0, 1.1],
            [0.1, 2.3, -0.2],
            [-0.1, 1.1, 3.3],
            [0.0, 0.0, 0.0],
            [3.3, 3.3, 3.3],
            [0.1, -0.1, -0.1],
            [-2.1, -3.4, 0.0],
        ];
        let tree = BallTree::<3, 2>::new3(&points, &mut thread_rng());
        println!("{tree:?}");
    }

    #[test]
    fn no_collision() {
        let points = [
            [0.0, 1.0, 1.1],
            [0.1, 2.3, -0.2],
            [-0.1, 1.1, 3.3],
            [0.0, 0.0, 0.0],
            [3.3, 3.3, 3.3],
            [0.1, -0.1, -0.1],
            [-2.1, -3.4, 0.0],
        ];
        let tree = BallTree::<3, 2>::new3(&points, &mut thread_rng());

        assert!(!tree.collides([0.0, 0.1, 0.0], 0.0));
    }

    #[test]
    fn in_collision() {
        let points = [
            [0.0, 1.0, 1.1],
            [0.1, 2.3, -0.2],
            [-0.1, 1.1, 3.3],
            [0.0, 0.0, 0.0],
            [3.3, 3.3, 3.3],
            [0.1, -0.1, -0.1],
            [-2.1, -3.4, 0.0],
        ];
        let tree = BallTree::<3, 2>::new3(&points, &mut thread_rng());

        println!("{tree:?}");
        assert!(tree.collides([0.0, 0.1, 0.0], 0.12));
    }

    #[test]
    fn quad_boundary() {
        let mut points = vec![
            [0.0, 0.0, 0.0],
            [-2.1, -3.4, 0.0],
            [-0.1, 1.1, 3.3],
            [0.0, 1.0, 1.1],
        ];
        let ball = Ball::covering3(&mut [], &mut points, &mut thread_rng());
        for point in points {
            assert!(ball.contains(&point));
        }
    }

    #[test]
    fn dont_go_over() {
        let mut points = [
            [0.0, 1.0, 1.1],
            [0.1, 2.3, -0.2],
            [-0.1, 1.1, 3.3],
            [0.0, 0.0, 0.0],
            [3.3, 3.3, 3.3],
            [0.1, -0.1, -0.1],
            [-2.1, -3.4, 0.0],
        ];
        let ball = Ball::covering3(&mut points, &mut Vec::new(), &mut thread_rng());
        for point in points {
            assert!(ball.contains(&point));
        }
    }
}
