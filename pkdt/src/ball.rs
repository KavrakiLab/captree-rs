//! Cache-friendly ball-trees.

use rand::Rng;

use crate::distsq;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Ball<const D: usize> {
    center: [f32; D],
    r_squared: f32,
}

#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::module_name_repetitions)]
pub struct BallTree<const D: usize> {
    balls: Box<[Ball<D>]>,
    points: Box<[[f32; D]]>,
}

impl<const D: usize> BallTree<D> {}

impl BallTree<3> {
    /// I was too lazy to try to figure out how to compute the circumsphere of an N-dimensional set of points.
    pub fn new3(points: impl IntoIterator<Item = [f32; 3]>, rng: &mut impl Rng) -> Self {
        fn ball_partition(
            points_to_partition: &mut [[f32; 3]],
            balls: &mut [Ball<3>],
            i: usize,
            rng: &mut impl Rng,
        ) {
            if points_to_partition.len() <= 1 {
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
            ball_partition(&mut points_to_partition[..halflen], balls, 2 * i + 1, rng);
            ball_partition(&mut points_to_partition[halflen..], balls, 2 * i + 2, rng);
        }

        let mut points_to_partition = points.into_iter().collect::<Box<_>>();
        let n2 = points_to_partition.len().next_power_of_two();
        let mut balls = vec![Ball::<3>::bad_ball(); n2];

        ball_partition(&mut points_to_partition, &mut balls, 0, rng);
        println!("{balls:?}");

        let mut points_buf = vec![[f32::NAN; 3]; n2].into_boxed_slice();

        todo!()
    }
}

impl<const D: usize> Ball<D> {
    const fn bad_ball() -> Self {
        Ball {
            center: [f32::NAN; D],
            r_squared: f32::NAN,
        }
    }

    fn contains(&self, point: &[f32; D]) -> bool {
        distsq(self.center, *point) <= self.r_squared
    }
}

impl Ball<3> {
    fn covering3(x: &mut [[f32; 3]], boundary: &mut Vec<[f32; 3]>, rng: &mut impl Rng) -> Self {
        println!("x = {x:?}, boundary = {boundary:?}");
        if boundary.len() == 4 {
            // Use Miroslav Fiedler's method
            // I don't have the patience to compute matrix inverses by hand
            let d12 = distsq(boundary[0], boundary[1]);
            let d13 = distsq(boundary[0], boundary[2]);
            let d14 = distsq(boundary[0], boundary[3]);
            let d23 = distsq(boundary[1], boundary[2]);
            let d24 = distsq(boundary[1], boundary[3]);
            let d34 = distsq(boundary[2], boundary[3]);

            // Cayley-Menger matrix
            let mut c_mat = nalgebra::matrix![
                0.0, 1.0, 1.0, 1.0, 1.0;
                1.0, 0.0, d12, d13, d14;
                1.0, d12, 0.0, d23, d24;
                1.0, d13, d23, 0.0, d34;
                1.0, d14, d24, d34, 0.0;
            ];
            println!("{c_mat:#?}");
            assert!(c_mat.try_inverse_mut());
            let big_m = -2.0 * c_mat;
            let little_m = big_m.row(0);

            let mut center = [0.0; 3];
            for d in 0..3 {
                center[d] += little_m
                    .iter()
                    .skip(1)
                    .zip(boundary.iter())
                    .map(|(weight, bound)| weight * bound[d])
                    .sum::<f32>();
                center[d] /= little_m.sum();
            }

            return Ball {
                center,
                r_squared: little_m.sum() / 4.0,
            };
        }
        if x.is_empty() {
            return match boundary.len() {
                0 => Ball {
                    center: [f32::NAN; 3],
                    r_squared: f32::NAN,
                },
                1 => Ball {
                    center: boundary[0],
                    r_squared: 0.0,
                },
                2 => {
                    let mut center = boundary[0];
                    for (c, b) in center.iter_mut().zip(boundary[1]) {
                        *c += b;
                    }
                    Ball {
                        center: center.map(|c| c / 2.0),
                        r_squared: distsq(boundary[0], boundary[1]),
                    }
                }
                3 => {
                    // https://en.wikipedia.org/wiki/Circumcircle#Higher_dimensions
                    let c = nalgebra::Vector3::from_row_slice(&boundary[0]);
                    let a = nalgebra::Vector3::from_row_slice(&boundary[1]) - c;
                    let b = nalgebra::Vector3::from_row_slice(&boundary[2]) - c;

                    let r_squared = (a.norm_squared() * b.norm_squared() * (a - b).norm_squared())
                        / (4.0 * a.cross(&b).norm_squared());
                    let center = (a.norm_squared() * b - b.norm_squared() * a).cross(&a.cross(&b))
                        / (2.0 * a.cross(&b).norm_squared())
                        + c;

                    Ball {
                        center: [center[0], center[1], center[2]],
                        r_squared,
                    }
                }
                _ => unreachable!("too many points on boundary"),
            };
        }

        x.swap(rng.gen_range(0..x.len()), 0);
        let sub_ball = Ball::covering3(&mut x[1..], boundary, rng);
        println!("x = {x:?}, boundary = {boundary:?}, sub_ball = {sub_ball:?}");
        if sub_ball.contains(&x[0]) {
            sub_ball
        } else {
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

    use super::{principal_component, BallTree};

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
        let points = [[0.0, 1.0, 1.1], [0.1, 2.3, -0.2], [-0.1, 1.1, 3.3]];
        let tree = BallTree::new3(points, &mut thread_rng());
        println!("{tree:?}");
    }
}
