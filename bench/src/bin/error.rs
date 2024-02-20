#![feature(portable_simd)]

use std::simd::{LaneCount, SupportedLaneCount};

use bench::{dist, fuzz_pointcloud, get_points, make_needles};
use kiddo::SquaredEuclidean;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const N: usize = 1 << 16;
const L: usize = 16;
const D: usize = 3;

fn main() {
    let mut rng = ChaCha20Rng::seed_from_u64(2707);
    let mut starting_points = get_points(N);
    fuzz_pointcloud(&mut starting_points, 0.001, &mut rng);
    measure_error::<D, L>(&starting_points, &mut rng, 1 << 16)
}

pub fn measure_error<const D: usize, const L: usize>(
    points: &[[f32; D]],
    rng: &mut impl Rng,
    n_trials: usize,
) where
    LaneCount<L>: SupportedLaneCount,
{
    let sp_clone = Box::from(points);

    let kdt = captree::PkdTree::new(&sp_clone);
    let mut kiddo_kdt = kiddo::KdTree::new();
    for pt in sp_clone.iter() {
        kiddo_kdt.add(pt, 0);
    }

    let (seq_needles, _) = make_needles(rng, n_trials);

    for seq_needle in seq_needles {
        let exact_kiddo_dist = kiddo_kdt
            .nearest_one::<SquaredEuclidean>(&seq_needle)
            .distance
            .sqrt();
        let approx_dist = dist(seq_needle, kdt.approx_nearest(seq_needle));
        let rel_error = approx_dist / exact_kiddo_dist - 1.0;
        println!("{seq_needle:?}\t{exact_kiddo_dist}\t{approx_dist}\t{rel_error}");
    }
}
