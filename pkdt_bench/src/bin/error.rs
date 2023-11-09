#![feature(portable_simd)]

use std::{
    env,
    path::Path,
    simd::{LaneCount, SupportedLaneCount},
};

use pkdt_bench::{dist, load_pointcloud, make_needles};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const N: usize = 1 << 16;
const L: usize = 16;
const D: usize = 3;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut rng = ChaCha20Rng::seed_from_u64(2707);
    let starting_points: Vec<[f32; D]> = if args.len() > 1 {
        eprintln!("Loading pointcloud from {}", &args[1]);
        load_pointcloud(Path::new(&args[1])).unwrap()
    } else {
        eprintln!("No pointcloud file! Using N={N}, L={L}, D={D}");
        eprintln!("generating random points...");
        (0..N)
            .map(|_| {
                [
                    rng.gen_range::<f32, _>(0.0..1.0),
                    rng.gen_range::<f32, _>(0.0..1.0),
                    rng.gen_range::<f32, _>(0.0..1.0),
                ]
            })
            .collect::<Vec<[f32; D]>>()
    };

    measure_error::<D, L>(&starting_points, &mut rng, 1 << 20);
}

pub fn measure_error<const D: usize, const L: usize>(
    points: &[[f32; D]],
    rng: &mut impl Rng,
    n_trials: usize,
) where
    LaneCount<L>: SupportedLaneCount,
{
    let sp_clone = Box::from(points);

    let kdt = pkdt::PkdTree::new(&sp_clone);
    let mut kiddo_kdt = kiddo::KdTree::new();
    for pt in sp_clone.iter() {
        kiddo_kdt.add(pt, 0);
    }

    let (seq_needles, simd_needles) = make_needles(rng, n_trials);

    for (i, simd_needle) in simd_needles.iter().enumerate() {
        let simd_idxs = kdt.query(simd_needle);
        for l in 0..L {
            let seq_needle = seq_needles[i * L + l];
            let q1 = simd_idxs[l];
            assert_eq!(q1, simd_idxs[l]);
            let exact_kiddo_dist = kiddo_kdt
                .nearest_one(&seq_needle, &kiddo::distance::squared_euclidean)
                .0
                .sqrt();
            let exact_dist = dist(kdt.get_point(kdt.query1_exact(seq_needle)), seq_needle);
            assert_eq!(exact_dist, exact_kiddo_dist);
            let approx_dist = dist(seq_needle, kdt.get_point(q1));
            let rel_error = approx_dist / exact_dist - 1.0;
            println!("{seq_needle:?}\t{exact_dist}\t{approx_dist}\t{rel_error}");
        }
    }
}
