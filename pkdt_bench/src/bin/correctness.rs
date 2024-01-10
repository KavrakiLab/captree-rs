#![feature(portable_simd)]

use std::simd::Simd;

use kiddo::SquaredEuclidean;
use pkdt::AffordanceTree;
use pkdt_bench::{dist, get_points, make_needles};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const N: usize = 1 << 12;
const R: f32 = 0.02;
const R_SQ: f32 = R * R;

fn main() {
    let points = get_points(N);
    let mut rng = ChaCha20Rng::seed_from_u64(27071);
    let n_trials = 1 << 20;

    let kdt = pkdt::PkdTree::new(&points);
    let mut kiddo_kdt = kiddo::KdTree::new();
    for pt in &points {
        kiddo_kdt.add(pt, 0);
    }

    let aff_tree = AffordanceTree::<3>::new(
        &points,
        (R_SQ - f32::EPSILON, R_SQ + f32::EPSILON),
        &mut rng,
    )
    .unwrap();

    let (needles, _) = make_needles::<3, 2>(&mut rng, n_trials);

    for (i, &needle) in needles.iter().enumerate() {
        println!("iter {i}: {needle:?}");
        let exact_kiddo_dist = kiddo_kdt
            .nearest_one::<SquaredEuclidean>(&needle)
            .distance
            .sqrt();
        let exact_dist = dist(kdt.get_point(kdt.query1_exact(needle)), needle);
        assert_eq!(exact_dist, exact_kiddo_dist);

        let simd_needle: [Simd<f32, 16>; 3] = [
            Simd::splat(needle[0]),
            Simd::splat(needle[1]),
            Simd::splat(needle[2]),
        ];
        if exact_dist <= R {
            assert!(aff_tree.collides(&needle, R_SQ));
            assert!(aff_tree.collides_simd(&simd_needle, Simd::splat(R)))
        } else {
            assert!(!aff_tree.collides(&needle, R_SQ));
            assert!(!aff_tree.collides_simd(&simd_needle, Simd::splat(R)))
        }
    }
}
