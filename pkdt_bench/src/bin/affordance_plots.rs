#![feature(portable_simd)]

use std::{cmp::min, hint::black_box, simd::Simd, time::Instant};

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use pkdt::AffordanceTree;
use pkdt_bench::make_needles;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const L: usize = 16;
const RADIUS: f32 = 0.05;
const RADIUS_RANGE_SQ: (f32, f32) = (0.0001, 0.0004);
const RADIUS_SQ: f32 = RADIUS * RADIUS;
const N_TRIALS: usize = 100_000;

fn main() {
    let mut rng = ChaCha20Rng::seed_from_u64(12345);
    for size in (4096..(1 << 20)).step_by(10_000) {
        let (seq_needles, simd_needles) = make_needles::<3, L>(&mut rng, N_TRIALS);
        let points = (0..size)
            .map(|_| {
                [
                    rng.gen_range::<f32, _>(0.0..1.0),
                    rng.gen_range::<f32, _>(0.0..1.0),
                    rng.gen_range::<f32, _>(0.0..1.0),
                ]
            })
            .collect::<Vec<_>>();

        let tic = Instant::now();
        let aff_tree = AffordanceTree::<3>::new(&points, RADIUS_RANGE_SQ, &mut rng).unwrap();
        let toc = Instant::now();
        let build_time = toc - tic;

        let tic = Instant::now();
        for needle in &seq_needles {
            black_box(aff_tree.collides(needle, RADIUS_SQ));
        }
        let toc = Instant::now();
        let seq_time = toc - tic;

        let tic = Instant::now();
        for needle in simd_needles {
            black_box(aff_tree.collides_simd(&needle, Simd::splat(RADIUS)));
        }
        let toc = Instant::now();
        let simd_time = toc - tic;

        let tic = Instant::now();
        let kdt = ImmutableKdTree::new_from_slice(&points);
        let toc = Instant::now();
        let kdt_build_time = toc - tic;

        let tic = Instant::now();
        for needle in &seq_needles {
            black_box(kdt.within_unsorted::<SquaredEuclidean>(needle, RADIUS_SQ));
        }
        let toc = Instant::now();
        let kdt_query0 = toc - tic;

        let tic = Instant::now();
        for needle in &seq_needles {
            black_box(kdt.nearest_one::<SquaredEuclidean>(needle).distance < RADIUS_SQ);
        }
        let toc = Instant::now();
        let kdt_query1 = toc - tic;

        let kdt_query_time = min(kdt_query0, kdt_query1);

        println!(
            "{},{},{},{},{},{}",
            size,
            build_time.as_secs_f64(),
            (seq_time / seq_needles.len() as u32).as_secs_f64(),
            (simd_time / seq_needles.len() as u32).as_secs_f64(),
            kdt_build_time.as_secs_f64(),
            (kdt_query_time / seq_needles.len() as u32).as_secs_f64()
        )
    }
}
