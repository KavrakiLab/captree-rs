#![feature(portable_simd)]

use std::{cmp::min, hint::black_box, simd::Simd};

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use pkdt::AffordanceTree;
use pkdt_bench::{make_needles, stopwatch};
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

        let (aff_tree, build_time) =
            stopwatch(|| AffordanceTree::<3>::new(&points, RADIUS_RANGE_SQ, &mut rng).unwrap());

        let (_, seq_time) = stopwatch(|| {
            for needle in &seq_needles {
                black_box(aff_tree.collides(needle, RADIUS_SQ));
            }
        });

        let (_, simd_time) = stopwatch(|| {
            for needle in simd_needles {
                black_box(aff_tree.collides_simd(&needle, Simd::splat(RADIUS)));
            }
        });

        let (kdt, kdt_build_time) = stopwatch(|| ImmutableKdTree::new_from_slice(&points));

        let (_, kdt_query0) = stopwatch(|| {
            for needle in &seq_needles {
                black_box(kdt.within_unsorted::<SquaredEuclidean>(needle, RADIUS_SQ));
            }
        });

        let (_, kdt_query1) = stopwatch(|| {
            for needle in &seq_needles {
                black_box(kdt.nearest_one::<SquaredEuclidean>(needle).distance < RADIUS_SQ);
            }
        });

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
