#![feature(portable_simd)]

use std::{hint::black_box, time::Instant};

use pkdt::BallTree;
use pkdt_bench::{get_points, make_needles};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const N: usize = 1 << 12;

fn main() {
    let points = get_points(N);
    let mut rng = ChaCha20Rng::seed_from_u64(2707);
    let n_trials = 1 << 18;

    let (seq_needles, _) = make_needles::<3, 2>(&mut rng, n_trials);
    println!("benchmarking...");

    bench_ball_tree::<1>(&points, &seq_needles, &mut rng);
}

fn bench_ball_tree<const LW: usize>(points: &[[f32; 3]], needles: &[[f32; 3]], rng: &mut impl Rng) {
    let tree = BallTree::<3, LW>::new3(points, rng);
    let tic = Instant::now();
    for &needle in needles {
        black_box(tree.collides(needle, 0.01f32));
    }
    let toc = Instant::now();
    println!(
        "completed ball tree (LW={LW}) in {:?} ({:?}/q)",
        toc - tic,
        (toc - tic) / needles.len() as u32,
    );
}
