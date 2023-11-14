#![feature(portable_simd)]

use std::{hint::black_box, simd::Simd, time::Instant};

use kiddo::SquaredEuclidean;
use pkdt::{BallTree, PkdForest};
use pkdt_bench::{get_points, make_needles};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const N: usize = 1 << 12;
const L: usize = 16;

fn main() {
    let points = get_points(N);
    let mut rng = ChaCha20Rng::seed_from_u64(2707);
    let n_trials = 1 << 20;

    println!("{} points", points.len());
    println!("generating PKDT...");
    let tic = Instant::now();
    let kdt = pkdt::PkdTree::new(&points);
    println!("generated tree in {:?}", Instant::now().duration_since(tic));

    println!("generating competitor's KDT");
    let tic = Instant::now();
    let kiddo_kdt = kiddo::ImmutableKdTree::new_from_slice(&points);
    println!(
        "generated kiddo tree in {:?}",
        Instant::now().duration_since(tic)
    );

    println!("testing for performance...");

    let (seq_needles, simd_needles) = make_needles(&mut rng, n_trials);
    // let (seq_needles, simd_needles) = make_correlated_needles(&mut rng, n_trials);

    println!("testing sequential...");

    let tic = Instant::now();
    for needle in &seq_needles {
        black_box(kiddo_kdt.within_unsorted::<SquaredEuclidean>(needle, 0.01f32.powi(2)));
    }
    let toc = Instant::now();
    let kiddo_time = toc.duration_since(tic);
    println!(
        "completed kiddo in {:?} ({:?}/q)",
        kiddo_time,
        kiddo_time / seq_needles.len() as u32
    );

    let tic = Instant::now();
    for needle in &seq_needles {
        black_box(kdt.query1_exact(*needle));
    }
    let toc = Instant::now();
    let exact_time = toc.duration_since(tic);
    println!(
        "completed exact in {:?} ({:?}/q)",
        exact_time,
        exact_time / seq_needles.len() as u32
    );

    let tic = Instant::now();
    for &needle in &seq_needles {
        black_box(kdt.might_collide(needle, 0.0001));
    }
    let toc = Instant::now();
    let seq_time = toc.duration_since(tic);
    println!(
        "completed sequential in {:?} ({:?}/q)",
        seq_time,
        seq_time / seq_needles.len() as u32
    );

    bench_forest::<1>(&points, &simd_needles, &mut rng);
    bench_forest::<2>(&points, &simd_needles, &mut rng);
    bench_forest::<3>(&points, &simd_needles, &mut rng);
    bench_forest::<4>(&points, &simd_needles, &mut rng);
    bench_forest::<5>(&points, &simd_needles, &mut rng);
    bench_forest::<6>(&points, &simd_needles, &mut rng);
    bench_forest::<7>(&points, &simd_needles, &mut rng);
    bench_forest::<8>(&points, &simd_needles, &mut rng);
    bench_forest::<9>(&points, &simd_needles, &mut rng);
    bench_forest::<10>(&points, &simd_needles, &mut rng);

    bench_ball_tree::<1>(&points, &seq_needles, &mut rng);
    bench_ball_tree::<2>(&points, &seq_needles, &mut rng);
    bench_ball_tree::<4>(&points, &seq_needles, &mut rng);
    bench_ball_tree::<8>(&points, &seq_needles, &mut rng);
    bench_ball_tree::<16>(&points, &seq_needles, &mut rng);
    bench_ball_tree::<32>(&points, &seq_needles, &mut rng);

    let tic = Instant::now();
    for needle in &simd_needles {
        black_box(kdt.might_collide_simd::<L>(needle, Simd::splat(0.0001)));
    }
    let toc = Instant::now();
    let simd_time = toc.duration_since(tic);
    println!(
        "completed simd in {:?}s, ({:?}/pt, {:?}/q)",
        simd_time,
        simd_time / seq_needles.len() as u32,
        simd_time / simd_needles.len() as u32
    );

    println!(
        "speedup: {}% vs single-query",
        (100.0 * (seq_time.as_secs_f64() / simd_time.as_secs_f64() - 1.0)) as u64
    );
    println!(
        "speedup: {}% vs kiddo",
        (100.0 * (kiddo_time.as_secs_f64() / simd_time.as_secs_f64() - 1.0)) as u64
    )
}

fn bench_forest<const T: usize>(
    points: &[[f32; 3]],
    simd_needles: &[[Simd<f32, L>; 3]],
    rng: &mut impl Rng,
) {
    let forest = PkdForest::<3, T>::new(points, rng);

    let tic = Instant::now();
    for needle in simd_needles {
        black_box(forest.might_collide_simd(needle, Simd::splat(0.01f32.powi(2))));
    }
    let toc = Instant::now();
    println!(
        "completed forest (T={T}) in {:?} ({:?}/pt, {:?}/q)",
        toc - tic,
        (toc - tic) / (simd_needles.len() * L) as u32,
        (toc - tic) / simd_needles.len() as u32,
    );
}

#[allow(dead_code)]
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

    let tic = Instant::now();
    for &needle in needles {
        black_box(tree.forward_only_collides(needle, 0.01f32));
    }
    let toc = Instant::now();
    println!(
        "completed ball tree (forward only, LW={LW}) in {:?} ({:?}/q)",
        toc - tic,
        (toc - tic) / needles.len() as u32,
    );
}
