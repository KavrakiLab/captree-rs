#![feature(portable_simd)]

use std::{hint::black_box, simd::Simd};

use afftree::{AffordanceTree, PkdForest};
use afftree_bench::{get_points, stopwatch};
use kiddo::SquaredEuclidean;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const N: usize = 1 << 12;
const L: usize = 16;

const R: f32 = 0.08;
const R_SQ: f32 = R * R;
const R_SQ_RANGE: (f32, f32) = (0.012 * 0.012, 0.08 * 0.08);

fn main() {
    let mut points = get_points(N);
    let mut rng = ChaCha20Rng::seed_from_u64(2707);
    points.shuffle(&mut rng);
    let n_trials = 1 << 20;

    println!("{} points", points.len());
    println!("generating PKDT...");
    let (kdt, kdt_gen_time) = stopwatch(|| afftree::PkdTree::new(&points));
    println!("generated tree in {:?}", kdt_gen_time);

    println!("generating competitor's KDT");
    let (kiddo_kdt, kiddo_gen_time) = stopwatch(|| kiddo::ImmutableKdTree::new_from_slice(&points));
    println!("generated kiddo tree in {:?}", kiddo_gen_time);

    println!("forward tree memory: {:?}B", kdt.memory_used());

    println!("testing for performance...");

    let (seq_needles, simd_needles) = afftree_bench::make_needles(&mut rng, n_trials);
    // let (seq_needles, simd_needles) = pkdt_bench::make_correlated_needles(&mut rng, n_trials);

    bench_affordance(&points, &simd_needles, &seq_needles, &mut rng);

    println!("testing sequential...");

    let (_, kiddo_range_time) = stopwatch(|| {
        for needle in &seq_needles {
            black_box(kiddo_kdt.within_unsorted::<SquaredEuclidean>(needle, R_SQ));
        }
    });

    println!(
        "completed kiddo (range) in {:?} ({:?}/q)",
        kiddo_range_time,
        kiddo_range_time / seq_needles.len() as u32
    );

    let (_, kiddo_exact_time) = stopwatch(|| {
        for needle in &seq_needles {
            black_box(kiddo_kdt.nearest_one::<SquaredEuclidean>(needle).distance < R_SQ);
        }
    });

    println!(
        "completed kiddo (exact) in {:?} ({:?}/q)",
        kiddo_exact_time,
        kiddo_exact_time / seq_needles.len() as u32
    );

    bench_forest::<1>(&points, &simd_needles, &mut rng);
    bench_forest::<2>(&points, &simd_needles, &mut rng);
    bench_forest::<3>(&points, &simd_needles, &mut rng);
    bench_forest::<4>(&points, &simd_needles, &mut rng);
    bench_forest::<5>(&points, &simd_needles, &mut rng);

    let (_, seq_time) = stopwatch(|| {
        for &needle in &seq_needles {
            black_box(kdt.might_collide(needle, R_SQ));
        }
    });
    println!(
        "completed forward sequential in {:?} ({:?}/q)",
        seq_time,
        seq_time / seq_needles.len() as u32
    );

    let (_, simd_time) = stopwatch(|| {
        for needle in &simd_needles {
            black_box(kdt.might_collide_simd::<L>(needle, Simd::splat(R_SQ)));
        }
    });
    println!(
        "completed forward SIMD in {:?}s, ({:?}/pt, {:?}/q)",
        simd_time,
        simd_time / seq_needles.len() as u32,
        simd_time / simd_needles.len() as u32
    );
}

fn bench_affordance(
    points: &[[f32; 3]],
    simd_needles: &[[Simd<f32, L>; 3]],
    seq_needles: &[[f32; 3]],
    rng: &mut impl Rng,
) {
    let (aff_tree, afftree_construct_time) = stopwatch(|| {
        AffordanceTree::<3, _, u64>::new(
            points,
            (R_SQ_RANGE.0, R_SQ_RANGE.1),
            // (0.0f32, 0.02f32.powi(2)),
            rng,
        )
        .unwrap()
    });
    println!(
        "constructed affordance tree in {:?}",
        afftree_construct_time
    );
    println!("affordance tree memory: {:?}B", aff_tree.memory_used());
    println!("affordance size: {}", aff_tree.affordance_size());

    let (_, aff_time) = stopwatch(|| {
        for needle in seq_needles {
            black_box(aff_tree.collides(needle, R_SQ));
        }
    });
    println!(
        "completed sequential queries for affordance trees in {:?} ({:?}/q)",
        aff_time,
        aff_time / seq_needles.len() as u32
    );

    let (_, aff_simd_time) = stopwatch(|| {
        for simd_needle in simd_needles {
            black_box(aff_tree.collides_simd(simd_needle, Simd::splat(R)));
        }
    });
    println!(
        "completed SIMD queries for affordance trees in {:?} ({:?}/pt, {:?}/q)",
        aff_simd_time,
        aff_simd_time / seq_needles.len() as u32,
        aff_simd_time / simd_needles.len() as u32
    );
}

fn bench_forest<const T: usize>(
    points: &[[f32; 3]],
    simd_needles: &[[Simd<f32, L>; 3]],
    rng: &mut impl Rng,
) {
    let forest = PkdForest::<3, T>::new(points, rng);

    let (_, time) = stopwatch(|| {
        for needle in simd_needles {
            black_box(forest.might_collide_simd(needle, Simd::splat(R_SQ)));
        }
    });
    println!(
        "completed forest (T={T}) in {:?} ({:?}/pt, {:?}/q)",
        time,
        time / (simd_needles.len() * L) as u32,
        time / simd_needles.len() as u32,
    );
}
