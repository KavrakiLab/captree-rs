#![feature(portable_simd)]

use std::error::Error;
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::simd::Simd;
use std::time::{Duration, Instant};

use kiddo::SquaredEuclidean;
use pkdt::{AffordanceTree, PkdTree};
use pkdt_bench::make_needles;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

const N_TRIALS: usize = 100_000;
const L: usize = 8;

const RADIUS_RANGE_SQ: (f32, f32) = (0.0001, 0.0004);
const QUERY_RADIUS_SQ: f32 = 0.00025;

fn main() -> Result<(), Box<dyn Error>> {
    let mut f_construct = File::create("construct_time.csv")?;
    let mut f_query = File::create("query_time.csv")?;

    for n_points in (1 << 8..1 << 16).step_by(1 << 8) {
        do_row(n_points, &mut f_construct, &mut f_query)?;
    }

    Ok(())
}

fn do_row(
    n_points: usize,
    f_construct: &mut File,
    f_query: &mut File,
) -> Result<(), Box<dyn Error>> {
    let mut rng = ChaCha20Rng::seed_from_u64(2707);
    let (seq_needles, simd_needles) = make_needles::<3, L>(&mut rng, N_TRIALS);

    let points = (0..n_points)
        .map(|_| {
            [
                rng.gen_range::<f32, _>(0.0..1.0),
                rng.gen_range::<f32, _>(0.0..1.0),
                rng.gen_range::<f32, _>(0.0..1.0),
            ]
        })
        .collect::<Vec<_>>();

    let (kdt, kdt_time) = time(|| kiddo::ImmutableKdTree::new_from_slice(&points));
    let (_, kdt_total_q_time) = time(|| {
        for seq_needle in &seq_needles {
            black_box(kdt.within_unsorted::<SquaredEuclidean>(seq_needle, QUERY_RADIUS_SQ));
        }
    });

    let (pkdt, pkdt_time) = time(|| PkdTree::new(&points));
    let (_, pkdt_total_seq_q_time) = time(|| {
        for &seq_needle in &seq_needles {
            black_box(pkdt.might_collide(seq_needle, QUERY_RADIUS_SQ));
        }
    });
    let (_, pkdt_total_simd_q_time) = time(|| {
        for simd_needle in &simd_needles {
            black_box(pkdt.might_collide_simd(simd_needle, Simd::splat(QUERY_RADIUS_SQ)));
        }
    });

    let (afftree, afftree_time) =
        time(|| AffordanceTree::<3>::new(&points, RADIUS_RANGE_SQ, &mut rng).unwrap());
    let (_, afftree_total_seq_q_time) = time(|| {
        for seq_needle in &seq_needles {
            black_box(afftree.collides(seq_needle, QUERY_RADIUS_SQ));
        }
    });
    let (_, afftree_total_simd_q_time) = time(|| {
        for simd_needle in &simd_needles {
            black_box(afftree.collides_simd(simd_needle, Simd::splat(QUERY_RADIUS_SQ)));
        }
    });

    writeln!(
        f_construct,
        "{n_points},{},{},{}",
        kdt_time.as_secs_f64(),
        pkdt_time.as_secs_f64(),
        afftree_time.as_secs_f64(),
    )?;
    writeln!(
        f_query,
        "{n_points},{},{},{},{},{}",
        kdt_total_q_time.as_secs_f64() / N_TRIALS as f64,
        pkdt_total_seq_q_time.as_secs_f64() / N_TRIALS as f64,
        pkdt_total_simd_q_time.as_secs_f64() / N_TRIALS as f64,
        afftree_total_seq_q_time.as_secs_f64() / N_TRIALS as f64,
        afftree_total_simd_q_time.as_secs_f64() / N_TRIALS as f64,
    )?;

    Ok(())
}

fn time<F: FnOnce() -> R, R>(f: F) -> (R, Duration) {
    let tic = Instant::now();
    let r = f();
    (r, Instant::now().duration_since(tic))
}
