use std::{hint::black_box, time::Instant};

use pkdt_bench::{get_points, make_needles};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const N: usize = 1 << 16;
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
    let mut kiddo_kdt = kiddo::KdTree::new();
    for pt in points.iter() {
        kiddo_kdt.add(pt, 0);
    }
    println!(
        "generated kiddo tree in {:?}",
        Instant::now().duration_since(tic)
    );

    println!("testing for performance...");

    let (seq_needles, simd_needles) = make_needles(&mut rng, n_trials);

    println!("testing sequential...");

    let tic = Instant::now();
    for needle in &seq_needles {
        black_box(kiddo_kdt.nearest_one(needle, &kiddo::distance::squared_euclidean));
    }
    let toc = Instant::now();
    let kiddo_time = (toc.duration_since(tic)).as_secs_f64();
    println!(
        "completed kiddo in {:?}s ({} qps)",
        kiddo_time,
        (simd_needles.len() as f64 / kiddo_time) as u64
    );

    let tic = Instant::now();
    for needle in &seq_needles {
        black_box(kdt.query1_exact(*needle));
    }
    let toc = Instant::now();
    let exact_time = (toc.duration_since(tic)).as_secs_f64();
    println!(
        "completed exact in {:?}s ({} qps)",
        exact_time,
        (simd_needles.len() as f64 / exact_time) as u64
    );

    let tic = Instant::now();
    for &needle in &seq_needles {
        black_box(kdt.query1(needle));
    }
    let toc = Instant::now();
    let seq_time = (toc.duration_since(tic)).as_secs_f64();
    println!(
        "completed sequential in {:?}s ({} qps)",
        seq_time,
        (seq_needles.len() as f64 / seq_time) as u64
    );

    for bail_height in 0..10 {
        let tic = Instant::now();
        for needle in &simd_needles {
            black_box(kdt.query_bail::<L>(needle, bail_height));
        }
        let toc = Instant::now();
        let seq_time = (toc.duration_since(tic)).as_secs_f64();
        println!(
            "completed bail{bail_height} in {:?}s ({} qps)",
            seq_time,
            (seq_needles.len() as f64 / seq_time) as u64
        );
    }

    let tic = Instant::now();
    for needle in &simd_needles {
        black_box(kdt.query(needle));
    }
    let toc = Instant::now();
    let simd_time = (toc.duration_since(tic)).as_secs_f64();
    println!(
        "completed simd in {:?}s, ({} qps)",
        simd_time,
        (seq_needles.len() as f64 / simd_time) as u64
    );

    println!(
        "speedup: {}% vs single-query",
        (100.0 * (seq_time / simd_time - 1.0)) as u64
    );
    println!(
        "speedup: {}% vs kiddo",
        (100.0 * (kiddo_time / simd_time - 1.0)) as u64
    )
}
