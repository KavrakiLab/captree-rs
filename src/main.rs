use std::{hint::black_box, time::Instant};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const N: usize = 1 << 12;
const L: usize = 16;
const D: usize = 3;

fn main() {
    println!("N={N}, L={L}, D={D}");
    let mut rng = ChaCha20Rng::seed_from_u64(2707);

    println!("generating random points...");
    let tic = Instant::now();
    let starting_points: Box<[[f32; D]; N]> = (0..N)
        .map(|_| {
            [
                rng.gen_range::<f32, _>(0.0..1.0),
                rng.gen_range::<f32, _>(0.0..1.0),
                rng.gen_range::<f32, _>(0.0..1.0),
            ]
        })
        .collect::<Box<[[f32; 3]]>>()
        .try_into()
        .unwrap();

    let mut sp_clone = starting_points.clone();

    println!("generating PKDT...");
    let kdt = pkdt::PkdTree::new(&mut sp_clone);
    println!("generated tree in {:?}", Instant::now().duration_since(tic));

    println!("generating competitor's KDT");
    let mut kiddo_kdt = kiddo::KdTree::new();
    for pt in sp_clone.iter() {
        kiddo_kdt.add(pt, 0);
    }

    let mut seq_needles = Vec::new();
    let mut simd_needles = Vec::new();

    println!("generating test values...");

    for _ in 0..1 << 16 {
        let mut simd_pts = [[0.0; L]; D];
        for l in 0..L {
            let mut seq_needle = [0.0; D];
            for d in 0..3 {
                let value = rng.gen_range::<f32, _>(0.0..1.0);
                seq_needle[d] = value;
                simd_pts[d][l] = value;
            }
            seq_needles.push(seq_needle);
        }
        simd_needles.push(simd_pts);
    }

    println!("testing for correctness...");
    for (i, simd_needle) in simd_needles.iter().enumerate() {
        let simd_idxs = kdt.query(simd_needle);
        for l in 0..L {
            assert_eq!(kdt.query1(seq_needles[i * L + l]), simd_idxs[l]);
        }
    }
    println!("simd and sequential implementations are consistent with each other.");

    println!("testing for performance...");
    println!("testing sequential...");

    let tic = Instant::now();
    for needle in &seq_needles {
        black_box(kiddo_kdt.nearest_one(needle, &kiddo::distance::squared_euclidean));
    }
    let toc = Instant::now();
    let seq_time = (toc.duration_since(tic)).as_secs_f64();
    println!(
        "completed kiddo in {:?}s ({} qps)",
        seq_time,
        (simd_needles.len() as f64 / seq_time) as u64
    );

    let tic = Instant::now();
    for needle in seq_needles {
        black_box(kdt.query1(needle));
    }
    let toc = Instant::now();
    let seq_time = (toc.duration_since(tic)).as_secs_f64();
    println!(
        "completed sequential in {:?}s ({} qps)",
        seq_time,
        (simd_needles.len() as f64 / seq_time) as u64
    );

    let tic = Instant::now();
    for needle in &simd_needles {
        black_box(kdt.query(needle));
    }
    let toc = Instant::now();
    let simd_time = (toc.duration_since(tic)).as_secs_f64();
    println!(
        "completed simd in {:?}s, ({} qps)",
        simd_time,
        (simd_needles.len() as f64 / simd_time) as u64
    );

    println!(
        "speedup: {}%",
        (100.0 * seq_time * (1.0 / simd_time - 1.0 / seq_time)) as u64
    )
}
