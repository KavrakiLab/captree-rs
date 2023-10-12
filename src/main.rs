use std::{hint::black_box, time::Instant};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const N: usize = 1 << 22;
const L: usize = 64;
const D: usize = 3;

fn main() {
    let mut rng = ChaCha20Rng::seed_from_u64(2707);

    println!("generating K-D tree...");
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

    let kdt = pkdt::PkdTree::new(&mut sp_clone);

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

    println!("testing for performance...");
    println!("testing sequential...");

    let tic = Instant::now();
    for needle in seq_needles {
        black_box(kdt.query1(needle));
    }
    let toc = Instant::now();
    let seq_time = (toc.duration_since(tic)).as_secs_f64();
    println!("completed sequential in {:?}s", seq_time);

    let tic = Instant::now();
    for needle in &simd_needles {
        black_box(kdt.query(needle));
    }
    let toc = Instant::now();
    let simd_time = (toc.duration_since(tic)).as_secs_f64();
    println!("completed simd in {:?}s", simd_time);

    println!(
        "speedup: {}%",
        ((seq_time - simd_time) * 100.0 / simd_time) as u64
    )
}
