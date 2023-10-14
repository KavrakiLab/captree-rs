use std::{env, path::Path};

use pkdt_bench::{load_pointcloud, run_benchmark};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const N: usize = 4096;
const L: usize = 16;
const D: usize = 3;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut rng = ChaCha20Rng::seed_from_u64(2707);
    let starting_points: Vec<[f32; D]> = if args.len() > 1 {
        println!("Loading pointcloud from {}", &args[1]);
        load_pointcloud(Path::new(&args[1])).unwrap()
    } else {
        println!("No pointcloud file! Using N={N}, L={L}, D={D}");
        println!("generating random points...");
        (0..N)
            .map(|_| {
                [
                    rng.gen_range::<f32, _>(0.0..1.0),
                    rng.gen_range::<f32, _>(0.0..1.0),
                    rng.gen_range::<f32, _>(0.0..1.0),
                ]
            })
            .collect::<Vec<[f32; D]>>()
    };

    run_benchmark::<D, L>(&starting_points, &mut rng, 1 << 20);
}
