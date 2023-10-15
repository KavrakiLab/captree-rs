use std::{env, path::Path};

use pkdt_bench::{load_pointcloud, measure_bail_error};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const N: usize = 1 << 16;
const L: usize = 16;
const D: usize = 3;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut rng = ChaCha20Rng::seed_from_u64(2707);
    let starting_points: Vec<[f32; D]> = if args.len() > 1 {
        eprintln!("Loading pointcloud from {}", &args[1]);
        load_pointcloud(Path::new(&args[1])).unwrap()
    } else {
        eprintln!("No pointcloud file! Using N={N}, L={L}, D={D}");
        eprintln!("generating random points...");
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

    measure_bail_error::<D, L>(&starting_points, &mut rng, 1 << 16);
}
