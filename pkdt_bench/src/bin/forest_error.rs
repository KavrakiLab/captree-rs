use std::{env, path::Path};

use pkdt::PkdForest;
use pkdt_bench::{load_pointcloud, make_needles};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const N: usize = 1 << 12;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut rng = ChaCha20Rng::seed_from_u64(2707);
    let starting_points: Vec<[f32; 3]> = if args.len() > 1 {
        eprintln!("Loading pointcloud from {}", &args[1]);
        load_pointcloud(Path::new(&args[1])).unwrap()
    } else {
        eprintln!("No pointcloud file! Using N={N}, D=3");
        eprintln!("generating random points...");
        (0..N)
            .map(|_| {
                [
                    rng.gen_range::<f32, _>(0.0..1.0),
                    rng.gen_range::<f32, _>(0.0..1.0),
                    rng.gen_range::<f32, _>(0.0..1.0),
                ]
            })
            .collect::<Vec<[f32; 3]>>()
    };

    err_forest::<1>(&starting_points, &mut rng);
    err_forest::<2>(&starting_points, &mut rng);
    err_forest::<3>(&starting_points, &mut rng);
    err_forest::<4>(&starting_points, &mut rng);
    err_forest::<5>(&starting_points, &mut rng);
    err_forest::<6>(&starting_points, &mut rng);
    err_forest::<7>(&starting_points, &mut rng);
    err_forest::<8>(&starting_points, &mut rng);
    err_forest::<9>(&starting_points, &mut rng);
    err_forest::<10>(&starting_points, &mut rng);
}

fn err_forest<const T: usize>(points: &[[f32; 3]], rng: &mut impl Rng) {
    let forest = PkdForest::<3, T>::new(points, rng);

    let mut kiddo_kdt = kiddo::KdTree::new();
    for pt in points {
        kiddo_kdt.add(pt, 0);
    }

    let (seq_needles, _) = make_needles::<3, 1>(rng, 10_000);

    let mut total_err = 0.0;
    for &needle in &seq_needles {
        let (_, forest_distsq) = forest.query1(needle);
        let (exact_distsq, _) = kiddo_kdt.nearest_one(&needle, &kiddo::distance::squared_euclidean);

        let exact_dist = exact_distsq.sqrt();
        let err = forest_distsq.sqrt() - exact_dist;
        total_err += err;
        let rel_err = err / exact_distsq.sqrt();
        println!("{T}\t{err}\t{rel_err}\t{exact_dist}");
    }

    eprintln!("T={T}: mean error {}", total_err / seq_needles.len() as f32);
}
