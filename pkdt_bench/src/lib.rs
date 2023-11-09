#![feature(portable_simd)]

use std::{
    env,
    path::Path,
};

use hdf5::{File, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;


pub fn get_points(n_points_if_no_cloud: usize) -> Vec<[f32; 3]> {
    let args: Vec<String> = env::args().collect();
    let mut rng = ChaCha20Rng::seed_from_u64(2707);

    if args.len() > 1 {
        println!("Loading pointcloud from {}", &args[1]);
        load_pointcloud(Path::new(&args[1])).unwrap()
    } else {
        println!("No pointcloud file! Using N={n_points_if_no_cloud}");
        println!("generating random points...");
        (0..n_points_if_no_cloud)
            .map(|_| {
                [
                    rng.gen_range::<f32, _>(0.0..1.0),
                    rng.gen_range::<f32, _>(0.0..1.0),
                    rng.gen_range::<f32, _>(0.0..1.0),
                ]
            })
            .collect()
    }
}

/// Generate some randomized numbers for us to benchmark against.
///
/// # Generic parameters
///
/// - `D`: the dimension of the space
/// - `L`: the number of SIMD lanes
///
/// # Returns
///
/// Returns a pair `(seq_needles, simd_needles)`, where `seq_needles` is correctly shaped for
/// sequential querying and `simd_needles` is correctly shaped for SIMD querying.
pub fn make_needles<const D: usize, const L: usize>(
    rng: &mut impl Rng,
    n_trials: usize,
) -> (Vec<[f32; D]>, Vec<[[f32; L]; D]>) {
    let mut seq_needles = Vec::new();
    let mut simd_needles = Vec::new();

    for _ in 0..n_trials / L {
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

    assert_eq!(seq_needles.len(), simd_needles.len() * L);

    (seq_needles, simd_needles)
}

/// Load a pointcloud as a vector of 3-d float arrays from a HDF5 file located at `pointcloud_path`.
pub fn load_pointcloud(pointcloud_path: impl AsRef<Path>) -> Result<Vec<[f32; 3]>> {
    let file = File::open(pointcloud_path)?;
    let dataset = file.dataset("pointcloud/points")?;
    let points = dataset.read_2d::<f32>()?;
    Ok(points
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect())
}

pub fn dist<const D: usize>(a: [f32; D], b: [f32; D]) -> f32 {
    a.into_iter()
        .zip(b)
        .map(|(x1, x2)| (x1 - x2).powi(2))
        .sum::<f32>()
        .sqrt()
}
