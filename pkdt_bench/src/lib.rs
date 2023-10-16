#![feature(portable_simd)]

use std::{
    env,
    path::Path,
    simd::{LaneCount, SupportedLaneCount},
};

use hdf5::{File, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

pub fn measure_error<const D: usize, const L: usize>(
    points: &[[f32; D]],
    rng: &mut impl Rng,
    n_trials: usize,
) where
    LaneCount<L>: SupportedLaneCount,
{
    let sp_clone = Box::from(points);

    let kdt = pkdt::PkdTree::new(&sp_clone);
    let mut kiddo_kdt = kiddo::KdTree::new();
    for pt in sp_clone.iter() {
        kiddo_kdt.add(pt, 0);
    }

    let (seq_needles, simd_needles) = make_needles(rng, n_trials);

    for (i, simd_needle) in simd_needles.iter().enumerate() {
        let simd_idxs = kdt.query(simd_needle);
        for l in 0..L {
            let seq_needle = seq_needles[i * L + l];
            let q1 = simd_idxs[l];
            assert_eq!(q1, simd_idxs[l]);
            let exact_kiddo_dist = kiddo_kdt
                .nearest_one(&seq_needle, &kiddo::distance::squared_euclidean)
                .0
                .sqrt();
            let exact_dist = dist(kdt.get_point(kdt.query1_exact(seq_needle)), seq_needle);
            assert_eq!(exact_dist, exact_kiddo_dist);
            let approx_dist = dist(seq_needle, kdt.get_point(q1));
            let rel_error = approx_dist / exact_dist - 1.0;
            println!("{seq_needle:?}\t{exact_dist}\t{approx_dist}\t{rel_error}");
        }
    }
}

/// Run a test of the error of the bail approach for SIMD KDT querying.
pub fn measure_bail_error<const D: usize, const L: usize>(
    points: &[[f32; D]],
    rng: &mut impl Rng,
    n_trials: usize,
) where
    LaneCount<L>: SupportedLaneCount,
{
    let sp_clone = Box::from(points);

    let kdt = pkdt::PkdTree::new(&sp_clone);
    let mut kiddo_kdt = kiddo::KdTree::new();
    for pt in sp_clone.iter() {
        kiddo_kdt.add(pt, 0);
    }

    let (seq_needles, simd_needles) = make_needles(rng, n_trials);

    for bail_height in 0..10 {
        for (i, simd_needle) in simd_needles.iter().enumerate() {
            let simd_idxs = kdt.query_bail(simd_needle, bail_height);
            for l in 0..L {
                let seq_needle = seq_needles[i * L + l];
                let q1 = simd_idxs[l];
                assert_eq!(q1, simd_idxs[l]);
                let exact_dist = kiddo_kdt
                    .nearest_one(&seq_needle, &kiddo::distance::squared_euclidean)
                    .0
                    .sqrt();
                let approx_dist = dist(seq_needle, kdt.get_point(q1));
                let rel_error = approx_dist / exact_dist - 1.0;
                println!("{bail_height}\t{seq_needle:?}\t{exact_dist}\t{approx_dist}\t{rel_error}");
            }
        }
    }
}

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
