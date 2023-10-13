#![feature(portable_simd)]

use std::{
    hint::black_box,
    path::Path,
    simd::{LaneCount, SupportedLaneCount},
    time::Instant,
};

use hdf5::{File, H5Type, Result};
use rand::Rng;

pub fn run_benchmark<const D: usize, const L: usize>(
    points: &[[f32; D]],
    rng: &mut impl Rng,
    n_trials: usize,
) where
    LaneCount<L>: SupportedLaneCount,
{
    let sp_clone = Box::from(points);

    println!("generating PKDT...");
    let tic = Instant::now();
    let kdt = pkdt::PkdTree::new(&sp_clone);
    println!("generated tree in {:?}", Instant::now().duration_since(tic));

    println!("generating competitor's KDT");
    let tic = Instant::now();
    let mut kiddo_kdt = kiddo::KdTree::new();
    for pt in sp_clone.iter() {
        kiddo_kdt.add(pt, 0);
    }
    println!(
        "generated kiddo tree in {:?}",
        Instant::now().duration_since(tic)
    );

    let mut seq_needles = Vec::new();
    let mut simd_needles = Vec::new();

    println!("generating test values...");

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

    let mut sum_approx_dist = 0.0;
    let mut sum_exact_dist = 0.0;
    println!("testing for correctness...");
    for (i, simd_needle) in simd_needles.iter().enumerate() {
        let simd_idxs = kdt.query(simd_needle);
        for l in 0..L {
            let seq_needle = seq_needles[i * L + l];
            let q1 = kdt.query1(seq_needle);
            assert_eq!(q1, simd_idxs[l]);
            sum_exact_dist += kiddo_kdt
                .nearest_one(&seq_needle, &kiddo::distance::squared_euclidean)
                .0
                .sqrt();
            sum_approx_dist += dist(seq_needle, kdt.get_point(q1));
        }
    }
    println!("simd and sequential implementations are consistent with each other.");
    println!(
        "mean exact distance: {}; mean approx dist: {}",
        sum_exact_dist / seq_needles.len() as f32,
        sum_approx_dist / seq_needles.len() as f32
    );

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
        "speedup: {}%",
        (100.0 * seq_time * (1.0 / simd_time - 1.0 / seq_time)) as u64
    )
}

fn dist<const D: usize>(a: [f32; D], b: [f32; D]) -> f32 {
    a.into_iter()
        .zip(b)
        .map(|(x1, x2)| (x1 - x2).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Load a pointcloud as a vector of 3-d float arrays from a HDF5 file located at `pointcloud_path`.
pub fn load_pointcloud(pointcloud_path: impl AsRef<Path>) -> Result<Vec<[f32; 3]>> {
    // TODO: Extend with radius for variable point size?
    #[derive(H5Type, Debug, Clone)]
    #[repr(C)]
    struct Point {
        x: f32,
        y: f32,
        z: f32,
    }

    impl From<Point> for [f32; 3] {
        fn from(p: Point) -> Self {
            [p.x, p.y, p.z]
        }
    }

    let file = File::open(pointcloud_path)?;
    let dataset = file.dataset("pointcloud/points")?;
    let points = dataset.read_1d::<Point>()?;
    Ok(points.mapv(|p| p.into()).into_raw_vec())
}
