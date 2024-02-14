#![feature(portable_simd)]

use std::simd::Simd;

use afftree::AffordanceTree;
use afftree_bench::{dist, parse_pointcloud_csv, parse_trace_csv, trace_rsq_range};
use kiddo::SquaredEuclidean;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const N: usize = 1 << 12;
const R: f32 = 0.02;
const R_SQ: f32 = R * R;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = rand::thread_rng();
    let args = std::env::args().collect::<Box<_>>();
    let points: Box<[[f32; 3]]> = if args.len() < 2 {
        (0..1 << 16)
            .map(|_| {
                [
                    rng.gen_range::<f32, _>(0.0..1.0),
                    rng.gen_range::<f32, _>(0.0..1.0),
                    rng.gen_range::<f32, _>(0.0..1.0),
                ]
            })
            .collect()
    } else {
        let mut p = parse_pointcloud_csv(&args[1])?.to_vec();
        p.shuffle(&mut rng);
        p.truncate(1 << 16);
        p.into_boxed_slice()
    };

    let trace: Box<[([f32; 3], f32)]> = if args.len() < 3 {
        (0..N)
            .map(|_| {
                (
                    [
                        rng.gen_range(0.0..1.0),
                        rng.gen_range(0.0..1.0),
                        rng.gen_range(0.0..1.0),
                    ],
                    rng.gen_range(0.0..=0.08),
                )
            })
            .collect()
    } else {
        parse_trace_csv(&args[2])?
    };

    let rsq_range = trace_rsq_range(&trace);

    let mut rng = ChaCha20Rng::seed_from_u64(27071);

    let kdt = afftree::PkdTree::new(&points);
    let mut kiddo_kdt = kiddo::KdTree::new();
    for pt in points.iter() {
        kiddo_kdt.add(pt, 0);
    }

    let aff_tree = AffordanceTree::<3>::new(&points, rsq_range, &mut rng).unwrap();

    for (i, (center, r)) in trace.iter().enumerate() {
        println!("iter {i}: {:?}", (center, r));
        let exact_kiddo_dist = kiddo_kdt
            .nearest_one::<SquaredEuclidean>(center)
            .distance
            .sqrt();
        let exact_dist = dist(kdt.get_point(kdt.query1_exact(*center)), *center);
        assert_eq!(exact_dist, exact_kiddo_dist);

        let simd_center: [Simd<f32, 8>; 3] = [
            Simd::splat(center[0]),
            Simd::splat(center[1]),
            Simd::splat(center[2]),
        ];
        if exact_dist <= R {
            assert!(aff_tree.collides(center, R_SQ));
            assert!(aff_tree.collides_simd(&simd_center, Simd::splat(R)))
        } else {
            assert!(!aff_tree.collides(center, R_SQ));
            assert!(!aff_tree.collides_simd(&simd_center, Simd::splat(R)))
        }
    }

    Ok(())
}
