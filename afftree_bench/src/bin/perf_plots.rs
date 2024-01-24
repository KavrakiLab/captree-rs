#![feature(portable_simd)]

use std::cmp::min;
use std::env::args;
use std::error::Error;
use std::fs::File;
use std::hint::black_box;
use std::io::Write;

use afftree::{AffordanceTree, PkdTree};
use afftree_bench::{
    parse_pointcloud_csv, parse_trace_csv, simd_trace_new, stopwatch, SimdTrace, Trace,
};
#[allow(unused_imports)]
use kiddo::SquaredEuclidean;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

const N_TRIALS: usize = 100_000;
const L: usize = 8;

const QUERY_RADIUS: f32 = 0.05;

fn main() -> Result<(), Box<dyn Error>> {
    let mut f_construct = File::create("construct_time.csv")?;
    let mut f_query = File::create("query_time.csv")?;

    let args: Vec<String> = args().collect();

    let mut rng = ChaCha20Rng::seed_from_u64(2707);
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

    let tests: Box<[([f32; 3], f32)]> = if args.len() < 3 {
        (0..N_TRIALS)
            .map(|_| {
                (
                    [
                        rng.gen_range(0.0..1.0),
                        rng.gen_range(0.0..1.0),
                        rng.gen_range(0.0..1.0),
                    ],
                    rng.gen_range(QUERY_RADIUS..=QUERY_RADIUS),
                )
            })
            .collect()
    } else {
        parse_trace_csv(&args[2])?
    };

    println!("number of points: {}", points.len());
    println!("number of tests: {}", tests.len());

    let simd_tests = simd_trace_new(&tests);

    let rsq_range = (
        tests
            .iter()
            .map(|x| x.1)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .ok_or("no points")?
            .powi(2),
        tests
            .iter()
            .map(|x| x.1)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .ok_or("no points")?
            .powi(2),
    );

    println!("radius-squared range: {rsq_range:?}");

    for n_points in (1 << 8..=points.len()).step_by(1 << 8) {
        do_row(
            &points[..n_points],
            &tests,
            &simd_tests,
            rsq_range,
            &mut f_construct,
            &mut f_query,
        )?;
    }

    Ok(())
}

fn do_row(
    points: &[[f32; 3]],
    trace: &Trace,
    simd_trace: &SimdTrace<L>,
    rsq_range: (f32, f32),
    f_construct: &mut File,
    f_query: &mut File,
) -> Result<(), Box<dyn Error>> {
    let (_kdt, kdt_time) = stopwatch(|| /*kiddo::ImmutableKdTree::new_from_slice(&points)*/ ());
    let (_, kdt_within_q_time) = stopwatch(|| {
        for (_center, _radius) in trace {
            black_box(
                // kdt.within_unsorted::<SquaredEuclidean>(center, radius.powi(2))
                //     .is_empty(),
                (),
            );
        }
    });
    let (_, kdt_nearest_q_time) = stopwatch(|| {
        for (_center, _radius) in trace {
            black_box(
                // kdt.nearest_one::<SquaredEuclidean>(center).distance <= radius.powi(2)
                (),
            );
        }
    });

    let kdt_total_q_time = min(kdt_within_q_time, kdt_nearest_q_time);

    let (pkdt, pkdt_time) = stopwatch(|| PkdTree::new(&points));
    let (_, pkdt_total_seq_q_time) = stopwatch(|| {
        for (center, radius) in trace {
            black_box(pkdt.might_collide(*center, radius.powi(2)));
        }
    });
    let (_, pkdt_total_simd_q_time) = stopwatch(|| {
        for (centers, radii) in simd_trace {
            black_box(pkdt.might_collide_simd(centers, radii * radii));
        }
    });

    let (afftree, afftree_time) = stopwatch(|| {
        AffordanceTree::<3>::new(&points, rsq_range, &mut rand::thread_rng()).unwrap()
    });
    let (_, afftree_total_seq_q_time) = stopwatch(|| {
        for (center, radius) in trace {
            black_box(afftree.collides(center, radius.powi(2)));
        }
    });
    let (_, afftree_total_simd_q_time) = stopwatch(|| {
        for (centers, radii) in simd_trace {
            black_box(afftree.collides_simd(centers, radii * radii));
        }
    });

    writeln!(
        f_construct,
        "{},{},{},{}",
        points.len(),
        kdt_time.as_secs_f64(),
        pkdt_time.as_secs_f64(),
        afftree_time.as_secs_f64(),
    )?;
    let trace_len = trace.len() as f64;
    writeln!(
        f_query,
        "{},{},{},{},{},{}",
        points.len(),
        kdt_total_q_time.as_secs_f64() / trace_len,
        pkdt_total_seq_q_time.as_secs_f64() / trace_len,
        pkdt_total_simd_q_time.as_secs_f64() / trace_len,
        afftree_total_seq_q_time.as_secs_f64() / trace_len,
        afftree_total_simd_q_time.as_secs_f64() / trace_len,
    )?;

    Ok(())
}
