#![feature(portable_simd)]

use std::cmp::min;
use std::env::args;
use std::error::Error;
use std::fs::{read, File};
use std::hint::black_box;
use std::io::Write;
use std::simd::Simd;

#[allow(unused_imports)]
use kiddo::SquaredEuclidean;
use pkdt::{AffordanceTree, PkdTree};
use pkdt_bench::stopwatch;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

const N_TRIALS: usize = 100_000;
const L: usize = 8;

type FVector = Simd<f32, L>;

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
        let mut p: Vec<_> = std::str::from_utf8(&read(&args[1])?)?
            .lines()
            .map(|l| {
                let mut split = l.split(',').flat_map(|s| s.parse::<f32>().ok());
                Ok::<_, Box<dyn Error>>([
                    split.next().ok_or("pc missing x")?,
                    split.next().ok_or("pc missing y")?,
                    split.next().ok_or("pc missing z")?,
                ])
            })
            .collect::<Result<_, _>>()?;
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
        std::str::from_utf8(&read(&args[2])?)?
            .lines()
            .map(|l| {
                let mut split = l.split(',').flat_map(|s| s.parse::<f32>().ok());
                Ok::<_, Box<dyn Error>>((
                    [
                        split.next().ok_or("trace missing x")?,
                        split.next().ok_or("trace missing y")?,
                        split.next().ok_or("trace missing z")?,
                    ],
                    split.next().ok_or("trace missing r")?,
                ))
            })
            .collect::<Result<_, _>>()?
    };

    println!("number of points: {}", points.len());
    println!("number of tests: {}", tests.len());

    let simd_tests = tests
        .chunks(L)
        .map(|w| {
            let mut centers = [[0.0; L]; 3];
            let mut radii = [0.0; L];
            for (l, ([x, y, z], r)) in w.iter().copied().enumerate() {
                centers[0][l] = x;
                centers[1][l] = y;
                centers[2][l] = z;
                radii[l] = r;
            }
            (centers.map(Simd::from_array), Simd::from_array(radii))
        })
        .collect::<Box<_>>();

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
    tests: &[([f32; 3], f32)],
    simd_tests: &[([FVector; 3], FVector)],
    rsq_range: (f32, f32),
    f_construct: &mut File,
    f_query: &mut File,
) -> Result<(), Box<dyn Error>> {
    let (_kdt, kdt_time) = stopwatch(|| /*kiddo::ImmutableKdTree::new_from_slice(&points)*/ ());
    let (_, kdt_within_q_time) = stopwatch(|| {
        for (_center, _radius) in tests {
            black_box(
                // kdt.within_unsorted::<SquaredEuclidean>(center, radius.powi(2))
                //     .is_empty(),
                (),
            );
        }
    });
    let (_, kdt_nearest_q_time) = stopwatch(|| {
        for (_center, _radius) in tests {
            black_box(
                // kdt.nearest_one::<SquaredEuclidean>(center).distance <= radius.powi(2)
                (),
            );
        }
    });

    let kdt_total_q_time = min(kdt_within_q_time, kdt_nearest_q_time);

    let (pkdt, pkdt_time) = stopwatch(|| PkdTree::new(&points));
    let (_, pkdt_total_seq_q_time) = stopwatch(|| {
        for (center, radius) in tests {
            black_box(pkdt.might_collide(*center, radius.powi(2)));
        }
    });
    let (_, pkdt_total_simd_q_time) = stopwatch(|| {
        for (centers, radii) in simd_tests {
            black_box(pkdt.might_collide_simd(centers, radii * radii));
        }
    });

    let (afftree, afftree_time) = stopwatch(|| {
        AffordanceTree::<3>::new(&points, rsq_range, &mut rand::thread_rng()).unwrap()
    });
    let (_, afftree_total_seq_q_time) = stopwatch(|| {
        for (center, radius) in tests {
            black_box(afftree.collides(center, radius.powi(2)));
        }
    });
    let (_, afftree_total_simd_q_time) = stopwatch(|| {
        for (centers, radii) in simd_tests {
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
    writeln!(
        f_query,
        "{},{},{},{},{},{}",
        points.len(),
        kdt_total_q_time.as_secs_f64() / N_TRIALS as f64,
        pkdt_total_seq_q_time.as_secs_f64() / N_TRIALS as f64,
        pkdt_total_simd_q_time.as_secs_f64() / N_TRIALS as f64,
        afftree_total_seq_q_time.as_secs_f64() / N_TRIALS as f64,
        afftree_total_simd_q_time.as_secs_f64() / N_TRIALS as f64,
    )?;

    Ok(())
}
