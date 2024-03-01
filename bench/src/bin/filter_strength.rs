use std::{fs::File, io::Write};

use bench::parse_pointcloud_csv;
use morton_filter::filter_permutation;

const PERMUTATIONS_3D: [[u8; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut f_results = File::create("filter_strength.csv")?;
    let args: Vec<_> = std::env::args().collect();
    let p = parse_pointcloud_csv(&args[1])?.to_vec();

    for i in 0..1000 {
        let r_filter = i as f32 / 10_000.0;
        write!(&mut f_results, "{r_filter}")?;
        let mut new_points = p.clone();
        for perm in PERMUTATIONS_3D {
            filter_permutation(&mut new_points, r_filter, perm);
            write!(&mut f_results, ",{}", new_points.len())?;
        }
        writeln!(&mut f_results)?;
    }

    Ok(())
}
