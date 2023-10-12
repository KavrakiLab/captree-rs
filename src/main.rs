use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const N: usize = 1 << 16;

fn main() {
    let mut rng = ChaCha20Rng::seed_from_u64(2707);

    let starting_points: Box<[[f32; 3]; N]> = (0..N)
        .map(|_| {
            [
                rng.gen_range::<f32, _>(0.0..1.0),
                rng.gen_range::<f32, _>(0.0..1.0),
                rng.gen_range::<f32, _>(0.0..1.0),
            ]
        })
        .collect::<Box<[[f32; 3]]>>()
        .try_into()
        .unwrap();

    let mut sp_clone = starting_points.clone();

    let kdt = pkdt::PkdTree::new(&mut sp_clone);

    println!("testing for correctness...");

    let neg1 = [-1.0, -1.0, -1.0];
    let neg1_idx = kdt.query1(neg1);
    assert_eq!(neg1_idx, 0);

    let pos1 = [1.0, 1.0, 1.0];
    let pos1_idx = kdt.query1(pos1);
    assert_eq!(pos1_idx, N - 1);

    let p = [0.5, 0.5, 0.5];

    let approx_closest = kdt.get_point(kdt.query(&[[p[0]], [p[1]], [p[2]]])[0]);
    println!(
        "approx closest is {:?} with distance {:?}",
        approx_closest,
        dist3d(&approx_closest, &p)
    );
    let mut true_closest = starting_points[0];
    for maybe_closer in &starting_points[1..] {
        if dist3d(&true_closest, &maybe_closer) < dist3d(&true_closest, &p) {
            true_closest = *maybe_closer;
        }
    }
    println!(
        "true closest is {true_closest:?} with distance {}",
        dist3d(&true_closest, &p)
    );
}

fn dist3d(x1: &[f32; 3], x2: &[f32; 3]) -> f32 {
    x1.iter()
        .zip(x2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}
