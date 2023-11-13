use pkdt_bench::{dist, get_points, make_needles};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const N: usize = 1 << 12;

fn main() {
    let points = get_points(N);
    let mut rng = ChaCha20Rng::seed_from_u64(27071);
    let n_trials = 1 << 20;

    let kdt = pkdt::PkdTree::new(&points);
    let mut kiddo_kdt = kiddo::KdTree::new();
    for pt in &points {
        kiddo_kdt.add(pt, 0);
    }

    let (needles, _) = make_needles::<3, 2>(&mut rng, n_trials);
    let ball_tree = pkdt::BallTree::<3, 2>::new3(&points, &mut rng);
    println!("{ball_tree:?}");
    assert!(ball_tree.is_valid());

    for (i, &needle) in needles.iter().enumerate() {
        println!("iter {i}");
        let exact_kiddo_dist = kiddo_kdt
            .nearest_one(&needle, &kiddo::distance::squared_euclidean)
            .0
            .sqrt();
        let exact_dist = dist(kdt.get_point(kdt.query1_exact(needle)), needle);
        assert_eq!(exact_dist, exact_kiddo_dist);

        assert!(ball_tree.collides(needle, exact_kiddo_dist + f32::EPSILON));

        assert!(!ball_tree.collides(
            needle,
            if exact_kiddo_dist < 0.02 {
                f32::EPSILON
            } else {
                exact_kiddo_dist - 0.02f32
            }
        ));
    }
}
