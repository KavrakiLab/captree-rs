use pkdt_bench::{dist, get_points, make_needles};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

const N: usize = 1 << 12;
const L: usize = 16;

fn main() {
    let points = get_points(N);
    let mut rng = ChaCha20Rng::seed_from_u64(27071);
    let n_trials = 1 << 20;

    let kdt = pkdt::PkdTree::new(&points);
    let mut kiddo_kdt = kiddo::KdTree::new();
    for pt in &points {
        kiddo_kdt.add(pt, 0);
    }

    let (seq_needles, simd_needles) = make_needles(&mut rng, n_trials);

    for (i, simd_needle) in simd_needles.iter().enumerate() {
        let simd_idxs = kdt.query::<L>(simd_needle);
        for l in 0..L {
            println!("iter {}", i * L + l);
            let seq_needle = seq_needles[i * L + l];
            let q1 = simd_idxs[l];
            assert_eq!(q1, simd_idxs[l]);
            let exact_kiddo_dist = kiddo_kdt
                .nearest_one(&seq_needle, &kiddo::distance::squared_euclidean)
                .0
                .sqrt();
            let exact_dist = dist(kdt.get_point(kdt.query1_exact(seq_needle)), seq_needle);
            assert_eq!(exact_dist, exact_kiddo_dist);
        }
    }
}
