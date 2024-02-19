#![warn(clippy::pedantic)]

//! A filtering algorithm for 3D point clouds.

use std::ops::BitOr;

use bitintr::Pdep;

/// Filter out `points` such that points within `min_sep` of each other may be removed.
pub fn morton_filter(points: &mut Vec<[f32; 3]>, min_sep: f32) {
    const PERMUTATIONS_3D: [[u8; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    let mut aabb_min = [f32::INFINITY; 3];
    let mut aabb_max = [f32::NEG_INFINITY; 3];

    for point in points.iter() {
        for k in 0..3 {
            if point[k] < aabb_min[k] {
                aabb_min[k] = point[k];
            }
            if point[k] > aabb_max[k] {
                aabb_max[k] = point[k];
            }
        }
    }

    let rsq = min_sep * min_sep;
    for permutation in PERMUTATIONS_3D {
        points.sort_by_cached_key(|point| morton_index(point, &aabb_min, &aabb_max, permutation));
        let mut i = 0;
        let mut j = 1;
        while j < points.len() {
            if distsq(&points[i], &points[j]) > rsq {
                i += 1;
                points[i] = points[j];
            }
            j += 1;
        }
        points.truncate(i + 1);
    }
}

fn distsq(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum()
}
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn morton_index(
    point: &[f32; 3],
    aabb_min: &[f32; 3],
    aabb_max: &[f32; 3],
    permutation: [u8; 3],
) -> u32 {
    const WIDTH: u32 = u32::BITS / 3;
    const MASK: u32 = 0b001_001_001_001_001_001_001_001_001_001;

    permutation
        .map(usize::from)
        .into_iter()
        .enumerate()
        .map(|(i, k)| {
            ((((point[k] - aabb_min[k]) / (aabb_max[k] - aabb_min[k])) * WIDTH as f32) as u32)
                .pdep(MASK << i)
        })
        .fold(0, BitOr::bitor)
}

#[cfg(test)]
mod tests {
    use crate::morton_filter;

    #[test]
    fn one_point() {
        let mut points = vec![[0.0; 3]];
        morton_filter(&mut points, 0.01);
        assert_eq!(points, vec![[0.0; 3]]);
    }

    #[test]
    fn duplicate() {
        let mut points = vec![[0.0; 3]; 2];
        morton_filter(&mut points, 0.01);
        assert_eq!(points, vec![[0.0; 3]]);
    }

    #[test]
    fn too_close() {
        let mut points = vec![[0.0; 3], [0.001; 3]];
        morton_filter(&mut points, 0.01);
        assert_eq!(points, vec![[0.0; 3]]);
    }

    #[test]
    fn too_far() {
        let mut points = vec![[0.0; 3], [0.01; 3]];
        morton_filter(&mut points, 0.01);
        assert_eq!(points, vec![[0.0; 3], [0.01; 3]]);
    }
}
