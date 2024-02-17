#![warn(clippy::pedantic)]

//! A filtering algorithm for 3D point clouds.

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

    let (aabb_min, aabb_max) = normalize(points);
    for permutation in PERMUTATIONS_3D {
        points.sort_by_cached_key(|point| morton_index(point, permutation));
        let mut i = 0;
        let mut j = 1;
        while j < points.len() {
            if distsq(&points[i], &points[j]) > min_sep {
                i += 1;
                points[i] = points[j];
            }
            j += 1;
        }
        points.truncate(i + 1);
    }
    denormalize(points, &aabb_min, &aabb_max);
}

fn distsq(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum()
}

fn normalize(points: &mut [[f32; 3]]) -> ([f32; 3], [f32; 3]) {
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

    for k in 0..3 {
        let width = aabb_max[k] - aabb_min[k];
        for point in &mut *points {
            point[k] = (point[k] - aabb_min[k]) / width;
        }
    }

    (aabb_min, aabb_max)
}

fn denormalize(points: &mut [[f32; 3]], aabb_min: &[f32; 3], aabb_max: &[f32; 3]) {
    for k in 0..3 {
        let width = aabb_max[k] - aabb_min[k];
        for point in &mut *points {
            point[k] = point[k] * width + aabb_min[k];
        }
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn morton_index(point: &[f32; 3], permutation: [u8; 3]) -> u32 {
    const WIDTH: u32 = u32::BITS / 3;
    const MASK: u32 = 0b001_001_001_001_001_001_001_001_001_001;
    let i0 = permutation[0] as usize;
    let i1 = permutation[1] as usize;
    let i2 = permutation[2] as usize;

    let x0 = (point[i0] * WIDTH as f32) as u32;
    let x1 = (point[i1] * WIDTH as f32) as u32;
    let x2 = (point[i2] * WIDTH as f32) as u32;

    x0.pdep(MASK) | x1.pdep(MASK << 1) | x2.pdep(MASK << 2)
}
