#![feature(portable_simd)]

use std::simd::Simd;

use afftree::{AffordanceTree, SquaredEuclidean};

#[allow(unused)]
struct AffTree3xf32(AffordanceTree<3, f32, u32, SquaredEuclidean, f32>);

fn afftree_3xf32_new(points: &[[f32; 3]], r_min: f32, r_max: f32) -> Box<AffTree3xf32> {
    Box::new(AffTree3xf32(
        AffordanceTree::new(
            points,
            (r_min * r_min, r_max * r_max),
            &mut rand::thread_rng(),
        )
        .unwrap(),
    ))
}

fn afftree_3xf32_collides(t: &AffTree3xf32, center: &[f32; 3], radius: f32) -> bool {
    t.0.collides(center, radius * radius)
}

fn afftree_3xf32_collides_simd_x8(
    t: &AffTree3xf32,
    centers: &[[f32; 8]; 3],
    radii: [f32; 8],
) -> bool {
    let r = Simd::from_array(radii);
    t.0.collides_simd(&centers.map(Simd::from_array), r * r)
}

#[cxx::bridge]
mod ffi {
    #[namespace = "vamp::ffi"]
    extern "Rust" {
        type AffTree3xf32;

        fn afftree_3xf32_new(points: &[[f32; 3]], r_min: f32, r_max: f32) -> Box<AffTree3xf32>;
        fn afftree_3xf32_collides(t: &AffTree3xf32, center: &[f32; 3], radius: f32) -> bool;
        fn afftree_3xf32_collides_simd_x8(
            t: &AffTree3xf32,
            centers: &[[f32; 8]; 3],
            radii: [f32; 8],
        ) -> bool;
    }
}
