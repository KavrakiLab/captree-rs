use std::path::Path;

use hdf5::{File, H5Type, Result};

// TODO: Extend with radius for variable point size?
#[derive(H5Type, Debug, Clone)]
#[repr(C)]
struct Point {
    x: f32,
    y: f32,
    z: f32,
}

impl From<Point> for [f32; 3] {
    fn from(p: Point) -> Self {
        [p.x, p.y, p.z]
    }
}

/// Load a pointcloud as a vector of 3-d float arrays from a HDF5 file located at `pointcloud_path`.
pub fn load_pointcloud(pointcloud_path: &Path) -> Result<Vec<[f32; 3]>> {
    let file = File::open(pointcloud_path)?;
    let dataset = file.dataset("pointcloud/points")?;
    let points = dataset.read_1d::<Point>()?;
    Ok(points.mapv(|p| p.into()).into_raw_vec())
}
