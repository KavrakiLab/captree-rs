from dataclasses import InitVar, dataclass, field
from functools import partial
from itertools import product
from pathlib import Path
from time import time

import numpy as np
from fire import Fire
from grapeshot.assets import ROBOTS
from grapeshot.model.camera import Camera, process_camera
from grapeshot.model.group import GroupABC
from grapeshot.model.robot import process_srdf
from grapeshot.model.world import Skeleton, World
from grapeshot.simulators.pybullet import PyBulletSimulator
from grapeshot.util.mesh import mesh_to_sampled_pointcloud, pointcloud_to_mesh
from grapeshot.util.worldpool import WorldPool
from h5py import File
from numpy.typing import NDArray


@dataclass(slots=True)
class RobotParams:
  name: str
  sensor_link: str
  sensor_group: str


@dataclass(slots=True)
class PointcloudParams:
  problem_name: str
  resolution: tuple[float, float]
  num_points: int


@dataclass(slots=True)
class SampleWorld(World):
  camera: Camera = field(init=False)
  sensor_group: GroupABC = field(init=False)
  robot_params: InitVar[RobotParams | None] = None
  pointcloud_params: InitVar[PointcloudParams | None] = None

  def __post_init__(self, robot_params, sample_params):
    World.__post_init__(self)
    if robot_params is not None and sample_params is not None:
      # Initial setup
      robot = ROBOTS[robot_params.name]
      robot_skel = self.add_skeleton(robot.urdf, name="robot")
      groups = process_srdf(robot_skel, robot.srdf)
      self.add_skeleton(robot.problems[sample_params.problem_name].environment, name="environment")
      # NOTE: I assume we do not need to load the plane here, since we don't care about
      # visualization and aren't running the sim (so gravity doesn't matter)
      self.setup_collision_filter()
      self.camera = process_camera(robot.cameras[robot_params.sensor_link], robot_skel)
      self.sensor_group = groups[robot_params.sensor_group]

  @property
  def robot(self) -> Skeleton:
    return self.get_skeleton("robot")

  @property
  def environment(self) -> Skeleton:
    return self.get_skeleton("environment")


def sample_pointcloud(q: NDArray, *, world: SampleWorld) -> NDArray:
  world.set_group_positions(world.sensor_group, q)
  # The ignore here is because we know the sim will be a PyBulletSim, which supports skeleton
  # filtering
  return world.sim.take_image(world.camera, [world.robot]).point_cloud  # type: ignore


def gather_pointcloud(num_points: int, *pointcloud_samples) -> NDArray:
  aggregate_cloud = np.concatenate(pointcloud_samples)
  # TODO: Parameterize the resolution
  cloud_mesh = pointcloud_to_mesh(aggregate_cloud, 0.005)
  return mesh_to_sampled_pointcloud(cloud_mesh, num_points)


def main(
    problem_name: str,
    robot_name: str = "fetch",
    sensor_link: str = "head",
    sensor_group: str = "head_with_torso",
    head_resolution: float = 0.15,
    torso_resolution: float = 1.0,
    num_points: int = 400_000,
    output_path: Path | None = None,
    show_progress: bool = True,
):
  print("Loading parameters...")
  robot_params = RobotParams(robot_name, sensor_link, sensor_group)
  pointcloud_params = PointcloudParams(
      problem_name, (torso_resolution, head_resolution), num_points
  )
  print("Setting up robot and scene...")
  world = SampleWorld(PyBulletSimulator(), {}, robot_params, pointcloud_params)

  def setup_clone(w: SampleWorld) -> SampleWorld:
    w.camera = world.camera
    w.sensor_group = world.sensor_group
    return w

  with WorldPool(world, world_setup_fn=setup_clone, show_progress=show_progress) as wp:
    print(
        f"Building pointcloud for {pointcloud_params.problem_name} with {pointcloud_params.num_points} points..."
    )
    start = time()
    torso_resolution, head_resolution = pointcloud_params.resolution
    mins = world.sensor_group.mins
    maxes = world.sensor_group.maxes
    pointcloud = wp(sample_pointcloud)(
        map_over=[[
            np.array([x, y, z]) for x,
            y,
            z in product(
                np.arange(mins[0], maxes[0], torso_resolution),
                np.arange(mins[1], maxes[1], head_resolution),
                np.arange(mins[2], maxes[2], head_resolution)
            )
        ]],
        gather=partial(gather_pointcloud, pointcloud_params.num_points)
    ).result()
    end = time()
    print(f"Pointcloud time: {end-start}s", f"Pointcloud shape: {pointcloud.shape}")
  if output_path is None:
    output_path = Path(
        "_".join([
            f"{robot_params.name}",
            f"{pointcloud_params.problem_name}",
            f"r{pointcloud_params.resolution[0]}x{pointcloud_params.resolution[1]}",
            f"n{pointcloud_params.num_points}.h5",
        ])
    )
  print(f"Saving pointcloud to {output_path}")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with File(output_path, "w") as output_file:
    pointcloud_group = output_file.create_group("pointcloud")
    pointcloud_dset = pointcloud_group.create_dataset(
        "points",
        (
            pointcloud_params.resolution[0],
            pointcloud_params.resolution[1],
            pointcloud_params.num_points
        ),
        dtype=np.float32
    )
    pointcloud_dset[...] = pointcloud


if __name__ == "__main__":
  Fire(main)
