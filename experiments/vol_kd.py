from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum, auto
from math import isclose
from sys import argv

import h5py
import numpy as np
from numpy.typing import NDArray


class Axis(Enum):
  X = auto()
  Y = auto()
  Z = auto()

  def next(self) -> Axis:
    if self == Axis.X:
      return Axis.Y

    if self == Axis.Y:
      return Axis.Z

    return Axis.X


@dataclass(slots=True)
class Bounds:
  x: NDArray
  y: NDArray
  z: NDArray

@dataclass(slots=True)
class Node:
  bounds: Bounds
  closest_dists: Bounds
  axis: Axis
  split_val: float
  rad_arr: InitVar[NDArray]
  inner: Bounds = field(init=False)
  parent: Node | None = None
  is_leaf: bool = False
  children: list[Node] = field(default_factory=list)
  affordance_volumes: list[Node] = field(default_factory=list)

  def __post_init__(self, rad_arr: NDArray):
    x = self.bounds.x - self.closest_dists.x + rad_arr
    y = self.bounds.y - self.closest_dists.y + rad_arr
    z = self.bounds.z - self.closest_dists.z + rad_arr
    self.inner = Bounds(x, y, z)


  def __repr__(self) -> str:
    return f"Node(b={self.bounds}, d={self.closest_dists}, a={self.axis}, s={self.split_val}, l={self.is_leaf})"


@dataclass(slots=True)
class Points:
  x: NDArray
  y: NDArray
  z: NDArray


def split(data: Points, axis: Axis) -> tuple[float, float, Points, Points]:
  if axis == Axis.X:
    dim_vals = data.x
  elif axis == Axis.Y:
    dim_vals = data.y
  else:
    dim_vals = data.z

  permutation = dim_vals.argsort()
  # NOTE: Assumes data are always a power of 2
  low_indices = permutation[:permutation.size // 2]
  high_indices = permutation[permutation.size // 2:]
  low_bound: int = low_indices[-1]
  high_bound: int = high_indices[0]
  split_val = (dim_vals[high_bound] + dim_vals[low_bound]) / 2.0
  split_dist = (dim_vals[high_bound] - dim_vals[low_bound]) / 2.0
  low_points = Points(data.x[low_indices], data.y[low_indices], data.z[low_indices])
  high_points = Points(data.x[high_indices], data.y[high_indices], data.z[high_indices])
  return split_val, split_dist, low_points, high_points


def build_kd(data: Points, max_rad: float) -> Node:
  rad_arr = np.array((-max_rad, max_rad))
  axis = Axis.X
  # Initial sort on axis
  if axis == Axis.X:
    dim_vals = data.x
  elif axis == Axis.Y:
    dim_vals = data.y
  else:
    dim_vals = data.z

  permutation = dim_vals.argsort()
  data.x = data.x[permutation]
  data.y = data.y[permutation]
  data.z = data.z[permutation]

  split_val, split_dist, low_set, high_set = split(data, axis)
  root = Node(
      Bounds(
          np.hstack((data.x.min(), data.x.max())),
          np.hstack((data.y.min(), data.y.max())),
          np.hstack((data.z.min(), data.z.max()))
      ),
      Bounds(np.zeros(2), np.zeros(2), np.zeros(2)),
      axis,
      split_val, rad_arr
  )
  queue = [(root, split_dist, low_set, high_set)]
  while queue:
    p, d, l, h = queue.pop()
    axis = axis.next()
    l_node = make_child(p, d, axis, rad_arr, True)
    if l.x.size == 1:
      l_node.is_leaf = True
      
    p.children.append(l_node)
    if not l_node.is_leaf:
      l_split_val, l_split_dist, l_low_set, l_high_set = split(l, axis)
      l_node.split_val = l_split_val
      queue.append((l_node, l_split_dist, l_low_set, l_high_set))

    h_node = make_child(p, d, axis, rad_arr, False)
    if h.x.size == 1:
      h_node.is_leaf = True
    p.children.append(h_node)
    if not h_node.is_leaf:
      h_split_val, h_split_dist, h_low_set, h_high_set = split(h, axis)
      h_node.split_val = h_split_val
      queue.append((h_node, h_split_dist, h_low_set, h_high_set))

  return root

def axis_value(n: Node, axis: Axis) -> float:
  if axis == Axis.X:
    return (n.bounds.x[1] + n.bounds.x[0]) / 2.0
  elif axis == Axis.Y:
    return (n.bounds.y[1] + n.bounds.y[0]) / 2.0
  else:
    return (n.bounds.z[1] + n.bounds.z[0]) / 2.0

def forward_volumes(tree: Node):
  leaf_count = 0
  leaf_vols = 0
  queue = [(tree.children[0], tree.children[1], tree.affordance_volumes)]
  #print(tree)
  while queue:
    l_sibling, r_sibling, parent_volumes = queue.pop()
    #print("L", l_sibling)
    #print("R", r_sibling)

    # Filter the parent volumes, refine to current tree level
    l_refined = refine_volumes(l_sibling, l_sibling.parent.axis, parent_volumes)
    r_refined = refine_volumes(r_sibling, r_sibling.parent.axis, parent_volumes)

    # Do the siblings intersect?
    if volumes_intersect(l_sibling, r_sibling):
      l_refined.append(r_sibling)
      r_refined.append(l_sibling)
  
    # l_refined.sort(key = lambda n: axis_value(n, l_sibling.parent.axis))
    # r_refined.sort(key = lambda n: axis_value(n, r_sibling.parent.axis))
    if not l_sibling.is_leaf:
      queue.append((l_sibling.children[0], l_sibling.children[1], l_refined))
    else:
      l_sibling.affordance_volumes = l_refined
      leaf_count += 1
      leaf_vols += len(l_sibling.affordance_volumes)
      if leaf_count % 1000 == 0:
        print(f"Processed {leaf_count} leaves")

    if not r_sibling.is_leaf:
      queue.append((r_sibling.children[0], r_sibling.children[1], r_refined))
    else:
      r_sibling.affordance_volumes = r_refined
      leaf_count += 1
      leaf_vols += len(r_sibling.affordance_volumes)
      if leaf_count % 1000 == 0:
        print(f"Processed {leaf_count} leaves")
    
  print(f"{leaf_count} total leaves")
  print(f"Average number of volumes: {leaf_vols / leaf_count}")

def volumes_intersect(l_node: Node, r_node: Node) -> bool:
  l_vol = l_node.inner
  r_vol = r_node.inner
  if l_vol.x[0] > r_vol.x[1] or r_vol.x[0] > l_vol.x[1]:
    return False
  
  if l_vol.y[0] > r_vol.y[1] or r_vol.y[0] > l_vol.y[1]:
    return False
  
  if l_vol.z[0] > r_vol.z[1] or r_vol.z[0] > l_vol.z[1]:
    return False
  
  return True


def refine_volumes(n: Node, parent_axis: Axis, parent_volumes: list[Node]) -> list[Node]:
  refined = []
  # TODO: Binary search this - parent_volumes can be sorted along parent split axis
  for v in parent_volumes:
    if not volumes_intersect(n, v):
      continue

    if v.is_leaf:
      refined.append(v)
    else:
      for c in v.children:
        if volumes_intersect(n, c):
          refined.append(c)

  return refined


def make_child(parent: Node, parent_split_dist: float, axis: Axis, rad_arr: NDArray, low_child: bool) -> Node:
  return Node(
      update_bounds(parent, low_child),
      update_dists(parent, parent_split_dist, low_child),
      axis,
      np.infty,
      rad_arr,
      parent=parent
  )


def update_bounds(node: Node, lower: bool) -> Bounds:
  return _update_bounds(node, lower, node.bounds, node.split_val)


def update_dists(node: Node, split_dist: float, lower: bool) -> Bounds:
  if not lower:
    split_dist = -split_dist

  return _update_bounds(node, lower, node.closest_dists, split_dist)


def _update_bounds(node: Node, lower: bool, bounds: Bounds, val: float) -> Bounds:
  bounds = Bounds(bounds.x.copy(), bounds.y.copy(), bounds.z.copy())
  if node.axis == Axis.X:
    dim_bounds = bounds.x
  elif node.axis == Axis.Y:
    dim_bounds = bounds.y
  else:
    dim_bounds = bounds.z

  if lower:
    dim_bounds[1] = val
  else:
    dim_bounds[0] = val

  return bounds

rng = np.random.default_rng()
MAX_SIZE = 2**11
if len(argv) > 1:
  points_data = h5py.File(argv[1])["pointcloud/points"]
  x = points_data[:, 0]
  y = points_data[:, 1]
  z = points_data[:, 2]
  print(points_data.size, x.size, y.size, z.size)
  if points_data.size > MAX_SIZE:
    indices = rng.choice(x.size, size=MAX_SIZE, replace=False)
    x = x[indices]
    y = y[indices]
    z = z[indices]
else:
  x = rng.uniform(-5.0, 5.0, MAX_SIZE)
  y = rng.uniform(-5.0, 5.0, MAX_SIZE)
  z = rng.uniform(0.0, 5.0, MAX_SIZE)

print(x.size, y.size, z.size)
points = Points(x, y, z)
tree = build_kd(points, 0.01)
print(tree)
forward_volumes(tree)