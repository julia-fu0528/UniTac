
import argparse
import sys
from datetime import datetime
import json
import numpy as np
from natsort import natsorted
# import pybullet as p
import matplotlib.pyplot as plt
import os
import trimesh
import trimesh.viewer
from pathlib import Path
import xml.etree.ElementTree as ET
import random
from collections import defaultdict
from urdfpy import URDF
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
import open3d as o3d
from src.utils.helpers import sample_points_from_mesh
from src.utils.visualize_mesh import create_viewing_parameters, visualize_with_camera

# from pykdl_utils.kdl_kinematics import KDLKinematics
def load_joint_torques(torque_path):
    # load the npy data
    # torque_path = "data/touch_back.npy"
    state = np.load(torque_path, allow_pickle=True)
    state_dict = {}
    torque_dict = {}
    for i in range(len(state)):
        state_dict[i] = state[i].kinematic_state.joint_states
        torque_dict[i] = []
        for joint in state_dict[i]:
            joint_name = getattr(joint, 'name', None)
            if joint_name is not None:
                if not joint_name.startswith("arm"):
                    # Store both the joint name and load value in the dictionary
                    torque_dict[i].append({
                        'name': joint_name,
                        'load': joint.load.value  # Assuming load has a 'value' attribute
                    })

    # Determine the number of entries and the maximum number of joints to dynamically handle varying joint counts
    num_entries = len(torque_dict)
    num_joints = max(len(torque_dict[i]) for i in torque_dict)

    # Initialize torque_data with NaN values in case different entries have different joint counts
    torque_data = np.full((num_entries, num_joints), np.nan, dtype=float)
    joint_names = []

    # Fill torque_data with torque values, ignoring missing joints for each entry
    for i in range(num_entries):
        for j, joint in enumerate(torque_dict[i]):
            torque_data[i, j] = joint['load']
            if i == 0:
                joint_names.append(joint['name'])
    # print(f"torque dict:{torque_dict}")
    return torque_data, num_entries, num_joints, joint_names


def load_joint_positions(joint_path):
    state = np.load(joint_path, allow_pickle=True)
    state_dict = {}
    joint_pos_dict = {}
    for i in range(len(state)):
        state_dict[i] = state[i].kinematic_state.joint_states
        joint_pos_dict[i] = []
        for joint in state_dict[i]:
            joint_name = getattr(joint, 'name', None)
            if joint_name is not None:
                if not joint_name.startswith("arm"):
                    # Store both the joint name and load value in the dictionary
                    joint_pos_dict[i].append({
                        'name': joint_name,
                        'angle': joint.position.value  # Assuming load has a 'value' attribute
                    })

    # Determine the number of entries and the maximum number of joints to dynamically handle varying joint counts
    num_entries = len(joint_pos_dict)
    num_joints = max(len(joint_pos_dict[i]) for i in joint_pos_dict)

    # Initialize torque_data with NaN values in case different entries have different joint counts
    joint_pos_data = np.full((num_entries, num_joints), np.nan, dtype=float)
    joint_names = []

    # Fill torque_data with torque values, ignoring missing joints for each entry
    for i in range(num_entries):
        for j, joint in enumerate(joint_pos_dict[i]):
            joint_pos_data[i, j] = joint['angle']
            if i == 0:
                joint_names.append(joint['name'])
    # print(f"torque dict:{torque_dict}")
    return joint_pos_data, num_entries, num_joints, joint_names


def vis_joint_pos_delta(joint_pos_path1, joint_pos_path2):
    joint_pos_data1, num_entries1, num_joints1, joint_names1 = load_joint_positions(joint_pos_path1)
    joint_pos_data2, num_entries2, num_joints2, joint_names2 = load_joint_positions(joint_pos_path2)

    # Create the plot
    plt.figure(figsize=(10, 6))
    num_entries = min(num_entries1, num_entries2)
    print(f"num_entries1: {num_entries1}")
    print(f"num_entries2: {num_entries2}")
    time_steps = np.arange(num_entries)
    joint_pos_data1 = joint_pos_data1[:num_entries]
    joint_pos_data2 = joint_pos_data2[:num_entries]
    print(f"joint_pos_data1: {joint_pos_data1}")
    print(f"joint_pos_data2: {joint_pos_data2}")
    joint_pos_data = joint_pos_data1 - joint_pos_data2
    print(f"joint_pos_data: {joint_pos_data}")

    for j in range(num_joints1):
        plt.plot(time_steps, joint_pos_data[:, j], label=f'Joint {joint_names1[j]}')

    # Add labels and legend
    plt.xlabel("Time")
    plt.ylabel("Joint Position")
    plt.title(f"Delta Joint Position Over Time for Each Joint for {joint_pos_path1.split('.')[0]} and {joint_pos_path2.split('.')[0]}")
    plt.legend()
    plt.grid(True)

    output_dir = "vis/joint_pos/1203"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{joint_pos_path1.split('.')[0].split('/')[-1]}-{joint_pos_path2.split('.')[0].split('/')[-1]}.png")
    print(f"save_path: {save_path}")
    plt.savefig(save_path, format="png", dpi=300)  # Save as a PNG file with 300 dpi resolution

    plt.show()


def vis_joint_torques(torque_path_list):
    # torque_data, num_entries, num_joints, joint_names = load_joint_torques(torque_path)
    all_torque_data = None
    total_entries = 0
    for torque_path in torque_path_list:
        torque_data, num_entries, num_joints, joint_names = load_joint_torques(torque_path)
        total_entries += num_entries
        if all_torque_data is None:
            all_torque_data = torque_data
        else:
            all_torque_data = np.vstack((all_torque_data, torque_data))
    # Create the plot
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(total_entries)

    for j in range(num_joints):
        plt.plot(time_steps, all_torque_data[:, j], label=f'Joint {joint_names[j]}')

    # Add labels and legend
    plt.xlabel("Time")
    plt.ylabel("Torque")
    # plt.title(f"Torque Over Time for Each Joint for {torque_path.split('.')[0]}")
    plt.title(f"Torque Over Time for Each Joint")
    plt.legend()
    plt.grid(True)

    output_dir = f"vis/1219/{torque_path_list[0].split('.')[0].split('/')[-2]}"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{torque_path.split('.')[0].split('/')[-1]}_torque.png")
    print(f"save_path: {save_path}")
    plt.savefig(save_path, format="png", dpi=300)  # Save as a PNG file with 300 dpi resolution

    plt.show()


def vis_joint_pos(joint_pos_path_list):
    all_joint_pos_data = None
    total_entries = 0
    num_joints = 0
    for joint_pos_path in joint_pos_path_list:
        joint_pos_data, num_entries, num_joints, joint_names = load_joint_positions(joint_pos_path)
        total_entries += num_entries
        if all_joint_pos_data is None:
            all_joint_pos_data = joint_pos_data
        else:
            all_joint_pos_data = np.vstack((all_joint_pos_data, joint_pos_data))
    # Create the plot
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(total_entries)

    for j in range(num_joints):
        plt.plot(time_steps, all_joint_pos_data[:, j], label=f'Joint {joint_names[j]}')

    # Add labels and legend
    plt.xlabel("Time")
    plt.ylabel("Joint Position")
    # plt.title(f"Joint Position Over Time for Each Joint for {joint_pos_path.split('.')[0]}")
    plt.title("Joint Position Over Time for Each Joint")
    plt.legend()
    plt.grid(True)
    output_dir = f"vis/1219/{joint_pos_path_list[0].split('.')[0].split('/')[-2]}"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{joint_pos_path_list[0].split('.')[0].split('/')[-1]}_joint.png")
    print(f"save_path: {save_path}")
    plt.savefig(save_path, format="png", dpi=300)  # Save as a PNG file with 300 dpi resolution

    plt.show()

def load_spot():
    robot = URDF.load('spot_description/spot.urdf')
    robot.show()

def find_closest_vertices(mesh, positions, num_points = 10000): 
    # Build a KD-tree for efficient nearest-neighbor search
    # if not isinstance(mesh_vertices, np.ndarray):
    #     mesh_vertices = np.asarray(mesh_vertices)
    sampled_points = sample_points_from_mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), num_points)

    # # visualize the points 
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(sampled_points)
    # o3d.visualization.draw_geometries([pcd])
    
    # kdtree = KDTree(mesh_vertices)
    kdtree = KDTree(sampled_points)
    
    # Query the closest vertex for each position
    closest_distances, closest_indices = kdtree.query(positions)
    
    # Extract the closest vertex positions
    # closest_vertices = mesh_vertices[closest_indices]
    closest_vertices = sampled_points[closest_indices]

    return closest_vertices, closest_indices


def create_red_markers(marker_positions, radius=0.01):
    """Create red spheres as markers at specified positions."""
    red_markers = []
    for pos in marker_positions:
        # Create a sphere and set its position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.paint_uniform_color([1, 0, 0])  # Red color
        sphere.translate(pos)
        red_markers.append(sphere)
    return red_markers


def add_red_dots(mesh, num_samples, z_threshold = 0.07, radius=0.01, output_path=None):
    ## OPEN3D NEAREST NEIGHBOR
    ## OPEN3D POINTCLOUD SAMPLING: RANDOM SAMPLING/POISSON DISK SAMPLING (BETTER)
    ## STORE JOINT ANGLES(POSE) IN ADDITION TO JOINT TORQUE
    ## REAL-TIME PREDICTION
    point_cloud = mesh.sample_points_poisson_disk(number_of_points=1000)
    points = np.asarray(point_cloud.points)
    points_to_filter_z = points[
        (points[:, 2] < z_threshold) & 
        (points[:, 1] > -0.09) & (points[:, 1] < 0.09) & 
        (points[:, 0] > -0.4) & (points[:, 0] < 0.40)]
    mask = np.ones(len(points), dtype=bool)  # Start with all points included
    for point in points_to_filter_z:
        mask = mask & ~np.all(points == point, axis=1)  # Exclude matching rows

    filtered_points = points[mask]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    resampled_pcd = filtered_pcd.random_down_sample(num_samples/ len(filtered_points))
    pcd_points = np.asarray(resampled_pcd.points)

    return pcd_points
    return create_red_markers(pcd_points, radius), pcd_points


def combine_meshes_o3d(mesh_list):
    combined_vertices = []
    combined_triangles = []
    vertex_offset = 0

    for mesh in mesh_list:
        if not isinstance(mesh, o3d.geometry.TriangleMesh):
            raise TypeError("All elements in the mesh list must be Open3D TriangleMesh objects.")
        
        # Add vertices and triangles from each mesh
        combined_vertices.append(np.asarray(mesh.vertices))
        combined_triangles.append(np.asarray(mesh.triangles) + vertex_offset)
        
        # Update the vertex offset for the next mesh
        vertex_offset += len(mesh.vertices)

    # Create a new TriangleMesh from the combined data
    combined_mesh = o3d.geometry.TriangleMesh()
    combined_mesh.vertices = o3d.utility.Vector3dVector(np.vstack(combined_vertices))
    combined_mesh.triangles = o3d.utility.Vector3iVector(np.vstack(combined_triangles))

    return combined_mesh

def load_spot_with_red_dots():
    # get the folder path of this file
    folder_path = Path(__file__).parent
    # Paths
    scene_path = os.path.join(folder_path, 'spot_description/meshes/base/visual/body.obj')
    updated_path = os.path.join(folder_path, 'spot_description/meshes/base/visual/body_updated.obj')

    scene = trimesh.load(scene_path, force='mesh')
    if hasattr(scene.visual, 'material'):
        scene.visual.material.alpha = 1.0  # Fully opaque

    # Convert to opaque vertex colors if no material
    if hasattr(scene.visual, 'to_color'):
        scene.visual = scene.visual.to_color()
        scene.visual.vertex_colors[:, 3] = 255  # Fully opaque
    if isinstance(scene, trimesh.Scene):
        # Combine all geometry into a single mesh if needed
        mesh = trimesh.util.concatenate([scene.geometry[g] for g in scene.geometry])
    else:
        mesh = scene

    # Convert Trimesh to Open3D
    o3d_mesh_list = convert_trimesh_to_open3d([mesh])
    if not o3d_mesh_list:
        raise ValueError("Failed to convert Trimesh to Open3D.")
    # Debug Open3D Mesh
    o3d_mesh = o3d_mesh_list[0]

    # Visualize in Open3D
    # o3d.visualization.draw_geometries([o3d_mesh])
    # sampled_points, face_indices = trimesh.sample.sample_surface_even(mesh, count=1000)
    vertices = []
    faces = []

    with open(scene_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Vertex line
                vertices.append([float(v) for v in line.split()[1:]])
            elif line.startswith('f '):  # Face line
                faces.append([int(f.split('/')[0]) - 1 for f in line.split()[1:]])
    vertices = trimesh.points.PointCloud(vertices)  # Convert to a Trimesh PointCloud for easier spatial operations

    # Group vertices by spatial orientation
    groups = defaultdict(list)
    for i, (x, y, z) in enumerate(vertices.vertices):
        if z > 0.08:  # Top
            groups['top'].append(i)
        elif x > 0.4:  # Front
            groups['front'].append(i)
        elif x < -0.41:  # Back
            groups['back'].append(i)
        elif y > 0.1:  # Right
            groups['right'].append(i)
        elif y < -0.1:  # Left
            groups['left'].append(i)

    # Sample 3 vertices from each group
    selected_vertices = []
    chosen_vertices = []

    # for group_name, indices in groups.items():
    #     # Get the vertex coordinates for the current group
    #     group_vertices = np.array([vertices.vertices[i] for i in indices])
    #     print(f"group_vertices: {len(group_vertices)}")
    #     # Determine the axis to sort by based on the face
    #     if group_name in ['top']:
    #         sort_axis = 0  # Sort by X (horizontal spread)
    #     elif group_name in ['left', 'right']:
    #         sort_axis = 1  # Sort by Y (depth spread)
    #     elif group_name in ['front', 'back']:
    #         sort_axis = 2  # Sort by Z (vertical spread)
    #     else:
    #         continue  # Skip if no clear sorting axis is defined

    #     # Sort vertices along the selected axis
    #     sorted_indices = group_vertices[:, sort_axis].argsort()
    #     sorted_group = group_vertices[sorted_indices]

    #     # Evenly select points from the sorted list
    #     num_points = 50  # Number of points to pick
    #     step = max(1, len(sorted_group) // num_points)  # Compute step size
    #     evenly_selected_indices = range(0, len(sorted_group), step)  # Pick points evenly


    #     # Map back to the original vertex indices
    #     # index
    #     selected_vertices.extend([indices[sorted_indices[idx]] for idx in evenly_selected_indices])
    #     # vertex
    #     chosen_vertices.extend([sorted_group[idx] for idx in evenly_selected_indices])

    chosen_vertices = np.array([[0.5, 0.0, -0.1], [-0.42, 0.0, 0.0], [0.0, 0.11, 0.0], [0.0, -0.11, 0.0]])
    robot = URDF.load(os.path.join(folder_path, 'spot_description/spot.urdf'))
    robot_meshes = []
    for link in robot.links:
        if link.visuals:
            for visual in link.visuals:
                if visual.geometry.mesh is not None:
                    # Load the mesh file (e.g., STL or OBJ)
                    mesh_path = os.path.join(folder_path, "spot_description", visual.geometry.mesh.filename)
                    o3d_mesh = convert_trimesh_to_open3d([trimesh.load(mesh_path, force='mesh')])[0]
                    if o3d_mesh== []:
                        print(f"Failed to load mesh from {mesh_path}")
                        continue
                    o3d_mesh.compute_vertex_normals()
                    robot_meshes.append(o3d_mesh)
    o3d.visualization.draw_geometries(robot_meshes)
    for chosen_vertex in chosen_vertices.tolist():
        # Add red dots to the selected vertices
        # red_dots = add_red_dots(mesh, [chosen_vertex], updated_path)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        sphere.compute_vertex_normals()

        sphere.paint_uniform_color([1, 0, 0])  # Red
        sphere.translate(chosen_vertex)
        geometry_list = [sphere] + robot_meshes
        o3d.visualization.draw_geometries(geometry_list)


        # print(f"Updated file written to {updated_path}")
        # robot.show()

def convert_trimesh_to_open3d(trimesh_fk):
        o3d_meshes = []
        tm_to_o3d_map = {}
        for tm in trimesh_fk:
            o3d_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(tm.vertices.copy()),
                triangles=o3d.utility.Vector3iVector(tm.faces.copy())
            )
            o3d_mesh.compute_vertex_normals()
            try:
                o3d_mesh.paint_uniform_color(tm.visual.material.main_color[:3] / 255.)
            except AttributeError:
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(tm.visual.vertex_colors[:, :3] / 255.)

            o3d_mesh.transform(trimesh_fk[tm])
            # self.prev_fks.append(trimesh_fk[tm]) # world -> T1

            o3d_meshes.append(o3d_mesh)
            tm_to_o3d_map[tm] = (o3d_mesh, np.array(trimesh_fk[tm]))
        return o3d_meshes, tm_to_o3d_map

def update_open3d(trimesh_fk, o3d_meshes):
        o3d_meshes = []
        tm_to_o3d_map = {}
        for tm in trimesh_fk:
            o3d_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(tm.vertices.copy()),
                triangles=o3d.utility.Vector3iVector(tm.faces.copy())
            )
            o3d_mesh.compute_vertex_normals()
            try:
                o3d_mesh.paint_uniform_color(tm.visual.material.main_color[:3] / 255.)
            except AttributeError:
                o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(tm.visual.vertex_colors[:, :3] / 255.)

            o3d_mesh.transform(trimesh_fk[tm])
            # self.prev_fks.append(trimesh_fk[tm]) # world -> T1

            o3d_meshes.append(o3d_mesh)
            tm_to_o3d_map[tm] = o3d_mesh
        return o3d_meshes, tm_to_o3d_map


# def update_meshes_with_fk(o3d_meshes, trimesh_fk):
#     for i, (mesh, new_geom) in enumerate(zip(o3d_meshes, trimesh_fk)):
#         if new_geom and new_geom.vertices is not None:
#             vertices = np.asarray(new_geom.vertices)
#             mesh.vertices = o3d.utility.Vector3dVector(vertices)
#             mesh.compute_vertex_normals()

def update_meshes_with_fk(base_meshes, transforms):
    points = []
    for mesh, transform in zip(base_meshes, transforms):
        mesh_points = np.asarray(mesh.vertices)
        transformed_points = (transform @ np.hstack([mesh_points, np.ones((len(mesh_points), 1))]).T).T[:, :3]
        points.extend(transformed_points)
    return np.array(points)

def load_robot_from_urdf(urdf_path):
    """Load robot meshes from a URDF file and convert to Open3D objects."""
    robot = URDF.load(urdf_path)
    robot_meshes = []

    # Convert URDF links to Open3D meshes
    for link in robot.links:
        if link.visuals:
            for visual in link.visuals:
                if visual.geometry.mesh is not None:
                    # Resolve the path to the mesh file
                    mesh_file = os.path.join(os.path.dirname(urdf_path), visual.geometry.mesh.filename)
                    if not os.path.exists(mesh_file):
                        print(f"Mesh file not found: {mesh_file}")
                        continue
                    
                    # Load the mesh with Open3D
                    o3d_mesh = convert_trimesh_to_open3d([trimesh.load(mesh_file, force='mesh')])[0]
                    if o3d_mesh.is_empty():
                        print(f"Failed to load mesh from {mesh_file}")
                        continue
                    
                    o3d_mesh.compute_vertex_normals()
                    
                    # Apply transformation if origin is defined
                    # if visual.origin is not None:
                    #     transform = np.eye(4)
                    #     transform[:3, :3] = visual.origin.rotation
                    #     transform[:3, 3] = visual.origin.xyz
                    #     o3d_mesh.transform(transform)
                    
                    robot_meshes.append(o3d_mesh)
    
    return robot_meshes


def visualize_robot_with_markers(robot_meshes, markers):
    """Visualize the robot and red markers using Open3D."""
    geometry_list = robot_meshes + markers
    o3d.visualization.draw_geometries(geometry_list)

def compute_joint_transform(joint, joint_position):
    """
    Compute the transformation matrix for a given joint and its position.

    Args:
        joint (urdfpy.Joint): The joint object.
        joint_position (float): The position of the joint (angle for revolute, displacement for prismatic).

    Returns:
        np.ndarray: The 4x4 transformation matrix.
    """
    transform = np.eye(4)  # Initialize as identity matrix

    # Apply the fixed transform from the joint's origin (if defined)
    if joint.origin is not None:
        origin_transform = np.eye(4)
        origin_transform = joint.origin
        # origin_transform[:3, :3] = joint.origin.rotation  # Rotation matrix from origin
        # origin_transform[:3, 3] = joint.origin.xyz        # Translation vector from origin
        transform = transform @ origin_transform

    # Compute joint-specific transformation
    if joint.joint_type in ["revolute", "continuous"]:
        # Rotation about the joint axis
        axis = np.array(joint.axis)
        angle = joint_position
        rotation_transform = trimesh.transformations.rotation_matrix(angle, axis)
        transform = transform @ rotation_transform
    elif joint.joint_type == "prismatic":
        # Translation along the joint axis
        axis = np.array(joint.axis)
        translation_transform = np.eye(4)
        translation_transform[:3, 3] = axis * joint_position
        transform = transform @ translation_transform
    elif joint.joint_type == "fixed":
        # No additional transformation for fixed joints
        pass
    else:
        raise ValueError(f"Unsupported joint type: {joint.joint_type}")

    return transform

def compute_forward_kinematics(robot, joint_positions):
    """
    Compute forward kinematics for a robot defined in a URDF file.

    Args:
        robot (URDF): The loaded URDF object.
        joint_positions (dict): Dictionary mapping joint names to their positions.

    Returns:
        dict: A dictionary mapping link names to their FK transformation matrices.
    """
    # Initialize the base transform as the identity matrix
    base_transform = np.eye(4)
    link_transforms = {robot.base_link.name: base_transform}

    # Map each joint to its parent and child link names
    joint_map = {joint.child: joint for joint in robot.joints}

    # Recursive function to compute FK for each link
    def compute_transform_for_link(link_name, parent_transform):
        if link_name in joint_map:
            # Get the joint connecting to this link
            joint = joint_map[link_name]
            joint_position = joint_positions.get(joint.name, 0.0)
            # Compute the joint transformation
            # joint_transform = joint.get_transform(joint_position)
            joint_transform = compute_joint_transform(joint, joint_position)
            current_transform = parent_transform @ joint_transform
        else:
            # This is the base link or a link with no parent joint
            current_transform = parent_transform

        # Store the transform for the current link
        link_transforms[link_name] = current_transform

        # Find child links connected via joints
        for joint in robot.joints:
            if joint.parent == link_name:
                compute_transform_for_link(joint.child, current_transform)

    # Start computation from the base link
    compute_transform_for_link(robot.base_link.name, base_transform)

    return link_transforms

def prepare_trimesh_fk(robot, link_fk_transforms, folder = None):
    """
    Prepare trimesh objects and their transforms for convert_trimesh_to_open3d.
    Returns a dictionary with trimesh objects as keys and FK transforms as values.
    """
    trimesh_fk = {}
    folder_path = Path(__file__).parent
    meshes = []
    for link in robot.links:
        for visual in link.visuals:
            if visual.geometry.mesh:
                # Load the mesh
                mesh_file = visual.geometry.mesh.filename
                mesh = trimesh.load(os.path.join(folder_path, folder, mesh_file), force='mesh')
                if hasattr(mesh.visual, 'material'):
                    mesh.visual.material.alpha = 1.0  # Fully opaque

                # Convert to opaque vertex colors if no material
                if hasattr(mesh.visual, 'to_color'):
                    mesh.visual = mesh.visual.to_color()
                    num_vertices = mesh.vertices.shape[0]
                    if mesh.visual.vertex_colors.ndim == 1:
                        # If vertex_colors is 1-dimensional, expand it
                        vertex_colors = np.tile(mesh.visual.vertex_colors, (num_vertices, 1))
                        vertex_colors = vertex_colors.reshape(-1, 4)  # Ensure correct shape
                        mesh.visual.vertex_colors = vertex_colors

                    mesh.visual.vertex_colors[:, 3] = 255  # Fully opaque
                # Assign the FK transform
                transform = link_fk_transforms[link.name]
                trimesh_fk[mesh] = transform
                meshes.append(mesh)

    return trimesh_fk, meshes

def visualize_robot_with_markers(robot_meshes, marker_positions):
    """
    Visualize the robot with markers, showing each marker from the appropriate viewing angle
    
    Args:
        robot_meshes (list): List of Open3D meshes for the robot
        marker_positions (dict): Dictionary of marker positions keyed by location name
    """
    for place, position in marker_positions.items():
        # Create marker for current position
        marker = create_red_markers([position], radius=1.0)[0]
        
        # Create camera parameters facing the marker
        camera_params = create_viewing_parameters(position)
        
        # Combine geometries
        geometries = robot_meshes + [marker]
        
        print(f"Viewing {place} marker. Press Ctrl+C in terminal to proceed to next view.")
        
        # Visualize with specific camera view
        visualize_with_camera(geometries, camera_params)

def update_joints(joint_pos_path):
    robot = URDF.load('spot_description/spot.urdf')

    joint_angles, num_entries, num_joints, joint_names = load_joint_positions(joint_pos_path)
    print(f"joint_angles: {joint_angles.shape}")

    simplified_to_full_name = {
        'fl.hx': 'front_left_hip_x',
        'fr.hx': 'front_right_hip_x',
        'hl.hx': 'rear_left_hip_x',
        'hr.hx': 'rear_right_hip_x',
        'fl.hy': 'front_left_hip_y',
        'fr.hy': 'front_right_hip_y',
        'hl.hy': 'rear_left_hip_y',
        'hr.hy': 'rear_right_hip_y',
        'fl.kn': 'front_left_knee',
        'fr.kn': 'front_right_knee',
        'hl.kn': 'rear_left_knee',
        'hr.kn': 'rear_right_knee',
    }
    joint_positions = {joint.name: 0.0 for joint in robot.joints}

    for i, joint_name in enumerate(joint_names):
        print(f"joint_name: {robot.joint_map.keys()}")
        print(f"joint_name: {joint_name}")
        joint = robot.joint_map[simplified_to_full_name.get(joint_name)]

        if joint:
            # if joint.type == 'revolute':
            joint.position = joint_angles[0, i]
            joint_positions[joint.name] = joint_angles[0, i]
            # else:
                # print(f"Joint {joint_name} is not a revolute joint.")
        else:
            print(f"Joint {joint_name} not found in URDF.")

    link_fk_transforms = compute_forward_kinematics(robot, joint_positions)
    trimesh_fk = prepare_trimesh_fk(robot, link_fk_transforms)
    robot_meshes = convert_trimesh_to_open3d(trimesh_fk)
    o3d.visualization.draw_geometries(robot_meshes)
    



if __name__ == "__main__":

    load_joint_torques("data/gouger1209/stand_h2/67.npy")
    update_joints("data/gouger1209/stand_h2/67.npy")
    # vis_joint_torques(torque_path2)
    vis_joint_torques(["data/gouger1209/stand_h2/67.npy"])
    # all directories in data/gouger1209
    all_dir = natsorted(os.listdir("data/gouger1209"))
    all_dir = [os.path.join("data/gouger1209", dir) for dir in all_dir]
    all_files = []
    for directory in all_dir:
        files = os.listdir(directory)
        all_files.extend([os.path.join(directory, file) for file in files])
    print(f"len all_files: {len(all_files)}")
    no_contact_files = [f for f in all_files if "no_contact.npy" in f]
    print(f"no_contact_files: {no_contact_files}")
    # all files that are named 95.npy
    files_95 = [f for f in all_files if "95.npy" in f]
    files_26 = [f for f in all_files if "26.npy" in f]
    # vis_joint_pos(no_contact_files)
    # vis_joint_pos(files_95)
    vis_joint_pos(files_26)
    # vis_joint_torques(no_contact_files)
    # vis_joint_torques(files_95)
    # vis_joint_torques(files_24)
    
    # vis_joint_pos_delta(torque_path1, torque_path2)


