# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Simple robot state capture tutorial."""
import sys
import time
import json
import os
import numpy as np
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.robot_state import RobotStateClient
from urdfpy import URDF
import open3d as o3d
import random
from pathlib import Path


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from src.utils.visualize_mesh import create_viewing_parameters, visualize_with_camera, look_at
from src.utils.visualize_robot_state import find_closest_vertices, add_red_dots, compute_forward_kinematics, prepare_trimesh_fk, \
convert_trimesh_to_open3d, create_red_markers, visualize_robot_with_markers, combine_meshes_o3d


def collect_data(output_path, hostname, command, duration=10):
    sdk = bosdyn.client.create_standard_sdk('RobotStateClient')
    robot = sdk.create_robot(hostname)
    bosdyn.client.util.authenticate(robot)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    # Make a robot state request
    if command == 'state':
        print("Collecting data\n")
        state = robot_state_client.get_robot_state()
        # create a dictionary with all the keys the same as state bu the values as empty lists
        state_dict = []
        start_time = time.time()

        while time.time() - start_time < duration:
            state = robot_state_client.get_robot_state()
            state_dict.append(state)
        
        # save data 
        print(f"Data collection complete, saved in {output_path}\n")
        np.save(output_path, state_dict)

    elif command == 'hardware':
        print(robot_state_client.get_hardware_config_with_link_info())
    elif command == 'metrics':
        print(robot_state_client.get_robot_metrics())

    return True

def get_vertex_normal_at_position(mesh, position):
    # Ensure the mesh has vertex normals computed
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Get all vertex positions
    vertices = np.asarray(mesh.vertices)

    # Compute the closest vertex index
    position = np.array(position)
    distances = np.linalg.norm(vertices - position, axis=1)
    closest_vertex_index = np.argmin(distances)

    # Get the normal of the closest vertex
    normals = np.asarray(mesh.vertex_normals)
    vertex_normal = normals[closest_vertex_index]

    return vertex_normal

def main():
    import argparse

    commands = {'state', 'hardware', 'metrics'}

    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('command', choices=list(commands), help='Command to run')
    parser.add_argument('--output_dir', required=True, help='Output directory for data')
    parser.add_argument('--markers_path', required=True, help='Path to markers positions')
    parser.add_argument('--robot_type', required=True, help='Robot type: spot or franka')
    parser.add_argument('--duration', type=int, default=10, help='Duration to collect data')
    options = parser.parse_args()

    # Create robot object with an image client.
    hostname = options.hostname
    command = options.command
    output_dir = options.output_dir
    markers_path = options.markers_path
    robot_type = options.robot_type
    duration = options.duration
    os.makedirs(output_dir, exist_ok=True)
    
    folder_path =  os.path.join(Path(__file__).parent.parent, f'{robot_type}_description')
    robot = URDF.load(os.path.join(folder_path, f'{robot_type}.urdf'))
    joint_positions = {joint.name: 0.0 for joint in robot.joints}  # Zero configuration
    link_fk_transforms = compute_forward_kinematics(robot, joint_positions)
    trimesh_fk, _ = prepare_trimesh_fk(robot, link_fk_transforms, folder=folder_path)
    robot_meshes, _ = convert_trimesh_to_open3d(trimesh_fk)
    total_mesh = o3d.geometry.TriangleMesh()
    for mesh in robot_meshes:
       total_mesh += mesh
    # FRANKA
    # x: 0.05 (-0.03 ~ 0.08) y: 0.02 (-0.005 ~ 0.05) -- sides, z: 0.5 (0.14 ~ 0.98) --height
    # Joint 1: 20
    theta = [np.pi/3 * i for i in range(6)]
    markers_pos = [[np.cos(t) * 0.05, np.sin(t) * 0.05, 0.17] for t in theta]
    markers_pos.extend([[np.cos(t) * 0.05, np.sin(t) * 0.05, 0.22] for t in theta])
    markers_pos.extend([[np.cos(t) * 0.05, np.sin(t) * 0.05, 0.27] for t in theta][0:1]
                       + [[np.cos(t) * 0.05, np.sin(t) * 0.05, 0.27] for t in theta][3:])
    markers_pos.extend([[np.cos(t) * 0.05, -0.08, 0.34 + np.sin(t) * 0.05] for t in theta][0:4])
    # Joint 2: 18
    markers_pos.extend([[np.cos(t) * 0.05, 0.08, 0.34 + np.sin(t) * 0.05] for t in theta])
    markers_pos.extend([[np.cos(t) * 0.05, np.sin(t) * 0.05, 0.44] for t in theta])
    markers_pos.extend([[np.cos(t) * 0.05, np.sin(t) * 0.05, 0.5] for t in theta])
    # Joint 3: 12
    markers_pos.extend([[np.cos(t) * 0.05, np.sin(t) * 0.05, 0.56] for t in theta])
    markers_pos.extend([[0.07 + np.cos(t) * 0.05, 0.05, 0.65 + np.sin(t) * 0.05] for t in theta])
    # Joint 4: 12
    markers_pos.extend([[0.07 + np.cos(t) * 0.05, -0.05, 0.65 + np.sin(t) * 0.05] for t in theta])
    markers_pos.extend([[np.cos(t) * 0.05, np.sin(t) * 0.05, 0.73] for t in theta])
    # Joint 5: 23
    markers_pos.extend([[np.cos(t) * 0.05, np.sin(t) * 0.05, 0.8] for t in theta])
    markers_pos.extend([[np.cos(t) * 0.05, np.sin(t) * 0.05, 0.85] for t in theta])
    markers_pos.extend([[np.cos(t) * 0.05, 0.07 + np.sin(t) * 0.05, 0.91] for t in theta][0:1]
                        + [[np.cos(t) * 0.05, 0.07 + np.sin(t) * 0.05, 0.91] for t in theta][3:4]
                        + [[np.cos(t + np.pi/6) * 0.05, 0.07 + np.sin(t+np.pi/6) * 0.05, 0.91] for t in theta][4:5])
    markers_pos.extend([[np.cos(t) * 0.05, 0.07 + np.sin(t) * 0.05, 0.96] for t in theta][0:1]
                        + [[np.cos(t) * 0.05, 0.07 + np.sin(t) * 0.05, 0.96] for t in theta][3:4])
    markers_pos.extend([[np.cos(t) * 0.05, 0.06, 1.04 + np.sin(t) * 0.05] for t in theta])
    # Joint 6: 15
    markers_pos.extend([[np.cos(t) * 0.05, -0.02, 1.04 + np.sin(t) * 0.05] for t in theta])
    markers_pos.extend([[0.09 + np.cos(t) * 0.05, np.sin(t) * 0.05, 1.08] for t in theta][:3]
                       + [[0.09 + np.cos(t) * 0.05, np.sin(t) * 0.05, 1.08] for t in theta][4:])
    markers_pos.extend([[0.07 + np.cos(t) * 0.05, np.sin(t) * 0.05, 1.01] for t in theta][:2]
                       + [[0.07 + np.cos(t) * 0.05, np.sin(t) * 0.05, 1.01] for t in theta][4:])
    markers_pos = np.array(markers_pos)
    num_points = 10000
    markers_pos, pos_indices = find_closest_vertices(total_mesh, markers_pos, num_points)
    markers = create_red_markers(markers_pos, radius=0.02)
    selected_idx = [6, 27, 40, 59, 76, 92]
    selected_markers = [markers[i] for i in selected_idx]
    markers_pos = markers_pos[selected_idx]
    o3d.visualization.draw_geometries(robot_meshes + selected_markers)



    # # SPOT
    # markers_pos = [[0.45, 0.06, -0.035],[0.45, -0.07, -0.035], # front
    #                [-0.45, 0.05, 0.05],[-0.45, -0.05, 0.05], # back
    #                [0.13, 0.14, 0.01], [-0.13, 0.14, -0.01], # left
    #                [0.1, -0.15, -0.01], [-0.13, -0.14, -0.01], # right
    #                [0.1, 0.05, 0.09], [-0.12, -0.01, 0.09],] # top
    # for i in range(8): # LEFT: 24
    #     x = -0.2 + i / 8 * (0.25 - (-0.2))
    #     for j in range(3):
    #         z = -0.04 + j / 3 * (0.1 - (-0.04))
    #         markers_pos.append([x, 0.105, z])
    # for i in range(8): # RIGHT: 24
    #     x = -0.2 + i / 8 * (0.25 - (-0.2))
    #     for j in range(3):
    #         z = -0.04 + j / 3 * (0.1 - (-0.04))
    #         markers_pos.append([x, -0.105, z])
    # for i in range(6):  # TOP: 24
    #     x = -0.37 + i / 6 * (0.04 - (-0.37))
    #     for j in range(3):
    #         y = -0.04 + j/3 * (0.08 - (-0.04))
    #         markers_pos.append([x, y, 0.08])
    # markers_pos.extend([[-0.2, 0.08, 0.08], [-0.12, 0.08, 0.08], [0.01, 0.08, 0.08], [-0.2, -0.07, 0.08], [0.01, -0.07, 0.08], [0.1, -0.07, 0.08]])
    # for i in range(4): # FRONT: 8
    #     y = -0.07 + i / 4 * (0.11 - (-0.07))
    #     markers_pos.append([0.39, y, -0.07])
    #     markers_pos.append([0.44, y, 0.04])
    # for i in range(4):  # BACK: 10
    #     y = -0.07 + i / 4 * (0.11 - (-0.07))
    #     markers_pos.append([-0.40, y, -0.06])
    #     markers_pos.append([-0.42, y, 0.04])
    # markers_pos.extend([[-0.42, -0.07, -0.01], [-0.42, 0.07, -0.01]])


    # np.savetxt(markers_path, markers_pos, delimiter=",", comments="")
    markers_pos = np.loadtxt(markers_path, delimiter=",")
    marker_positions = {f"{i}": pos for i, pos in enumerate(markers_pos)}
    print("DON'T TOUCH YET! COLLECTING NO CONTACT DATA")
    collect_data(os.path.join(output_dir, f"no_contact.npy"), hostname, command, duration=20)
    vertices = np.asarray(robot_meshes[0].vertices)
    robot_meshes[0].compute_vertex_normals()

    for idx, pos in marker_positions.items():
        distances = np.linalg.norm(vertices - pos, axis=1)
        index = np.argmin(distances)

        normal_at_vertex = robot_meshes[0].vertex_normals[index]
        distance = 2.0

        # Create marker for current position
        marker = create_red_markers([pos], radius=0.02)[0]
        geometries = robot_meshes + [marker]
        
        print(f"Viewing marker {idx} . Press Ctrl+C in terminal to proceed to next view.")
        
        # Visualize with specific camera view
        up = np.array([0.0, 0.0, 1.0])
        marker_idx = int(idx)
        if marker_idx < 2 or 81 < marker_idx < 90:
            front = np.array([-1.0, 0.0, 0.0])
        if 1 < marker_idx < 4 or marker_idx > 89:
            front = np.array([1.0, 0.0, 0.0])
        if 3 < marker_idx < 6 or 9 < marker_idx < 34:
            front = np.array([0.0, -1.0, 0.0])
        if 5 < marker_idx < 8 or 33 < marker_idx < 58:
            front = np.array([0.0, 1.0, 0.0])
        if 7 < marker_idx < 10 or 57 < marker_idx < 82:
            front = np.array([0.0, 0.0, -1.0])
            up = np.array([0.0, 1.0, 0.0])
        o3d.visualization.draw_geometries(geometries, zoom=0.5, front = -front, lookat=pos, up = up)
        # user_input = input(f"Is marker position {idx} legit? Enter 'y' for yes, 'n' for no (default: 'y'): \n").strip().lower()
    
        # if user_input == 'n':
        #     print(f"Marker position {idx} deemed not legit. Skipping this marker...\n")
        #     continue
        output_path = os.path.join(output_dir, f"{idx}.npy")
        print(f"YOU CAN TOUCH THE SPOT NOW. Data collection will start in 5 seconds, please make sure you are touching the Spot.\n")
        time.sleep(2)
        collect_data(output_path, hostname, command, duration)
        print(f"Touch Data Collected for marker position {idx}, saved in {output_path}\n")
        
        



if __name__ == '__main__':
    main()
