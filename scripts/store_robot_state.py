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
from urdfpy import URDF
import open3d as o3d
import random
from pathlib import Path
import rospy
from rospy import Subscriber, Rate
from sensor_msgs.msg import JointState

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from src.utils.visualizer import RobotVisualizer
from src.utils.visualize_mesh import create_viewing_parameters, visualize_with_camera, look_at
from src.utils.visualize_robot_state import find_closest_vertices, add_red_dots, compute_forward_kinematics, prepare_trimesh_fk, \
convert_trimesh_to_open3d, create_red_markers, visualize_robot_with_markers, combine_meshes_o3d


class Robot:
    def __init__(self, output_dir, markers_path, robot_type):
        folder_path =  os.path.join("../", f'{robot_type}_description')

        self.output_dir = output_dir
        self.markers_path = markers_path
        self.robot_type = robot_type
        self.robot = URDF.load(os.path.join(folder_path, f'{robot_type}.urdf'))

        self.markers_pos = []
        self.markers_dict = {}

        # load robot mesh from urdf
        joint_positions = {joint.name: 0.0 for joint in self.robot.joints}  # Zero configuration
        link_fk_transforms = compute_forward_kinematics(self.robot, joint_positions)
        trimesh_fk, _ = prepare_trimesh_fk(self.robot, link_fk_transforms, folder=folder_path)
        self.robot_meshes, _ = convert_trimesh_to_open3d(trimesh_fk)
        self.total_mesh = o3d.geometry.TriangleMesh()
        for mesh in self.robot_meshes:
            self.total_mesh += mesh
    
    def save_markers(self, original_colors = None, visualizer=None):
        np.savetxt(self.markers_path, self.markers_pos, delimiter=",", comments="")

        # visualize markers
        if visualizer == None:
            markers = create_red_markers(self.markers_pos, radius=0.02)
            o3d.visualization.draw_geometries(self.robot_meshes + markers)
        else:
            for pcd, orig_color in zip(visualizer.point_clouds, original_colors):
                pcd.colors = o3d.utility.Vector3dVector(orig_color)
            
            start_time = time.time()

            # while time.time() - start_time < 100:
            #     joint_positions = {joint.name: 0.0 for joint in visualizer.robot.joints}
            #     joint_position = self.current_state.position
            #     # joint_position = np.array([-1.71, 1.40, 2.07, -2.44, -1.04, 1.36, -0.74, 0.0, 0.0,])
            #     cfg = {joint: pos for joint, pos in zip(joint_positions.keys(), joint_position)}
            #     visualizer.visualize(cfg=cfg)

            #     for pcd_indices, local_indices in zip(visualizer.pcd_indices, visualizer.local_indices):
            #         for pcd_idx, local_idx in zip(pcd_indices, local_indices):
            #             colors = np.asarray(visualizer.point_clouds[pcd_idx].colors)
            #             colors[local_idx] = [1, 0, 0]
            #             visualizer.point_clouds[pcd_idx].colors = o3d.utility.Vector3dVector(colors)

            

    
    def load_markers(self):
        self.markers_pos = np.loadtxt(self.markers_path, delimiter=",")
        self.markers_dict = {f"{i}": pos for i, pos in enumerate(self.markers_pos)}
        
        return self.markers_dict
    
class Spot(Robot):
    def __init__(self, output_dir, markers_path, robot_type, hostname, command):
        super().__init__(output_dir, markers_path, robot_type)
        self.hostname = hostname
        self.command = command
        import bosdyn.client
        import bosdyn.client.util
        from bosdyn.client.robot_state import RobotStateClient
    
    def choose_markers(self, num_points = 10000):
        markers_pos = [[0.45, 0.06, -0.035],[0.45, -0.07, -0.035], # front
                    [-0.45, 0.05, 0.05],[-0.45, -0.05, 0.05], # back
                    [0.13, 0.14, 0.01], [-0.13, 0.14, -0.01], # left
                    [0.1, -0.15, -0.01], [-0.13, -0.14, -0.01], # right
                    [0.1, 0.05, 0.09], [-0.12, -0.01, 0.09],] # top
        for i in range(8): # LEFT: 24
            x = -0.2 + i / 8 * (0.25 - (-0.2))
            for j in range(3):
                z = -0.04 + j / 3 * (0.1 - (-0.04))
                markers_pos.append([x, 0.105, z])
        for i in range(8): # RIGHT: 24
            x = -0.2 + i / 8 * (0.25 - (-0.2))
            for j in range(3):
                z = -0.04 + j / 3 * (0.1 - (-0.04))
                markers_pos.append([x, -0.105, z])
        for i in range(6):  # TOP: 24
            x = -0.37 + i / 6 * (0.04 - (-0.37))
            for j in range(3):
                y = -0.04 + j/3 * (0.08 - (-0.04))
                markers_pos.append([x, y, 0.08])
        markers_pos.extend([[-0.2, 0.08, 0.08], [-0.12, 0.08, 0.08], [0.01, 0.08, 0.08], 
                            [-0.2, -0.07, 0.08], [0.01, -0.07, 0.08], [0.1, -0.07, 0.08]])
        for i in range(4): # FRONT: 8
            y = -0.07 + i / 4 * (0.11 - (-0.07))
            markers_pos.append([0.39, y, -0.07])
            markers_pos.append([0.44, y, 0.04])
        for i in range(4):  # BACK: 10
            y = -0.07 + i / 4 * (0.11 - (-0.07))
            markers_pos.append([-0.40, y, -0.06])
            markers_pos.append([-0.42, y, 0.04])
        markers_pos.extend([[-0.42, -0.07, -0.01], [-0.42, 0.07, -0.01]])
        markers_pos = np.array(markers_pos)
        self.markers_pos, pos_indices = find_closest_vertices(self.total_mesh, markers_pos, num_points)
        self.markers_dict = {f"{i}": pos for i, pos in enumerate(self.markers_pos)}


        return self.markers_pos

    def collect_data(self, output_path, duration=10):
        sdk = bosdyn.client.create_standard_sdk('RobotStateClient')
        robot = sdk.create_robot(self.hostname)
        bosdyn.client.util.authenticate(robot)
        robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        # Make a robot state request
        if self.command == 'state':
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

        elif self.command == 'hardware':
            print(robot_state_client.get_hardware_config_with_link_info())
        elif self.command == 'metrics':
            print(robot_state_client.get_robot_metrics())

        return True
    
    def vis_markers(self, idx, pos):
        # Create marker for current position
        marker = create_red_markers([pos], radius=0.02)[0]
        geometries = self.robot_meshes + [marker]
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

    def save_data(self, idx, output_path, duration=10):
        # user_input = input(f"Is marker position {idx} legit? Enter 'y' for yes, 'n' for no (default: 'y'): \n").strip().lower()
        # if user_input == 'n':
        #     print(f"Marker position {idx} deemed not legit. Skipping this marker...\n")
        #     continue
        output_path = os.path.join(self.output_dir, f"{idx}.npy")
        print(f"YOU CAN TOUCH THE SPOT NOW. Data collection will start in 5 seconds, please make sure you are touching the Spot.\n")
        time.sleep(2)
        self.collect_data(output_path, self.hostname, self.command, duration)
        print(f"Touch Data Collected for marker position {idx}, saved in {output_path}\n")







class Franka(Robot):
    def __init__(self, output_dir, markers_path, robot_type):
        super().__init__(output_dir, markers_path, robot_type)
        rospy.init_node("save_franka_state")
        self.joint_sub = Subscriber("/right_arm/joint_states", JointState, self.joint_callback)
        self.save_rate = Rate(30)
        self.current_state = None

    def joint_callback(self, state: JointState):
        self.current_state = state

    def choose_markers(self, num_points = 10000):
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
        self.markers_pos, pos_indices = find_closest_vertices(self.total_mesh, markers_pos, num_points)

        
        return self.markers_pos
    

    def vis_markers(self, idx, pos):
        # Create marker for current position
        marker = create_red_markers([pos], radius=0.02)[0]
        geometries = self.robot_meshes + [marker]
        print(f"Viewing marker {idx} . Press Ctrl+C in terminal to proceed to next view.")

        o3d.visualization.draw_geometries(geometries)


    def save_data(self, idx, output_path, original_colors, duration=2, visualizer=None):

        if idx != -1:
            output_path = os.path.join(self.output_dir, f"{idx}.npy")
        else:
            output_path = os.path.join(self.output_dir, f"no_contact.npy")
        # create a dictionary with all the keys the same as state bu the values as empty lists
        state_dict = []

        for pcd, orig_color in zip(visualizer.point_clouds, original_colors):
            pcd.colors = o3d.utility.Vector3dVector(orig_color)

        # VISUALIZE
        joint_position = self.current_state.position
        # joint_position = np.array([-1.71, 1.40, 2.07, -2.44, -1.04, 1.36, -0.74])
        cfg = {joint: joint_position[idx] for idx, joint in enumerate(visualizer.robot.actuated_joint_names)}
        visualizer.visualize(cfg=cfg)
        if idx != -1:
            pcd_indices = visualizer.pcd_indices[int(idx)]
            local_indices = visualizer.local_indices[int(idx)]

            cur_pcd = visualizer.point_clouds[pcd_indices[0]]
            points = np.asarray(cur_pcd.points)
            pos = points[local_indices[0]]
            if not cur_pcd.has_normals():
                cur_pcd.compute_normals()
            normals = np.asarray(cur_pcd.normals)
            normal = normals[local_indices[0]]

            for pcd_idx, local_idx in zip(pcd_indices, local_indices):
                colors = np.asarray(cur_pcd.colors)
                colors[local_idx] = [1, 0, 0]
                visualizer.point_clouds[pcd_idx].colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries(visualizer.point_clouds, zoom=0.7, front = normal, lookat=pos, up = [0, 0, 1])

            print(f"YOU CAN TOUCH THE SPOT NOW. Data collection will start in 5 seconds, please make sure you are touching the Spot.\n")
        else:
            o3d.visualization.draw_geometries(visualizer.point_clouds, zoom = 0.7, front = [1, 0, 0], lookat=[0, 0, 0.5], up = [0, 1, 0])
            print("DON'T TOUCH YET! COLLECTING NO CONTACT DATA")

        print("Collecting data\n")
        
        start_time = time.time()
        while time.time() - start_time < duration:          
            joint_torque = self.current_state.effort
            state = np.hstack([joint_torque[:7], joint_position[:7]], )
            state_dict.append(state)
            self.save_rate.sleep()
        
        # save data 
        print(f"Data collection complete, saved in {output_path}\n")
        np.save(output_path, state_dict)

        print(f"Touch Data Collected for marker position {idx}, saved in {output_path}\n")



def main():
    import argparse

    commands = {'state', 'hardware', 'metrics'}

    parser = argparse.ArgumentParser()
    try:
        bosdyn.client.util.add_base_arguments(parser)
    except:
        pass

    parser.add_argument('command', choices=list(commands), help='Command to run')
    parser.add_argument('--hostname', required=False, help='Hostname of the robot')
    parser.add_argument('--output_dir', required=True, help='Output directory for data')
    parser.add_argument('--markers_path', required=True, help='Path to markers positions')
    parser.add_argument('--robot_type', required=True, help='Robot type: spot or franka')
    parser.add_argument('--duration', type=int, default=2, help='Duration to collect data')
    options = parser.parse_args()

    output_dir = options.output_dir
    markers_path = options.markers_path
    robot_type = options.robot_type
    duration = options.duration
    os.makedirs(output_dir, exist_ok=True)
    
    if robot_type == 'spot':
        # Create robot object with an image client.
        hostname = options.hostname
        command = options.command
        robot = Spot(output_dir, markers_path, robot_type, hostname, command)
        markers_positions = robot.load_markers()
    elif robot_type == 'franka':
        robot = Franka(output_dir, markers_path, robot_type)
        markers_pos = robot.choose_markers()

        markers = create_red_markers(markers_pos, radius=0.02)
        selected_idx = [6, 13, 27, 30, 40, 59, 67, 76, 80, 92]
        robot.markers_pos = [robot.markers_pos[i] for i in selected_idx]

        vis = o3d.visualization.Visualizer() 
        vis.create_window()
        # visualizer = SpotVisualizer(robot_type = "franka", vis=vis)
        visualizer = RobotVisualizer(robot_type = "franka")
        original_colors = [np.asarray(pcd.colors).copy() for pcd in visualizer.point_clouds]
        visualizer.marker_2vert(robot.markers_pos, radius = 0.02)
        


        robot.markers_dict = {f"{i}": pos for i, pos in enumerate(robot.markers_pos)}
        print(robot.markers_dict)
        robot.save_markers(original_colors=original_colors, visualizer=visualizer)


    if robot_type == 'spot':
        robot.collect_data(os.path.join(output_dir, f"no_contact.npy"), duration=10)
    elif robot_type == 'franka':
        robot.save_data(-1,  os.path.join(output_dir, f"no_contact.npy"),  original_colors=original_colors, duration=10, visualizer = visualizer)
    # vertices = np.asarray(robot.robot_meshes[0].vertices)
    # robot.robot_meshes[0].compute_vertex_normals()

    for idx, pos in robot.markers_dict.items():
        if robot_type == 'spot':
            robot.vis_markers(idx, pos)
            robot.save_data(idx, output_dir, duration=5)
        elif robot_type == 'franka':
            robot.save_data(idx, output_dir, original_colors=original_colors, duration=duration, visualizer = visualizer)
        
        



if __name__ == '__main__':
    main()
