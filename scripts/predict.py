import os
import time
import sys
import numpy as np
import open3d as o3d
from pathlib import Path
from collections import Counter
from scipy.spatial import cKDTree
from natsort import natsorted
from urdfpy import URDF
import matplotlib.pyplot as plt

import torch
from network import LitRobot

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from src.utils.visualizer import RobotVisualizer
from src.utils.helpers import sample_points_from_mesh
from src.utils.visualize_mesh import create_viewing_parameters, visualize_with_camera
from src.utils.visualize_robot_state import update_meshes_with_fk, combine_meshes_o3d, create_red_markers, compute_forward_kinematics, find_closest_vertices, load_joint_torques, prepare_trimesh_fk, convert_trimesh_to_open3d


class RealtimeRobot:
    def __init__(self, markers_path, data_dir, classify, ckpts_path, seq, device, robot_type):
        self.markers_path = markers_path
        self.classify = classify
        self.ckpts_path = ckpts_path
        self.seq = seq
        self.device = device
        self.robot_type = robot_type

        # Load marker positions
        self.markers_pos = np.loadtxt(markers_path, delimiter=",")
        self.marker_positions = {f"{i}": pos for i, pos in enumerate(self.markers_pos)}
        print(f"Loaded marker positions: {self.marker_positions}")
        # sys.exit()

        folder_path =  Path(__file__).parent
        torque_dir = os.path.join(folder_path, data_dir)
        subdirs = natsorted(os.listdir(torque_dir))
        torque_files = [f for f in os.listdir(os.path.join(torque_dir, subdirs[0])) if f.endswith('.npy')]
        torque_files = natsorted(torque_files)
        # get all the file names
        self.classes = [f.split('.')[0] for f in torque_files]
        self.coordinates = {}
        print(f"class: {self.classes}")
        for c in self.classes:
            if self.marker_positions.get(c) is None:
                self.coordinates[str(self.markers_pos.shape[0])] = np.array([0, 0, 0])
                print(f"key:{str(self.markers_pos.shape[0])}")
            else:
                self.coordinates[c] = self.marker_positions.get(c)

        # Load the trained model
        print("Loading the model...")
        if classify:
            output_dim = self.markers_pos.shape[0] + 1
            print(f"Output dim: {output_dim}")
        else:
            output_dim = 3
        if self.robot_type == 'spot':
            self.model = self.load_from_checkpoint(input_dim=24 * seq, output_dim=output_dim * seq)
        elif self.robot_type == 'franka':
            self.model = self.load_from_checkpoint( input_dim=14 * seq, output_dim=output_dim * seq)
        print("Model loaded successfully.")


        # visualizer
        vis = o3d.visualization.Visualizer() 
        vis.create_window()
        self.visualizer = RobotVisualizer(robot_type=robot_type, vis=vis)
        self.original_colors = [np.asarray(pcd.colors).copy() for pcd in self.visualizer.point_clouds]
        # self.visualizer = RobotVisualizer(robot_type=robot_type)
    
    def load_from_checkpoint(self, input_dim, output_dim):
        """
        Load a model from a checkpoint file.
        """
        if self.device == "gpu":
            self.device = "cuda"
        # checkpoint = torch.load(self.ckpts_path, map_location=torch.device(self.device))
        # model = LitRobot(input_dim=input_dim, output_dim=output_dim, markers_path=self.markers_path, 
        #                 classify=self.classify, seq = self.seq, robot_type=self.robot_type)
        print(f"loading state dict")
        # model.load_state_dict(checkpoint['state_dict'])
        model = LitRobot.load_from_checkpoint(
            self.ckpts_path,
            input_dim=input_dim,
            output_dim=output_dim,
            markers_path=self.markers_path,
            classify=self.classify,
            seq=self.seq,
            robot_type=self.robot_type,
            map_location=torch.device(self.device)
        )
        print(f"finished loading state dict")
        model.to(self.device)
        print(f"finish to device")
        model.eval()
        print(f"return from load from checkpoint")
        return model
    
    def create_buffers(self, seq_win, radius=0.04, alpha=0.95, sliding_win=3):
        seq_win = self.seq
        if self.classify:
            buffer = np.zeros((sliding_win, len(self.classes)))
        else:
            buffer = np.zeros((sliding_win, 3))
        if self.robot_type == 'spot':
            data_buffer = np.zeros((seq_win, 24))
        elif self.robot_type == 'franka':
            data_buffer = np.zeros((seq_win, 14))
        weights = np.power((1 - alpha), np.arange(sliding_win))
        weights = alpha * weights
        # normalize - in the regression case, weighted average
        if not self.classify:
            weights = weights / np.sum(weights)


        self.data_buffer = data_buffer
        self.buffer = buffer
        self.weights = weights
        return self.data_buffer, self.buffer, self.weights
    
    def predict(self):
        # Real time prediction
        self.buffer = np.roll(self.buffer, 1, axis=0) 
        # processed_data = data_buffer.flatten()
        processed_data_tensor = torch.tensor(self.data_buffer.flatten(), dtype=torch.float32).to(self.model.device).reshape(1, -1)
        with torch.no_grad():
            if self.device == "gpu":
                self.buffer[0:self.seq] = self.model.predict(processed_data_tensor).cpu().numpy().reshape(self.seq, -1)
            else:
                result = self.model.predict(processed_data_tensor).numpy()
                self.buffer[0:self.seq] = self.model.predict(processed_data_tensor).numpy().reshape(self.seq, -1)
        print(f"Buffer shape: {self.buffer}")
        predictions = np.dot(self.weights, self.buffer)

        if self.classify:
            predictions = predictions.reshape(-1, len(self.classes)+1)
            predicted_class_index = np.argmax(predictions)
            confidence = np.max(predictions)
            predicted_class = self.classes[predicted_class_index]
            print(f"Prediction: {predicted_class}, Confidence: {confidence:.2f}")
            if predicted_class == "no_contact":
                pos = np.array([0, 0, 0])
            else:
                pos = self.marker_positions.get(predicted_class)
        else:
            pos = predictions
            # Compute the weighted variance
            weighted_mean = predictions
            differences = self.buffer - weighted_mean  # Difference between each row and the mean
            squared_differences = differences**2
            weighted_variance = np.dot(self.weights, np.mean(squared_differences, axis=1))  # Average squared differences
            confidence = 1 / (1 + np.sqrt(weighted_variance))  # Inverse relation: lower variance → higher confidence

        return pos


    def vis_prediction(self, pos, threshold=0.1):
        # pos = np.array([0.03444629, 0.0472558 , 0.4486532 ])
        # original_vertex_colors = np.asarray(total_mesh.vertex_colors).copy()
        print(f"pos: {pos}")
        for pcd, orig_color in zip(self.visualizer.point_clouds, self.original_colors):
            pcd.colors = o3d.utility.Vector3dVector(orig_color)

        cur_marker_pcd_indices, cur_marker_local_indices = self.visualizer.pos_2pcd(pos, radius=0.02)
        print(f"cur_marker_pcd_indices: {cur_marker_pcd_indices}")
        print(f"cur_marker_local_indices: {cur_marker_local_indices}")
        for pcd_idx, local_idx in zip(cur_marker_pcd_indices, cur_marker_local_indices):
            colors = np.asarray(self.visualizer.point_clouds[pcd_idx].colors)
            colors[local_idx] = [1, 0, 0]
            self.visualizer.point_clouds[pcd_idx].colors = o3d.utility.Vector3dVector(colors)
        
        # if self.vis:
        #     self.vis.poll_events()
        #     self.vis.update_renderer()
       
    


class RealtimeSpot(RealtimeRobot):
    def __init__(self, markers_path, data_dir, classify, ckpts_path, seq, device, hostname, choreo, choreography_filepaths=None):
        # imports
        from bosdyn.client.robot_state import RobotStateClient
        from bosdyn.client import create_standard_sdk
        import bosdyn.client.util

        from bosdyn.api.spot import choreography_sequence_pb2
        from bosdyn.client import create_standard_sdk
        from bosdyn.choreography.client.choreography import (ChoreographyClient,
                                                            load_choreography_sequence_from_txt_file)
        from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
        from bosdyn.api import lease_pb2
        from google.protobuf.timestamp_pb2 import Timestamp
        from bosdyn.api import header_pb2
        from bosdyn.client import ResponseError, RpcError, create_standard_sdk
        from bosdyn.client.exceptions import UnauthenticatedError
        from bosdyn.client.license import LicenseClient
        
        super().__init__(markers_path, data_dir, classify, ckpts_path, seq, device, robot_type="spot")
        self.hostname = hostname
        self.choreo = choreo
        self.choreo_files = choreography_filepaths
        self.choreos = []
        self.choreography_client = None

        self.simplified_to_full_name = {'fl.hx': 'front_left_hip_x', 'fr.hx': 'front_right_hip_x',
            'hl.hx': 'rear_left_hip_x', 'hr.hx': 'rear_right_hip_x', 'fl.hy': 'front_left_hip_y',
            'fr.hy': 'front_right_hip_y', 'hl.hy': 'rear_left_hip_y', 'hr.hy': 'rear_right_hip_y',
            'fl.kn': 'front_left_knee', 'fr.kn': 'front_right_knee', 'hl.kn': 'rear_left_knee',
            'hr.kn': 'rear_right_knee', 'arm0.sh0': 'arm_sh0', 'arm0.sh1': 'arm_sh1',
            'arm0.el0': 'arm_el0', 'arm0.el1': 'arm_el1', 'arm0.wr0': 'arm_wr0', 'arm0.wr1': 'arm_wr1',
            'arm0.f1x': 'arm_f1x', 'arm0.hr0': 'arm_hr0'}

        # Initialize robot and client
        sdk = create_standard_sdk('RobotStateClient')
        sdk.register_service_client(ChoreographyClient)


        robot = sdk.create_robot(hostname)
        bosdyn.client.util.authenticate(robot)
        self.robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        
        if self.choreo:
            # License 
            license_client = self.init_license(robot)
            # Check that an estop is connected with the robot so that the robot commands can be executed.
            assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                            'such as the estop SDK example, to configure E-Stop.'

            # Get lease client and take control
            lease, lk = self.init_lease(robot)
            self.init_choreo(robot)

        

    def init_license(self, robot):
        license_client = robot.ensure_client(LicenseClient.default_service_name)
        if not license_client.get_feature_enabled([ChoreographyClient.license_name
                                                ])[ChoreographyClient.license_name]:
            print('This robot is not licensed for choreography.')
            sys.exit(1)
        return license_client

    def init_lease(self, robot):
        lease_client = robot.ensure_client(LeaseClient.default_service_name)    
        lease = lease_client.take()
        lk = LeaseKeepAlive(lease_client)
        return lease, lk
    
    def init_choreo(self, robot):
        # Create choreography client
        choreography_client = robot.ensure_client(ChoreographyClient.default_service_name)
        available_moves = choreography_client.list_all_moves()
        for choreo_file in self.choreo_files:
            try: # step, trot, turn_2step, twerk, unstow
                self.choreos.append(load_choreography_sequence_from_txt_file(choreo_file))
            except Exception as excep:
                print(f'Failed to load choreography. Raised exception: {excep}')
                return True
        # upload the routine to the robot
        for choreo in self.choreos:
            try:
                upload_response = choreography_client.upload_choreography(choreo,
                                                                            non_strict_parsing=True)
            except UnauthenticatedError as err:
                print(
                    'The robot license must contain \'choreography\' permissions to upload and execute dances. ')
                return True
            except ResponseError as err:
                error_msg = 'Choreography sequence upload failed. The following warnings were produced: '
                for warn in err.response.warnings:
                    error_msg += warn
                print(error_msg)
                return True
        
        sequences_on_robot = choreography_client.list_all_sequences()
        self.choreography_client = choreography_client
        known_sequences = '\n'.join(sequences_on_robot.known_sequences)
        print(f'Sequence uploaded. All sequences on the robot:\n{known_sequences}')

        robot.power_on()
    
    def update_vis(self):
        self.data_buffer = np.roll(self.data_buffer, 1, axis=0) 
        state = self.robot_state_client.get_robot_state()

        # Preprocess the data for inference
        processed_data = self.preprocess_realtime_data(state, normalize=True)
        self.data_buffer[0] = processed_data
        print(f"Processed data shape: {processed_data.shape}")
        joint_positions = {joint.name: 0.0 for joint in self.visualizer.robot.joints}
        joint_states = state.kinematic_state.joint_states
        for joint_info in joint_states:
            joint = self.visualizer.robot.joint_map[self.simplified_to_full_name.get(joint_info.name)]
            if joint:
                joint_positions[self.simplified_to_full_name.get(joint_info.name)] = joint_info.position.value
            else:
                print(f"Joint {joint_info['name']} not found in URDF.")
        self.visualizer.visualize(cfg=joint_positions)


        

    def preprocess_realtime_data(self, data, normalize=True):
        """
        Preprocess the real-time data for inference.
        """
        state_dict = data.kinematic_state.joint_states
        torque_dict = []
        for joint in state_dict:
            joint_name = getattr(joint, 'name', None)
            if joint_name is not None:
                if not joint_name.startswith("arm"):
                    torque_dict.append({
                        'name': joint_name,
                        'position': joint.position.value,
                        'load': joint.load.value  
                    })
        num_joints = len(torque_dict)
        torque_data = np.full((1, num_joints), np.nan, dtype=float)
        pos_data = np.full((1, num_joints), np.nan, dtype=float)
        joint_names = []
        for j, joint in enumerate(torque_dict):
                torque_data[0, j]  = joint['load']
                pos_data[0, j] = joint['position']
                joint_names.append(joint['name'])
        data = np.hstack((torque_data, pos_data))
        if normalize:
            data = 2 * ((data - data.min()) / (data.max() - data.min())) - 1
        data_processed = np.array(data)

        return data_processed

    def exe_choreo(self, choreography):
        routine_name = choreography.name
        delayed_start = 2.0
        client_start_time = time.time() + delayed_start
        # begins at the very beginning.
        start_slice = 0
        # Issue the command to the robot's choreography service.
        self.choreography_client.execute_choreography(choreography_name=routine_name,
                                                client_start_time=client_start_time,
                                                choreography_starting_slice=start_slice)
        # Estimate how long the choreographed sequence will take.
        total_choreography_slices = 0
        for move in choreography.moves:
            # Calculate the slice when the move will end
            end_slice = move.start_slice + move.requested_slices

            #  Store the highest end_slice value of all the moves.
            if total_choreography_slices < end_slice:
                total_choreography_slices = end_slice
        estimated_time_seconds = delayed_start + total_choreography_slices / choreography.slices_per_minute * 60.0

        # Sleep for the duration of the dance, plus an extra second.
        time.sleep(estimated_time_seconds + 1.0)

    def respond(self, pos, threshold=1):
        distance = np.linalg.norm(pos - self.coordinates.get(str(len(self.markers_pos))))
        print(f"pos: {pos}, distance: {distance}")
        # if distance < threshold:
        # # step, trot, turn_2step, twerk, unstow
        # left
        if pos[1] > 0.09 and -0.35 < pos[0] < 0.3 and pos[2] < 0.11:
            self.exe_choreo(self.choreos[0]) # step
        # right
        elif pos[1] < -0.11 and -0.35 < pos[0] < 0.3 and pos[2] < 0.11:
            self.exe_choreo(self.choreos[1]) # trot
        # front
        elif pos[0] > 0.15 and pos[2] < 0.1 and -0.14 < pos[1] < 0.14:
            self.exe_choreo(self.choreos[2]) # turn_2step
        # back
        elif pos[0] < -0.45 and pos[2] < 0.1:
        # and -0.14 < pos[1] < 0.14:
            self.exe_choreo(self.choreos[3]) # twerk
        # top
        elif pos[2] > 0.11:
            self.exe_choreo(self.choreos[4]) # unstow
        # else: 
        #     continue






class RealtimeFranka(RealtimeRobot):
    def __init__(self, markers_path, data_dir, classify, ckpts_path, seq, device):
        # import rospy
        # from rospy import Subscriber, Rate
        # from sensor_msgs.msg import JointState

        super().__init__(markers_path, data_dir, classify, ckpts_path, seq, device, robot_type="franka")
        # rospy.init_node("save_franka_state")
        # self.joint_sub = Subscriber("/right_arm/joint_states", JointState, self.joint_callback)
        # self.save_rate = Rate(30)
        self.current_state = None
        self.idx = 0
    
    def joint_callback(self, state):
        self.current_state = state

    def update_vis(self):
        # if self.current_state is None:
        #     return
        self.data_buffer = np.roll(self.data_buffer, 1, axis=0) 
        # joint_position = self.current_state.position
        joint_position = np.array([-1.71, 1.40, 2.07, -2.44, -1.04, 1.36, -0.74, 0.0, 0.0,])
        # joint_torque = self.current_state.effort
        # state = np.hstack([joint_torque[:7], joint_position[:7]], )
        states = np.load("../data/franka_right/test12/no_contact.npy", allow_pickle=True)
        if self.idx >= len(states):
            self.idx = 0
        state = states[self.idx]
        self.idx += 1
        # self.save_rate.sleep()

        self.data_buffer[0] = state
        print(f"Processed data shape: {state.shape}")
        cfg = {joint: joint_position[idx] for idx, joint in enumerate(self.visualizer.robot.actuated_joint_names)}
        self.visualizer.visualize(cfg=cfg)




def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpts_path', required=True, help='Path to the trained model')
    parser.add_argument('--markers_path', required=True, help='Path to markers positions')
    parser.add_argument('--data_dir', required=True, help='Path to the directory containing torque data')
    parser.add_argument('--device', required=True, help='gpu or cpu')
    parser.add_argument('--robot_type', required=True, help='Robot type: spot or franka')
    parser.add_argument('--classify', action='store_true', help='Run classification model instead of regression')
    parser.add_argument('--seq', type=int, help='Train on sequence data, length of sequence')
    # spot: optional
    parser.add_argument('--hostname', required=False, help='Hostname of the robot')
    parser.add_argument('--choreo', action='store_true', help='Run choreography')
    parser.add_argument('--choreography-filepaths', required=False, nargs='+',
                    help='List of filepath(s) to load the choreographed sequence text files from.')


    options = parser.parse_args()
    classify = options.classify
    device  = options.device
    seq = options.seq
    robot_type = options.robot_type
    markers_path = options.markers_path
    data_dir = options.data_dir

    if robot_type == 'spot':    
        # choreography
        if options.choreo:
            if options.choreography_filepaths:
                choreo_files = options.choreography_filepaths
            else:
                print("No choreography files provided.")
                sys.exit(1)
        else:
            choreo_files = None
        realtime_robot = RealtimeSpot(markers_path, data_dir, classify, options.ckpts_path, seq, device, options.hostname, options.choreo, choreo_files)
    elif robot_type == 'franka':
        realtime_robot = RealtimeFranka(markers_path, data_dir, classify, options.ckpts_path, seq, device)
    # Create buffers
    data_buffer, buffer, weights = realtime_robot.create_buffers(seq, radius=0.04, alpha=0.95, sliding_win=3)

    try:
        while True:
            # Real-time prediction
            realtime_robot.update_vis()
            pos = realtime_robot.predict()
            # Visualize the predictions
            realtime_robot.vis_prediction(pos)
            if robot_type == 'spot' and options.choreo:
                # Respond to the predictions
                realtime_robot.respond(pos)
    except KeyboardInterrupt:
        print("Exiting real-time inference...")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()