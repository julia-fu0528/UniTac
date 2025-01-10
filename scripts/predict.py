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
# import tensorflow as tf
# from tensorflow.keras.models import load_model


import torch
from network import LitSpot

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))

from src.utils.visualizer import SpotVisualizer
from src.utils.helpers import sample_points_from_mesh
from src.utils.visualize_mesh import create_viewing_parameters, visualize_with_camera
from src.utils.visualize_robot_state import update_meshes_with_fk, combine_meshes_o3d, create_red_markers, compute_forward_kinematics, find_closest_vertices, load_joint_torques, prepare_trimesh_fk, convert_trimesh_to_open3d

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



def collect_realtime_data(robot_state_client, duration=2):
    """
    Collect real-time data from the robot for a specified duration.
    """
    start_time = time.time()
    data = []

    while time.time() - start_time < duration:
        state = robot_state_client.get_robot_state()
        data.append(state)

    return np.array(data)

def preprocess_realtime_data(data, markers_path, normalize=True):
    """
    Preprocess the real-time data for inference.
    """
    markers_pos = np.loadtxt(markers_path, delimiter=",")
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

def infer_realtime(model, data, classes):
    """
    Perform real-time inference using the trained model.
    """
    predictions = model.predict(data)

    if predictions.shape[0] == 2:  # Multiple samples
        predicted_classes = [np.argmax(pred) for pred in predictions]
        
        # Find the most common class
        most_common_class_index, count = Counter(predicted_classes).most_common(1)[0]
        predicted_class = classes[most_common_class_index]
        
        # Confidence can be calculated as the proportion of predictions for the most common class
        confidence = count / len(predicted_classes)
    else:  # Single sample
        predicted_class_index = np.argmax(predictions)
        confidence = np.max(predictions)
        predicted_class = classes[predicted_class_index]

    print(f"Prediction: {predicted_class}, Confidence: {confidence:.2f}")
    return predicted_class, confidence


def visualize_prediction(marker_positions, predicted_class, robot_meshes):
    """
    Visualize the prediction on the robot with markers.
    """
    pos = marker_positions.get(predicted_class)
    marker = create_red_markers([pos], radius=0.04)[0]
    if marker is None:
        print(f"No marker for predicted class: {predicted_class}")
        return

    # Create camera parameters for the marker
    camera_params = create_viewing_parameters(pos)
    geometries = robot_meshes + [marker]
    # Visualize the robot with the predicted marker
    print(f"Visualizing prediction for class: {predicted_class}")
    visualize_with_camera(geometries, camera_params)


# @tf.keras.utils.register_keras_serializable()
# def euclidean_distance(y_true, y_pred):
#     return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)))

# @tf.keras.utils.register_keras_serializable()
# def regression_accuracy(y_true, y_pred, tolerance=0.1):
#     # Calculate the Euclidean distance between true and predicted coordinates
#     distance = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))
#     # Check if the distance is within the tolerance
#     correct_predictions = tf.cast(distance <= tolerance, tf.float32)
#     # Return the mean accuracy
#     return tf.reduce_mean(correct_predictions)


def load_from_checkpoint(checkpoint_path, input_dim, output_dim, markers_path, device, classify, seq):
    """
    Load a model from a checkpoint file.
    """
    if device == "gpu":
        device = "cuda"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model = LitSpot(input_dim=input_dim, output_dim=output_dim, markers_path=markers_path, classify=classify, seq = seq)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model

def exe_choreo(choreography, choreography_client):
    routine_name = choreography.name
    delayed_start = 2.0
    client_start_time = time.time() + delayed_start
    # begins at the very beginning.
    start_slice = 0
    # Issue the command to the robot's choreography service.
    choreography_client.execute_choreography(choreography_name=routine_name,
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

def main():
    import argparse
    from bosdyn.client.robot_state import RobotStateClient
    from bosdyn.client import create_standard_sdk
    import bosdyn.client.util

    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--ckpts_path', required=True, help='Path to the trained model')
    parser.add_argument('--markers_path', required=True, help='Path to markers positions')
    parser.add_argument('--data_dir', required=True, help='Path to the directory containing torque data')
    parser.add_argument('--device', required=True, help='gpu or cpu')
    parser.add_argument('--choreography-filepaths', required=True, nargs='+',
                    help='List of filepath(s) to load the choreographed sequence text files from.')
    parser.add_argument('--classify', action='store_true', help='Run classification model instead of regression')
    parser.add_argument('--seq', type=int, help='Train on sequence data, length of sequence')



    options = parser.parse_args()
    classify = options.classify
    device  = options.device
    seq = options.seq
    choreo_files = options.choreography_filepaths


     # Load marker positions
    markers_path = options.markers_path
    markers_pos = np.loadtxt(markers_path, delimiter=",")

     # Load the trained model
    print("Loading the model...")
    # model = load_model(options.model_path)
    if classify:
        output_dim = 101
    else:
        output_dim = 3
    model = load_from_checkpoint(options.ckpts_path, input_dim=24 * seq, output_dim=output_dim * seq, markers_path=markers_path, classify=classify, device=device, seq=seq)
    print("Model loaded successfully.")

    # Initialize robot and client
    sdk = create_standard_sdk('RobotStateClient')
    sdk.register_service_client(ChoreographyClient)

    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    # License 
    license_client = robot.ensure_client(LicenseClient.default_service_name)
    if not license_client.get_feature_enabled([ChoreographyClient.license_name
                                              ])[ChoreographyClient.license_name]:
        print('This robot is not licensed for choreography.')
        sys.exit(1)
     # Check that an estop is connected with the robot so that the robot commands can be executed.
    assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                    'such as the estop SDK example, to configure E-Stop.'

    # Get lease client and take control
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    sys.exit()
    lease = lease_client.take()
    lk = LeaseKeepAlive(lease_client)

    # client_lease = lease_response
    # lease_proto = lease_response.lease_proto
    # lease_client.retain_lease(client_lease)

    # Create choreography client
    choreography_client = robot.ensure_client(ChoreographyClient.default_service_name)
    available_moves = choreography_client.list_all_moves()
    choreos = [] # step, trot, turn_2step, twerk, unstow
    for choreo_file in choreo_files:
        try:
            choreos.append(load_choreography_sequence_from_txt_file(choreo_file))
        except Exception as excep:
            print(f'Failed to load choreography. Raised exception: {excep}')
            return True
    # upload the routine to the robot
    for choreo in choreos:
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
    known_sequences = '\n'.join(sequences_on_robot.known_sequences)
    print(f'Sequence uploaded. All sequences on the robot:\n{known_sequences}')

    robot.power_on()

    # Sit the robot down and power off the robot.
    # robot.power_off()
    # return True
    # sys.exit()
    # sequence = choreography_sequence_pb2.ChoreographySequence()
    # sequence.name = "my_dance"
    # sequence.slices_per_minute = 60

    # move = sequence.moves.add()
    # move.type = "trot" 
    # move.start_slice = 0
    # move.requested_slices = 5
    # Upload the sequence
    # upload_response = choreography_client.upload_choreography(sequence, non_strict_parsing=True)
    # print("Upload response:", upload_response) 
    # available_moves = choreography_client.list_all_moves()
    # print("Available moves:", available_moves) 
    # sys.exit() 
    # markers_pos = [   
    #     # front
    #     [0.45, 0.06, -0.035],
    #     [0.45, -0.07, -0.035],
    #     # back
    #     [-0.45, 0.05, 0.05],
    #     [-0.45, -0.05, 0.05],
    #     # left
    #     [0.13, 0.14, 0.01],
    #     [-0.13, 0.14, -0.01],
    #     # right
    #     [0.1, -0.15, -0.01],
    #     [-0.13, -0.14, -0.01],
    #     # top
    #     [0.1, 0.05, 0.09],
    #     [-0.12, -0.01, 0.09],
    # ]
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
        'arm0.sh0': 'arm_sh0',
        'arm0.sh1': 'arm_sh1',
        'arm0.el0': 'arm_el0',
        'arm0.el1': 'arm_el1',
        'arm0.wr0': 'arm_wr0',
        'arm0.wr1': 'arm_wr1',
        'arm0.f1x': 'arm_f1x',
        'arm0.hr0': 'arm_hr0',
    }
    
    # markers_pos, pos_indices = find_closest_vertices(robot_meshes[0], markers_pos)
    marker_positions = {f"{i}": pos for i, pos in enumerate(markers_pos)}
    print(f"Loaded marker positions: {len(marker_positions)}")


    folder_path =  Path(__file__).parent
    torque_dir = os.path.join(folder_path, options.data_dir)
    subdirs = natsorted(os.listdir(torque_dir))
    torque_files = [f for f in os.listdir(os.path.join(torque_dir, subdirs[0])) if f.endswith('.npy')]
    torque_files = natsorted(torque_files)
    # get all the file names
    classes = [f.split('.')[0] for f in torque_files]
    coordinates = {}
    print(f"class: {classes}")
    for c in classes:
        if marker_positions.get(c) is None:
            coordinates['100'] = np.array([0, 0, 0])
            # coordinates.append(np.array([0, 0, 0]))
        else:
            # coordinates.append(marker_positions.get(c))
            coordinates[c] = marker_positions.get(c)

   
    vis = o3d.visualization.Visualizer() 
    vis.create_window()
    visualizer = SpotVisualizer(vis=vis)

    radius = 0.04
    alpha = 0.95
    sliding_win = 3
    seq_win = seq
    # sliding_win = 5
    if classify:
        buffer = np.zeros((sliding_win, 101))
    else:
        buffer = np.zeros((sliding_win, 3))
    data_buffer = np.zeros((seq_win, 24))
    weights = np.power((1 - alpha), np.arange(sliding_win))
    weights = alpha * weights
    # normalize - in the regression case, weighted average
    if not classify:
        weights = weights / np.sum(weights)

    original_colors = [np.asarray(pcd.colors).copy() for pcd in visualizer.point_clouds]
    # original_vertex_colors = np.asarray(total_mesh.vertex_colors).copy()
   
    # Add the combined mesh to the visualizer
    all_points = np.concatenate([np.asarray(pcd.points) for pcd in visualizer.point_clouds])
    kdtree = cKDTree(all_points)

    point_cloud_sizes = [len(np.asarray(pcd.points)) for pcd in visualizer.point_clouds]
    point_cloud_boundaries = np.cumsum([0] + point_cloud_sizes) 


    try:
        while True:
            start = time.time()
            data_buffer = np.roll(data_buffer, 1, axis=0) 
            state = robot_state_client.get_robot_state()

            # Preprocess the data for inference
            processed_data = preprocess_realtime_data(state, markers_path)
            data_buffer[0] = processed_data
            print(f"Processed data shape: {processed_data.shape}")
            joint_positions = {joint.name: 0.0 for joint in visualizer.robot.joints}
            joint_states = state.kinematic_state.joint_states
            for joint_info in joint_states:
                joint = visualizer.robot.joint_map[simplified_to_full_name.get(joint_info.name)]
                if joint:
                    joint_positions[simplified_to_full_name.get(joint_info.name)] = joint_info.position.value
                else:
                    print(f"Joint {joint_info['name']} not found in URDF.")
            visualizer.visualize(cfg=joint_positions)

            # Real time prediction
            buffer = np.roll(buffer, 1, axis=0) 
            # processed_data = data_buffer.flatten()
            processed_data_tensor = torch.tensor(data_buffer.flatten(), dtype=torch.float32).to(model.device).reshape(1, -1)
            with torch.no_grad():
                if device == "gpu":
                    buffer[0:seq] = model.predict(processed_data_tensor).cpu().numpy().reshape(seq, -1)
                else:
                    result = model.predict(processed_data_tensor).numpy()
                    buffer[0:seq] = model.predict(processed_data_tensor).numpy().reshape(seq, -1)
            predictions = np.dot(weights, buffer)
            if classify:
                    predictions = predictions.reshape(-1, 101)
                    predicted_class_index = np.argmax(predictions)
                    confidence = np.max(predictions)
                    predicted_class = classes[predicted_class_index]
                    print(f"Prediction: {predicted_class}, Confidence: {confidence:.2f}")
                    if predicted_class == "no_contact":
                        pos = np.array([0, 0, 0])
                    else:
                        pos = marker_positions.get(predicted_class)
            else:
                pos = predictions
                # Compute the weighted variance
                weighted_mean = predictions
                differences = buffer - weighted_mean  # Difference between each row and the mean
                squared_differences = differences**2
                weighted_variance = np.dot(weights, np.mean(squared_differences, axis=1))  # Average squared differences
                confidence = 1 / (1 + np.sqrt(weighted_variance))  # Inverse relation: lower variance → higher confidence

            for pcd, orig_color in zip(visualizer.point_clouds, original_colors):
                pcd.colors = o3d.utility.Vector3dVector(orig_color)

            indices = kdtree.query_ball_point(pos, radius)
            print(f"pos: {pos}")
            for idx in indices:
                pcd_idx = np.searchsorted(point_cloud_boundaries, idx, side='right') - 1
                local_idx = idx - point_cloud_boundaries[pcd_idx]
                    
                colors = np.asarray(visualizer.point_clouds[pcd_idx].colors)
                colors[local_idx] = [1, 0, 0]
                visualizer.point_clouds[pcd_idx].colors = o3d.utility.Vector3dVector(colors)

            vis.poll_events()
            vis.update_renderer()
            threshold = 1
            # distance = np.linalg.norm(pos - coordinates.get("100"))
            # print(f"pos: {pos}, distance: {distance}")
            # if distance < threshold:
            # step, trot, turn_2step, twerk, unstow
            # left
            if pos[1] > 0.09 and -0.35 < pos[0] < 0.3 and pos[2] < 0.11:
                exe_choreo(choreos[0], choreography_client) # step
            # right
            elif pos[1] < -0.11 and -0.35 < pos[0] < 0.3 and pos[2] < 0.11:
                exe_choreo(choreos[1], choreography_client) # trot
            # front
            elif pos[0] > 0.15 and pos[2] < 0.1 and -0.14 < pos[1] < 0.14:
                exe_choreo(choreos[2], choreography_client) # turn_2step
            # back
            elif pos[0] < -0.45 and pos[2] < 0.1:
            # and -0.14 < pos[1] < 0.14:
                exe_choreo(choreos[3], choreography_client) # twerk
            # top
            elif pos[2] > 0.11:
                exe_choreo(choreos[4], choreography_client) # unstow
            else: 
                continue
                
    except KeyboardInterrupt:
        print("Exiting real-time inference...")
    # except (KeyboardInterrupt, Exception) as e:
    #     print("Stopping...")
    #     # Try to stop any ongoing choreography
    #     try:
    #         choreography_client.stop_choreography(lease=lease_proto)
    #     except Exception as e:
    #         print(f"Error stopping choreography: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    # finally:
    #     # Always clean up lease management
    #     if 'lease_client' in locals() and 'lease' in locals():
    #         print("Returning lease...")
    #         lease_client.return_lease(client_lease)
    #     if 'lease_keep_alive' in locals():
    #         lease_keep_alive.shutdown()
    #     print("Lease returned successfully")


if __name__ == "__main__":
    main()