import numpy as np

def load_joint_torques(torque_path, franka=False):
    # load the npy data
    # torque_path = "data/touch_back.npy"
    state = np.load(torque_path, allow_pickle=True)
    if franka:
        return state[:, :7], state.shape[0], 7, ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
    state_dict = {}
    torque_dict = {}
    for i in range(len(state)):
        state_dict[i] = state[i].kinematic_state.joint_states
        torque_dict[i] = []
        for joint in state_dict[i]:
            joint_name = getattr(joint, 'name', None)
            if joint_name is not None:
                # if not joint_name.startswith("arm"):
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
                # if not joint_name.startswith("arm"):
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



def sample_points_from_mesh(vertices, faces, num_points):
    # Compute triangle areas
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    cross_product = np.cross(v2 - v1, v3 - v1)
    triangle_areas = np.linalg.norm(cross_product, axis=1) / 2

    # Normalize areas to sum to 1
    area_cumsum = np.cumsum(triangle_areas)
    area_cumsum /= area_cumsum[-1]

    # Sample triangles based on area
    samples = np.random.rand(num_points)
    triangle_indices = np.searchsorted(area_cumsum, samples)

    # Sample points within triangles
    r1 = np.sqrt(np.random.rand(num_points))
    r2 = np.random.rand(num_points)
    u = 1 - r1
    v = r1 * (1 - r2)
    w = r1 * r2

    sampled_triangles = faces[triangle_indices]
    p1 = vertices[sampled_triangles[:, 0]]
    p2 = vertices[sampled_triangles[:, 1]]
    p3 = vertices[sampled_triangles[:, 2]]

    points = u[:, None] * p1 + v[:, None] * p2 + w[:, None] * p3
    return points

