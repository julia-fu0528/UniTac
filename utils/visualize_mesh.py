import numpy as np
import open3d as o3d


def create_viewing_parameters(target_position, distance=1.0):
    """
    Create camera parameters to look at a specific point
    
    Args:
        target_position (np.ndarray): The point to look at [x, y, z]
        distance (float): Distance from the target point
    """
    target = np.array(target_position)
    print(f"type of target: {target}")
    print(f"type of distance: {distance}")
    # Default camera position and orientation
    eye = target + np.array([distance, distance, distance])
    print("after eye\n")
    up = np.array([0, 0, 1])
    
    # Custom views based on specific target positions
    if np.isclose(target[0], 0.445) and np.isclose(target[1], 0.0) and np.isclose(target[2], 0.05):  # Front
        eye = target + np.array([-distance, 0, 0.2])  # Look from front
    elif np.isclose(target[0], -0.42) and np.isclose(target[1], 0.0) and np.isclose(target[2], 0.02):  # Back
        eye = target + np.array([distance, 0, 0.2])  # Look from back
    elif np.isclose(target[1], 0.11):  # Right
        eye = target + np.array([0, -distance, 0.2])  # Look from right
    elif np.isclose(target[1], -0.11):  # Left
        eye = target + np.array([0, distance, 0.2])  # Look from left
    
    # Calculate vectors for view matrix
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    # Ensure the up vector is not parallel to the forward vector
    if np.dot(forward, up) > 0.99:  # If too parallel, adjust
        up = np.array([0, 1, 0])
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)  # Recalculate up vector for orthogonality
    
    # Create rotation matrix (each row is a basis vector)
    R = np.vstack([right, up, -forward])
    
    # Create extrinsic matrix
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = -R @ eye  # Translation part
    
    # Initialize visualization to get default intrinsics
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    vis.destroy_window()
    
    # Set camera parameters
    new_params = o3d.camera.PinholeCameraParameters()
    new_params.intrinsic = params.intrinsic
    new_params.extrinsic = extrinsic
    
    return new_params

def visualize_with_camera(geometries, camera_params):
    """
    Visualize geometries with specific camera parameters
    
    Args:
        geometries (list): List of Open3D geometries to visualize
        camera_params: Camera parameters from create_viewing_parameters
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=2048, height=1500)
    
    # Add geometries
    for geometry in geometries:
        vis.add_geometry(geometry)
    
    # Set the camera parameters
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(camera_params)
    
    # Set rendering options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.8, 0.8, 0.8])  # Light gray background
    opt.point_size = 5.0
    
    # Update geometry to ensure proper rendering
    for geometry in geometries:
        vis.update_geometry(geometry)
    
    vis.poll_events()
    vis.update_renderer()
    
    vis.run()
    vis.destroy_window()


def look_at(camera_pos, target_pos, up_vector):
    forward = target_pos - camera_pos
    forward /= np.linalg.norm(forward)
    
    right = np.cross(up_vector, forward)
    right = right / np.linalg.norm(right)

    up = np.cross(forward, right)

    view_matrix = np.eye(4)
    view_matrix[:3, :3] = np.stack([right, up, forward], axis=1)
    view_matrix[:3, 3] = -view_matrix[:3, :3] @ camera_pos
    return view_matrix
