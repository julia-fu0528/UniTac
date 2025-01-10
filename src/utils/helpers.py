import numpy as np
import open3d as o3d

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

