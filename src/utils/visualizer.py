from urdfpy import URDF
import trimesh
import numpy as np
import open3d as o3d
from collections import OrderedDict
from scipy.spatial import cKDTree

class RobotVisualizer():
    def __init__(self, robot_type, vis=None):
        self.vis = vis
        self.prev_fks = []
        self.default_pcds = []
        self.point_clouds = []
        self.robot_type = robot_type

        self.points_per_mesh = 300 if robot_type == "spot" else 3000
        self.robot = URDF.load(f"../{robot_type}_description/{robot_type}.urdf")
        
        # for markers
        self.pcd_indices = []
        self.local_indices = []
        
        # link -> body
        if robot_type == "spot":
            joint_positions = {'base_link_joint': 0.0, 'front_rail_joint': 0.0, 'rear_rail_joint': 0.0, 'front_left_hip_x': 0.0, 
                               'front_left_hip_y': 0.0, 'front_left_knee': 0.0, 'front_right_hip_x': 0.0, 'front_right_hip_y': 0.0, 
                               'front_right_knee': 0.0, 'rear_left_hip_x': 0.0, 'rear_left_hip_y': 0.0, 'rear_left_knee': 0.0, 
                               'rear_right_hip_x': 0.0, 'rear_right_hip_y': 0.0, 'rear_right_knee': 0.0, 'arm_sh0': -0.0005052089691162109, 
                               'arm_sh1': -3.119394302368164, 'arm_hr0': 0.0, 'arm_el0': 3.126546859741211, 'arm_el1': 1.569286823272705, 
                               'arm_wr0': -0.0010634660720825195, 'arm_wr1': -1.5685768127441406, 'arm_f1x': -0.0076329708099365234}
            self.fk_default = self.robot.visual_trimesh_fk(cfg=joint_positions)
        else:
            self.fk_default = self.robot.visual_trimesh_fk(cfg=None)
        # link -> world
        self.fk_meshes = []
        for tm in self.fk_default:
            self.fk_meshes.append(tm)
            self.fk_default[tm] = self.fk_default[tm]

        self.o3d_meshes_default = self.convert_trimesh_to_open3d(self.fk_default)

        # self.vis = vis
        if self.vis:
            # for geometry in self.o3d_meshes_default:
                # self.vis.add_geometry(geometry)

            # add ground plane
            ground_plane = o3d.geometry.TriangleMesh.create_box(width=5.0, height=5.0, depth=0.1)
            ground_plane.translate([-2.5, -2.5, -0.53-0.26])
            # self.vis.add_geometry(ground_plane)
            self.vis.poll_events()
            self.vis.update_renderer()


    def convert_trimesh_to_open3d(self, trimesh_fk):
        o3d_meshes = []
        if self.point_clouds == []:
            initializing = True
        else:
            initializing = False
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
            self.prev_fks.append(trimesh_fk[tm]) # world -> T1
            if len(tm.vertices) > 1000 and self.robot_type == "spot":
                pcd = o3d_mesh.sample_points_uniformly(number_of_points= 70 * self.points_per_mesh)  #1300
            else:
                pcd = o3d_mesh.sample_points_uniformly(number_of_points=self.points_per_mesh)
            if self.vis:
                self.vis.add_geometry(pcd)
            self.point_clouds.append(pcd)
            o3d_meshes.append(o3d_mesh)
        
        if initializing:
            self.default_pcds = self.point_clouds.copy()
            # self.default_pcds = [pcd.transform(trimesh_fk[tm]) for pcd, tm in zip(self.point_clouds, trimesh_fk)]

            self.all_points = np.concatenate([np.asarray(pcd.points) for pcd in self.default_pcds])
            self.kdtree = cKDTree(self.all_points)

            self.point_cloud_sizes = [len(np.asarray(pcd.points)) for pcd in self.default_pcds]
            self.point_cloud_boundaries = np.cumsum([0] + self.point_cloud_sizes) 

        return o3d_meshes
    
    def update_open3d_meshes(self, trimesh_fk):    
        for idx, tm in enumerate(trimesh_fk):
            current_transform = trimesh_fk[tm] # world -> T2

            # we want to transform T1 -> T2
            self.o3d_meshes_default[idx].transform(np.linalg.inv(self.prev_fks[idx]))
            self.o3d_meshes_default[idx].transform(current_transform)

            self.point_clouds[idx].transform(np.linalg.inv(self.prev_fks[idx]))
            self.point_clouds[idx].transform(current_transform)

            if self.vis:
                self.vis.update_geometry(self.point_clouds[idx])
            self.prev_fks[idx] = current_transform

    def visualize(self, cfg=None, odom=None):

        fk = self.robot.visual_trimesh_fk(cfg=cfg)
        if odom is None:
            odom = np.eye(4)
        for tm in fk:
            fk[tm] = odom @ fk[tm]

        # o3d.visualization.draw_geometries(self.o3d_meshes_default)
        if False:
            scene = trimesh.Scene()
            for tm in fk:
                scene.add_geometry(tm, transform=fk[tm])
            if show:
                scene.show()
            # scene.export(saved_name)
        else:
            self.update_open3d_meshes(fk)
            if self.vis:

                for geometry in self.o3d_meshes_default:
                    self.vis.update_geometry(geometry)
                self.vis.poll_events()
                self.vis.update_renderer()
            # else:
            #     o3d.visualization.draw_geometries(self.o3d_meshes_default)
            #     o3d.visualization.draw_geometries(self.point_clouds)
    
    # def marker_2vert(self, markers_pos, radius = 0.04):
    #     all_points = np.concatenate([np.asarray(pcd.points) for pcd in self.point_clouds])
    #     kdtree = cKDTree(all_points)

    #     point_cloud_sizes = [len(np.asarray(pcd.points)) for pcd in self.point_clouds]
    #     point_cloud_boundaries = np.cumsum([0] + point_cloud_sizes) 
        
    #     for marker_pos in markers_pos:
    #         indices = kdtree.query_ball_point(marker_pos, radius)
    #         cur_marker_pcd_indices = []
    #         cur_marker_local_indices = []
    #         for idx in indices:
    #             pcd_idx = np.searchsorted(point_cloud_boundaries, idx, side='right') - 1
    #             local_idx = idx - point_cloud_boundaries[pcd_idx]

    #             cur_marker_pcd_indices.append(pcd_idx)
    #             cur_marker_local_indices.append(local_idx)
    #         self.pcd_indices.append(cur_marker_pcd_indices)
    #         self.local_indices.append(cur_marker_local_indices)
            
    def marker_2vert(self, markers_pos, radius=0.04):
        all_points = np.concatenate([np.asarray(pcd.points) for pcd in self.point_clouds])
        kdtree = cKDTree(all_points)

        point_cloud_sizes = [len(np.asarray(pcd.points)) for pcd in self.point_clouds]
        point_cloud_boundaries = np.cumsum([0] + point_cloud_sizes) 

        print(f"markers_pos:{len(markers_pos)}")
        count = 0
        for marker_pos in markers_pos:
            print(f"idx = {count}")
            count += 1
            indices = kdtree.query_ball_point(marker_pos, radius)
            
            cur_marker_pcd_indices = []
            cur_marker_local_indices = []

            print(f"Marker Position: {marker_pos}")  # <-- Check what position is being searched

            for idx in indices:
                pcd_idx = np.searchsorted(point_cloud_boundaries, idx, side='right') - 1
                local_idx = idx - point_cloud_boundaries[pcd_idx]

                cur_marker_pcd_indices.append(pcd_idx)
                cur_marker_local_indices.append(local_idx)

                # print(f"   Closest Point: {all_points[idx]} from Point Cloud {pcd_idx}")  # <-- Debug nearest point

            self.pcd_indices.append(cur_marker_pcd_indices)
            self.local_indices.append(cur_marker_local_indices)


    def pos_2pcd(self, pos, radius = 0.04):
        indices = self.kdtree.query_ball_point(pos, radius)
        cur_marker_pcd_indices = []
        cur_marker_local_indices = []
        for idx in indices:
            pcd_idx = np.searchsorted(self.point_cloud_boundaries, idx, side='right') - 1
            local_idx = idx - self.point_cloud_boundaries[pcd_idx]
        
            cur_marker_pcd_indices.append(pcd_idx)
            cur_marker_local_indices.append(local_idx)
        
        return cur_marker_pcd_indices, cur_marker_local_indices






if __name__ == "__main__":
    visualizer = RobotVisualizer()
    visualizer.visualize()