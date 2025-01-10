from urdfpy import URDF
import trimesh
import numpy as np
import open3d as o3d
from collections import OrderedDict

class SpotVisualizer():
    def __init__(self, vis=None):
        self.vis = vis
        self.prev_fks = []
        self.point_clouds = []
        self.points_per_mesh = 300
        self.robot = URDF.load("spot_description/spot.urdf")
        
        # link -> body
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
        self.point_clouds = []
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
            if len(tm.vertices) > 1000:
                pcd = o3d_mesh.sample_points_uniformly(number_of_points=90 * self.points_per_mesh)
            else:
                pcd = o3d_mesh.sample_points_uniformly(number_of_points=self.points_per_mesh)
            self.vis.add_geometry(pcd)
            self.point_clouds.append(pcd)
            o3d_meshes.append(o3d_mesh)
        return o3d_meshes
    
    def update_open3d_meshes(self, trimesh_fk):    
        for idx, tm in enumerate(trimesh_fk):
            current_transform = trimesh_fk[tm] # world -> T2

            # we want to transform T1 -> T2
            self.o3d_meshes_default[idx].transform(np.linalg.inv(self.prev_fks[idx]))
            self.o3d_meshes_default[idx].transform(current_transform)

            self.point_clouds[idx].transform(np.linalg.inv(self.prev_fks[idx]))
            self.point_clouds[idx].transform(current_transform)

            self.vis.update_geometry(self.point_clouds[idx])
            self.prev_fks[idx] = current_transform

    def visualize(self, cfg=None, odom=None):

        fk = self.robot.visual_trimesh_fk(cfg=cfg)
        if odom is None:
            odom = np.eye(4)
        for tm in fk:
            fk[tm] = odom @ fk[tm]

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
            else:
                o3d.visualization.draw_geometries(self.o3d_meshes_default)
        


if __name__ == "__main__":
    visualizer = SpotVisualizer()
    visualizer.visualize()