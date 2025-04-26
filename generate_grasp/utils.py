
import open3d as o3d
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

def create_pose(tcp_length, grasp_offset, x_angle, angle, glob_angle, orient_object):
    # gripper offset
    gripper_offset = np.eye(4)
    gripper_offset[2, 3] = tcp_length
    
    grasp_offset = np.array([[1,0,0,grasp_offset[0]], [0,1,0,grasp_offset[1]], [0,0,1,grasp_offset[2]], [0,0,0,1]])
    grasp_offset[:3, :3] = R.from_euler('x', [x_angle], degrees=True).as_matrix()
    
    # rotating finger
    theta_rot = np.eye(4)
    theta_rot[:3, :3] = R.from_euler('y', [angle], degrees=True).as_matrix()

    # global rotation
    glob_theta_rot = np.eye(4)
    glob_theta_rot[:3, :3] = R.from_euler('z', [glob_angle], degrees=True).as_matrix()

    tcp2obj = gripper_offset @ grasp_offset @ glob_theta_rot @ theta_rot
    
    obj2tcp = np.linalg.inv( tcp2obj @ orient_object )

    return obj2tcp

def write_data(grasp_pose_dictionary, angle, glob_angle, finger_width, obj2tcp):
    unique_index = len(grasp_pose_dictionary)
    grasp_pose_dictionary["uni"+str(unique_index)+"_"+str(glob_angle)+"_"+str(angle)] = [obj2tcp, finger_width]

def compute_obj_orientation(model_name_def, center_offset, rotation_info):
    fuze_trimesh = trimesh.load( model_name_def )
    bounding_transform = fuze_trimesh.bounding_box_oriented.primitive.transform #TODO bounding_box_?

    # object center
    center_object = np.eye(4)
    center_object[:3, 3] = -bounding_transform[:3, 3] 
    center_object[:3, 3] += center_offset

    # object rotation is computed from center and orientation
    r = R.from_euler(rotation_info[0], rotation_info[1], degrees=True)
    rotate_object = np.eye(4)
    rotate_object[:3, :3] = r.as_matrix()

    orient_object = rotate_object @ center_object

    return orient_object

def visualize_pose(finger_opening, tcp_length, obj2tcp, mesh):
    gun_metal_grey = [0.16470588, 0.20392157, 0.22352941]

    finger_width = 8
    mesh_flange = o3d.geometry.TriangleMesh.create_box(width=finger_opening*2,
                                                         height=finger_width*2,
                                                         depth=finger_width)
    mesh_flange.paint_uniform_color([1.0, 0, 0])
    mesh_flange.transform(np.array([[1.0, 0, 0, -(finger_opening)], [0, 1, 0, -finger_width/2], [0, 0, 1, finger_width/2], [0, 0, 0, 1]]))
    mesh_flange.transform(obj2tcp)
    
    mesh_cylinder = o3d.geometry.TriangleMesh.create_box(width=finger_width,
                                                         height=3.0,
                                                         depth=(tcp_length))

    mesh_cylinder.paint_uniform_color(gun_metal_grey)
    mesh_cylinder.transform(np.array([[1.0, 0, 0, -(finger_width/2)], [0, 1, 0, -3 -finger_opening], [0, 0, 1, 0], [0, 0, 0, 1]]))
    mesh_cylinder.transform(np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    mesh_cylinder.transform(obj2tcp)

    mesh_cylinder2 = o3d.geometry.TriangleMesh.create_box(width=finger_width,
                                                          height=3.0,
                                                          depth=(tcp_length))
    mesh_cylinder2.paint_uniform_color(gun_metal_grey)
    mesh_cylinder2.transform(np.array([[1.0, 0, 0, -(finger_width/2)], [0, 1, 0, finger_opening], [0, 0, 1, 0], [0, 0, 0, 1]]))
    mesh_cylinder2.transform(np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    mesh_cylinder2.transform(obj2tcp)

    o3d.visualization.draw_geometries([mesh, mesh_cylinder, mesh_cylinder2, mesh_flange])