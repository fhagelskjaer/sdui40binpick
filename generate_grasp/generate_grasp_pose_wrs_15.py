import numpy as np
import trimesh
import open3d as o3d
import joblib
from scipy.spatial.transform import Rotation as R

from utils import compute_obj_orientation, create_pose, write_data, visualize_pose

def main():
    # visualize = True
    visualize = False

    angle_resolution = 90 if visualize else 5

    grasp_pose_dictionary = {}

    tcp_length = 230

    output_name = "grasp_pose_obj_wrs_15.pickle" 
    model_name_def = "../data/15_SBARB6200ZZ_30.stl"

    object_model = o3d.io.read_triangle_mesh(model_name_def)

    grasp_offset = [0, 0, -6.0]
    finger_opening = 20
    orient_object = compute_obj_orientation(model_name_def, center_offset=[0, 0, -10], rotation_info=['x', 90])


    for glob_angle, finger_width in [[0, finger_opening]]:
        for angle in range(0, 360, angle_resolution):
            for x_angle in [0, 30, 60, 90]:
                obj2tcp = create_pose(tcp_length, grasp_offset, x_angle, angle, glob_angle, orient_object)
                write_data(grasp_pose_dictionary, angle, glob_angle, finger_width, obj2tcp)
                if visualize:
                    visualize_pose(finger_width, tcp_length, obj2tcp, object_model)

    for glob_angle, finger_width in [[180, finger_opening]]:
        for angle in range(0, 360, angle_resolution):
            for x_angle in [0, -30, -60, -90]:
                obj2tcp = create_pose(tcp_length, grasp_offset, x_angle, angle, glob_angle, orient_object)
                write_data(grasp_pose_dictionary, angle, glob_angle, finger_width, obj2tcp)
                if visualize:
                    visualize_pose(finger_width, tcp_length, obj2tcp, object_model)

    grasp_offset = [0, 0, 20]
    finger_opening = 5
    orient_object = compute_obj_orientation(model_name_def, center_offset=[0, 0, 11], rotation_info=['x', 90])

    for glob_angle, finger_width in [[90, finger_opening], [270, finger_opening]]:
        for angle in range(0, 360, angle_resolution):
            for x_angle in [0]:
                obj2tcp = create_pose(tcp_length, grasp_offset, x_angle, angle, glob_angle, orient_object)
                write_data(grasp_pose_dictionary, angle, glob_angle, finger_width, obj2tcp)
                if visualize:
                    visualize_pose(finger_width, tcp_length, obj2tcp, object_model)

    grasp_offset = [-18, 0, 5]
    finger_opening = 15
    orient_object = compute_obj_orientation(model_name_def, center_offset=[0, 0, 0], rotation_info=['x', 90])

    for glob_angle, finger_width in [[0, finger_opening]]:
        for angle in range(0, 360, angle_resolution):
            for x_angle in [-90]:
                obj2tcp = create_pose(tcp_length, grasp_offset, x_angle, angle, glob_angle, orient_object)
                write_data(grasp_pose_dictionary, angle, glob_angle, finger_width, obj2tcp)
                if visualize:
                    visualize_pose(finger_width, tcp_length, obj2tcp, object_model)

    for glob_angle, finger_width in [[180, finger_opening]]:
        for angle in range(0, 360, angle_resolution):
            for x_angle in [90]:
                obj2tcp = create_pose(tcp_length, grasp_offset, x_angle, angle, glob_angle, orient_object)
                write_data(grasp_pose_dictionary, angle, glob_angle, finger_width, obj2tcp)
                if visualize:
                    visualize_pose(finger_width, tcp_length, obj2tcp, object_model)


    joblib.dump(grasp_pose_dictionary, output_name) 



if __name__ == "__main__":
    main()
