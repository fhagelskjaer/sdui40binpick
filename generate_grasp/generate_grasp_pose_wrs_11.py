import open3d as o3d
import joblib

from utils import compute_obj_orientation, create_pose, write_data, visualize_pose


def main():
    grasp_pose_dictionary = {}

    # visualize = True
    visualize = False

    tcp_length = 230


    output_name = "grasp_pose_obj_wrs_11.pickle" 
    model_name_def = "../data/11_KZAF1075NA4WA55GA20AA0.stl"

    object_model = o3d.io.read_triangle_mesh(model_name_def)

    angle_resolution = 90 if visualize else 5

    grasp_pose_dictionary = {}

    grasp_offset = [0, 0, -3.0]
    finger_opening = 7
    orient_object = compute_obj_orientation(model_name_def, center_offset=[0, 0, 0], rotation_info=['z', 90])
    x_angle = 0

    for glob_angle, finger_width in [[0, finger_opening], [180, finger_opening]]:
        for angle in range(0,360,angle_resolution):
            obj2tcp = create_pose(tcp_length, grasp_offset, x_angle, angle, glob_angle, orient_object)
            write_data(grasp_pose_dictionary, angle, glob_angle, finger_width, obj2tcp)
            if visualize:
                visualize_pose(finger_width, tcp_length, obj2tcp, object_model)

    joblib.dump(grasp_pose_dictionary, output_name)

if __name__ == "__main__":
    main()