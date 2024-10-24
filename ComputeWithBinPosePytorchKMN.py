import torch
import torch.nn as nn
from sklearn.neighbors import KDTree
import trimesh
import numpy as np
import open3d as o3d
import cv2
from dgl.geometry import farthest_point_sampler
import copy
import joblib
import time
import uuid
import sys

from scipy.spatial import ConvexHull

from to_cloud import create_point_cloud, create_point_cloud_rgb, point_cloud_to_depth_map
from compute_inner_bin import create_new_cad
from edgecheck import loadModel, computeSingleCheck

from keymatchnet.model import DGCNN_gpvn
from keymatchnet.pe_utils import compute, mm_by_keypoint, addi
from keymatchnet.data import normalize_1d, normalize_2d, pc_center2cp

np.set_printoptions(suppress=True)


def visualize_coordinate(transform):
    coordinate_z = o3d.geometry.TriangleMesh.create_cylinder(radius=5, height=100.0)
    coordinate_z.compute_vertex_normals()
    coordinate_z.paint_uniform_color([0.1, 0.1, 0.9])
    coordinate_z.transform(np.array([[1.0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 50.0], [0, 0, 0, 1]]))
    coordinate_z.transform(transform)

    coordinate_x = o3d.geometry.TriangleMesh.create_cylinder(radius=5, height=100.0)
    coordinate_x.compute_vertex_normals()
    coordinate_x.paint_uniform_color([0.9, 0.1, 0.1])
    coordinate_x.transform(np.array([[1.0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 50.0], [0, 0, 0, 1]]))
    coordinate_x.transform(np.array([[0.0, 0, 1, 0], [0, 1, 0, 1], [-1, 0, 0, 0], [0, 0, 0, 1]]))
    coordinate_x.transform(transform)

    coordinate_y = o3d.geometry.TriangleMesh.create_cylinder(radius=5, height=100.0)
    coordinate_y.compute_vertex_normals()
    coordinate_y.paint_uniform_color([0.1, 0.9, 0.1])
    coordinate_y.transform(np.array([[1.0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 50.0], [0, 0, 0, 1]]))
    coordinate_y.transform(np.array([[1.0, 0, 0, 0], [0, 0, -1, 1], [0, 1, 0, 0], [0, 0, 0, 1]]))
    coordinate_y.transform(transform)

    return coordinate_z, coordinate_y, coordinate_x


def center_point_cloud_in_z(target, min_dist_z, max_dist_z):
    points = np.asarray(target.points)
    target = target.select_by_index(np.where(points[:, 2] < max_dist_z)[0])
    points = np.asarray(target.points)
    target = target.select_by_index(np.where(points[:, 2] > min_dist_z)[0])
    return target


class ParaPose:
    def __init__(self, grasp_pose_file_name, model_name_def, batch_size, num_point,
                 number_of_keypoints, min_num_point, camera_mat, image_size):

        self.num_point = num_point
        self.min_num_point = min_num_point
        self.batch_size = batch_size

        self.model_name_def = model_name_def

        self.number_of_keypoints = number_of_keypoints

        cad_string = model_name_def
        
        # load the object
        obj_cad = o3d.io.read_triangle_mesh(cad_string)
        self.obj_pc = obj_cad.sample_points_poisson_disk(num_point)
        self.obj_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30), False)

        self.radius = np.max(np.max(self.obj_pc.points, axis=0) - np.min(self.obj_pc.points, axis=0))

        self.model_center = self.obj_pc.get_center()

        self.scene, self.nc, self.r = loadModel(model_name_def,
                                               camera_mat,
                                               image_size
                                               )

        self.grasp_pose_dictionary = joblib.load(grasp_pose_file_name)
        
        self.bin_tree = None
        self.bin_tree_smaller = None
        self.hull = None


    def computePointsInBin(self, target, source_original, source_mesh_original, source_inner, source_mesh_inner, transformation_bin, voxel_grid_search_size, viz=False):
        
        start_time_seconds = time.time()
        
        source = copy.deepcopy(source_inner)
        source_mesh = copy.deepcopy(source_mesh_inner)

        result = o3d.pipelines.registration.registration_icp(source_original, target, 10, transformation_bin,
                                                             o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                             o3d.pipelines.registration.ICPConvergenceCriteria(
                                                             max_iteration=10))

        transformation_bin_corrected = result.transformation

        source.transform(transformation_bin_corrected)
    
        source_mesh.transform(transformation_bin_corrected)


        self.source_mesh = source_mesh

        if viz:
            print('ICP and T took %0.3f s' % (time.time() - start_time_seconds))
            o3d.visualization.draw_geometries([target, source_mesh])

        convex_hull = source_mesh.compute_convex_hull()
        bb = convex_hull[0].get_oriented_bounding_box()
        indicies = bb.get_point_indices_within_bounding_box(target.points)

        target_show = o3d.geometry.PointCloud()
        target_show.points = o3d.utility.Vector3dVector(np.asarray(target.points)[indicies, :])
        target_show.colors = o3d.utility.Vector3dVector(np.asarray(target.colors)[indicies, :])

        if viz:
            o3d.visualization.draw_geometries([target_show])

        bin_mean_xyz = np.reshape(np.mean(np.asarray(source.points),axis=0), (1,3))

        vertices = np.asarray(convex_hull[0].vertices)
        
        self.hull = ConvexHull( source.points )

        smaller_point_cloud = np.reshape(bin_mean_xyz, (3)) + (( np.asarray(source.points) - np.reshape(bin_mean_xyz, (3)) )*0.90)

        self.bin_tree = KDTree(np.asarray(source.points), leaf_size=2)
        self.bin_tree_smaller = KDTree(smaller_point_cloud, leaf_size=2)

        if viz:
            print('ICP and KDE tree took %0.3f s' % (time.time() - start_time_seconds))


        target_show_voxel = target_show.voxel_down_sample(voxel_grid_search_size)


        source_distance, indecies_source = self.bin_tree.query(np.asarray(target_show_voxel.points), k=1)
        small_distance, indecies_small = self.bin_tree_smaller.query(np.asarray(target_show_voxel.points), k=1)

        inside_the_bin = source_distance > small_distance 
        inside_the_bin = np.reshape(inside_the_bin, (-1))

        pcd_o3d_show_voxel = o3d.geometry.PointCloud()
        pcd_o3d_show_voxel.points = o3d.utility.Vector3dVector(np.asarray(target_show_voxel.points)[inside_the_bin, :])
        # pcd_o3d_show_voxel.colors = o3d.utility.Vector3dVector(np.asarray(target_show_voxel.colors)[inside_the_bin, :])

        if viz:
            # o3d.visualization.draw_geometries([pcd_o3d_show])
            print('Inside bin took %0.3f s' % (time.time() - start_time_seconds))

        
        pcd_o3d = target_show.voxel_down_sample(1)

        pcd_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
        pcd_o3d.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, -1.0]))


        pointcloud_pointnet_pvn = np.concatenate([np.asarray(pcd_o3d.points), np.asarray(pcd_o3d.normals)], axis=1)

        filtered_points = np.asarray(pcd_o3d_show_voxel.points)

        index_filter = list(range(filtered_points.shape[0]))
        np.random.shuffle(index_filter)

        filtered_points = filtered_points[index_filter[:self.batch_size], :]

        return filtered_points, pointcloud_pointnet_pvn, target_show, transformation_bin_corrected

    def computePoseList(self, device, model, target, depth, filtered_points, pointcloud_pointnet_pvn, start_time_seconds,
                        feature_threshold_for_vote,
                        min_addi_distance,
                        min_depth_count,
                        depth_background_distance,
                        depth_acc_dist,
                        depth_scale,
                        viz=False):

        if viz:
            print("Start of network")

        detections = []
        sorting_list = []
        center_list = []
        

        for cloud_index in range(len(filtered_points)):
            best_detection_scores = []
            
            object_xyz = np.asarray(self.obj_pc.points)
            
            # prepare the object point cloud
            x = torch.FloatTensor(object_xyz)
            x = np.reshape(x, (1, -1, 3))
            fpi = farthest_point_sampler(x, self.number_of_keypoints)

            object_xyz_feature = object_xyz[fpi[0], :]

            obj_pc_temp = copy.deepcopy(self.obj_pc)
            R = obj_pc_temp.get_rotation_matrix_from_xyz(
              (np.random.uniform(0, np.pi * 2), np.random.uniform(0, np.pi * 2), np.random.uniform(0, np.pi * 2)))
            obj_pc_temp.rotate(R, center=(0, 0, 0))
            model_pc_out = np.concatenate([np.asarray(obj_pc_temp.points), np.asarray(obj_pc_temp.normals)], axis=1)

            model_pc_out = normalize_2d(model_pc_out)

            obj_model = self.obj_pc
            obj = model_pc_out.astype('float32')
            fpi = fpi[0].cpu().numpy()

            mm_dist = mm_by_keypoint(np.asarray(obj_model.points)[fpi, :3])*4

            # compute point sphere
            point_check = [filtered_points[cloud_index, :3]]

            centerTreeFilter = KDTree(np.asarray([point_check[0]]), leaf_size=2)
            distFilter, indeciesFilter = centerTreeFilter.query(pointcloud_pointnet_pvn[:, :3], k=1)
            pointlist = pointcloud_pointnet_pvn[distFilter.flatten() < self.radius, :]

            if len(pointlist) < self.min_num_point:
                continue

            while len(pointlist) < self.num_point:
                pointlist = np.array(list(pointlist) + list(pointlist))
            np.random.shuffle(pointlist)

            data = pc_center2cp(np.array(pointlist[:self.num_point])[:, :6], point_check[0])
            data, _ = normalize_1d(data)
            data = data.astype('float32')
            scene_info = pointlist

            data, obj, fpi = torch.from_numpy(np.array([data])), torch.from_numpy(np.array([obj])), torch.from_numpy(np.array([fpi]))
            data, obj, fpi = data.to(device), obj.to(device), fpi.to(device)

            data = data.permute(0, 2, 1)
            obj = obj.permute(0, 2, 1)

            # run the model
            seg_pred, key_pred = model(data, obj, fpi, device)

            # compute the RANSAC
            result, inliers = compute( seg_pred, [scene_info], key_pred, [obj_model], fpi, self.number_of_keypoints, feature_threshold_for_vote, device, mm_dist=mm_dist)

            # variation of the inside box verification created for opening of maersk2
            if self.hull is not None:
                model_center_h = np.append(self.model_center, np.array([1]))
                center_points = result @ model_center_h
                
                center_list.append(center_points[:3]) # TODO <<<
                
                center_points = np.array([ center_points[:3] ])
            
                tolerance=1e-12
                in_hull = np.all(np.add(np.dot(center_points, self.hull.equations[:,:-1].T),
                        self.hull.equations[:,-1]) <= tolerance, axis=1)

                source_distance, indecies_source = self.bin_tree.query(np.asarray(center_points), k=1)
                small_distance, indecies_small = self.bin_tree_smaller.query(np.asarray(center_points), k=1)
                
                if not in_hull[0] or source_distance[0,0] < small_distance[0,0]:
                    if viz:
                        print("Object center outside bin rejected")
                    continue
            
            # perform a depth render comparison            
            if result[2,3] < 50:
                continue
    
            detection = [result, {'fit': inliers, 'depth_count': 0, 'id': str(uuid.uuid1().hex)}]
                                
            detection = computeSingleCheck(detection, 
                                            depth,
                                            self.scene,
                                            self.nc,
                                            self.r,
                                            depth_scale,
                                            depth_background_distance,
                                            depth_acc_dist)

            detections.append(detection)
            sorting_list.append(-detection[1]["depth_count"])
        
        if viz:
            print('Init PE lasted %0.3f s' % (time.time() - start_time_seconds))

        # sort according to depth overlap
        sorted_index = np.argsort(sorting_list)
        
        # non maximum suppresion according to addi overlap
        accepted_poses = []
        for pose_idx in sorted_index:
            pose = detections[pose_idx]
            
            if pose[1]['depth_count'] < min_depth_count:
                break
            
            unique_pose = True
            for acc_pose in accepted_poses:
                addi_score = addi(pose[0], acc_pose[0], self.obj_pc)
                if addi_score < min_addi_distance:
                    unique_pose = False
            if unique_pose:
                accepted_poses.append(pose)

        if viz:
            print('NMS lasted %0.3f s' % (time.time() - start_time_seconds))

        if viz:
            newtarget = copy.deepcopy(target)
            r = newtarget.get_rotation_matrix_from_xyz((0, np.pi, 0))
            newtarget.rotate(r, center=(0, 0, 0))

            to_draw = []

            for detection in accepted_poses:

                print("Final Fitness", detection[1]['fit'], detection[1]['depth_count'])
                source_temp = o3d.io.read_triangle_mesh(self.model_name_def)
                source_temp.paint_uniform_color([0, 1.0, 0])
                source_temp.transform(detection[0])
                source_temp.rotate(r, center=(0, 0, 0))
                to_draw.append(source_temp)

            scene_pc = o3d.geometry.PointCloud()
            scene_pc.points = o3d.utility.Vector3dVector(np.array(center_list))
            scene_pc.paint_uniform_color([1,0,0])
            scene_pc.rotate(r, center=(0, 0, 0))

            self.source_mesh.rotate(r, center=(0, 0, 0))

            o3d.visualization.draw_geometries(to_draw + [newtarget] + [scene_pc])


        return accepted_poses

    def computeGraspPoses(self, accepted_poses, bin_name, bin_transformation, target_show, finger_color,
                          minimum_angle_to_camera, z_distance_offset_for_collision_grasp, 
                          tcp_length, finger_width, finger_depth, viz=False):

        fuze_trimesh = trimesh.load(bin_name)
        fuze_trimesh = fuze_trimesh.apply_transform(bin_transformation)
        collision_manager = trimesh.collision.CollisionManager()
        collision_manager.add_object("bin", fuze_trimesh)

        mesh_bin = o3d.io.read_triangle_mesh(bin_name)
        mesh_bin.transform(bin_transformation)
        mesh_bin.paint_uniform_color([0.1, 0.1, 0.9])

        full_collision_free_list = []



        mesh_cylinder_og = o3d.geometry.TriangleMesh.create_box(width=finger_width,
                                                                height=finger_depth,
                                                                depth=(tcp_length))
        mesh_cylinder_og.paint_uniform_color(finger_color)
        mesh_cylinder2_og = o3d.geometry.TriangleMesh.create_box(width=finger_width,
                                                                 height=finger_depth,
                                                                 depth=(tcp_length))
        mesh_cylinder2_og.paint_uniform_color(finger_color)

        for detection in accepted_poses:

            detection_index_to_save = detection[1]['id']
            
            not_in_collision_list = []

            for grasp_index in self.grasp_pose_dictionary.keys():
                obj2tcp, finger_opening = self.grasp_pose_dictionary[str(grasp_index)]

                cam2tcp = np.dot(detection[0], obj2tcp)

                # get z vector from rotation matrix
                z_vector = cam2tcp[:3, 2]
                angle = np.arccos(z_vector[2])
                if angle > minimum_angle_to_camera:
                    continue

                mesh_cylinder = copy.deepcopy(mesh_cylinder_og)
                finger2opening = np.array([[1.0, 0, 0, -(finger_width / 2)], [0, 1, 0, -finger_depth -finger_opening], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
                finger2rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
                final_transform_1 = np.dot(np.dot(cam2tcp, finger2rotation), finger2opening)
                mesh_cylinder.transform(final_transform_1)

                mesh_cylinder2 = copy.deepcopy(mesh_cylinder2_og)
                finger2opening = np.array([[1.0, 0, 0, -(finger_width / 2)], [0, 1, 0, finger_opening], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
                finger2rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
                final_transform_2 = np.dot(np.dot(cam2tcp, finger2rotation), finger2opening)
                mesh_cylinder2.transform(final_transform_2)

                finger2opening = np.array([[1.0, 0, 0, 0], [0, 1, 0, -(finger_depth/2) - finger_opening], [0, 0, 1, tcp_length/2], [0, 0, 0, 1]], np.float64)
                finger2rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
                final_transform_1 = np.dot(np.dot(cam2tcp, finger2rotation), finger2opening)
                cyl1_trimesh = trimesh.creation.box(extents=(finger_width, finger_depth, tcp_length), transform=final_transform_1)
                
                finger2opening = np.array([[1.0, 0, 0, 0], [0, 1, 0, (finger_depth/2) + finger_opening], [0, 0, 1, tcp_length/2], [0, 0, 0, 1]], np.float64)
                finger2rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
                final_transform_2 = np.dot(np.dot(cam2tcp, finger2rotation), finger2opening)
                cyl2_trimesh = trimesh.creation.box(extents=(finger_width, finger_depth, tcp_length), transform=final_transform_2)

                if not collision_manager.in_collision_single(cyl1_trimesh) and \
                        not collision_manager.in_collision_single(cyl2_trimesh):
                    bb1 = mesh_cylinder.get_oriented_bounding_box()
                    indices_1 = bb1.get_point_indices_within_bounding_box(target_show.points)
                    bb2 = mesh_cylinder2.get_oriented_bounding_box()
                    indices_2 = bb2.get_point_indices_within_bounding_box(target_show.points)
                    
                    not_in_collision_list.append(
                        [grasp_index, cam2tcp, finger_opening, detection[0], len(indices_1) + len(indices_2),
                         detection_index_to_save])
                else:
                    # transform back a short distance
                    obj2tcp_distance = np.array(obj2tcp)

                    offset_2cm = np.eye(4)
                    offset_2cm[2, 3] = z_distance_offset_for_collision_grasp

                    obj2tcp_distance = np.dot(obj2tcp_distance, offset_2cm)

                    # if this is collision free
                    cam2tcp = np.dot(detection[0], obj2tcp_distance)

                    # make sure the grasp approach starts inside the bin
                    obj2tcp_distance_new = np.array(obj2tcp_distance)
                    offset_tcp = np.eye(4)
                    offset_tcp[2, 3] = tcp_length
                    obj2tcp_distance_new = np.dot(obj2tcp_distance_new, offset_tcp)
                    cam2tcp_test = np.dot(detection[0], obj2tcp_distance_new)                

                    source_distance, indecies_source = self.bin_tree.query( np.array([ cam2tcp_test[:3, 3] ]), k=1)
                    small_distance, indecies_small = self.bin_tree_smaller.query( np.array([ cam2tcp_test[:3, 3] ]), k=1)
                    if source_distance[0,0] < small_distance[0,0]:
                        continue

                    finger2opening = np.array([[1.0, 0, 0, 0], [0, 1, 0, -(finger_depth/2) - finger_opening], [0, 0, 1, tcp_length/2], [0, 0, 0, 1]], np.float64)
                    finger2rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
                    final_transform_1 = np.dot(np.dot(cam2tcp, finger2rotation), finger2opening)
                    cyl1_trimesh = trimesh.creation.box(extents=(finger_width, finger_depth, tcp_length), transform=final_transform_1)

                    finger2opening = np.array([[1.0, 0, 0, 0], [0, 1, 0, (finger_depth/2) + finger_opening], [0, 0, 1, tcp_length/2], [0, 0, 0, 1]], np.float64)
                    finger2rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
                    final_transform_2 = np.dot(np.dot(cam2tcp, finger2rotation), finger2opening)
                    cyl2_trimesh = trimesh.creation.box(extents=(finger_width, finger_depth, tcp_length), transform=final_transform_2)

                    # add to the list with a further index of the last to use
                    if not collision_manager.in_collision_single(cyl1_trimesh) and \
                            not collision_manager.in_collision_single(cyl2_trimesh):
                        not_in_collision_list.append(
                            [grasp_index, cam2tcp, finger_opening, detection[0], -1, detection_index_to_save])

            if viz:
                print("Number out of collision",
                    len(not_in_collision_list))

            if len(not_in_collision_list) > 0:
                full_collision_free_list += not_in_collision_list

        if len(full_collision_free_list) == 0:
            return None, None

        pose_estimation_dictionary = {"gel": full_collision_free_list,
                                      "bp": bin_transformation}

        if viz:
            cam2tcp = full_collision_free_list[0][1]
            finger_opening = full_collision_free_list[0][2]

            to_draw = [target_show]

            mesh_cylinder = o3d.geometry.TriangleMesh.create_box(width=finger_width,
                                                                 height=3.0,
                                                                 depth=tcp_length)
            mesh_cylinder.paint_uniform_color(finger_color)

            finger2opening = np.array(
                [[1.0, 0, 0, -(finger_width / 2)], [0, 1, 0, -finger_depth - finger_opening], [0, 0, 1, 0], [0, 0, 0, 1]],
                np.float64)
            finger2rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
            final_transform_1 = np.dot(np.dot(cam2tcp, finger2rotation), finger2opening)
            mesh_cylinder.transform(final_transform_1)

            mesh_cylinder2 = o3d.geometry.TriangleMesh.create_box(width=finger_width,
                                                                  height=3.0,
                                                                  depth=tcp_length)
            mesh_cylinder2.paint_uniform_color(finger_color)
            finger2opening = np.array(
                [[1.0, 0, 0, -(finger_width / 2)], [0, 1, 0, finger_opening], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
            finger2rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float64)
            final_transform_2 = np.dot(np.dot(cam2tcp, finger2rotation), finger2opening)
            mesh_cylinder2.transform(final_transform_2)

            to_draw.append(mesh_cylinder)
            to_draw.append(mesh_cylinder2)

            o3d.visualization.draw_geometries(to_draw)
            
        return pose_estimation_dictionary


def paramInitializer(args_model_root, args_k, args_emb_dims, num_key):

    cuda = torch.cuda.is_available()

    device = torch.device("cuda" if cuda else "cpu")
    model = DGCNN_gpvn(args_k, args_emb_dims, num_key, 0).to(device)
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model.load_state_dict(torch.load(args_model_root, map_location=torch.device("cuda" if cuda else "cpu")))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, device
    
    

def main():
    depth_image_file = sys.argv[1]
    
    if len(sys.argv) == 3:
        rgb_image_file = sys.argv[2]
        
    np.random.seed(0)

    from config import number_of_keypoints, batch_size, num_point, min_num_point, min_depth_count, model_root, camera_mat, image_size, gun_metal_grey, bin_name, bin_transformation, pose_estimation_parameter_list, min_pc_dist, max_pc_dist
    bin_transformation = np.array(bin_transformation)
    camera_mat = np.array(camera_mat)

    # load model
    model, device = paramInitializer(args_model_root=model_root, args_k=number_of_keypoints, args_emb_dims=1024, num_key=number_of_keypoints)

    # load the bin
    source_mesh_original = o3d.io.read_triangle_mesh(bin_name)
    source_mesh_inner = create_new_cad(source_mesh_original)
    
    source_mesh_original.paint_uniform_color(gun_metal_grey)
    source_original = source_mesh_original.sample_points_uniformly(number_of_points=2000)

    source_mesh_inner.paint_uniform_color(gun_metal_grey)
    source_inner = source_mesh_inner.sample_points_uniformly(number_of_points=20000)
    
    # load all objects
    parapose_list = []
    for name, grasp_pose_file_name in pose_estimation_parameter_list:
        print(name, grasp_pose_file_name)
        parapose_list.append(ParaPose(grasp_pose_file_name, name,
                                      batch_size, num_point, number_of_keypoints, min_num_point,
                                      camera_mat, image_size))

    print("Start of algorithm")

    while True:
        for input_idx in range(len(pose_estimation_parameter_list)):
            print(
                "Idx: {} : {}".format(input_idx, pose_estimation_parameter_list[input_idx][0].split("/")[-1]))
        print("'q' to exit")
        input_string = input("Enter idx for objects: ")
        if input_string == 'q':
            break
        try:
            requested_object = int(input_string)
            parapose = parapose_list[requested_object]
        except:
            print("Use accepted input integer")
            continue

        start_time_seconds = time.time()

        # read data
        
        intrinsics = [camera_mat[0,0], camera_mat[1,1], camera_mat[0,2], camera_mat[1,2]]
    
        if depth_image_file.split(".")[-1] == "pcd":
            target = o3d.io.read_point_cloud(depth_image_file)
            depth = point_cloud_to_depth_map(np.asarray(target.points), intrinsics, (image_size[1], image_size[0]) )
        else:
            depth = cv2.imread(depth_image_file, cv2.IMREAD_ANYDEPTH)
            depth = depth.astype(np.float32)/10        
            if len(sys.argv) == 3:
                bgr = cv2.imread(rgb_image_file)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                target = create_point_cloud_rgb(depth, rgb, intrinsics)
            else:
                target = create_point_cloud(depth, intrinsics)
                target.paint_uniform_color(gun_metal_grey)

        # remove unnessesary part of point cloud  
        target = center_point_cloud_in_z(target, min_pc_dist, max_pc_dist)

        print('Data collection took %0.3f s' % (time.time() - start_time_seconds))

        filtered_points, pointcloud_pointnet_pvn, target_show, transformation_bin_corrected = \
            parapose.computePointsInBin(target,
                                        source_original,
                                        source_mesh_original,
                                        source_inner,
                                        source_mesh_inner,
                                        bin_transformation,
                                        voxel_grid_search_size=2,
                                        viz = False)

        print('Initial processing took %0.3f s' % (time.time() - start_time_seconds))

        accepted_poses = parapose.computePoseList(device,
                                                  model,
                                                  target,
                                                  depth, 
                                                  filtered_points, 
                                                  pointcloud_pointnet_pvn,
                                                  start_time_seconds,
                                                  feature_threshold_for_vote=0.7,
                                                  min_addi_distance=5,
                                                  min_depth_count=min_depth_count,
                                                  depth_background_distance=0.005,
                                                  depth_acc_dist=0.0025,
                                                  depth_scale=1,
                                                  viz=True)

        if len(accepted_poses) == 0:
            continue

        pose_estimation_dictionary = parapose.computeGraspPoses(accepted_poses,
                                                                 bin_name,
                                                                 transformation_bin_corrected,
                                                                 target_show,
                                                                 finger_color=gun_metal_grey,
                                                                 minimum_angle_to_camera=0.52359877559,
                                                                 z_distance_offset_for_collision_grasp=-20,
                                                                 tcp_length=230,
                                                                 finger_width=8.0, 
                                                                 finger_depth=3.0,
                                                                 viz=True)
                                                                 
        print('Grasp estimation took %0.3f s' % (time.time() - start_time_seconds))


if __name__ == "__main__":
    main()
