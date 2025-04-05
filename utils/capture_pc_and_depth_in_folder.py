"""
Capture point clouds, with color, from the Zivid camera.

For scenes with high dynamic range we combine multiple acquisitions to get an HDR point cloud.
"""

import zivid
from zivid.experimental import calibration
import open3d as o3d
import numpy as np
import cv2
import os

def _point_cloud_to_cv_depth(point_cloud):
    depth_map = point_cloud.copy_data("z")
    
    depth_map[np.isnan(depth_map)[:, :]] = 0
    
    depth_map[ depth_map >= (2**16)/10 ] = 0

    return depth_map

class ZividPcCam:
    def __init__(self, settings_file = "utils/nov13_spec.yml"):
        app = zivid.Application()

        print("Connecting to camera")
        self.camera = app.connect_camera()

        print("Configuring settings")
        self.settings = zivid.Settings.load(settings_file)

        camera_intrinsics = calibration.intrinsics(self.camera)
        self.mtx = np.array([[camera_intrinsics.camera_matrix.fx, 0.0, camera_intrinsics.camera_matrix.cx], [0.0, camera_intrinsics.camera_matrix.fy, camera_intrinsics.camera_matrix.cy], [0, 0, 1.0]])
        self.dist = np.array([camera_intrinsics.distortion.k1, camera_intrinsics.distortion.k2, camera_intrinsics.distortion.p1,camera_intrinsics.distortion.p2, camera_intrinsics.distortion.k3])

        self.width, self.height = (1944,1200)

        self.newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,(self.width,self.height),0, (self.width,self.height))

    def compute(self):
        print("Capturing frame (HDR)")
        with self.camera.capture(self.settings) as frame:
            point_cloud = frame.point_cloud()

        depth_map = _point_cloud_to_cv_depth(point_cloud)

        xyz = point_cloud.copy_data("xyz")
        rgba = point_cloud.copy_data("rgba")
        xyz = np.nan_to_num(xyz).reshape(-1, 3)
        rgb=rgba[:, :, :3]
        
        rgb_vector = rgb.reshape(-1, 3)

        point_cloud_open3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        point_cloud_open3d.colors = o3d.utility.Vector3dVector(rgb_vector / 255)

        return point_cloud_open3d, depth_map, rgb

    def camera_matrix(self):
        return self.mtx, self.dist, self.newcameramtx

def main():

    zivid_cam = ZividPcCam() 
    
    print( zivid_cam.camera_matrix() )

    while True:
        input_string = input("write output folder name: ('q' to exit)\n") 

        if input_string == 'q':
            break

        parent_dir = "."
        path = os.path.join(parent_dir, input_string)  
        os.mkdir(path)
            
        target, depth, rgb = zivid_cam.compute()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        cv2.imwrite( path + '/' + "color.png", bgr)
        cv2.imwrite( path + "/depth_image_01mm_resolution.png", (depth*10).astype(np.uint16))

if __name__ == "__main__":
    main()
