import numpy as np
import cv2
import open3d as o3d
import sys

def point_cloud_to_depth_map(point_cloud, intrinsics, image_size):
    """
    Convert a point cloud to a depth map.

    Parameters:
        point_cloud (numpy.ndarray): A Nx3 array of 3D points.
        intrinsics (tuple): Camera intrinsics (fx, fy, cx, cy).
        image_size (tuple): Size of the output depth map (height, width).

    Returns:
        numpy.ndarray: A 2D array representing the depth map.
    """
    # Extract camera intrinsics
    fx, fy, cx, cy = intrinsics
    height, width = image_size
    
    # Create an empty depth map
    depth_map = np.zeros((height, width), dtype=np.float32)

    # Project 3D points back to 2D pixel coordinates
    for point in point_cloud:
        X, Y, Z = point
        
        # Skip points with zero or negative depth
        if Z <= 0:
            continue
            
        if Z >= (2**16)/10:
            continue
            
        # Project 3D points to 2D image plane
        # x = int((fx * X / Z) + cx)
        # y = int((fy * Y / Z) + cy)

        # Project 3D points to 2D image plane
        x = int(round((fx * X / Z) + cx))
        y = int(round((fy * Y / Z) + cy))

        
        # Check if projected coordinates are within the image boundaries
        if 0 <= x < width and 0 <= y < height:
            # Update depth map at the projected location
            depth_map[y, x] = Z  # Z is the depth value

    return depth_map


def depth_image_to_point_cloud(depth_image, intrinsics):
    # Get the dimensions of the depth image
    height, width = depth_image.shape
    
    # Create a meshgrid of pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()
    z = depth_image.flatten()
    
    # Filter out points with zero depth
    valid_indices = z > 0
    x = x[valid_indices]
    y = y[valid_indices]
    z = z[valid_indices]
    
    # Convert pixel coordinates to camera coordinates
    fx, fy, cx, cy = intrinsics
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z
    
    # Stack to create a point cloud
    points = np.vstack((X, Y, Z)).T
    
    return points


def create_point_cloud(depth_image, intrinsics):
    # Convert depth image to point cloud
    points = depth_image_to_point_cloud(depth_image, intrinsics)
    
    # Create Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    return point_cloud
    
    
def depth_image_and_rgb_to_point_cloud(depth_image, rgb_image, intrinsics):
    # Get the dimensions of the depth image
    height, width = depth_image.shape
    
    # Create a meshgrid of pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()
    z = depth_image.flatten()
    
    # Filter out points with zero depth
    valid_indices = z > 0
    x = x[valid_indices]
    y = y[valid_indices]
    z = z[valid_indices]
    
    # Convert pixel coordinates to camera coordinates
    fx, fy, cx, cy = intrinsics
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z
    
    # Stack to create a point cloud
    points = np.vstack((X, Y, Z)).T
    
    colors = rgb_image[y,x,:]/255
    
    return points, colors


def create_point_cloud_rgb(depth_image, rgb_image, intrinsics):
    # Convert depth image to point cloud
    points, colors = depth_image_and_rgb_to_point_cloud(depth_image, rgb_image, intrinsics)
    
    # Create Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    return point_cloud
    

# Example usage
if __name__ == "__main__":

    intrinsics = (1.78657788e+03, 1.78574548e+03, 9.84213745e+02, 6.09480774e+02)
    image_size = (1200,1944)

    point_cloud_name = sys.argv[1]

    bin_name = "/".join(point_cloud_name.split("/")[:-1]) 
    
    target = o3d.io.read_point_cloud(point_cloud_name)

     
    depth_image_10 = point_cloud_to_depth_map(np.asarray(target.points), intrinsics, image_size)
    
    point_cloud = create_point_cloud(depth_image_10, intrinsics)
       
    o3d.visualization.draw_geometries([point_cloud, target])
    
    cv2.imwrite(bin_name+"/depth_image_01mm_resolution.png", (depth_image_10*10).astype(np.uint16))
    
    ###### 
    depth_map = cv2.imread(bin_name+"/depth_image_01mm_resolution.png", cv2.IMREAD_ANYDEPTH)
    depth_map = depth_map.astype(np.float32)/10

    point_cloud = create_point_cloud(depth_map, intrinsics)
        
    o3d.visualization.draw_geometries([point_cloud, target])
