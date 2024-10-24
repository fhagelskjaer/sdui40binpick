import numpy as np
import cv2
import open3d as o3d
import sys

from to_cloud import create_point_cloud_rgb, create_point_cloud

def select_word_precense_new(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    # vis.create_window()
    vis.create_window(
        window_name="Open3D -- Select bin position",
        width=1920 // 2,
        height=1080,
        left=1920 // 2,
        top=0,
        visible=True,
    )
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    return vis.get_picked_points()
   
   
def create_matrix_string(transformation):
    output_string = "["
    for i in range(4):
        output_string += "["
        for j in range(4):
            output_string += str(transformation[i,j])
            output_string += ","
        output_string += "],"
    output_string += "]"
    return output_string


intrinsics = (1.78657788e+03, 1.78574548e+03, 9.84213745e+02, 6.09480774e+02)
image_size = (1200,1944)
    
print( sys.argv[0] )
    
depth_image_file = sys.argv[2]

if depth_image_file.split(".")[-1] == "pcd":
    target = o3d.io.read_point_cloud(depth_image_file)
else:
    depth = cv2.imread(depth_image_file, cv2.IMREAD_ANYDEPTH)
    depth = depth.astype(np.float32)/10        
    if len(sys.argv) == 4:
        rgb_image_file = sys.argv[2]
        bgr = cv2.imread(rgb_image_file)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        target = create_point_cloud_rgb(depth, rgb, intrinsics)
    else:
        target = create_point_cloud(depth, intrinsics)
    
# target = target.scale(1000.0, np.array([0.0,0.0,0.0])) # only for realsense

source_mesh = o3d.io.read_triangle_mesh(sys.argv[1])
source_mesh.paint_uniform_color([0.16470588, 0.20392157, 0.22352941]) # Gun-metal gray
source = source_mesh.sample_points_uniformly(number_of_points=20000)


source_points = select_word_precense_new(source)
target_points = select_word_precense_new(target)

corres = o3d.utility.Vector2iVector() 
        
for n, point_pred in enumerate(target_points):
    #corres.append( np.array([target_points[n], source_points[n] ]) )
    #print( np.array([target_points[n], source_points[n] ]) )
    
    print( "First object, then scene", np.array([source_points[n], target_points[n]] ) )
    corres.append( np.array([source_points[n], target_points[n]]) )

print( corres )

max_correspondence_distance = 20

result = o3d.pipelines.registration.registration_ransac_based_on_correspondence( source, target, corres, max_correspondence_distance,
    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n = 3, 
    checkers=[
        #o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(
        #    np.pi),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
            max_correspondence_distance)
    ],
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(100,0.999) ) 
    
print( result.transformation )
    
result = o3d.pipelines.registration.registration_icp(source, target, 2.5, result.transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 10))
     
print( "Found Transform:" )

output_string = create_matrix_string(result.transformation)

print( output_string )

source_mesh.transform(result.transformation)
o3d.visualization.draw_geometries([source_mesh, target])
