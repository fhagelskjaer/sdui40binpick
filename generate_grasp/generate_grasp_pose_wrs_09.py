import numpy as np
import trimesh
import open3d as o3d
# import json
import joblib

output_name = "grasp_pose_obj_wrs_000009.pickle" 

model_name_def = "../data/09_BGPSL6-9-L30-F7.stl"

mesh = o3d.io.read_triangle_mesh(model_name_def) 
mesh.paint_uniform_color([1, 0.706, 0])

fuze_trimesh = trimesh.load( model_name_def )
bounding_transform = fuze_trimesh.bounding_box_oriented.primitive.transform

tcp_length = 230

print( bounding_transform )


#mesh_cylinder.transform(np.array([[1.0,0,0, bounding_transform[0,3] ],[0,1,0, bounding_transform[1,3] ],[0,0,1, bounding_transform[2,3] + 50.0 ],[0,0,0,1]]))


grasp_pose_dictionary = {} # json.loads(json_string)



for glob_theta, finger_width in [[0, 5], [np.pi, 5]]:
# for glob_theta, finger_width in [[np.pi, 20]]:
    
    for offsets in [8,0,-10]:
    
        for angle in range(0,360,5):

            theta = np.pi * angle / 180.0

            tcp2obj =  np.dot( np.dot( np.dot( np.array([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]]), 
             np.array([[1.0,0,0,offsets],
                [0, np.cos(theta),-np.sin(theta),0],
                [0, np.sin(theta),np.cos(theta),tcp_length-3], #28.0],
                [0,0,0,1]])),
                np.array([[np.cos(glob_theta),0,-np.sin(glob_theta),0],[0,1,0,0],[np.sin(glob_theta),0,np.cos(glob_theta),0],[0,0,0,1]])),
                np.array([[1.0,0,0,-bounding_transform[0,3]],
                [0,1,0,-bounding_transform[1,3]],
                [0,0,1,-bounding_transform[2,3]],
                [0,0,0,1]]))

            obj2tcp = np.linalg.inv(tcp2obj)
                
            # grasp_pose_dictionary[str(angle)] = obj2tcp
            grasp_pose_dictionary[str(offsets)+"_"+str(glob_theta)+"_"+str(angle)] = [obj2tcp, finger_width]

            # visulizePose(finger_width, obj2tcp, mesh)


        
# json.dump(output_name, grasp_pose_dictionary)   
joblib.dump(grasp_pose_dictionary, output_name) 
