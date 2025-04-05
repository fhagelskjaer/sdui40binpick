import numpy as np
import open3d as o3d
import trimesh
import sys

def line_intersects_triangle(p1, p2, v0, v1, v2):
    """
    Check if the line segment defined by points p1 and p2 intersects the triangle defined by v0, v1, v2.

    :param p1: Starting point of the line (numpy array)
    :param p2: Ending point of the line (numpy array)
    :param v0: First vertex of the triangle (numpy array)
    :param v1: Second vertex of the triangle (numpy array)
    :param v2: Third vertex of the triangle (numpy array)
    :return: True if intersects, False otherwise
    """
    # Edge vectors
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(p2 - p1, e2)
    a = np.dot(e1, h)

    # If the determinant is near zero, the line is parallel to the triangle
    if abs(a) < 1e-10:
        return False

    f = 1.0 / a
    s = p1 - v0
    u = f * np.dot(s, h)

    # Check the barycentric coordinate u
    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, e1)
    v = f * np.dot(p2 - p1, q)

    # Check the barycentric coordinate v
    if v < 0.0 or u + v > 1.0:
        return False

    t = f * np.dot(e2, q)

    # Check if the intersection point is on the line segment
    if t >= 0 and t <= 1:  # Check bounds of the line segment
        return True

    return False


def line_intersects_mesh(p1, p2, triangles):
    """
    Check if the line segment intersects any triangle in the mesh.

    :param p1: Starting point of the line (numpy array)
    :param p2: Ending point of the line (numpy array)
    :param triangles: List of triangles defined by their vertices (list of numpy arrays)
    :return: True if intersects, False otherwise
    """
    for triangle in triangles:
        v0, v1, v2 = triangle
        if line_intersects_triangle(p1, p2, v0, v1, v2):
            return True
    return False

def create_new_cad(source_mesh):
    source = source_mesh.sample_points_uniformly(number_of_points=2000)

    bin_pointcloud = np.asarray(source.points)

    bin_mean_xyz = np.reshape(np.mean(np.asarray(source.points),axis=0), (3))


    vertice_list = np.asarray(source_mesh.vertices)
    triangle_list = np.asarray(source_mesh.triangles)

    triangles_not_in_collision = []

    points_inside = []

    for triangle_idx in range(len(triangle_list)):
        
        triangle_to_test = [vertice_list[triangle_list[triangle_idx][0]],vertice_list[triangle_list[triangle_idx][1]],vertice_list[triangle_list[triangle_idx][2]]]
        triangle_center = np.mean(triangle_to_test, axis=0)

        # print(bin_mean_xyz, triangle_center)
       
        triangles = []

        for c_t_i, triangle in enumerate(triangle_list):
        
            if c_t_i == triangle_idx:
                continue
            
            tri = []
            for veridx in triangle:
                tri.append(vertice_list[veridx])
            
            triangles.append(tri)

        # print( line_intersects_mesh(bin_mean_xyz, triangle_center, triangles) )
        if( not line_intersects_mesh(bin_mean_xyz, triangle_center, triangles) ):
            triangles_not_in_collision.append(triangle_list[triangle_idx])
            
    source_mesh.triangles = o3d.utility.Vector3iVector( np.array(triangles_not_in_collision, int) )

    return source_mesh

def main():
    bin_name = sys.argv[1]

    source_mesh = o3d.io.read_triangle_mesh(bin_name)

    source_mesh = create_new_cad(source_mesh)

    source_new = source_mesh.sample_points_uniformly(number_of_points=2000)
    o3d.visualization.draw_geometries([source_new])

if __name__ == "__main__":
    main()


