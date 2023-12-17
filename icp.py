import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
def icp(A, B, max_iterations=100, tolerance=1e-4):
    """
    Iterative Closest Point (ICP) algorithm for 3D point clouds.

    Parameters:
    - A: numpy array, source point cloud (3 x N)
    - B: numpy array, target point cloud (3 x M)
    - max_iterations: int, maximum number of iterations
    - tolerance: float, convergence criterion

    Returns:
    - R: numpy array, rotation matrix (3 x 3)
    - T: numpy array, translation vector (3 x 1)
    - A_aligned: numpy array, aligned source point cloud (3 x N)
    """

    A_aligned = np.copy(A)
    for iteration in range(max_iterations):
        # Find the nearest neighbors between A_aligned and B
        kdtree = KDTree(B.T)
        distances, indices = kdtree.query(A_aligned.T)

        # Extract corresponding points from A_aligned and B
        correspondences_A = A_aligned[:, indices]
        correspondences_B = B[:, indices]

        # Calculate the transformation (R: rotation matrix, T: translation vector)
        R, T = estimate_rigid_transform(correspondences_A, correspondences_B)

        # Apply the transformation to A_aligned
        A_aligned = np.dot(R, A_aligned) + T

        # Check for convergence
        mean_distance = np.mean(distances)
        if mean_distance < tolerance:
            break

    return R, T, A_aligned

def estimate_rigid_transform(A, B):
    """
    Estimate rigid transformation (rotation matrix and translation vector) between two point clouds.

    Parameters:
    - A: numpy array, source points (3 x N)
    - B: numpy array, target points (3 x N)

    Returns:
    - R: numpy array, rotation matrix (3 x 3)
    - T: numpy array, translation vector (3 x 1)
    """
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    H = np.dot((A - centroid_A), (B - centroid_B).T)
    U, S, Vt = np.linalg.svd(H)

    R = np.dot(Vt.T, U.T)
    T = centroid_B - np.dot(R, centroid_A)

    return R, T

def load_point_cloud(file_path):
    """
    Load point cloud from a txt file.

    Parameters:
    - file_path: str, path to the point cloud file

    Returns:
    - point_cloud: open3d.geometry.PointCloud, loaded point cloud
    """
    points = np.loadtxt(file_path)  # Assuming points are separated by spaces
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud


def visualize_point_clouds(A, B, A_aligned):
    """
    Visualize the original and aligned point clouds.

    Parameters:
    - A: open3d.geometry.PointCloud, source point cloud
    - B: open3d.geometry.PointCloud, target point cloud
    - A_aligned: open3d.geometry.PointCloud, aligned source point cloud
    """
    # Visualize original point clouds
    A.paint_uniform_color([1, 0, 0])  # Red color for source point cloud
    B.paint_uniform_color([0, 0, 1])  # Blue color for target point cloud

    # Visualize aligned point cloud
    A_aligned.paint_uniform_color([0, 1, 0])  # Green color for aligned point cloud

    # Combine point clouds for visualization
    o3d.visualization.draw_geometries([A, B, A_aligned])


# 示例用法
if __name__ == "__main__":
    # 从文件加载点云
    file_path_A = "1_A.txt"
    file_path_B = "1_B.txt"
    
    A = load_point_cloud(file_path_A)
    B = load_point_cloud(file_path_B)

    # 应用ICP算法
    R_result, T_result, A_aligned = icp(np.asarray(A.points).T, np.asarray(B.points).T)

    # 可视化结果
    visualize_point_clouds(A, B, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(A_aligned.T)))