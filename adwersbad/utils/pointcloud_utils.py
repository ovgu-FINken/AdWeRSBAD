import numpy as np
import open3d as o3d


def labels_to_rgb(labels):
    """Convert integer labels to RGB-like values."""
    # Normalize labels to a [0, 1] range for Open3D colors
    labels = np.asarray(labels, dtype=np.float32)

    # Handle the case where labels are not already RGB values
    unique_labels = np.unique(labels)
    label_to_rgb = {
        label: [label / unique_labels.max(), 0, 0] for label in unique_labels
    }

    rgb_labels = np.array([label_to_rgb[label] for label in labels])
    return rgb_labels, label_to_rgb


def rgb_to_labels(rgb_labels, label_to_rgb):
    """Convert RGB values back to integer labels."""
    inverse_label_map = {tuple(v): k for k, v in label_to_rgb.items()}
    labels = np.array(
        [inverse_label_map[tuple(rgb)] for rgb in rgb_labels], dtype=np.uint8
    )
    return labels


def downsample(cloud, labels, npts=16384):
    # Convert labels to RGB-like values
    rgb_labels, label_to_rgb = labels_to_rgb(labels)

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    pcd.colors = o3d.utility.Vector3dVector(rgb_labels)  # Attach labels as colors

    # Downsample the point cloud using farthest point sampling
    downsampled_pcd = pcd.farthest_point_down_sample(num_samples=npts)

    # Retrieve downsampled points and colors
    downsampled_cloud = np.asarray(downsampled_pcd.points, dtype=np.float32)
    downsampled_rgb_labels = np.asarray(downsampled_pcd.colors)

    # Convert RGB values back to original integer labels
    downsampled_labels = rgb_to_labels(downsampled_rgb_labels, label_to_rgb)

    return downsampled_cloud, downsampled_labels
