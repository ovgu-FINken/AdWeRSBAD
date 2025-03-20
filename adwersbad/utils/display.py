import random
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion

from adwersbad.class_helpers import adwersbad_labels, create_adwersbad_label_map
from adwersbad.utils.transform import (
    project_points_to_image,
    project_vehicle_to_image_waymo,
    transform_pcl_to_camera,
    transform_pcl_to_ego,
)


def show_pcl_with_o3d(
    points: np.ndarray, second_points: Optional[np.ndarray] = None
) -> None:
    """
    Displays the point cloud using Open3D and optionally overlays a second point cloud in red.

    Args:
        points (np.ndarray): The first point cloud data in shape (N, 3).
        second_points (Optional[np.ndarray]): The second point cloud data in shape (M, 3), displayed in red.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 1, 0])  # Green

    if second_points is not None:
        pcd_second = o3d.geometry.PointCloud()
        pcd_second.points = o3d.utility.Vector3dVector(second_points)
        pcd_second.paint_uniform_color([1, 0, 0])  # Red
        o3d.visualization.draw_geometries(
            [pcd, pcd_second], window_name="Point Cloud", width=800, height=600
        )
    else:
        o3d.visualization.draw_geometries(
            [pcd], window_name="Point Cloud", width=800, height=600
        )


def show_multiple_pcls(pointclouds: List[np.ndarray]) -> None:
    """
    Displays multiple point clouds using Open3D, assigning each a different random color.

    Args:
        pointclouds (List[np.ndarray]): A list of point cloud data arrays, each of shape (N, 3).
    """
    geometries = []

    for points in pointclouds:

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Generate a random color for each point cloud
        color = [random.random() for _ in range(3)]
        pcd.paint_uniform_color(color)

        geometries.append(pcd)

    o3d.visualization.draw_geometries(
        geometries, window_name="Point Clouds", width=800, height=600
    )


def create_corners_from_box_data(bbox: Dict) -> np.ndarray:
    """
    Creates an 8-point bounding box for visualization in Open3D.

    Args:
        bbox (Dict): Bounding box data containing x, y, z, width, length, height, and yaw.

    Returns:
        np.ndarray: An array of 8 corner points defining the bounding box.
    """
    from adwersbad.utils.transform import rotate

    x, y, z = bbox["x"], bbox["y"], bbox["z"]
    width, length, height = bbox["width"], bbox["length"], bbox["height"]
    bottom, top = z - height * 0.5, z + height * 0.5
    yaw = bbox["yaw"]

    point_offsets = np.array(
        [
            [-0.5 * length, -0.5 * width, 0],
            [-0.5 * length, 0.5 * width, 0],
            [0.5 * length, -0.5 * width, 0],
            [0.5 * length, 0.5 * width, 0],
        ]
    )

    rot = Quaternion(axis=(0, 0, 1), radians=yaw).rotation_matrix.T
    point_offsets = rotate(point_offsets, rot)

    points = np.zeros((8, 3))
    for offset in range(point_offsets.shape[0]):
        newx, newy = x + point_offsets[offset][0], y + point_offsets[offset][1]
        points[offset] = [newx, newy, bottom]
        points[offset + 4] = [newx, newy, top]

    return points


def show_pcl_with_labels(points: np.ndarray, boxes: List[Dict]) -> None:
    """
    Displays a point cloud along with labeled bounding boxes using Open3D.

    Args:
        points (np.ndarray): The point cloud data in shape (N, 3).
        boxes (List[Dict]): List of bounding box data.
    """
    line_segments = [
        [0, 1],
        [1, 3],
        [3, 2],
        [2, 0],
        [4, 5],
        [5, 7],
        [7, 6],
        [6, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    geometries = [pcl]

    for label in boxes:
        bbox_points = create_corners_from_box_data(label)
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bbox_points),
            lines=o3d.utility.Vector2iVector(line_segments),
        )
        line_set.colors = o3d.utility.Vector3dVector(
            [[1, 0, 0] for _ in range(len(line_segments))]
        )
        geometries.append(line_set)

    o3d.visualization.draw_geometries(geometries)


def create_corners_from_box_data(bbox: Dict) -> np.ndarray:
    """
    Creates an 8-point bounding box for visualization in Open3D.

    Args:
        bbox (Dict): Bounding box data containing x, y, z, width, length, height, and yaw.

    Returns:
        np.ndarray: An array of 8 corner points defining the bounding box.
    """
    from adwersbad.utils.transform import rotate

    x, y, z = bbox["x"], bbox["y"], bbox["z"]
    width, length, height = bbox["width"], bbox["length"], bbox["height"]
    bottom, top = z - height * 0.5, z + height * 0.5
    yaw = bbox["yaw"]

    point_offsets = np.array(
        [
            [-0.5 * length, -0.5 * width, 0],
            [-0.5 * length, 0.5 * width, 0],
            [0.5 * length, -0.5 * width, 0],
            [0.5 * length, 0.5 * width, 0],
        ]
    )

    rot = Quaternion(axis=(0, 0, 1), radians=yaw).rotation_matrix.T
    point_offsets = rotate(point_offsets, rot)

    points = np.zeros((8, 3))
    for offset in range(point_offsets.shape[0]):
        newx, newy = x + point_offsets[offset][0], y + point_offsets[offset][1]
        points[offset] = [newx, newy, bottom]
        points[offset + 4] = [newx, newy, top]

    return points


def show_pcl_with_labels(points: np.ndarray, boxes: List[Dict]) -> None:
    """
    Displays a point cloud along with labeled bounding boxes using Open3D.

    Args:
        points (np.ndarray): The point cloud data in shape (N, 3).
        boxes (List[Dict]): List of bounding box data.
    """
    line_segments = [
        [0, 1],
        [1, 3],
        [3, 2],
        [2, 0],
        [4, 5],
        [5, 7],
        [7, 6],
        [6, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    geometries = [pcl]

    for label in boxes:
        bbox_points = create_corners_from_box_data(label)
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bbox_points),
            lines=o3d.utility.Vector2iVector(line_segments),
        )
        line_set.colors = o3d.utility.Vector3dVector(
            [[1, 0, 0] for _ in range(len(line_segments))]
        )
        geometries.append(line_set)

    o3d.visualization.draw_geometries(geometries)


def show_pcl_in_image(
    points: np.ndarray,
    labels: np.ndarray,
    image: np.ndarray,
    ego_pose_lidar: Dict[str, Any],
    camera_params: Dict[str, Any],
    ego_pose_camera: Dict[str, Any],
    dataset: str,
) -> None:
    """
    Projects and visualizes point cloud data onto an image depending on the dataset.

    Args:
        points (np.ndarray): Nx3 array of 3D points.
        labels (np.ndarray): Nx1 array of labels.
        image (np.ndarray): Image array.
        ego_pose_lidar (Dict[str, Any]): Vehicle pose from LiDAR perspective.
        camera_params (Dict[str, Any]): Camera intrinsic and extrinsic parameters.
        dataset (str): The dataset name to determine the projection method.
    """
    width, height = image.shape[2], image.shape[1]
    from torchvision.transforms.functional import to_pil_image

    image = to_pil_image(image)

    if dataset.lower() == "waymo":
        from adwersbad.utils.transform import project_vehicle_to_image_waymo

        points_projected = project_vehicle_to_image_waymo(
            ego_pose_lidar, camera_params, width, height, points
        )
    else:
        intrinsic = np.array(camera_params["intrinsic"]).reshape(3, 3)
        points = transform_pcl_to_camera(
            points, ego_pose_lidar, camera_params, ego_pose_camera
        )
        points_projected = project_points_to_image(points, image, intrinsic)
    mask = points_projected[:, 2].astype(bool)
    points_filtered = points_projected[mask, :]
    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    ax.imshow(image)
    if labels:
        labels_filtered = labels[mask]

        l2c = create_adwersbad_label_map("id", "color")
        label_colors = np.array([l2c[l] for l in labels_filtered])
        ax.scatter(points_filtered[:, 0], points_filtered[:, 1], c=label_colors, s=5)
    else:
        ax.scatter(points_filtered[:, 0], points_filtered[:, 1], s=5)
    ax.axis("off")
    plt.show()


if __name__ == "__main__":
    import json

    import numpy as np
    import torch
    import torch.utils.data as torch_data

    from adwersbad import Adwersbad

    cols = {
        "lidar": [
            "points",
            "lidar_parameters",
            "lidar_vehicle_pose",
            "lidar_id",
            "lidar_uid",
        ],
        "camera": [
            "image",
            "camera_parameters",
            "camera_vehicle_pose",
            "camera_id",
            "camera_uid",
        ],
        # "lidar_box": ["lidar_box"],
        # "lidar_segmentation": ["lidar_segmentation"],
        # "camera_segmentation": ["camera_segmentation"],
    }
    # cols ={'lidar' : ['points'], 'lidar_segmentation' : ['lidar_segmentation']}
    # cols = {"lidar_box": ["lidar_uid", "lidar_box"]}
    dataset = "waymo"
    ds = Adwersbad(
        data=cols,
        splits=["all"],
        datasets=[dataset],
        # scenario="twilight",
        offset=0,
        # limit=100,
    )
    dl = iter(torch_data.DataLoader(ds))
    pcls = []
    used_ids = ["waymo_1"]
    lidid = "waymo_1"
    camid = "waymo_1"
    for _ in range(15):
        (
            pts,
            lidp,
            ego_lid,
            lidid,
            luid,
            image,
            camp,
            ego_cam,
            camid,
            camuid,
            # boxes,
            # labels,
            # camseg,
        ) = next(dl)
        while lidid in used_ids or camid in used_ids:
            (
                pts,
                lidp,
                ego_lid,
                lidid,
                luid,
                image,
                camp,
                ego_cam,
                camid,
                camuid,
                # boxes,
                # labels,
                # camseg,
            ) = next(dl)
            lidid = lidid[0]
            camid = camid[0]
        # show_pcl_with_labels(pts[0], json.loads(boxes[0]))
        print(lidid, camid)
        print(luid, camuid)
        proj = show_pcl_in_image(
            pts[0].numpy(),
            # labels[0].numpy(),
            None,
            image[0],
            json.loads(ego_lid[0]),
            json.loads(camp[0]),
            json.loads(ego_cam[0]),
            dataset,
        )
        # print(luid)
        # print(lidid)
        # pcl = pts[0].numpy()
        # print(pcl.shape)
        # pts_t = transform_pcl_to_ego(pcl, json.loads(lidp[0]))
        # pcls.append(pcl)
    # show_multiple_pcls(pcls)
