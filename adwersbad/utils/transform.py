from typing import Any, Dict, Tuple

import numpy as np
import pyquaternion as pyq


def build_transformation_matrix(
    rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """
    Converts rotation and translation into a 4x4 transformation matrix.

    Args:
        rotation (np.ndarray): 3x3 rotation matrix.
        translation (np.ndarray): 3x1 translation vector.

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    assert rotation.shape == (3, 3)
    assert translation.shape == (
        3,
    ), f"translation incorrect shape: {translation.shape}"

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation

    return transformation_matrix


def transformation_matrix_from_extrinsic(
    extrinsic: Dict[str, Any], inverse: bool = False
) -> np.ndarray:
    """
    Generates a transformation matrix from an extrinsic parameter dictionary.

    Args:
        extrinsic (Dict[str, Any]): Dictionary containing 'rotation' (quaternion) and 'translation' (vector).
        inverse (bool, optional): If True, computes the inverse transformation. Defaults to False.

    Returns:
        np.ndarray: 4x4 transformation matrix.
    """
    if inverse:
        rot = pyq.Quaternion(extrinsic["rotation"]).rotation_matrix.T
        trans = rot.dot(-np.array(extrinsic["translation"]))
    else:
        rot = pyq.Quaternion(extrinsic["rotation"]).rotation_matrix
        trans = np.array(extrinsic["translation"])
    return build_transformation_matrix(rot, trans)


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> pyq.Quaternion:
    """
    Converts a rotation matrix to a quaternion.

    Args:
        rotation (np.ndarray): 3x3 rotation matrix.

    Returns:
        pyq.Quaternion: Corresponding quaternion.
    """
    return pyq.Quaternion(matrix=rotation)


def deconstruct_transformation_matrix(
    transform: np.ndarray,
) -> Tuple[pyq.Quaternion, np.ndarray]:
    """
    Decomposes a 4x4 transformation matrix into its rotation (as a quaternion) and translation components.

    Args:
        transform (np.ndarray): A 4x4 transformation matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - A 4-element quaternion (x, y, z, w).
            - A 3-element translation vector (x, y, z).

    Raises:
        ValueError: If the input matrix is not 4x4.
    """
    if transform.shape != (4, 4):
        raise ValueError(
            f"Expected a 4x4 transformation matrix, but got shape {transform.shape}"
        )

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    q = rotation_matrix_to_quaternion(rotation)  # Convert to quaternion

    return q, translation


def quaternion_to_rotation_matrix(quat: np.ndarray | pyq.Quaternion) -> np.ndarray:
    """
    Converts a quaternion to a rotation matrix.

    Args:
        quat (np.ndarray | pyq.Quaternion): Quaternion representation.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    q = pyq.Quaternion(quat)
    return q.rotation_matrix


def transform(points: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
    """
    Applies a 4x4 transformation matrix to a set of 3D points.

    Args:
        points (np.ndarray): Nx3 array of 3D points.
        transformation_matrix (np.ndarray): 4x4 transformation matrix.

    Returns:
        np.ndarray: Transformed Nx3 points.
    """
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_global = points_homogeneous @ transformation_matrix.T
    return points_global[:, :3]


def rotate(points: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Applies a rotation matrix to a set of points.

    Args:
        points (np.ndarray): Nx3 array of 3D points.
        rotation_matrix (np.ndarray): 3x3 rotation matrix.

    Returns:
        np.ndarray: Rotated points.
    """
    return np.dot(points, rotation_matrix)


def translate(points: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
    """
    Applies a translation to a set of points.

    Args:
        points (np.ndarray): Nx3 array of 3D points.
        translation_vector (np.ndarray): 3-element translation vector.

    Returns:
        np.ndarray: Translated points.
    """
    return points + np.array(translation_vector)


def translate_nu(points: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Applies translation to each coordinate axis separately.

    Args:
        points (np.ndarray): 3xN array of points.
        v (np.ndarray): 3-element translation vector.

    Returns:
        np.ndarray: Translated points.
    """
    for i in range(3):
        points[i, :] = points[i, :] + v[i]
    return points


def rotate_nu(points: np.ndarray, rot_matrix: np.ndarray) -> np.ndarray:
    """
    Applies a rotation matrix to a set of points.

    Args:
        points (np.ndarray): 3xN array of points.
        rot_matrix (np.ndarray): 3x3 rotation matrix.

    Returns:
        np.ndarray: Rotated points.
    """
    return np.dot(rot_matrix, points[:3, :])


def transform_pcl_to_ego(points: np.ndarray, lid_params: Dict[str, Any]) -> np.ndarray:
    """
    Transforms point cloud to ego frame.

    Args:
        points (np.ndarray): Nx3 point cloud.
        lid_params (Dict[str, Any]): Lidar parameters containing rotation and translation.

    Returns:
        np.ndarray: Transformed point cloud.
    """
    points = rotate(points, pyq.Quaternion(lid_params["rotation"]).rotation_matrix)
    points = translate(points, -np.array(lid_params["translation"]))
    return points


def transform_pcl_to_camera(
    points: np.ndarray,
    ego_pose_lidar: Dict[str, Any],
    cam_params: Dict[str, Any],
    ego_pose_camera: Dict[str, Any],
) -> np.ndarray:
    """
    Transforms point cloud from lidar to camera coordinate system.
    Transforms to word coordinate system first to account for vehicle movement in between lidar and image timestamps.

    Args:
        points (np.ndarray): Nx3 point cloud.
        ego_pose_lidar (Dict[str, Any]): Lidar extrinsic parameters.
        cam_params (Dict[str, Any]): Camera intrinsic and extrinsic parameters.
        ego_pose_camera (Dict[str, Any]): Ego vehicle camera pose.

    Returns:
        np.ndarray: Transformed point cloud in camera coordinates.
    """
    p2w_mat = transformation_matrix_from_extrinsic(ego_pose_lidar)
    p2wp = transform(points, p2w_mat)
    w2wc_mat = transformation_matrix_from_extrinsic(ego_pose_camera, inverse=True)
    p2wc = transform(p2wp, w2wc_mat)
    wc2c_mat = transformation_matrix_from_extrinsic(cam_params, inverse=True)
    return transform(p2wc, wc2c_mat)


# TODO:Add generic projections function that calls the correct projection depending on dataset


def project_points_to_image(
    points: np.ndarray, image: Any, intrinsic: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """
    Projects 3D points onto a 2D image plane using intrinsic parameters.

    Args:
        points (np.ndarray): Nx3 array of 3D points.
        image (Any): Image object with size attribute.
        intrinsic (np.ndarray): Camera intrinsic matrix (up to 4x4).
        normalize (bool, optional): If True, normalizes projected points. Defaults to True.

    Returns:
        np.ndarray: Nx3 array of projected 2D points and a validity mask.
    """
    assert intrinsic.shape[0] <= 4
    assert intrinsic.shape[1] <= 4
    assert points.shape[1] == 3

    proj_mat = np.eye(4)
    proj_mat[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic

    depths = points[:, 2]
    nbr_points = points.shape[0]
    points_homogeneous = np.hstack((points, np.ones((nbr_points, 1))))
    points_proj = np.dot(points_homogeneous, proj_mat.T)
    points_proj = points_proj[:, :3]

    if normalize:
        points_proj = points_proj / points_proj[:, 2].repeat(3).reshape(
            points_proj.shape
        )
    print(image.size)
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1.0)
    mask = np.logical_and(mask, points_proj[:, 0] > 1)
    mask = np.logical_and(mask, points_proj[:, 0] < image.size[0] - 1)
    mask = np.logical_and(mask, points_proj[:, 1] > 1)
    mask = np.logical_and(mask, points_proj[:, 1] < image.size[1] - 1)

    points_proj = points_proj[:, :2]
    result = np.hstack((points_proj, mask.reshape(-1, 1).astype(int)))
    return result


def project_vehicle_to_image_waymo(
    lidar_vehicle_pose: Dict[str, Any],
    camera_calibration: Dict[str, Any],
    width: int,
    height: int,
    points: np.ndarray,
) -> np.ndarray:
    """
    Projects 3D points from vehicle coordinate system to image coordinates using Waymo camera operations.

    Args:
        lidar_vehicle_pose (Dict[str, Any]): Vehicle pose transformation from vehicle to world coordinates.
        calibration (Dict[str, Any]): Camera calibration parameters (intrinsic and extrinsic).
        width (int): Image width.
        height (int): Image height.
        points (np.ndarray): Nx3 array of points in the vehicle coordinate system.

    Returns:
        np.ndarray: Nx3 array containing (u, v, valid) projected coordinates.
    """
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2 as open_dataset
    from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops

    transform = transformation_matrix_from_extrinsic(lidar_vehicle_pose)
    world_points = np.zeros_like(points)
    for i, point in enumerate(points):
        cx, cy, cz, _ = np.matmul(transform, [*point, 1])
        world_points[i] = (cx, cy, cz)

    extrinsic = transformation_matrix_from_extrinsic(camera_calibration)
    extrinsic = tf.reshape(tf.constant(list(extrinsic), dtype=tf.float32), [4, 4])
    intrinsic = np.array(camera_calibration["intrinsic"])
    intrinsic = tf.constant(list(intrinsic.flatten()), dtype=tf.float32)
    metadata = tf.constant(
        [
            width,
            height,
            open_dataset.CameraCalibration.GLOBAL_SHUTTER,
        ],
        dtype=tf.int32,
    )
    camera_image_metadata = list(transform.flatten()) + [0.0] * 10

    points_in_image = py_camera_model_ops.world_to_image(
        extrinsic, intrinsic, metadata, camera_image_metadata, world_points
    ).numpy()
    u, v, ok = points_in_image.transpose()
    ok = ok.astype(bool)
    u = u[ok]
    v = v[ok]
    u = np.clip(u, 0, width)
    v = np.clip(v, 0, height)
    filtered_points = np.zeros((len(ok), 3))
    filtered_points[ok, 0] = u
    filtered_points[ok, 1] = v
    filtered_points[ok, 2] = 1  # Mark valid points with 1 (True)
    return filtered_points
