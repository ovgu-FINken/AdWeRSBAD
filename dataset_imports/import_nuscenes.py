import json
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Tuple

import numpy as np
import psycopg
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, LidarSegPointCloud
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image
from pyquaternion import Quaternion
from tqdm.contrib.concurrent import process_map

import adwersbad.db_helpers as adwersbaddb
from adwersbad.class_helpers import (chain_maps, create_adwersbad_label_map,
                                     nuscenes_to_adwersbad_label_map)
from adwersbad.config import config
from adwersbad.log import setup_logger
from adwersbad.utils.transform import build_transformation_matrix
from adwersbad.weather_import import (check_if_twilight,
                                      get_weather_from_timestamp)

logger = setup_logger("import_nuscenes")

loc_to_gps = {
    "boston-seaport": [42.336849169438615, -71.05785369873047],
    "singapore-onenorth": [1.2882100868743724, 103.78475189208984],
    "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555],
    "singapore-queenstown": [1.2782562240223188, 103.76741409301758],
}


def get_pointcloud_data(
    sample: Dict[str, Any]
) -> Tuple[
    BytesIO, BytesIO, BytesIO, str, Dict[str, List[float]], Dict[str, List[float]]
]:
    """
    Retrieve and process point cloud data, including associated metadata, from a NuScenes sample.

    Args:
        sample (dict): A NuScenes sample dictionary containing data tokens and metadata.

    Returns:
            - xyz_bytes (BytesIO): Byte-stream containing the XYZ coordinates of the points in the lidar frame.
            - intensities_bytes (BytesIO): Byte-stream containing the intensity values of the lidar points.
            - semantic_bytes (BytesIO): Byte-stream containing a semantic segmentation image (PNG format) of the point cloud.
            - boxes_json (str): JSON string encoding a list of bounding boxes
            - extrinsic (dict): A dictionary containing the rotation (as a quaternion) and translation of the lidar sensor relative to the vehicle coordinate system.
            - ego_pose (dict): A dictionary containing the rotation (as a quaternion) and translation of the ego vehicle in the global coordinate system.

    Raises:
        KeyError: If the required keys are missing in the provided sample or NuScenes metadata.
        ValueError: If any data transformation fails due to unexpected data formats or contents.

    Example:
        sample = nusc.get('sample', sample_token)
        xyz_bytes, intensities_bytes, semantic_bytes, boxes_json, extrinsic, ego_pose = get_pointcloud_data(sample)

    Notes:
        - This function uses the NuScenes SDK to retrieve and process point cloud and related metadata.
        - Lidar point cloud data is rotated and translated to the vehicle coordinate system.
        - Bounding boxes are transformed from the sensor frame to the ego vehicle frame.
        - The semantic segmentation is encoded as a PNG image, and bounding boxes are encoded in JSON format.
    """
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    lidar_segmentation_token = lidar_data["token"]
    lidar_segmentation_path = (
        nusc.dataroot + "/" + nusc.get("lidarseg", lidar_segmentation_token)["filename"]
    )

    ego_pose_token = lidar_data["ego_pose_token"]
    ego_pose = nusc.get("ego_pose", ego_pose_token)
    ego_rotation = ego_pose["rotation"]
    ego_translation = ego_pose["translation"]
    ego_pose = {"rotation": ego_rotation, "translation": ego_translation}

    lidar_file_path, boxes, _ = nusc.get_sample_data(lidar_token)
    cs_data = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    quaternion = np.array(cs_data["rotation"]).tolist()
    translation = np.array(cs_data["translation"]).tolist()
    extrinsic = {'type': 'nuscenes', "rotation": quaternion, "translation": translation}
    box_list = []
    for box in boxes:
        # transform from sensor into ego coord system, aka invert:
        # box.translate(-np.array(cs_record['translation']))
        # box.rotate(Quaternion(cs_record['rotation']).inverse)
        box.rotate(Quaternion(cs_data["rotation"]))
        box.translate(np.array(cs_data["translation"]))
        x, y, z = box.center
        w, l, h = box.wlh
        n_points = nusc.get("sample_annotation", box.token)["num_lidar_pts"]
        box_dict = {
            "x": x,
            "y": y,
            "z": z,
            "width": w,
            "length": l,
            "height": h,
            "n_points": n_points,
            "rotation": box.orientation.elements.tolist(),
            "label": nuscenes_to_adwersbad_label_map[box.name],
        }
        box_list.append(box_dict)

    boxes_json = json.dumps(box_list)
    pointcloud = LidarSegPointCloud(lidar_file_path, lidar_segmentation_path)
    points = LidarPointCloud.from_file(lidar_file_path)
    points.rotate(Quaternion(cs_data["rotation"]).rotation_matrix)
    points.translate(np.array(cs_data["translation"]))
    points = points.points.T
    xyz = points[:, :3]
    intensities = points[:, 3]
    semantic = nusc2adwersbad_id(pointcloud.labels)
    semantic_bytes = BytesIO()
    Image.fromarray(semantic).save(semantic_bytes, format="PNG")
    xyz_bytes = BytesIO()
    intensities_bytes = BytesIO()
    np.save(xyz_bytes, xyz)
    np.save(intensities_bytes, intensities)
    return xyz_bytes, intensities_bytes, semantic_bytes, boxes_json, extrinsic, ego_pose


def get_image_data(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Retrieves image data and associated metadata for a given sample from multiple camera channels.

    Args:
        sample (dict): A NuScenes sample dictionary containing data tokens and metadata.

    Returns:
        List[dict]: A list of dictionaries, each containing image data and metadata for a camera channel. Each dictionary includes:
            - "image_file_path" (str): Path to the image file.
            - "boxes" (str): JSON-encoded list of bounding boxes, each with attributes such as dimensions, position, orientation, and label.
            - "intrinsic" (list): List representation of the camera intrinsic matrix.
            - "rotation" (list): Quaternion rotation of the camera relative to the vehicle coordinate system.
            - "translation" (list): Translation of the camera relative to the vehicle coordinate system.
            - "channel" (str): The camera channel name (e.g., "CAM_FRONT").

    """
    # List of all camera types in the sample (can add/remove based on your need)
    camera_channels = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    camera_data_list = []
    for camera_channel in camera_channels:
        if camera_channel not in sample["data"]:
            logger.debug(
                f"Skipping missing {camera_channel=} in sample {sample['token']}"
            )
            continue  # Skip if the camera channel doesn't exist for the sample
        cam_token = sample["data"][camera_channel]
        cam_data = nusc.get("sample_data", cam_token)

        ego_pose_token = cam_data["ego_pose_token"]
        ego_pose = nusc.get("ego_pose", ego_pose_token)
        ego_rotation = ego_pose["rotation"]
        ego_translation = ego_pose["translation"]
        ego_pose = {"rotation": ego_rotation, "translation": ego_translation}

        image_file_path, boxes, cam_intrinsic = nusc.get_sample_data(cam_token)
        calibrated_sensor = nusc.get(
            "calibrated_sensor", cam_data["calibrated_sensor_token"]
        )
        # ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])
        quaternion = np.array(calibrated_sensor["rotation"]).tolist()
        translation = np.array(calibrated_sensor["translation"]).tolist()

        box_list = []
        for box in boxes:
            x, y, z = box.center
            w, l, h = box.wlh
            box_dict = {
                "x": x,
                "y": y,
                "z": z,
                "width": w,
                "length": l,
                "height": h,
                "yaw": box.orientation.degrees,
                "label": nuscenes_to_adwersbad_label_map[box.name],
            }
            box_list.append(box_dict)

        boxes_json = json.dumps(box_list)

        camera_data = {
            "image_file_path": image_file_path,
            "boxes": boxes_json,
            "intrinsic": cam_intrinsic.tolist(),
            "rotation": quaternion,
            "translation": translation,
            "channel": camera_channel,
            "ego_pose": ego_pose,
        }
        camera_data_list.append(camera_data)
    return camera_data_list


def get_metadata(sample: dict) -> Tuple[int, str, float, float, int, str, bool]:
    """
    Retrieve metadata for a given sample, including timestamp, location, GPS coordinates,
    and weather details.

    Args:
        sample (dict): A NuScenes sample dictionary containing metadata and tokens.

    Returns:
            - timestamp (int): The timestamp of the sample in microseconds since epoch.
            - location (str): The location name corresponding to the sample.
            - lat (float): Latitude of the sample's location.
            - lon (float): Longitude of the sample's location.
            - weather_code (int): Weather condition code based on the timestamp and location.
            - weather_string (str): Descriptive weather condition (e.g., "Clear", "Rainy").
            - isday (bool): Indicates whether it is daytime at the given timestamp and location.
    """
    timestamp = sample["timestamp"]
    # get log token
    scene_token = sample["scene_token"]
    scene = nusc.get("scene", scene_token)
    log_token = scene["log_token"]
    log = nusc.get("log", log_token)

    location = log["location"]
    lat, lon = loc_to_gps[location]
    weather_code, weather_string, isday = get_weather_from_timestamp(
        timestamp, lat, lon
    )
    return timestamp, location, lat, lon, weather_code, weather_string, isday


def process_sample(sample: dict, split: str, conn: psycopg.Connection) -> None:
    """
    Processes a NuScenes sample, extracting and storing point cloud, image, and weather data into a database.

    Args:
        sample (dict): A NuScenes sample containing metadata, sensor data, and annotations.
        split (str): The dataset split (e.g., "train", "val", "test") for categorizing the data.
        conn (psycopg.Connection): Database connection object for saving processed data.

    Returns:
        None

    Workflow:
        1. Extracts point cloud data (xyz coordinates, intensities, semantic labels, boxes, extrinsic, ego pose).
        2. Extracts camera image data and their associated metadata and annotations.
        3. Retrieves weather and time-of-day metadata for the sample.
        4. Determines the time of day (day, night, twilight) based on timestamp, location, and twilight checks.
        5. Saves weather data to the database.
        6. Saves camera images, their metadata, and labels to the database.
        7. Saves LiDAR point cloud data and labels to the database.

    """
    xyz, intensities, semantic, boxes_json, extrinsic, ego_lidar = get_pointcloud_data(
        sample
    )
    camera_data_list = get_image_data(sample)
    time, loc, lat, lon, _, weather_string, isday = get_metadata(sample)
    tod = "night" if isday == 0 else "day"
    time = datetime.fromtimestamp(time / 1e6)
    is_twilight, ttod = check_if_twilight(time, lat, lon)
    if is_twilight:
        tod = ttod
    weather_uid = adwersbaddb.save_weather_to_db(
        conn, time, loc, weather_string, tod, "nuscenes", split
    )
    if weather_uid == -1:
        logger.debug(
            f"skipping image processing due to error in weather data insertion (uid=-1)"
        )
    else:
        for camera_data in camera_data_list:
            with open(camera_data["image_file_path"], "rb") as img_file:
                image_bytes = BytesIO(img_file.read())
            camera_name = "nuscenes_" + camera_data["channel"]
            cam_params = {
                'type': 'nuscenes',
                key: camera_data[key]
                for key in ("intrinsic", "rotation", "translation")
            }
            image_uid = adwersbaddb.save_image_to_db(
                conn,
                weather_uid,
                image_bytes,
                camera_name,
                json.dumps(cam_params),
                json.dumps(camera_data["ego_pose"]),
            )
            adwersbaddb.save_image_labels_to_db(conn, image_uid, None, camera_data["boxes"])
        lidar_uid = adwersbaddb.save_pointcloud_to_db(
            conn,
            weather_uid,
            xyz,
            intensities,
            "nuscenes_top",
            json.dumps(extrinsic),
            json.dumps(ego_lidar),
        )
        adwersbaddb.save_pointcloud_labels_to_db(conn, lidar_uid, semantic, boxes_json)


def process_scene(scene):
    # scene = nusc.get('scene', scene_name)
    split = scene2split[scene["name"]]
    next_sample_token = scene["first_sample_token"]
    conn = adwersbaddb.get_connection(dbinfo)
    while next_sample_token:
        current_sample = nusc.get("sample", next_sample_token)
        process_sample(current_sample, split, conn)
        next_sample_token = current_sample["next"]
    conn.commit()
    conn.close()


data_root = config(section="paths")["data_root"]

nusc = NuScenes(version="v1.0-trainval", dataroot=f"{data_root}/nuscenes", verbose=True)
nusc_id2n = nusc.lidarseg_idx2name_mapping
nusc2adwersbad_n = nuscenes_to_adwersbad_label_map
adwersbad_n2id = create_adwersbad_label_map("name", "id")

nusc2adwersbad_id = np.vectorize(
    chain_maps(nusc_id2n, nusc2adwersbad_n, adwersbad_n2id).get, otypes=[np.uint8]
)


dbinfo = "psycopg@local"
n_cpu = int(config(section="base")["n_cpu"])

split2scenes = create_splits_scenes()
scene2split = {}
for k, v in split2scenes.items():
    if k in ["train", "val"]:
        for scene_name in v:
            split = "training" if k == "train" else "validation"
            scene2split[scene_name] = split

process_map(process_scene, nusc.scene, max_workers=n_cpu, chunksize=1)
