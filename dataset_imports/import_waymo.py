import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
from datetime import datetime
from enum import IntEnum
from io import BytesIO

import numpy as np
import psycopg
import quaternion
import tensorflow.compat.v1 as tf
from file_helpers import get_file_paths
from google import protobuf
from PIL import Image
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import (
    box_utils,
    camera_segmentation_utils,
    frame_utils,
    transform_utils,
)

import adwersbad.db_helpers as adwersbaddb
from adwersbad.class_helpers import (
    chain_maps,
    create_adwersbad_label_map,
    waymo_to_adwersbad_label_map_bbox,
    waymo_to_adwersbad_label_map_camera,
    waymo_to_adwersbad_label_map_lidar,
)
from adwersbad.config import config
from adwersbad.log import setup_logger
from adwersbad.utils.transform import rotate, rotation_matrix_to_quaternion, translate

tf.logging.set_verbosity(tf.logging.ERROR)

tf.enable_eager_execution()

data_root = config(section="paths")["data_root"]
DATA_ROOT = data_root + "/waymov1/individual_files/"
# we do not use the testing split as it does not have labeled data
SPLITS = ["training", "validation"]

n2id = create_adwersbad_label_map("name", "id")
w2adwersbad_id_cam = np.vectorize(
    chain_maps(waymo_to_adwersbad_label_map_camera, n2id).get, otypes=[np.uint8]
)
w2adwersbad_id_lidar = np.vectorize(
    chain_maps(waymo_to_adwersbad_label_map_lidar, n2id).get, otypes=[np.uint8]
)


class WaymoLidarNames(IntEnum):
    UNKNOWN = 0
    TOP = 1
    FRONT = 2
    SIDE_LEFT = 3
    SIDE_RIGHT = 4
    REAR = 5


def get_paths_for_split(split: str = "training") -> list[str]:
    """
    Retrieves file paths for a given dataset split.

    Args:
        split (str): The dataset split to retrieve paths for.
                     Defaults to 'training'.

    Returns:
        list[str]: A list of file paths corresponding to the specified split.
    """
    split_path = DATA_ROOT + split + "/"
    filepaths = get_file_paths(data_path=split_path, extension="tfrecord")
    return filepaths


# Configure the logging system
# logging.basicConfig(
#     level=logging.INFO,  # Change to DEBUG for more detailed output
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("waymo.log"),  # Log to a file
#         # logging.StreamHandler()  # Also log to the console
#     ]
# )

logger = setup_logger("import_waymo")


def heading_to_quaternion(heading: float) -> np.quaternion:
    """
    Convert a heading angle in radians to a quaternion representation.

    This function calculates the quaternion corresponding to a rotation about the Z-axis.
    The heading angle should be provided in radians.

    Args:
        heading (float): The heading(yaw) angle in radians.

    Returns:
        numpy.ndarray: A 1x4 numpy array representing the quaternion in the format [qw, qx, qy, qz]
    """
    q = quaternion.from_rotation_vector((0, 0, heading))
    # qw = np.cos(heading / 2.0)
    # qx = qy = 0.0
    # qz = np.sin(heading / 2.0)
    return q  # np.quaternion(qw, qx, qy, qz)


def count_points_in_box(points, box):
    # we transformed the pcl to box space for this, so we need to also translate the box
    # box_center = np.array(
    #     [box["box"]["centerX"], box["box"]["centerY"], box["box"]["centerZ"]]
    # )
    # box_yaw = box["box"]["heading"]  # Assume this is in degrees
    # pcl_translated = pcl - box_center
    dimensions = np.array([box["width"], box["length"], box["height"]])
    half_dims = dimensions / 2.0
    within_box = np.all(np.abs(points) <= half_dims, axis=1)
    return np.sum(within_box)


def convert_range_image_to_point_cloud_labels(
    frame: open_dataset.Frame,
    range_images: dict,
    segmentation_labels: dict,
    ri_index: int = 0,
) -> list[np.ndarray | None]:
    """
    converts segmentation labels from range images to point cloud labels.

    args:
        frame (open_dataset.Frame): the frame containing the range images.
        range_images (dict): a dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        segmentation_labels (dict): dictionary of segmentation labels.
        ri_index (int): index indicating the return to use (0 for first,
                        1 for second). defaults to 0.

    returns:
        list[np.ndarray]:  nx2 list of 3d lidar points' segmentation labels. 0 for unlabeled points

    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims
        )
        range_image_mask = range_image_tensor[..., 0] > 0
        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
            point_labels.append(sl_points_tensor.numpy())
        else:
            point_labels.append(np.array([]))
            # num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            # sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

    return point_labels


def process_image(
    frame: open_dataset.Frame, image: open_dataset.CameraImage
) -> tuple[BytesIO, str, BytesIO, str, dict]:
    """
    processes the camera images from a dataset frame

    args:
        frame (open_dataset.Frame): the frame containing the camera image.
        image (open_dataset.CameraImage): the camera image data to process.

    returns:
        tuple[BytesIO, str, BytesIO, str, dict]: a tuple containing the image data in a BytesIO object,
        camera id, segmentation png image data stored in a BytesIO object, bounding boxes in json format,
        and camera parameters as a dictionary containing intrinsic and extrinsic matrices, and resolution.
    """
    camera_id = image.name
    image_bytes = BytesIO(image.image)
    image_seg = image.camera_segmentation_label
    if image_seg.panoptic_label:
        semantic, _ = (
            camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
                tf.image.decode_png(image_seg.panoptic_label, dtype=tf.uint16).numpy(),
                image_seg.panoptic_label_divisor,
            )
        )
        # semantic = np.vectorize(convert_waymo_label_to_adwersbad_label_camera, otypes=[str])(semantic)
        # n2id = create_adwersbad_label_map("name", "id")
        # semantic = np.vectorize(n2id.get, otypes=[np.uint8])(semantic)
        semantic = w2adwersbad_id_cam(semantic)
        sem_png = BytesIO(tf.io.encode_png(semantic).numpy())
    else:
        sem_png = 0

    boxes = ""
    for camera_labels in frame.camera_labels:
        # ignore camera labels that do not correspond to this camera.
        if camera_labels.name != camera_id:
            continue
        try:
            boxes = json.loads(protobuf.json_format.MessageToJson(camera_labels))[
                "labels"
            ]
        except KeyError:
            # no boxes for this camera image
            boxes = ""
        box_list = []
        for box in boxes:
            box_dict = {
                "x": box["box"]["centerX"],
                "y": box["box"]["centerY"],
                "length": box["box"]["length"],
                "width": box["box"]["width"],
                "label": waymo_to_adwersbad_label_map_bbox[box["type"]],
            }
            box_list.append(box_dict)

    camera_params = {}
    for calibration in frame.context.camera_calibrations:
        if calibration.name == camera_id:
            # intrinsic parameters
            intrinsic_matrix = np.array(calibration.intrinsic).reshape(3, 3).tolist()

            # extrinsic parameters
            extrinsic_matrix = np.array(calibration.extrinsic.transform).reshape(4, 4)
            quaternion = rotation_matrix_to_quaternion(extrinsic_matrix)
            v_translation = extrinsic_matrix[:3, 3].tolist()

            resolution = {"width": calibration.width, "height": calibration.height}

            camera_params = {
                "type": "waymo",
                "intrinsic": intrinsic_matrix,
                "rotation": quaternion.elements.tolist(),
                "translation": v_translation,
                "resolution": resolution,
            }
            break
    return image_bytes, camera_id, sem_png, box_list, camera_params


def process_pointcloud(
    frame: open_dataset.Frame,
) -> tuple[BytesIO, BytesIO, BytesIO, str]:
    """
    processes a point cloud from the dataset.

    args:
        frame (open_dataset.Frame): the frame containing the point cloud data.

    returns:
        tuple[BytesIO, BytesIO, BytesIO, str]: a tuple containing the point cloud bytes,
        features bytes, segmentation labels bytes, and bounding boxes in json format.
    """
    (range_images, camera_projections, segmentation_labels, range_image_top_pose) = (
        frame_utils.parse_range_image_and_camera_projection(frame)
    )
    points, points_cp = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        keep_polar_features=True,
    )
    # points_ri2, _ = frame_utils.convert_range_image_to_point_cloud(
    # frame, range_images, camera_projections, range_image_top_pose, ri_index=1, keep_polar_features=True)

    point_labels = convert_range_image_to_point_cloud_labels(
        frame, range_images, segmentation_labels
    )
    print(points.shape)
    print(points_cp.shape)
    exit()
    # point_labels_ri2 = convert_range_image_to_point_cloud_labels(
    # frame, range_images, segmentation_labels, ri_index=1)
    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    # points_all_ri2 = np.concatenate(points_ri2, axis=0)
    # points_all = np.concatenate((points_all, points_all_ri2), axis=0)
    point_labels_all = np.concatenate(point_labels, axis=0)
    # point_labels_all_ri2 = np.concatenate(point_labels_ri2, axis=0)
    # points_labels_all = np.concatenate((point_labels_all, point_labels_all_ri2), axis=0)
    lidar_features = points_all[:, :3]
    points_all = points_all[:, 3:]
    # points_all2 = points_all_ri2[:,3:]
    # camera projection corresponding to each point.
    # cp_points_all = np.concatenate(cp_points, axis=0)
    # cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)
    if np.any(point_labels_all):
        labels = point_labels_all[:, 1].reshape(-1, 1)
        # semantic = np.vectorize(convert_waymo_label_to_adwersbad_label_lidar, otypes=[str])(labels)
        # n2id = create_adwersbad_label_map("name", "id")
        # labels = np.squeeze(np.vectorize(n2id.get, otypes=[np.uint8])(semantic))
        labels = np.squeeze(w2adwersbad_id_lidar(labels))
        labels_png = BytesIO()
        image.fromarray(labels).save(labels_png, format="png")
        # labels_png = BytesIO(tf.io.encode_png(labels).numpy())
    else:
        labels_png = 0
    try:
        lidar_boxes = json.loads(protobuf.json_format.MessageToJson(frame))[
            "laserlabels"
        ]
        unused_keys = ["metadata", "mostvisiblecameraname", "camerasyncedbox"]
        for box in lidar_boxes:
            for key in unused_keys:
                del box[key]
        lidar_boxes = json.dumps(lidar_boxes)
    except keyerror as e:
        logger.error(f"no laser labels found in frame")
        lidar_boxes = 0
    points_bytes = BytesIO()
    np.save(points_bytes, points_all)
    features_bytes = BytesIO()
    np.save(features_bytes, lidar_features)
    return points_bytes, features_bytes, labels_png, lidar_boxes


def count_points_in_boxes(pcl, lidar_boxes):
    """
    count how many points from each point cloud fall inside the provided rotated bounding boxes.

    args:
        points (list): a list of points
        lidar_boxes (list): a list of bounding boxes with center, dimensions, and rotation.

    returns:
        list: a list of counts, where each element corresponds to the number of points inside a specific box.
    """
    boxes_with_counts = []
    for box in lidar_boxes:
        box_measurements = np.array(
            [
                [
                    box["box"]["centerX"],
                    box["box"]["centerY"],
                    box["box"]["centerZ"],
                    box["box"]["length"],
                    box["box"]["width"],
                    box["box"]["height"],
                    box["box"]["heading"],
                ]
            ],
        )
        # pcl_translated = pcl - box_center
        # pcl_rotated = rotate_pcl(pcl_translated, -box_yaw)
        # pcl_inbox = pcl_rotated + box_center
        # overlay_pcls([pcl, pcl_rotated, pcl_inbox], box)
        # count = count_points_in_box(pcl_rotated, box["box"])
        count = box_utils.compute_num_points_in_box_3d(pcl, box_measurements)
        count = int(count.numpy()[0])
        # print()
        # print(box)
        # print(count)
        # print()
        # exit()
        # todo box label conversion
        if count > 0:
            box_dict = {
                "x": box["box"]["centerX"],
                "y": box["box"]["centerY"],
                "z": box["box"]["centerZ"],
                "length": box["box"]["length"],
                "width": box["box"]["width"],
                "height": box["box"]["height"],
                "yaw": box["box"]["heading"],
                "points_in_box": count,
                "label": waymo_to_adwersbad_label_map_bbox[box["type"]],
            }
            boxes_with_counts.append(box_dict)
    return boxes_with_counts


def process_pointcloud_individually(
    frame: open_dataset.Frame,
) -> list[tuple[BytesIO, BytesIO, BytesIO, str | None, str, str]]:
    """
    processes point clouds from the dataset and returns a list of point cloud and label pairs.

    args:
        frame (open_dataset.Frame): the frame containing the point cloud data.

     returns:
        list[tuple[bytes, bytes, bytes, str, str]]: a list of tuples where each tuple contains the point cloud bytes,
        the point cloud features(intensity and elongation) bytes, the segmentation label bytes, and the lidar sensor name for each sensor.
    """
    # parse and convert pcl and labels
    (range_images, camera_projections, segmentation_labels, range_image_top_pose) = (
        frame_utils.parse_range_image_and_camera_projection(frame)
    )
    points, _ = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        keep_polar_features=True,
    )
    point_labels = convert_range_image_to_point_cloud_labels(
        frame, range_images, segmentation_labels
    )
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    frame_json = json.loads(protobuf.json_format.MessageToJson(frame))
    lidar_boxes = frame_json.get("laserLabels")
    pcl_label_box_pairs = []

    for pcl, label, calib in zip(points, point_labels, calibrations):
        # pcl = range, intensity, elongation, x, y, z
        # projections = {}
        # for c_id in range(1, 9):
        #     projections[c_id] = cp[(cp[:3, 0] == c_id) | (cp[3:, 3] == c_id)]
        # print(projections)
        xyz, features = pcl[:, 3:], pcl[:, :3]
        extrinsic_matrix = np.reshape(np.array(calib.extrinsic.transform), [4, 4])
        quaternion = rotation_matrix_to_quaternion(extrinsic_matrix)
        v_translation = extrinsic_matrix[:3, 3]
        # points_in_vehicle_frame = rotate(xyz, quaternion)
        # points_in_vehicle_frame = translate(points_in_vehicle_frame, -v_translation)
        # xyz = points_in_vehicle_frame
        name = "waymo_" + str(calib.name)
        boxes_with_point_counts = (
            json.dumps(count_points_in_boxes(xyz, lidar_boxes)) if lidar_boxes else None
        )

        calib_params = {
            "type": "waymo",
            "rotation": quaternion.elements.tolist(),
            "translation": v_translation.tolist(),
            # "beam_inclinations": calib.beam_inclinations,
            # "beam_inclination_min": calib.beam_inclination_min,
            # "beam_inclination_max": calib.beam_inclination_max,
        }
        labels_png = None
        if label is not None and np.any(label):
            labels = w2adwersbad_id_lidar(np.squeeze(label[:, 1].reshape(-1, 1)))
            labels_png = BytesIO()
            Image.fromarray(labels).save(labels_png, format="png")
        if labels_png or boxes_with_point_counts:
            points_bytes = BytesIO()
            features_bytes = BytesIO()
            np.save(points_bytes, xyz)
            np.save(features_bytes, features)
            pcl_label_box_pairs.append(
                (
                    points_bytes,
                    features_bytes,
                    labels_png,
                    boxes_with_point_counts,
                    name,
                    calib_params,
                )
            )
    return pcl_label_box_pairs


def process_file(filepath: str) -> None:
    """
    processes a single tfrecord file, extracting and saving weather, camera,
    and lidar data into the database.

    args:
        filepath (str): path to the tfrecord file to process.
    """
    dataset = tf.data.TFRecordDataset(filepath, compression_type="")
    conn = None
    failed_frames = 0
    try:
        params = config(section=dbinfo)
        conn = psycopg.connect(**params)
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            context = frame.context
            vehicle_pose = np.array(
                json.loads(protobuf.json_format.MessageToJson(frame.pose))["transform"]
            ).reshape(4, 4)
            quaternion = rotation_matrix_to_quaternion(vehicle_pose)
            translation = vehicle_pose[:3, 3].tolist()
            # context_json = protobuf.json_format.messagetojson(context)
            vehicle_pose = {
                "rotation": quaternion.elements.tolist(),
                "translation": translation,
            }
            location = context.stats.location
            weather = context.stats.weather
            time = frame.timestamp_micros
            time = datetime.fromtimestamp(time / 1e6)
            tod = context.stats.time_of_day
            weather_uid = adwersbaddb.save_weather_to_db(
                conn, time, location, weather, tod, "waymo", split
            )
            for image in frame.images:
                image_bytes, camera_id, sem_png, boxes, cam_params = process_image(
                    frame, image
                )
                image_uid = adwersbaddb.save_image_to_db(
                    conn,
                    weather_uid,
                    image_bytes,
                    "waymo_" + str(camera_id),
                    json.dumps(cam_params),
                    json.dumps(vehicle_pose),
                )
                adwersbaddb.save_image_labels_to_db(
                    conn, image_uid, sem_png, json.dumps(boxes)
                )
            pcl_label_pairs = process_pointcloud_individually(frame)
            for pcl, features, labels, boxes, name, lidar_params in pcl_label_pairs:
                lidar_uid = adwersbaddb.save_pointcloud_to_db(
                    conn,
                    weather_uid,
                    pcl,
                    features,
                    name,
                    json.dumps(lidar_params),
                    json.dumps(vehicle_pose),
                )
                adwersbaddb.save_pointcloud_labels_to_db(conn, lidar_uid, labels, boxes)
        conn.commit()
        logger.info(
            f"processed waymo file {filepath} with {failed_frames} failed frames"
        )
    except psycopg.DatabaseError as e:
        logger.error(f"error processing record {filepath}. \n error: {e}")
        raise
    finally:
        if conn:
            conn.close()


def import_wmo(location, timestamp):
    from adwersbad.weather_import import get_weather_from_timestamp

    if location == "location_sf":
        lat, lon = [37.773972, -122.431297]
    elif location == "location_phx":
        lat, lon = [33.448376, -112.074036]
    else:
        return None
    weather, weather_string, isday = get_weather_from_timestamp(timestamp, lat, lon)
    return weather_string


if __name__ == "__main__":
    from tqdm.contrib.concurrent import process_map

    from adwersbad.class_helpers import (
        convert_waymo_label_to_adwersbad_label_camera,
        convert_waymo_label_to_adwersbad_label_lidar,
        create_adwersbad_label_map,
    )

    dbinfo = "psycopg@local"
    filepaths = ""
    for split in SPLITS:
        filepaths = get_paths_for_split(split)
        if not filepaths:
            print("no dataset files found")
            exit()
        # process_file(filepaths[0])
        process_map(process_file, filepaths, max_workers=12, chunksize=1)
