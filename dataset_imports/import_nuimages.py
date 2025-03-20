import json
from datetime import datetime
from io import BytesIO

import numpy as np
import psycopg
import tqdm
from nuimages import NuImages
from nuimages.utils.utils import name_to_index_mapping
from PIL import Image
from tqdm.contrib.concurrent import process_map

from adwersbad import config
from adwersbad.class_helpers import (
    create_adwersbad_label_map,
    nuscenes_to_adwersbad_label_map,
)
from adwersbad.db_helpers import (
    get_connection,
    save_image_labels_to_db,
    save_image_to_db,
    save_weather_to_db,
)
from adwersbad.log import setup_logger
from adwersbad.weather_import import check_if_twilight, get_weather_from_timestamp

# Configure the logging system
logger = setup_logger("import_nuimages")

loc_to_gps = {
    "boston-seaport": [42.336849169438615, -71.05785369873047],
    "singapore-onenorth": [1.2882100868743724, 103.78475189208984],
    "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555],
    "singapore-queenstown": [1.2782562240223188, 103.76741409301758],
}


def get_image_and_segmentation(nuim, sample, label_map):
    # Get the sample_data entry for the image
    sample_data = nuim.get("sample_data", sample["key_camera_token"])
    sd_token = sample_data["token"]
    calib = nuim.get("calibrated_sensor", sample_data["calibrated_sensor_token"])

    quaternion = np.array(calib["rotation"]).tolist()
    translation = np.array(calib["translation"]).tolist()
    cam_intrinsic = calib["camera_intrinsic"]
    cam_dist = calib["camera_distortion"]
    channel = nuim.get("sensor", calib["sensor_token"])["channel"]
    channel = "nuimages_" + channel
    image_file_path = nuim.dataroot + "/" + sample_data["filename"]
    # Load the image using PIL
    with open(image_file_path, "rb") as img_file:
        img_bytes = img_file.read()
    # Check if this image has a corresponding semantic segmentation
    # nuim.render_image(sample['key_camera_token'], out_path='test.jpg')
    semantic, _ = nuim.get_segmentation(sd_token)
    semantic = label_map(semantic)
    semantic_bytes = BytesIO()
    cam_params = {
        "rotation": quaternion,
        "translation": translation,
        "intrinsic": cam_intrinsic,
        "distortion": cam_dist,
    }
    Image.fromarray(semantic).save(semantic_bytes, format="PNG")
    return BytesIO(img_bytes), semantic_bytes, channel, cam_params


def get_metadata(nuim, sample):
    timestamp = sample["timestamp"]
    log = nuim.get("log", sample["log_token"])
    location = log["location"]
    lat, lon = loc_to_gps[location]
    weather_code, weather_string, isday = get_weather_from_timestamp(
        timestamp, lat, lon
    )
    return timestamp, location, lat, lon, weather_code, weather_string, isday


def process_sample(sample):
    nuim_n2id = name_to_index_mapping(nuim.category)
    nuim_id2n = {v: k for k, v in nuim_n2id.items()}
    adwersbad_n2id = create_adwersbad_label_map("name", "id")
    label_map = {
        key: adwersbad_n2id.get(nuscenes_to_adwersbad_label_map.get(sub_key))
        for key, sub_key in nuim_id2n.items()
    }
    #
    void_label = adwersbad_n2id["void"]
    # void label aka unlabeled, 0 in nuscenes
    # no idea why they dont have it in ther index mapping
    label_map.update({0: void_label})
    label_map = np.vectorize(label_map.get, otypes=[np.uint8])
    with get_connection(dbinfo) as conn:
        try:
            im, seg, channel, cam_params = get_image_and_segmentation(
                nuim, sample, label_map
            )
            time, loc, lat, lon, wmo, weather_string, isday = get_metadata(nuim, sample)
            time = datetime.fromtimestamp(time / 1e6)
            if isday == 0:
                tod = "Night"
            else:
                tod = "Day"
            is_twilight, ttod = check_if_twilight(time, lat, lon)
            if is_twilight:
                tod = ttod
            weather_uid = save_weather_to_db(
                conn, time, loc, weather_string, tod, "nuimages", split
            )
            if weather_uid == -1:
                logger.debug(
                    f"Skipping image processing due to error in weather data insertion (uid=-1)"
                )
            else:
                image_uid = save_image_to_db(
                    conn, weather_uid, im, channel, json.dumps(cam_params)
                )
                save_image_labels_to_db(conn, image_uid, seg, False)
            conn.commit()
            logger.debug(f"Completed processing sample with {weather_uid=}")
        except psycopg.DatabaseError as e:
            logger.error(f"Error processing record {filepath}. \n Error: {e}")
            raise
    return


def process_samples():
    process_map(process_sample, (nuim.sample), max_workers=12, chunksize=6)
    # for sample in tqdm.tqdm(nuim.sample):


dbinfo = "psycopg@local"
for split in ["training", "validation"]:
    if split == "training":
        version = "v1.0-train"
    elif split == "validation":
        version = "v1.0-val"
    else:
        logger.error(f"invalid {split=}, aborting")
        exit()
    # TODO: get from .ini
    nuim = NuImages(
        dataroot="/adwersbaddata/datasets/nuimages",
        version=version,
        verbose=True,
        lazy=False,
    )
    nuim_n2id = name_to_index_mapping(nuim.category)
    nuim_id2n = {v: k for k, v in nuim_n2id.items()}
    adwersbad_n2id = create_adwersbad_label_map("name", "id")
    label_map = {
        key: adwersbad_n2id.get(nuscenes_to_adwersbad_label_map.get(sub_key))
        for key, sub_key in nuim_id2n.items()
    }
    void_label = adwersbad_n2id["void"]
    # void label aka unlabeled, 0 in nuscenes, it is not specified in the name_to_index_mapping provided by nuscenes
    label_map.update({0: void_label})
    logger.info(f"created label map for nuimages:")
    for k, v in label_map.items():
        logger.info(f"{k} : {v}")
    process_samples()
