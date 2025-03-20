import glob
import logging
from os.path import join
from pathlib import Path


def get_file_paths(data_path: str = "", extension: str = "") -> list[str]:
    if not extension.startswith("."):
        extension = "." + extension
    path = data_path + "*" + extension
    return glob.glob(path)


def display_image(file_name_camera):
    import cv2
    import matplotlib.pyplot as plt

    image_front_center = cv2.imread(file_name_camera)
    image_front_center = cv2.cvtColor(image_front_center, cv2.COLOR_BGR2RGB)
    plt.fig = plt.figure(figsize=(15, 15))

    plt.imshow(image_front_center)
    plt.axis("off")
    plt.title("front center")

    plt.show()
    plt.waitforbuttonpress(0)
    plt.close()


def get_other_files_from_label_file(
    label_file_path: Path,
) -> (Path, Path, Path, Path, Path):
    """Given a label file path, return the corresponding image, image info, bus, lidar and bboxes file paths

    Args:
        label_file_path (Path): path to label file

    Returns:
        image_file (Path): path to image file
        image_json (Path): path to image json file
        bus_file (Path): path to bus file
        lidar_file (Path): path to lidar file
        bboxes_file (Path): path to bboxes file
    """

    assert Path(label_file_path).exists()
    label_path = Path(label_file_path)
    time_and_day = label_path.parent.parent.parent
    sensor_name = str(label_path.parent.parts[-1]).replace("cam_", "")
    # image file
    image_file = label_path.relative_to(time_and_day).with_suffix(".png")
    image_file = time_and_day / str(image_file).replace("label", "camera")

    # image json
    image_json = label_path.relative_to(time_and_day).with_suffix(".json")
    image_json = time_and_day / str(image_json).replace("label", "camera")

    # bus file
    bus_file = (
        time_and_day / "bus" / str(label_path.name).replace("label", "bus_signals")
    )
    bus_file = Path(str(bus_file).split("bus_signals")[0] + "bus_signals.json")

    # lidar file
    lidar_file = label_path.relative_to(time_and_day).with_suffix(".npz")
    lidar_file = time_and_day / str(lidar_file).replace("label", "lidar")

    # bboxes file
    bboxes_file = label_path.relative_to(time_and_day).with_suffix(".json")
    bboxes_file = time_and_day / str(bboxes_file).replace("label", "label3D")

    # check is all files exist, if not log and set to None
    if not image_file.exists():
        logging.debug(f"image file does not exist: {image_file}")
        image_file = None
    if not image_json.exists():
        logging.debug(f"image json does not exist: {image_json}")
        image_json = None
    if not bus_file.exists():
        logging.debug(f"bus file does not exist: {bus_file}")
        bus_file = None
    if not lidar_file.exists():
        logging.debug(f"lidar file does not exist: {lidar_file}")
        lidar_file = None
    if not bboxes_file.exists():
        logging.debug(f"bboxes file does not exist: {bboxes_file}")
        bboxes_file = None
    return sensor_name, image_file, image_json, bus_file, lidar_file, bboxes_file


def extract_image_file_name_from_lidar_file_name(file_name_lidar, root_path="./a2d2/"):
    file_name_image = file_name_lidar.split("/")
    file_name_image = file_name_image[-1].split(".")[0]
    file_name_image = file_name_image.split("_")
    file_name_image = (
        file_name_image[0]
        + "_"
        + "camera_"
        + file_name_image[2]
        + "_"
        + file_name_image[3]
        + ".png"
    )
    seq_name = file_name_lidar.split("/")[2]
    file_name_image = join(
        root_path, seq_name, "camera/cam_front_center/", file_name_image
    )
    return file_name_image


def extract_image_file_name_from_lidar_file_name(file_name_lidar, root_path="./a2d2/"):
    """example file name: ./a2d2/20180807_145028/lidar/
        cam_front_center/
        20180807145028_lidar_frontcenter_000000091.npz
    ->
        20180807145028_lidar_frontcenter_000000091
    ->
        [20180807145028,lidar,frontcenter,000000091]
    ->
        20180807145028_camera_frontcenter_000000091.png

    """

    # split into parts
    file_name_image = file_name_lidar.split("/")
    # get the last part without the extension
    file_name_image = file_name_image[-1].split(".")[0]
    # split into date and time, separated by _
    file_name_image = file_name_image.split("_")

    file_name_image = (
        file_name_image[0]
        + "_"
        + "camera_"
        + file_name_image[2]
        + "_"
        + file_name_image[3]
        + ".png"
    )
    seq_name = file_name_lidar.split("/")[2]
    file_name_image = join(
        root_path, seq_name, "camera/cam_front_center/", file_name_image
    )
    return file_name_image


def extract_bboxes_file_name_from_image_file_name(file_name_image, root_path="./a2d2/"):
    # split into parts
    file_name_bboxes = file_name_image.split("/")
    # get the last part without the extension
    file_name_bboxes = file_name_bboxes[-1].split(".")[0]
    # split into date and time, separated by _
    file_name_bboxes = file_name_bboxes.split("_")

    file_name_bboxes = (
        file_name_bboxes[0]
        + "_"
        + "label3D_"
        + file_name_bboxes[2]
        + "_"
        + file_name_bboxes[3]
        + ".json"
    )
    seq_name = file_name_image.split("/")[2]
    file_name_bboxes = join(
        root_path, seq_name, "label3D/cam_front_center/", file_name_bboxes
    )

    return file_name_bboxes


def extract_bus_file_name_from_lidar_file_name(file_name_lidar, root_path="./a2d2/"):
    """example file name: ./a2d2/20180807_145028/lidar/
        cam_front_center/
        20180807145028_lidar_frontcenter_000000091.npz
    ->

    """

    file_name_bus = file_name_lidar.split("/")
    file_name_bus = file_name_bus[-1].split(".")[0]
    file_name_bus = file_name_bus.split("_")
    file_name_bus = file_name_bus[0] + "_" + "bus_signals.json"
    seq_name = file_name_lidar.split("/")[2]
    file_name_bus = join(root_path, seq_name, "bus/", file_name_bus)
    return file_name_bus


def extract_json_file_name_from_image_file_name(file_name_image, root_path="./a2d2/"):
    file_name_json = file_name_image[:-3] + "json"
    return file_name_json


def get_dataset_files(dataset="a2d2", root_path="./a2d2"):
    if dataset == "a2d2":
        return get_a2d2_files(root_path)
    elif dataset == "waymo":
        return get_waymo_context_names(root_path)
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return file_names


def get_a2d2_files(root_path="./a2d2"):
    path = join(root_path, "*/camera/cam_front_center/*.npz")
    # print(f"looking for files matching: {path}")
    file_names = sorted(glob.glob(path))
    print(f"found {len(file_names)} files")
    return file_names


def get_waymo_context_names(root_path="./waymo"):
    train_path = join(root_path, "training/lidar/*.parquet")
    val_path = join(root_path, "validation/lidar/*.parquet")
    test_path = join(root_path, "testing/lidar/*.parquet")
    test_loc_path = join(root_path, "testing_location/lidar/*.parquet")
    file_names = {
        "training": sorted(glob.glob(train_path)),
        "validation": sorted(glob.glob(val_path)),
        "testing": sorted(glob.glob(test_path)),
        "testing_location": sorted(glob.glob(test_loc_path)),
    }
    # we need to extract the context name from files named like this:
    # ./waymo/lidar/1005081002024129653_5725_000_5745_000.parquet
    paths = []
    for file_name in file_names:
        context_name = file_name.split("/")[-1]
        context_name = context_name.split(".")[0]
        paths.append((context_name, file_name))
    # sanity check, each file name should contain the context name
    for context_name, file_name in paths:
        assert context_name in file_name
    return paths


def move_processed_file(file_name):
    """move file to processed folder leaving parents intact"""
    if file_name is None:
        return
    file_name = Path(file_name)
    new_file_name = "processed" / file_name
    Path.mkdir(new_file_name.parent, parents=True, exist_ok=True)
    file_name.replace(new_file_name)


def remove_empty_dirs(root):
    """remove empty directories"""
    for path in reversed(list(Path(root).rglob("*"))):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError as e:
                pass
