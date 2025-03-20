import json
from collections import namedtuple
from enum import IntEnum
from io import SEEK_SET
from random import seed, shuffle
from typing import Dict, Iterable, Union

import distinctipy

from adwersbad.config import config

CLASS_LIST_PATH = "a2d2_semantic/class_list.json"
RGB_OUTPUT_PATH = "a2d2_semantic/a2d2_rgb.json"
LABELS_OUTPUT_PATH = "adwersbad_labels.json"
LABEL_COLORS_OUTPUT_PATH = "label_colors.png"
SEED_VALUE = 69420


def convert_a2d2_to_rgb():
    with open("a2d2_semantic/class_list.json") as f:
        data = json.load(f)
    reduced_classes = reduce_a2d2_classes(data)
    print(reduced_classes)
    a2d2_rgb = {hex_to_rgb(key): value for key, value in data.items()}
    with open("a2d2_semantic/a2d2_rgb.json", "w") as fp:
        json.dump(a2d2_rgb, fp)


def reduce_a2d2_classes(a2d2_dict):
    class_list = []
    class_dict = {}
    for key, value in a2d2_dict.items():
        o = value.split(" ")[0]
        if o not in class_list:
            class_list.append(o)
            class_dict[o] = key
    return class_dict


def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip("#")
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def chain_maps(*label_maps: Iterable[Dict]) -> Dict:
    """Chain label maps.

    The first dictionary's keys map to the values of the final dictionary, using
    all intermediate dictionaries in sequence.
    Note that currently there is no check if the dictionaries are actually compatible maps.

    Args:
        *dicts: An arbitrary number of dictionaries to chain.

    Returns:
        A combined dictionary where keys from the first dictionary map to
        values from the last dictionary.
    """
    if len(label_maps) < 2:
        raise ValueError("At least two dictionaries are needed to chain.")

    combined_map = label_maps[0]

    for next_dict in label_maps[1:]:
        combined_map = {k: next_dict[v] for k, v in combined_map.items()}

    return combined_map


adwersbad_label = namedtuple(
    "adwersbadlabel", ["name", "id", "category", "cat_id", "color"]
)

a2d2_to_adwersbad_label_map = {
    "rd normal street": "road",
    "drivable cobblestone": "road",
    "parking ares": "road",
    "slow drive area": "road",
    "painted driv. instr.": "road",
    "zebra crossing": "road",
    "speed bumper": "road",
    "solid line": "lane marking",
    "dashed line": "lane marking",
    "sidewalk": "sidewalk",
    "curbstone": "sidewalk",
    "non-drivable street": "sidewalk",
    "buildings": "building",
    "sidebars": "wall",
    "grid structure": "fence",
    "road blocks": "pole",
    "poles": "pole",
    "signal corpus": "traffic light",
    "traffic signal 1": "traffic light",
    "traffic signal 2": "traffic light",
    "traffic signal 3": "traffic light",
    "traffic sign 1": "traffic sign",
    "traffic sign 2": "traffic sign",
    "traffic sign 3": "traffic sign",
    "traffic guide obj.": "traffic sign",
    "nature object": "vegetation",
    "irrelevant signs": "terrain",
    "obstacles / trash": "terrain",
    "sky": "sky",
    "pedestrian 1": "person",
    "pedestrian 2": "person",
    "pedestrian 3": "person",
    "car 1": "car",
    "car 2": "car",
    "car 3": "car",
    "car 4": "car",
    "small vehicles 1": "car",
    "small vehicles 2": "car",
    "small vehicles 3": "car",
    "electronic traffic": "car",
    "truck": "truck",
    "utility vehicle": "truck",
    "tractor": "truck",
    "bycicle 1": "bicycle",
    "bycicle 2": "bicycle",
    "bycicle 3": "bicycle",
    "bycicle 4": "bicycle",
    "ego car": "void",
    "rd restricted area": "void",
    "blurred area": "void",
}


class WaymoLabelLidar(IntEnum):
    UNDEFINED = 0
    CAR = 1
    TRUCK = 2
    BUS = 3
    OTHER_VEHICLE = 4
    MOTORCYCLIST = 5
    BICYCLIST = 6
    PEDESTRIAN = 7
    SIGN = 8
    TRAFFIC_LIGHT = 9
    POLE = 10
    CONSTRUCTION_CONE = 11
    BICYCLE = 12
    MOTORCYCLE = 13
    BUILDING = 14
    VEGETATION = 15
    TREE_TRUNK = 16
    CURB = 17
    ROAD = 18
    LANE_MARKER = 19
    OTHER_GROUND = 20
    WALKABLE = 21
    SIDEWALK = 22


class WaymoLabelCamera(IntEnum):
    UNDEFINED = 0
    EGO_VEHICLE = 1
    CAR = 2
    TRUCK = 3
    BUS = 4
    OTHER_LARGE_VEHICLE = 5
    BICYCLE = 6
    MOTORCYCLE = 7
    TRAILER = 8
    PEDESTRIAN = 9
    CYCLIST = 10
    MOTORCYCLIST = 11
    BIRD = 12
    GROUND_ANIMAL = 13
    CONSTRUCTION_CONE_POLE = 14
    POLE = 15
    PEDESTRIAN_OBJECT = 16
    SIGN = 17
    TRAFFIC_LIGHT = 18
    BUILDING = 19
    ROAD = 20
    LANE_MARKER = 21
    ROAD_MARKER = 22
    SIDEWALK = 23
    VEGETATION = 24
    SKY = 25
    GROUND = 26
    DYNAMIC = 27
    STATIC = 28


# class WaymoLabelBox(IntEnum):
#     TYPE_UNKNOWN = 0
#     TYPE_VEHICLE = 1
#     TYPE_PEDESTRIAN = 2
#     TYPE_SIGN = 3
#     TYPE_CYCLIST = 4


waymo_to_adwersbad_label_map_bbox = {
    "TYPE_UNKNOWN": "void",
    "TYPE_VEHICLE": "vehicle",
    "TYPE_PEDESTRIAN": "person",
    "TYPE_SIGN": "traffic sign",
    "TYPE_CYCLIST": "bicycle",
}

waymo_to_adwersbad_label_map_lidar = {
    WaymoLabelLidar.UNDEFINED: "void",
    WaymoLabelLidar.CAR: "car",
    WaymoLabelLidar.TRUCK: "truck",
    WaymoLabelLidar.BUS: "bus",
    WaymoLabelLidar.OTHER_VEHICLE: "vehicle",
    WaymoLabelLidar.MOTORCYCLIST: "rider",
    WaymoLabelLidar.BICYCLIST: "rider",
    WaymoLabelLidar.PEDESTRIAN: "person",
    WaymoLabelLidar.SIGN: "traffic sign",
    WaymoLabelLidar.TRAFFIC_LIGHT: "traffic light",
    WaymoLabelLidar.POLE: "pole",
    WaymoLabelLidar.CONSTRUCTION_CONE: "pole",
    WaymoLabelLidar.BICYCLE: "bicycle",
    WaymoLabelLidar.MOTORCYCLE: "motorcycle",
    WaymoLabelLidar.BUILDING: "building",
    WaymoLabelLidar.VEGETATION: "vegetation",
    WaymoLabelLidar.TREE_TRUNK: "vegetation",
    WaymoLabelLidar.CURB: "sidewalk",
    WaymoLabelLidar.ROAD: "road",
    WaymoLabelLidar.LANE_MARKER: "lane marking",
    WaymoLabelLidar.OTHER_GROUND: "terrain",
    WaymoLabelLidar.WALKABLE: "sidewalk",
    WaymoLabelLidar.SIDEWALK: "sidewalk",
}


waymo_to_adwersbad_label_map_camera = {
    WaymoLabelCamera.UNDEFINED: "void",
    WaymoLabelCamera.EGO_VEHICLE: "ego vehicle",
    WaymoLabelCamera.CAR: "car",
    WaymoLabelCamera.TRUCK: "truck",
    WaymoLabelCamera.BUS: "bus",
    WaymoLabelCamera.OTHER_LARGE_VEHICLE: "vehicle",
    WaymoLabelCamera.BICYCLE: "bicycle",
    WaymoLabelCamera.MOTORCYCLE: "motorcycle",
    WaymoLabelCamera.TRAILER: "vehicle",
    WaymoLabelCamera.PEDESTRIAN: "person",
    WaymoLabelCamera.CYCLIST: "rider",
    WaymoLabelCamera.MOTORCYCLIST: "rider",
    WaymoLabelCamera.BIRD: "bird",
    WaymoLabelCamera.GROUND_ANIMAL: "animal",
    WaymoLabelCamera.CONSTRUCTION_CONE_POLE: "pole",
    WaymoLabelCamera.POLE: "pole",
    WaymoLabelCamera.PEDESTRIAN_OBJECT: "person",
    WaymoLabelCamera.SIGN: "traffic sign",
    WaymoLabelCamera.TRAFFIC_LIGHT: "traffic light",
    WaymoLabelCamera.BUILDING: "building",
    WaymoLabelCamera.ROAD: "road",
    WaymoLabelCamera.LANE_MARKER: "lane marking",
    WaymoLabelCamera.ROAD_MARKER: "lane marking",
    WaymoLabelCamera.SIDEWALK: "sidewalk",
    WaymoLabelCamera.VEGETATION: "vegetation",
    WaymoLabelCamera.SKY: "sky",
    WaymoLabelCamera.GROUND: "ground",
    WaymoLabelCamera.DYNAMIC: "dynamic",
    WaymoLabelCamera.STATIC: "static",
}

nuscenes_to_adwersbad_label_map = {
    "animal": "animal",
    "human.pedestrian.stroller": "person",
    "human.pedestrian.wheelchair": "person",
    "human.pedestrian.adult": "person",
    "human.pedestrian.child": "person",
    "human.pedestrian.construction_worker": "person",
    "human.pedestrian.police_officer": "person",
    "human.pedestrian.child": "person",
    "human.pedestrian.construction_worker": "person",
    "human.pedestrian.personal_mobility": "person",
    "vehicle.bicycle": "bicycle",
    "vehicle.emergency.ambulance": "truck",
    "vehicle.emergency.police": "car",
    "vehicle.truck": "truck",
    "vehicle.car": "car",
    "vehicle.ego": "ego vehicle",
    "vehicle.motorcycle": "motorcycle",
    "noise": "void",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "truck",
    "vehicle.construction": "vehicle",
    "vehicle.bus.bendy": "bus",
    "flat.other": "ground",
    "flat.terrain": "terrain",
    "flat.sidewalk": "sidewalk",
    "flat.driveable_surface": "road",
    "static.vegetation": "vegetation",
    "static.manmade": "building",
    "static_object.bicycle_rack": "bicycle",
    "static.other": "building",
    "movable_object.barrier": "fence",
    "movable_object.debris": "static",
    "movable_object.pushable_pullable": "static",
    "movable_object.trafficcone": "pole",
    "void": "void",
}


def convert_waymo_label_to_adwersbad_label_lidar(waymo_label):
    return waymo_to_adwersbad_label_map_lidar[WaymoLabelLidar(waymo_label)]


def convert_waymo_label_to_adwersbad_label_camera(waymo_label):
    return waymo_to_adwersbad_label_map_camera[WaymoLabelCamera(waymo_label)]


def get_category(name):
    name = name.lower()
    if "person" in name or "rider" in name:
        return "person"
    elif (
        "car" in name
        or "bus" in name
        or "truck" in name
        or "bicycle" in name
        or "motorcycle" in name
        or "vehicle" in name
    ):
        return "vehicle"
    elif "building" in name or "wall" in name or "fence" in name:
        return "structure"
    elif "road" in name or "sidewalk" in name:
        return "road and sidewalk"
    elif "lane marking" in name:
        return "lane marking"
    elif "vegetation" in name or "terrain" in name:
        return "Nature"
    elif "sky" in name:
        return "sky"
    elif "traffic light" in name or "traffic sign" in name:
        return "traffic signal"
    elif "pole" in name:
        return "pole"
    elif "void" in name or "ego vehicle" in name:
        return "void"
    else:
        return "other"


def print_colors(adwersbad_labels):
    """Display a legend of adwersbad label colors using matplotlib.

    This function creates a visual representation of adwersbad label colors and saves it as a PNG file.

    Args:
        adwersbad_labels (List[adwersbad_label]): The list of adwersbad labels to display colors for.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    color_names = {}
    for l in adwersbad_labels:
        color_names[l.name] = l.color

    fig, ax = plt.subplots()
    rect_height = 1 / len(color_names)

    for i, (name, color) in enumerate(color_names.items()):
        rect = mpatches.Rectangle(
            (0, i * rect_height), 1, rect_height, color=color, label=name
        )
        ax.add_patch(rect)

    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("label_colors.png")


def generate_adwersbad_labels():
    """
    Generate adwersbad labels from existing label mappings.

    Ensures the 'void' label always has ID 0, aggregates unique label names from datasets,
    assigns unique IDs, categorizes them, generates distinct colors, and returns a list of adwersbad labels.

    Returns:
        List[adwersbad_label]: A list of adwersbad labels, each containing the name, id, category,
                         category ID, and assigned color.
    """
    unique_values = set().union(
        a2d2_to_adwersbad_label_map.values(),
        waymo_to_adwersbad_label_map_lidar.values(),
        waymo_to_adwersbad_label_map_camera.values(),
        waymo_to_adwersbad_label_map_bbox.values(),
        nuscenes_to_adwersbad_label_map.values(),
    )

    # Ensure 'void' is always present and has ID 0
    unique_values = sorted(unique_values)
    if "void" in unique_values:
        unique_values.remove("void")
    unique_values.insert(0, "void")

    labels_with_categories = [(name, get_category(name)) for name in unique_values]

    # Sort by category, then alphabetically, but keep 'void' as the first label
    labels_with_categories = sorted(
        labels_with_categories,
        key=lambda x: (
            (x[1], x[0]) if x[0] != "void" else ("", "")
        ),  # Ensure 'void' stays at the top
    )

    unique_labels_with_ids = {
        name: idx for idx, (name, _) in enumerate(labels_with_categories)
    }
    categories = sorted({category for _, category in labels_with_categories})
    category_ids = {name: idx for idx, name in enumerate(categories)}

    # Generate distinct colors for the labels
    seed(SEED_VALUE)
    colors = distinctipy.get_colors(len(unique_labels_with_ids))
    shuffle(colors)

    # Generate final adwersbad labels
    adwersbad_labels = [
        adwersbad_label(
            name=name,
            id=id,
            category=get_category(name),
            cat_id=category_ids[get_category(name)],
            color=colors[id],
        )
        for name, id in unique_labels_with_ids.items()
    ]

    return adwersbad_labels


conf = config(section="paths")
label_file = conf["project_root"] + "/adwersbad_labels.json"

try:
    with open(label_file) as f:
        adwersbad_labels = [adwersbad_label(**d) for d in json.load(f)]
except FileNotFoundError as e:
    print(f"label file {label_file} not found, generating labels")
    adwersbad_labels = generate_adwersbad_labels()
# create a map from any field to another field in adwersbadlabel


def create_adwersbad_label_map(
    from_field: str, to_field: str
) -> Dict[Union[str, int], Union[str, int]]:
    """Create a mapping from one field to another in adwersbad labels.

    This function creates a dictionary mapping the values of a specified field
    to the values of another specified field from the adwersbad labels.

    Args:
        from_field (str): The field name to map from.
        to_field (str): The field name to map to.

    Returns:
        Dict[Union[str, int], Union[str, int]]: A dictionary mapping from_field values to to_field values.
    """
    label_map = {}
    for label in adwersbad_labels:
        label_map[getattr(label, from_field)] = getattr(label, to_field)
    return label_map


"""
n2id = create_adwersbad_label_map("name", "id")
id2n = create_adwersbad_label_map("id", "name")
n2c = create_adwersbad_label_map("name", "category")
c2n = create_adwersbad_label_map("category", "name")
n2cid = create_adwersbad_label_map("name", "cat_id")
cid2n = create_adwersbad_label_map("cat_id", "name")
n2col = create_adwersbad_label_map("name", "color")
col2n = create_adwersbad_label_map("color", "name")
print("name to id:")
print(n2id)
print("id to name:")
print(id2n)
print("name to category:")
print(n2c)
print("category to name:")
print(c2n)
print("name to category id:")
print(n2cid)
print("category id to name:")
print(cid2n)
print("name to color:")
print(n2col)
print("color to name:")
print(col2n)
"""
"""
print("after serialization:")

with open('adwersbad_name_to_id_map.json') as f:
    adwersbad_labels = json.load(f)
#convert to named tuple
adwersbad_labels = [adwersbad_label(**x) for x in adwersbad_labels]
"""

if __name__ == "__main__":
    # adwersbad_labels = generate_adwersbad_labels()
    print("adwersbad_labels:")
    for label in adwersbad_labels:
        print(label)
    print("cateogries")
    cur_cat = adwersbad_labels[0].category
    print(cur_cat)
    for label in adwersbad_labels:
        if label.category == cur_cat:
            print(label)
        else:
            cur_cat = label.category
            print("------")
            print(cur_cat)
            print(label)
    with open("adwersbad_labels.json", "w") as f:
        json.dump([x._asdict() for x in adwersbad_labels], f)

    print("total number of labels: " + str(len(adwersbad_labels)))
    create_adwersbad_label_map("name", "id")
    # with open('adwersbad_name_to_id_map.json', "w") as f:
    # for label in sorted(adwersbad_labels, key=lambda x:x.id):
    #    print(label)
