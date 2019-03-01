import os

import copy

from typing import Dict, List

config_defaults = {
    "outputs": [],
    "inputs": [],
    "children": False
}

io_defaults = {
    "negative": False
}

image_treat_defaults = {
    "type": "as_is",
    "threshold": 0
}

negative_mask_dataset = {
    "inputs": [{
        "name": "default",

        "data_type": "float32",

        "negative": True,

        "bindings": [{
            "reader": "RGBA",
            "path": "",
            "bind": [0, 1, 2],
            "treat": {
                "type": "as_is"
            }
        }]
    }]
}

color_mask_dataset = {
    "inputs": [{
        "name": "default",

        "data_type": "float32",

        "bindings": [{
            "reader": "RGBA",
            "path": "",
            "bind": [0, 1, 2],
            "treat": {
                "type": "as_is"
            }
        }]
    }],

    "outputs": [{
        "name": "default",

        "data_type": "bool",

        "bindings": [{
            "reader": "RGBA",
            "path": "",
            "treat": {
                "type": "binary_mask"
            }
        }]
    }]
}

simple_mask_dataset = {
    "inputs": [{
        "name": "default",

        "data_type": "uint8",

        "bindings": [{
            "reader": "RGBA",
            "path": "",
            "bind": [0, 1, 2],
            "treat": {
                "type": "as_is"
            }
        }]
    }],

    "outputs": [{
        "name": "default",

        "data_type": "bool",

        "bindings": [{
            "reader": "monochrome",
            "path": "",
            "bind": [0],
            "treat": {
                "type": "binary_mask"
            }
        }]
    }]
}

def unpack_config(name, config, from_dir):
    initial_config = config[name]

    if type(initial_config) is list:
        initial_config = {
            "children": initial_config
        }

    cfg: Dict = copy.deepcopy(initial_config)

    if "input_path" in cfg.keys():
        cfg["inputs"] = get_simple_mask_dataset(os.path.join(from_dir, cfg.pop("input_path")), "dummy")["inputs"]

    if "output_path" in cfg.keys():
        cfg["outputs"] = get_simple_mask_dataset("dummy", os.path.join(from_dir, cfg.pop("output_path")))["outputs"]

    cfg["inputs"] = cfg.pop("inputs", [])
    cfg["outputs"] = cfg.pop("outputs", [])
    cfg["children"] = cfg.pop("children", False)

    children = cfg["children"]

    if children:
        unpacked_children = []

        for item in children:
            if item == name:
                continue

            unpacked_children.append(unpack_config(item, config, from_dir))

        cfg["children"] = unpacked_children

    bindings = []

    for item in (cfg["inputs"] + cfg["outputs"]):
        bindings += item["bindings"]

    for binding in bindings:
        if os.path.isabs(binding["path"]):
            continue

        binding["path"] = os.path.join(from_dir, binding["path"])

    return cfg

def get_negative_mask_dataset(input_path):
    result = copy.deepcopy(negative_mask_dataset)

    result["inputs"][0]["bindings"][0]["path"] = input_path

    return result

def get_color_mask_dataset(input_path, output_path, labels, bind = None):
    result = copy.deepcopy(color_mask_dataset)

    result_bind = None

    if bind:
        result_bind = bind
    else:
        result_bind = list(range(len(labels)))

    result["inputs"][0]["bindings"][0]["path"] = input_path
    result["outputs"][0]["bindings"][0]["path"] = output_path
    result["outputs"][0]["bindings"][0]["bind"] = result_bind
    result["outputs"][0]["bindings"][0]["treat"]["colors"] = labels

    return result

def get_simple_mask_dataset(input_path, output_path):
    result = copy.deepcopy(simple_mask_dataset)

    result["inputs"][0]["bindings"][0]["path"] = input_path
    result["outputs"][0]["bindings"][0]["path"] = output_path

    return result