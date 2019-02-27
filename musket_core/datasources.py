import os

import numpy as np
import imageio

import dsconfig

from typing import List, Dict

def get_data_reader(config):
    reader = DATA_READERS[config["reader"]]

    return reader(config)

def get_value(key, dict: Dict, defaults: Dict):
    if key in dict.keys():
        return dict[key]

    if key in defaults:
        return defaults[key]

    return None

class Treat:
    def __init__(self, cfg, defaults = {}):
        self.type = get_value("type", cfg, defaults)
        self.colors = get_value("colors", cfg, defaults)
        self.threshold = get_value("threshold", cfg, defaults)
        self.normalize = get_value("normalize", cfg, defaults)

class AbstractDataReader:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_ids(self):
        pass

    def read(self, id):
        pass

class ImageDataReader(AbstractDataReader):
    def __init__(self, cfg: Dict):
        self.cfg = cfg

        self.path = cfg["path"]

        self.ids = []
        self.exts = {}

        self.treat = Treat(cfg["treat"], dsconfig.image_treat_defaults)

        if self.treat.colors:
            self.colors = self.fill_alpha(self.treat.colors)

        for item in os.listdir(self.path):
            ext = item[item.rindex(".") + 1:]

            if ext in ["png", "jpeg", "jpg", "gif", "bmp"]:
                id = item[:item.rindex(".")]

                self.ids.append(id)

                self.exts[id] = ext

    def get_ids(self):
        return self.ids

    def fill_alpha(self, list):
        result = []

        for item in list:
            new_item = []

            for v in item:
                new_item.append(v)

            while len(new_item) < 4:
                new_item.append(0)

            result.append(new_item)

        return result

    def pick_by_colors(self, data):
        masks = []

        for item in self.colors:
            mask = np.expand_dims(np.all(data == item, axis=-1), 2)

            masks.append(mask)

        return np.concatenate(masks, 2)

    def normalize(self, data, value):
        max = np.max(data)

        if max > 0:
            return self.treat.normalize * data / max
        return data

    def read(self, id):
        path = os.path.join(self.path, id + "." + self.exts[id])

        result = imageio.imread(path)

        if len(result.shape) < 3:
            result = np.expand_dims(result, 2)

        if self.cfg["reader"] == "monochrome":
            result = np.expand_dims(np.sum(result, 2), 2) / result.shape[-1]

        if self.cfg["reader"] == "RGBA":
            if result.shape[-1] == 3:
                result = np.concatenate((result, np.zeros(result.shape[:-1] + (1,))), 2)
            if result.shape[-1] == 1:
                result = np.concatenate((result, result, result, result), 2)

        if self.treat.type == "as_is":
            if self.treat.normalize:
                return normalize(result)

            return result

        if self.treat.type == "binary_mask" and not self.treat.colors:
            result = result > self.treat.threshold

        if self.treat.type == "binary_mask" and self.treat.colors:
            result = self.pick_by_colors(result.astype(np.uint8))

        if self.treat.normalize:
            max = np.max(result)

            if max > 0:
                result = self.treat.normalize * result / max

        return result

DATA_READERS = {
    "monochrome": ImageDataReader,
    "RGBA": ImageDataReader
}

class DataSourceItem:
    def __init__(self, id, inputs: List, outputs: List):
        self.id = id
        self.inputs = inputs
        self.outputs = outputs

class GenericDataSource:
    def __init__(self, config: Dict, unwrap = False):
        self.unwrap = unwrap

        self.config = self.parse_config(config)

        self.ids = self.resolve_ids()

        pass

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        id = self.ids[item]

        inputs = []
        outputs = []

        for input in self.config["inputs"]:
            item = self.read_io_item(id, input)

            if input["negative"]:
                outputs.append(np.zeros(item.shape[:-1] + (input["negative"] + 0, ), np.bool))

            inputs.append(item)

        for output in self.config["outputs"]:
            outputs.append(self.read_io_item(id, output))

        if self.unwrap:
            return DataSourceItem(id, inputs[0], outputs[0])

        return DataSourceItem(id, inputs, outputs)

    def read_io_item(self, id, io_config):
        channels = []

        for binding in io_config["bindings"]:
            reader: AbstractDataReader = binding["reader"]

            data: np.ndarray = self.bind(reader.read(id), binding["bind"])

            channels.append(data.astype(io_config["data_type"]))

        channels_size = 0

        return np.concatenate(channels, len(channels[0].shape) - 1)

    def bind(self, data, channels):
        result = np.take(data, channels, len(data.shape) - 1)

        return result

    def resolve_ids(self):
        all_ids = []

        for io in (self.config["inputs"] + self.config["outputs"]):
            for binding in io["bindings"]:
                all_ids.append(binding["reader"].get_ids())

        result = set(all_ids[0])

        for ids in all_ids:
            result = result & set(ids)

        result = list(result)

        result.sort()

        return result

    def parse_config(self, config):
        result = {
            "inputs": [],
            "outputs": []
        }

        for input in config["inputs"]:
            result["inputs"].append(self.parse_io_config(input))

        for output in get_value("outputs", config, dsconfig.config_defaults):
            result["outputs"].append(self.parse_io_config(output))

        return result

    def parse_io_config(self, io_config):
        result = {
            "name": io_config["name"],
            "data_type": io_config["data_type"],
            "bindings": [],
            "negative": get_value("negative", io_config, dsconfig.io_defaults)
        }

        for binding in io_config["bindings"]:
            parsed_binding = {
                "bind": binding["bind"],
                "treat": binding["treat"],
                "reader": get_data_reader(binding)
            }

            result["bindings"].append(parsed_binding)

        return result

class SimpleMaskDataSet(GenericDataSource):
    def __init__(self, input_path, output_path):
        GenericDataSource.__init__(self, dsconfig.get_simple_mask_dataset(input_path, output_path), True)

class ColorMaskDataSet(GenericDataSource):
    def __init__(self, input_path, output_path, label_colors, bind = None):
        GenericDataSource.__init__(self, dsconfig.get_color_mask_dataset(input_path, output_path, label_colors, bind), True)

class NegativeMaskDataSet(GenericDataSource):
    def __init__(self, input_path,):
        GenericDataSource.__init__(self, dsconfig.get_negative_mask_dataset(input_path), True)

#gds = ColorMaskDataSet("./picsart/train", "./picsart/train", [[210, 208, 195], [211, 209, 196], [210, 206, 194]])

# gds = NegativeMaskDataSet("./picsart/train")
#
# for i in range(10):
#     item = gds[i]
#
#     o = item.outputs.astype(np.uint8) * 255
#
#     imageio.imsave("./dstst/" + str(i) + ".png", o)