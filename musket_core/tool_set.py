import os

import numpy as np

import pandas as pd

import imageio

import imgaug

from typing import Dict, List

from musket_core.utils import ensure

class ImageWriter:
    def __init__(self, root, id_set = [], parameters = {}):
        self.root = os.path.join(root, id_set[0] + "_" + id_set[1])

        self.parameters = parameters

        ensure(self.root)

    def write(self, id, image):
        imageio.imsave(os.path.join(self.root, id + ".png"), image.astype(np.uint8))

class MetricsWriter:
    def __init__(self, root, id_set = [], parameters = {}):
        self.fname = os.path.join(root, id_set[0] + "_" + id_set[1] + ".csv")

        self.metrics = {}

        self.metrics_calculators = {
            "iou": self.iou,
            "id": self.id,
            "test": self.test
        }

        self.scale_aug = None

    def write(self, name, data, metrics_calculator = None):
        self.write_by_name("id", data)
        self.write_by_name(name, data, metrics_calculator)

    def write_all(self, names, data, metrics_calculator = {}):
        self.write_by_name("id", data)

        for name in names:
            self.write_by_name(name, data, metrics_calculator.get(name, None))

    def write_by_name(self, name, data, metrics_calculator = None):
        self.metrics[name] = self.metrics.pop(name, [])

        if metrics_calculator:
            self.metrics[name].append(metrics_calculator(data))
        else:
            self.metrics[name].append(self.metrics_calculators[name](data))

    def end(self):
        df = pd.DataFrame(self.metrics)

        df.to_csv(self.fname, index=False)

    def get_scale_aug(self, data):
        if self.scale_aug:
            return self.scale_aug
        else:
            self.scale_aug = imgaug.augmenters.Scale({"height": data.p_y.shape[0], "width": data.p_y.shape[1]})

        return self.scale_aug

    def id(self, data):
        return data.id

    def test(self, data):
        return 100

    def iou(self, data):
        yt = data.y

        yp = data.p_y

        yt = self.get_scale_aug(data).augment_image(yt.astype(np.uint8))

        yt_max = np.max(yt)
        yp_max = np.max(yp)

        if yt_max <= 0: yt_max = 1
        if yp_max <= 0: yp_max = 1

        yt = yt / np.max(yt)
        yp = yp / np.max(yp)

        intersection = np.sum(yt * yp)
        union = np.sum(yt) + np.sum(yp)

        if union <= 0:
            return 1

        return 2.0 * intersection / union