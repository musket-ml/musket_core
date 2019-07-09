from musket_core import datasets

import numpy as np

import keras

import tensorflow as tf

import lightgbm

import os

class OutputMeta:
    def __init__(self, shape, owner):
        self.output_meta = True
        self.model = owner

class GradientBoosting:
    def __init__(self, output_dim, boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, class_weight=None, min_split_gain=0., min_child_weight=1e-3, min_child_samples=20, subsample=1., subsample_freq=0, colsample_bytree=1., reg_alpha=0., reg_lambda=0., random_state=None, n_jobs=-1, silent=True, importance_type='split'):
        if output_dim > 1:
            objective = "multiclass"
        else:
            objective = "regression"

        self.model = lightgbm.LGBMModel(boosting_type, num_leaves, max_depth, learning_rate, n_estimators, subsample_for_bin, objective, class_weight, min_split_gain, min_child_weight, min_child_samples, subsample, subsample_freq, colsample_bytree, reg_alpha, reg_lambda, random_state, n_jobs, silent, importance_type, n_classes=output_dim)

        self.output_dim = output_dim

        self.result = None

    def __call__(self, *args, **kwargs):
        return OutputMeta(self.output_dim, self)

    def compile(self, *args, **kwargs):
        pass

    def convert_data(self, generator):
        result_x = []
        result_y = []

        for item in generator:
            result_x.append(item[0])
            result_y.append(item[1])

        result_x = np.concatenate(result_x)
        result_y = np.concatenate(result_y)

        result_x = np.reshape(result_x, (len(result_x), -1))
        result_y = np.reshape(result_y, (len(result_y), -1))

        if self.output_dim > 1:
            result_y = np.argmax(result_y, 1)

        return result_x.astype(np.float32), result_y.astype(np.int32)

    def predict(self, *args, **kwargs):
        input = np.array(args)[0]

        input = np.reshape(input, (len(input), -1))

        self.model._n_features = input.shape[1]

        predictions = self.model.predict(input)

        return predictions

    def load_weights(self, path, val):
        if os.path.exists(path):
            self.model._Booster = lightgbm.Booster(model_file=path)

    def fit_generator(self, *args, **kwargs):
        callbacks = kwargs["callbacks"]

        file_path = None

        for item in callbacks:
            if hasattr(item, "filepath"):
                file_path = item.filepath

        generator_train = args[0]
        generator_test = kwargs["validation_data"]

        train_x, train_y = self.convert_data(generator_train)
        val_x, val_y = self.convert_data(generator_test)

        self.model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=2)

        self.model.booster_.save_model(file_path)