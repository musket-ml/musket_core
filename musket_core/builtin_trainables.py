from musket_core import datasets, generic_config

import numpy as np

import keras

import tensorflow as tf

import lightgbm

import os


class OutputMeta:
    def __init__(self, shape, owner):
        self.output_meta = True
        self.model = owner

class Log():
    def __init__(self):
        pass

    def get(self, monitor):
        pass

class RGetter:
    def get(self, key):
        return self.__dict__[key]

    def __getitem__(self, item):
        if not item in self.__dict__.keys():
            print("KEY: " + str(item))

        return self.__dict__[item]

    def keys(self):
        return self.__dict__.keys()

    def has_key(self, k):
        return k in self.__dict__

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __unicode__(self):
        return repr(self.__dict__)

class GradientBoosting:
    def __init__(self, output_dim, boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, class_weight=None, min_split_gain=0., min_child_weight=1e-3, min_child_samples=20, subsample=1., subsample_freq=0, colsample_bytree=1., reg_alpha=0., reg_lambda=0., random_state=None, n_jobs=-1, silent=True, importance_type='split'):
        if output_dim > 2:
            objective = "multiclass"
        else:
            objective = "regression"

        self.model = lightgbm.LGBMModel(boosting_type, num_leaves, max_depth, learning_rate, n_estimators, subsample_for_bin, objective, class_weight, min_split_gain, min_child_weight, min_child_samples, subsample, subsample_freq, colsample_bytree, reg_alpha, reg_lambda, random_state, n_jobs, silent, importance_type, n_classes=output_dim, first_metric_only=True)

        self.output_dim = output_dim

        self.custom_metrics = {}

        self.result = None

        self.rgetter = RGetter()

        self.stop_training = False

        self.custom_loss_callable = None

    def __call__(self, *args, **kwargs):
        return OutputMeta(self.output_dim, self)

    def loss_to_gb(self, loss):
        result = np.array(loss)

        if len(result.shape) == 2:
            result = result.transpose([0, 1]).flatten()

        return result

    def compile(self, *args, **kwargs):
        custom_loss = args[1]

        if not custom_loss in ["multiclass", "regression"]:
            custom_loss_tf = keras.losses.get(custom_loss)

            t_true = keras.layers.Input((self.output_dim,))
            t_pred = keras.layers.Input((self.output_dim,))

            def grad1(y_true, y_pred):
                return tf.gradients(custom_loss_tf(y_true, y_pred), [y_true, y_pred], stop_gradients=[y_true])

            def grad2(y_true, y_pred):
                return tf.gradients(grad1(y_true, y_pred), [y_true, y_pred], stop_gradients=[y_true])

            def custom_loss_func(y_true, y_pred):
                true, pred = self.to_tf(y_true, y_pred)

                pred[np.where(pred == 0)] = 0.000001

                pred[np.where(pred == 1)] = 1.0 - 0.000001

                s = tf.get_default_session()

                res_1 = self.eval_func(true, pred, [grad1(t_true, t_pred), t_true, t_pred], s, False)[1]
                res_2 = self.eval_func(true, pred, [grad2(t_true, t_pred), t_true, t_pred], s, False)[1]

                return self.loss_to_gb(res_1), self.loss_to_gb(res_2)

            self.custom_loss_callable = custom_loss_func

        for item in args[2]:
            self.custom_metrics[item] = self.to_tensor(keras.metrics.get(item))

    def eval_func(self, y_true, y_pred, f, session, mean=True):
        func = f[0]

        arg1 = f[1]
        arg2 = f[2]

        if mean:
            return np.mean(session.run(func, {arg1: y_true, arg2: y_pred}))

        return session.run(func, {arg1: y_true, arg2: y_pred})

    def eval_metrics(self, y_true, y_pred, session):
        result = {}

        for item in self.custom_metrics.keys():
            preds = y_pred

            if generic_config.need_threshold(item):
                preds = (preds > 0.5).astype(np.float32)

            result[item] = self.eval_func(y_true, preds, self.custom_metrics[item], session)

        return result

    def to_tensor(self, func):
        i1 = keras.layers.Input((self.output_dim,))
        i2 = keras.layers.Input((self.output_dim,))

        return func(i1, i2), i1, i2

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
        else:
            result_y = (result_y > 0.5).flatten()

        return result_x.astype(np.float32), result_y.astype(np.int32)

    def predict(self, *args, **kwargs):
        input = np.array(args)[0]

        input = np.reshape(input, (len(input), -1))

        self.model._n_features = input.shape[1]

        predictions = self.model.predict(input)

        if self.output_dim in [1, 2]:
            return self.groups_to_vectors(predictions, len(predictions))

        return predictions

    def load_weights(self, path, val):
        if os.path.exists(path):
            self.model._Booster = lightgbm.Booster(model_file=path)

    def numbers_to_vectors(self, numbers):
        result = np.zeros((len(numbers), self.output_dim))

        count = 0

        if self.output_dim == 1:
            for item in numbers:
                result[count, 0] = item

                count += 1

            return result

        for item in numbers:
            result[count, item] = 1

            count += 1

        return result

    def groups_to_vectors(self, data, length):
        result = np.zeros((length, self.output_dim))

        if self.output_dim == 1:
            result[:, 0] = data

            return result

        if self.output_dim == 2:
            ids = np.array(range(length), np.int32)

            ids = [ids, (data > 0.5).astype(np.int32)]

            result[ids] = 1

            return result

        for item in range(self.output_dim):
            result[:, item] = data[length * item : length * (item + 1)]

        return result

    def to_tf(self, numbers, data):
        y_true = self.numbers_to_vectors(numbers)

        y_pred = self.groups_to_vectors(data, len(numbers))

        return y_true, y_pred

    def save(self, file_path, overwrite):
        if hasattr(self.model, "booster_"):
            self.model.booster_.save_model(file_path)

    def fit_generator(self, *args, **kwargs):
        callbacks = kwargs["callbacks"]

        file_path = None
        early_stopping_rounds = None

        for item in callbacks:
            if hasattr(item, "filepath"):
                file_path = item.filepath

        generator_train = args[0]
        generator_test = kwargs["validation_data"]

        generator_test.batchSize = len(generator_test.indexes)

        train_x, train_y = self.convert_data(generator_train)
        val_x, val_y = self.convert_data(generator_test)

        self.model.n_estimators = kwargs["epochs"]

        checkpoint_cb = None

        for item in callbacks:
            item.set_model(self)
            item.on_train_begin()

            if "ModelCheckpoint" in str(item):
                checkpoint_cb = item

        def custom_metric(y_true, y_pred):
            true, pred = self.to_tf(y_true, y_pred)

            results = self.eval_metrics(true, pred, tf.get_default_session())

            for item in list(results.keys()):
                results["val_" + item] = results[item]

            self.rgetter.__dict__ = results

            return checkpoint_cb.monitor, np.mean(results[checkpoint_cb.monitor]), "great" in str(checkpoint_cb.monitor_op)

        def custom_callback(*args, **kwargs):
            iter = args[0][2]

            self.model._Booster = args[0][0]

            for item in callbacks:
                if "ReduceLROnPlateau" in str(item):
                    continue
                item.on_epoch_end(iter, self.rgetter)

        if self.custom_loss_callable:
            self.model.objective = self.custom_loss_callable
            self.model._objective = self.custom_loss_callable

        self.model.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks = [custom_callback], eval_metric = custom_metric)

        for item in callbacks:
            item.on_train_end()