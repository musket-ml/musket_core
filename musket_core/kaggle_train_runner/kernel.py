import os

import yaml

from musket_core.kaggle_train_runner import utils

import socketserver

KERNEL_STATUS_RUNNING = "running"
KERNEL_STATUS_ERROR = "error"
KERNEL_STATUS_CANCELED = "cancel"
KERNEL_STATUS_COMPLETE = "complete"
KERNEL_STATUS_UNKNOWN = "404"
KERNEL_STATUS_NOINTERNET = "nointernet"

class Project:
    def __init__(self, root):
        self.root = root

        self.meta = None

        self.kernels = []

        self.folds = 0

        self.meta = utils.read_project_meta(self.root)

        self.collect_experiment()

        self.load()

        self.server: socketserver.TCPServer = None

        self.folds = None

    def metadataFolder(self):
        return os.path.join(self.root, ".metadata")

    def collect_experiment(self):
        utils.remove(os.path.join(self.metadataFolder(), "kaggle", "project"))

        utils.ensure(os.path.join(self.metadataFolder(), "kaggle"))
        utils.ensure(os.path.join(self.metadataFolder(), "kaggle", "project"))
        utils.ensure(os.path.join(self.metadataFolder(), "kaggle", "project", "experiments"))

        def filter(item):
            if not os.path.isdir(os.path.join(self.root, item)):
                return False

            if item == "experiments":
                return False

            if item == "kaggle":
                return False

            if item == "data":
                return False

            if item == "Data":
                return False

            if item == ".metadata":
                return False

            return True

        directories = [item for item in os.listdir(self.root) if filter(item)]

        for item in directories:
            utils.copy(os.path.join(self.root, item), os.path.join(self.metadataFolder(), "kaggle", "project", item))

        utils.copy(os.path.join(self.root, "experiments", self.meta["experiment"]), os.path.join(self.metadataFolder(), "kaggle", "project", "experiments", "experiment"))

        data_name_path = os.path.join(os.path.join(self.metadataFolder(), "kaggle", "project"), "dataset_id.txt")

        if not os.path.exists(data_name_path):
            with open(data_name_path, "w+") as f:
                dataset_id = self.meta["dataset_sources"][0] if len(self.meta["dataset_sources"]) > 0 else self.meta["competition_sources"][0]

                f.write(dataset_id)

    def experiment_folder(self):
        return os.path.join(self.root, "experiments", self.meta["experiment"])

    def load(self):
        utils.ensure(os.path.join(self.metadataFolder(), "kaggle"))

        with open(os.path.join(self.root, "experiments", self.meta["experiment"], "config.yaml")) as cfg:
            self.folds = yaml.load(cfg).pop("folds_count", None)

        if self.meta["split_by_folds"]:
            for item in range(self.folds):
                self.kernels.append(Kernel(item, self, item))

            return

        for item in range(self.meta["kernels"]):
            self.kernels.append(Kernel(item, self))

    def kernel(self, id):
        for item in self.kernels:
            if item.id == id:
                return item

class Kernel:
    def __init__(self, id, project, fold=-1):
        self.id = id

        self.project = project

        self.meta = None

        self.fold=fold

        self.load()

        self.run_count = 0

    def load(self):
        self.meta = utils.kernel_meta(self.get_path(), self.id, self.project.meta["username"], self.project.meta["server"], self.get_title(), self.project.meta["dataset_sources"], self.project.meta["competition_sources"], self.project.meta["kernel_sources"], self.fold, self.project.meta["time"])

    def get_path(self):
        return os.path.join(self.project.metadataFolder(), "kaggle", "kernels", "kernel_" + str(self.id))

    def downloaded_experiment(self):
        return os.path.join(self.get_path(), "project", "experiments", "experiment")

    def get_title(self):
        return self.project.meta["project_id"] + "-" + str(self.id)

    def get_status(self, after_run=False):
        return self.parse_status(utils.get_status(self.meta["id"]))

    def parse_status(self, status_text):
        if "404 - Not Found" in status_text:
            return KERNEL_STATUS_UNKNOWN

        if "Failed to establish a new connection" in status_text:
            return KERNEL_STATUS_NOINTERNET

        if 'has status "error"' in status_text:
            return KERNEL_STATUS_ERROR

        if 'has status "running"' in status_text:
            return KERNEL_STATUS_RUNNING

        if 'has status "complete"' in status_text:
            return KERNEL_STATUS_COMPLETE

        if 'has status "cancelAcknowledged"' in status_text:
            return KERNEL_STATUS_CANCELED

        return status_text

    def archive(self, initial=False):
        generated_project_path = os.path.join(self.get_path(), "project")

        if initial:
            utils.archive(os.path.join(self.project.metadataFolder(), "kaggle", "project"), generated_project_path)

    def log(self, bytes):
        utils.log(os.path.join(self.get_path(), "kernel.log"), bytes)

    def is_complete(self):
        if self.run_count > 2:
            return True

        return utils.is_complete(self.get_path())

    def push(self):
        response = utils.run_kernel(self.get_path())

        return response

    def assemble(self):
        original_experiment = self.project.experiment_folder()

        original_metrics = os.path.join(original_experiment, "metrics")
        original_weights = os.path.join(original_experiment, "weights")

        downloaded_experiment = self.downloaded_experiment()

        downloaded_metrics = os.path.join(downloaded_experiment, "metrics")
        downloaded_weights = os.path.join(downloaded_experiment, "weights")

        if not os.path.exists(downloaded_metrics):
            return

        if not os.path.exists(downloaded_weights):
            return

        metrics = utils.listdir(downloaded_metrics)
        weights = utils.listdir(downloaded_weights)

        utils.ensure(original_metrics)
        utils.ensure(original_weights)

        for item in metrics:
            src = os.path.join(downloaded_metrics, item)
            dst = os.path.join(original_metrics, item)

            if os.path.exists(dst):
                continue

            utils.copy_file(src, dst)

        for item in weights:
            src = os.path.join(downloaded_weights, item)
            dst = os.path.join(original_weights, item)

            if os.path.exists(dst):
                continue

            utils.copy_file(src, dst)

        for item in utils.listdir(downloaded_experiment):
            src = os.path.join(downloaded_experiment, item)
            dst = os.path.join(original_experiment, item)

            if os.path.exists(dst):
                continue

            utils.copy_file(src, dst)

    def download(self):
        project_path = os.path.join(self.get_path(), "project")

        utils.remove(project_path)

        utils.download(self.meta["id"], self.get_path())

        self.run_count += 1










