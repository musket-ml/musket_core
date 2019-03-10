import os

import runpy as rp

import inspect

import imageio

from typing import Dict, List

from musket_core.utils import ensure

class TaskRunner:
    def __init__(self, task, root, parameters = {}):
        self.task = task

        self.root = root

        self.parameters = parameters

        self.args = self.create_args(inspect.getfullargspec(task))

    def run(self, data):
        args = self.compose_args(data)

        self.task(*args)

    def compose_args(self, data):
        result = [data]

        for key in self.args.keys():
            result.append(self.args[key])

        return result

    def create_args(self, spec):
        handlers = {}

        for arg in spec.annotations:
            if arg == "parameters":
                handlers[arg] = self.parameters

                continue

            if arg == "data":
                continue

            handler_type = spec.annotations[arg]

            handlers[arg] = handler_type(self.root, self.parameters)

        return handlers

class ImageWriter:
    def __init__(self, root, parameters = {}):
        self.root = root

        self.parameters = parameters

        ensure(self.root)

    def write(self, id, image):
        imageio.imsave(os.path.join(self.root, id + ".png"), image)

def load_task_sets(path, names):
    functions = {}

    for item in names:
        task_set = os.path.join(path, item + ".py")

        module = rp.run_path(task_set)

        for function in [(name, module[name]) for name in module.keys() if hasattr(module[name], '__call__')]:
            functions[function[0]] = function[1]

    return functions

def create_task_runner(task_name, task_set_id, parameters, root, tasks_map: Dict):
    runner_root = os.path.join(root, task_name + "_" + str(task_set_id))

    callback = tasks_map[task_name]

    return TaskRunner(callback, runner_root, parameters)

def eval_task_for_item(item, task_name, tasks_runners: Dict[str, TaskRunner]):
    tasks_runners[task_name].run(item)
