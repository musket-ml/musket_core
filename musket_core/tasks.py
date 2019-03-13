import os

import inspect

import runpy as rp

from typing import Dict, List

class TaskRunner:
    def __init__(self, task, task_root, id_set = [], parameters = {}):
        self.task = task

        self.root = task_root

        self.id_set = id_set

        self.parameters = parameters

        self.args = self.create_args(inspect.getfullargspec(task))

    def run(self, data):
        self.task(data, **self.args)

    def end(self):
        for handler in self.args.values():
            if hasattr(handler, "end"):
                handler.__getattribute__("end")()

    def create_args(self, spec):
        handlers = {}

        for arg in spec.annotations:
            if arg == "parameters":
                handlers[arg] = self.parameters

                continue

            if arg == "data":
                continue

            handler_type = spec.annotations[arg]

            handlers[arg] = handler_type(self.root, self.id_set, self.parameters)

        return handlers

def load_task_sets(path, names):
    functions = {}

    for item in names:
        task_set = os.path.join(path, item + ".py")

        module = rp.run_path(task_set)

        for function in [(name, module[name]) for name in module.keys() if hasattr(module[name], '__call__')]:
            functions[function[0]] = function[1]

    return functions

def create_task_runner(task_name, task_set_id, parameters, root, tasks_map: Dict):
    id_set = [task_name, str(task_set_id)]

    callback = tasks_map[task_name]

    return TaskRunner(callback, root, id_set, parameters)

def eval_task_for_item(item, task_name, tasks_runners: Dict[str, TaskRunner]):
    tasks_runners[task_name].run(item)
