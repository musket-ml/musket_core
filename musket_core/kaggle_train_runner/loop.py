import time

from musket_core.kaggle_train_runner.kernel import Project, Kernel, KERNEL_STATUS_UNKNOWN, KERNEL_STATUS_CANCELED, KERNEL_STATUS_COMPLETE, KERNEL_STATUS_ERROR, KERNEL_STATUS_NOINTERNET, KERNEL_STATUS_RUNNING

from musket_core.kaggle_train_runner import connection

import threading

from async_promises import Promise

class Task:
    def __init__(self, task, sleep=None, on_complete=None):
        self.task = task
        self.on_complete = on_complete
        self.time_to_run = time.time() + sleep

    def run(self):
        def rejection(cause):
            print(cause)

        Promise(lambda resolve, reject: resolve(self.task() or True)).then(lambda success: self.on_complete() or True if self.on_complete else True, rejection)

def kernel_status_request_task(kernel: Kernel, on_complete, wait=300, after_run=False):
    status = [KERNEL_STATUS_NOINTERNET]

    def complete():
        on_complete(kernel, status[0])

        print("kernel_status_request_task complete")

    def task():
        while True:
            status[0] = kernel.get_status(after_run)

            if status[0] != KERNEL_STATUS_NOINTERNET:
                break

            else:
                print(status[0])

            time.sleep(1)

    print("shedule kernel_status_request_task...")

    return Task(task, wait, complete)

def kernel_run_request_task(kernel: Kernel, on_complete, wait=300):
    def task():
        result = kernel.push()

        if len(result) > 0:
            print(result);

            if("Maximum batch GPU session count") in result:
                print("retry will be started after " + str(wait) + " seconds")

                time.sleep(wait)

                task()

    print("shedule kernel_run_request_task...")

    def complete(k):
        on_complete(k)

        print("kernel_run_request_task complete")

    return Task(task, wait, lambda: complete(kernel))

class MainLoop:
    def __init__(self, project: Project):
        self.project = project

        self.queue = []

        self.running = 0

        self.wait = self.project.meta["requests_delay"]

        self.kernels_queue = None

    def add_task(self, task):
        self.queue.insert(0, task)

    def run_server(self):
        def do_run_server():
            connection.run_server(self.project)

        threading.Thread(target=do_run_server).start()

    def shutdown(self):
        self.project.server.shutdown()
        self.project.server.server_close()

        for item in self.project.kernels:
            item.assemble()

    def start(self):
        self.run_server()

        self.kernels_queue = list(self.project.kernels)

        self.add_kernels()

        while True:
            if len(self.queue) > 0:
                task = self.queue.pop()

                if time.time() > task.time_to_run:
                    task.run()
                else:
                    self.add_task(task)

            elif self.running == 0:
                self.shutdown()

                break

            time.sleep(1)

    def add_kernels(self):
        while len(self.kernels_queue) and self.running < self.project.meta["kernels"]:
            self.running += 1

            self.add_task(kernel_status_request_task(self.kernels_queue.pop(), self.on_kernel_status, 10));

    def run_kernel(self, kernel, wait_after_run, wait_after_status, is_initial):
        kernel.archive(is_initial)

        self.add_task(kernel_run_request_task(kernel, lambda k: self.add_task(kernel_status_request_task(k, self.on_kernel_status, wait_after_status)), wait_after_run))

    def on_kernel_status(self, kernel, status):
        print("status: " + status)

        if status == KERNEL_STATUS_UNKNOWN:
            self.run_kernel(kernel, 10, 20, True)

            return

        if status == KERNEL_STATUS_COMPLETE or status == KERNEL_STATUS_ERROR:
            kernel.download()

            if kernel.is_complete():
                self.running -= 1

                self.add_kernels()
            else:
                self.run_kernel(kernel, self.wait, self.wait, False)
            return

        if status == KERNEL_STATUS_RUNNING:
            self.add_task(kernel_status_request_task(kernel, self.on_kernel_status))

            return




