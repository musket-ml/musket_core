import os
import http.server
import socketserver

from musket_core.kaggle_train_runner.kernel import Project

class CustomHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/zip')
        self.end_headers()

        if "kernel" in self.path:
            kernel = self.server.project.kernel(self.parse_kernel())

            zip_path = os.path.join(kernel.get_path(), "project.zip")

            with open(zip_path, "rb") as f:
                print("requested: " + zip_path)

                self.wfile.write(f.read())

            os.remove(zip_path)

    def do_POST(self):
        print("POST: " + self.path)

        if "kernel" in self.path:
            kernel = self.server.project.kernel(self.parse_kernel())

            kernel.log(self.rfile.read(int(self.headers['Content-Length'])))

    def parse_kernel(self):
        return int(self.path.split("/").pop())

def run_server(project: Project):
    with socketserver.TCPServer(("", project.meta["port"]), CustomHandler) as httpd:
        httpd.project = project

        project.server = httpd

        httpd.serve_forever()
