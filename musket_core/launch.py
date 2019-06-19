import sys
from musket_core import tools,projects
import yaml
import threading
from musket_core import utils
config = sys.argv[1]
out = sys.argv[2]
def main():
        global config
        with open(config,"r") as f:
            config=f.read()
        config = config[1:].replace("!!com.onpositive", "!com.onpositive")
        obj = yaml.load(config)
        print(obj)
        results = obj.perform(projects.Workspace(), tools.ProgressMonitor())
        with open(out,"w") as f:
            f.write(yaml.dump(results))

if __name__ == '__main__':            
    main()