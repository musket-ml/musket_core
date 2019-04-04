from musket_core import tools
from musket_core import projects
t=tools.Introspect("C:/work/runtime-EclipseApplication/c")
r=t.perform(projects.Workspace(),tools.ProgressMonitor())
print(r)