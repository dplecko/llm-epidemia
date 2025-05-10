import sys
import os
import importlib
sys.path.append(os.path.abspath("workspace/tasks"))

def load_tasks(module_name):
    try:
        module = importlib.import_module(module_name)
        return module
    except ModuleNotFoundError:
        return []

task_specs = []
task_specs_hd = []

# task files without the "task_" prefix
task_files = [
    "acs", "labor", "fbi", "edu", 
    "nhanes", "gss", "meps", "scf", 
    "brfss", "nsduh"
]

for task_file in task_files:
    # Load the module
    module = load_tasks(f"tasks_{task_file}")
    
    # Extract standard tasks
    if module and hasattr(module, f"tasks_{task_file}"):
        task_specs.extend(getattr(module, f"tasks_{task_file}"))
    
    # Extract high-dimensional tasks
    if module and hasattr(module, f"tasks_{task_file}_hd"):
        task_specs_hd.extend(getattr(module, f"tasks_{task_file}_hd"))
