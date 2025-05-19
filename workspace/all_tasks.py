
import sys, os
sys.path.append(os.path.join(os.getcwd(), "workspace"))
import pandas as pd
from task_spec import task_specs, task_specs_hd
from helpers import hd_taskname
# low-dimensional

res = []
for i in range(len(task_specs)):
    res.append({
        "task": i+1,
        "task_name": task_specs[i]['name'],
        "setting": "low"
    })

res = pd.DataFrame(res)
res.to_latex(index=False, escape = False)

res_hd = []
for i in range(len(task_specs_hd)):
    res_hd.append({
        "task": i+1,
        "task_name": hd_taskname(task_specs_hd[i]),
        "setting": "high"
    })

print(pd.DataFrame(res_hd).to_latex(index=False, escape = False))
