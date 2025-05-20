
from workspace.common import *

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
