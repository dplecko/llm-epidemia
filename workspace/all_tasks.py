
from workspace.common import *

# low-dimensional
res = []
for i in range(len(task_specs)):
    res.append({
        "task": i+1,
        "task_name": task_specs[i]['name']
    })

print(pd.DataFrame(res).to_latex(index=False, escape = False))

# high-dimensional
gpt_idx = [0,  1,  2,  3,  4,  6,  7, 13, 14, 15, 16, 17, 18, 19, 20, 26, 27, 28, 29, 31, 32, 33, 39, 40, 42,
  44, 45, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 68, 69, 70, 72, 73, 74, 75, 76,
  77, 78, 80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91]
res_hd = []
for i in range(len(task_specs_hd)):
    res_hd.append({
        "task": i+1,
        "task_name": hd_taskname(task_specs_hd[i]),
        "dimension": len(task_specs_hd[i]['v_cond']),
        "inc": "\\ding{52}" if i in gpt_idx else "\\ding{56}"
    })

print(pd.DataFrame(res_hd).to_latex(index=False, escape = False))
