
import importlib
import pandas as pd
import sys, os
import numpy as np
import json
import pdb
from tqdm import tqdm
sys.path.append(os.path.join(os.getcwd(), "workspace"))
from helpers import model_name, model_unname
from build_eval_df import build_eval_df
from bench_eval import eval_task, eval_to_score
from task_spec import task_specs

eval_df, eval_map = build_eval_df(["llama3_8b_instruct"], task_specs[0:12])

# for i in range(len(eval_df)):
#     eval_df.loc[i, "task_name"] = task_specs[eval_df.loc[i, "task_id"]].get("name", eval_df.loc[i, "task_id"])

eval_task("llama3_8b_instruct", task_specs[23])