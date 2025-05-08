
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
from task_spec import task_specs

eval_df = build_eval_df(["llama3_8b_instruct"], task_specs[0:1])