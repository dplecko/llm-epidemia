
import sys, os
sys.path.append(os.path.join(os.getcwd(), "workspace"))
import pandas as pd
import numpy as np
import re, json
from plotnine import *
from tqdm import tqdm
import pdbpp

# our own code
from bench_eval import build_eval_df
from task_spec import task_specs, task_specs_hd
from helpers import model_name, dts_map, task_to_filename, hd_tasksize, hd_taskname, name_and_sort, model_colors
from hd_helpers import fit_lgbm, promptify, gen_prob_lvls, decode_prob_lvl
from model_load import load_model, MODEL_PATHS
from evaluator_helpers import extract_pv, compress_vals, extract_pv_batch