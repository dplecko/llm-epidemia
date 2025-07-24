
import sys, os
sys.path.append(os.path.join(os.getcwd(), "workspace"))
sys.path.append(os.path.join(os.getcwd(), "workspace/utils"))
import pandas as pd
import numpy as np
import re, json
from plotnine import *
from tqdm import tqdm
import pdbpp

# our own code
from eval import build_eval_df
from task_spec import task_specs, task_specs_hd
from helpers import model_name, dts_map, task_to_filename, model_colors, hd_taskname
from plot_helpers import name_and_sort
from hd_helpers import fit_lgbm, promptify, gen_prob_lvls, decode_prob_lvl, hd_tasksize, decode_prob_matrix
from model_load import load_model, MODEL_PATHS
from extract_helpers import extract_pv, compress_vals, extract_pv_batch
from qual_inspect import make_detail_plotnine
