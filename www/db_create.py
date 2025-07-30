
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from workspace.common import *

models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", 
          "gemma3_27b_instruct", "deepseek_7b_chat"]

dfs = []
for prob in [False, True]:
    if prob:
        models = models + ["gpt-4.1"]

    df_ld, _ = build_eval_df(models, task_specs, prob=prob)
    df_hd, _ = build_eval_df(models, task_specs_hd, prob=prob)
    df_hd["task_name"] = df_hd["task_id"].apply(lambda x: hd_taskname(task_specs_hd[x]))
    dfs.append(df_ld)
    dfs.append(df_hd)

df_all = pd.concat(dfs, axis=0, ignore_index=True)
df_all["prob"] = ["Likelihood" if prob else "Q&A" for prob in df_all["prob"]]

df_all["task_name"] = df_all["task_name"].astype(str)
df_all["dataset"] = df_all["dataset"].map(dts_map)
df_all["model_display"] = model_name(df_all["model"])
df_all.to_json("www/viz_data.json", orient="records", lines=False)

