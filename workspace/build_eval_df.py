
import sys
import os
sys.path.append(os.path.abspath("workspace"))
import pandas as pd
import pdb
import traceback
from bench_eval import eval_task, eval_to_score

def build_eval_df(models, tasks):
    rows = []
    for model in models:
        for i, task in enumerate(tasks):
            try:
                df_eval = eval_task(model, task)
                score = eval_to_score(df_eval)
            except Exception as e:
                print(f"[ERROR] model={model}, task={task.get('name', i)}")
                traceback.print_exc()
                df_eval, score = None, None
            
            rows.append({
                "model": model,
                "task_id": i,
                "score": score,
                "eval": df_eval,
            })
    return pd.DataFrame(rows)
