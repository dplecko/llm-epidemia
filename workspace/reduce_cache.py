
# compress columns

from workspace.common import *

info_vars = ['weight', 'lgbm_pred', 'llm_pred']
cnt = 0
size = 0

for prob in [False, True]:
    for model_name in MODEL_PATHS.keys():
        for i in tqdm(range(len(task_specs_hd))):
            fl = task_to_filename(model_name, task_specs_hd[i])
            if prob:
                fl = "PROB_" + fl
            path = os.path.join("data", "benchmark", fl)
            if os.path.exists(path):
                cnt = cnt+1
                size = size + os.path.getsize(path) / (1024 * 1024)
                out_var = [task_specs_hd[i]['v_out']]
                cond_vars = task_specs_hd[i]['v_cond']
                data = pd.read_parquet(path)
                ncol = data.shape[1]
                data = data[out_var + cond_vars + info_vars]
                ncol_red = data.shape[1]
                print("Reduced", ncol-ncol_red, " out of ", ncol, " columns.\n")
                data.to_parquet(path) 
            
            
            

