from workspace.common import *

models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", 
          "gemma3_27b_instruct", "deepseek_7b_chat"]

df_ld, _ = build_eval_df(models, task_specs)

for prob in [True, False]:
    df_hd, _ = build_eval_df(models, task_specs_hd, prob=prob)
    df_hd["task_name"] = df_hd["task_id"].apply(lambda x: hd_taskname(task_specs_hd[x]))
    df_ld = pd.concat([df_ld, df_hd], axis=0, ignore_index=True)

df_ld["prob"] = ["Likelihood" if prob else "Q&A" for prob in df_ld["prob"]]


df_ld["task_name"] = df_ld["task_name"].astype(str)
df_ld["dataset"] = df_ld["dataset"].map(dts_map)
df_ld["model_display"] = model_name(df_ld["model"])
df_ld.to_json("www/viz_data.json", orient="records", lines=False)

df_ld["dataset"].unique()

q_num = []
for i in range(len(task_specs_hd)):
    q_num.append(hd_tasksize(task_specs_hd[i])[0])
