from workspace.common import *

def safe_id(x):
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(x))

regen_plots = False
tasks = task_specs
models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4",
          "gemma3_27b_instruct", "deepseek_7b_chat"]

df, eval_map = build_eval_df(models, tasks)
df = df[["task_id", "task_name", "model", "score", "dataset"]]
df_wide = df.pivot(index=["task_id", "task_name", "dataset"], columns="model", values="score").reset_index()
model_cols = [col for col in df_wide.columns if col not in {"task_id", "task_name", "dataset"}]

if regen_plots:
    os.makedirs("www/data", exist_ok=True)
    for i in tqdm(range(df.shape[0])):
        task_idx = df.iloc[i]["task_id"]
        model = df.iloc[i]["model"]
        task = task_specs[task_idx]
        eval = eval_map[i]

        if hasattr(eval, "attrs") and "distr" in eval.attrs:
            distr = eval.attrs["distr"].copy()
            distr["prop"] = distr["prop"].astype(float)
            distr = distr[["lvl_names", "prop", "type", "cond"]]
            
            # Read the first variable for 'lvl_names' ordering
            ord_ser = pd.read_parquet(task_specs[task_idx]["dataset"])[task_specs[task_idx]["variables"][0]]
            if isinstance(ord_ser.dtype, pd.CategoricalDtype):
                # Extract categories from the reference Series
                lvl_order = ord_ser.cat.categories
                # Apply ordering to 'lvl_names' in distr
                distr["lvl_names"] = pd.Categorical(
                    distr["lvl_names"], 
                    categories=lvl_order, 
                    ordered=True
                )

            # Read the second variable for 'cond' ordering
            ord_ser = pd.read_parquet(task_specs[task_idx]["dataset"])[task_specs[task_idx]["variables"][1]]
            if isinstance(ord_ser.dtype, pd.CategoricalDtype):
                # Extract categories from the reference Series
                cond_order = ord_ser.cat.categories
                # Apply ordering to 'cond' in distr
                distr["cond"] = pd.Categorical(
                    distr["cond"], 
                    categories=cond_order, 
                    ordered=True
                )

            distr = distr.sort_values(["cond", "lvl_names"]).reset_index(drop=True)
            prompt_tpl = task_specs[task_idx].get("prompt", f"What is the distribution of {task['name']}?")
            distr["prompt"] = distr["cond"].apply(lambda c: prompt_tpl.format(c))

            fname = f"www/data/l1_task{task_idx}_{model}.json"
            with open(fname, "w") as f:
                json.dump(distr.to_dict(orient="records"), f, indent=2)


# Add dataset column
# df_wide["dataset"] = df_wide["task_name"].str.extract(r'^(.*?):')

# ---- Build HTML ----
html = open("www/template/head.html").read()
html += "<div class='table-responsive'>\n<table class='table table-dark table-bordered text-center align-middle'>\n"
html += """<thead class='table-light text-dark'><tr>
<th>
  Task<br>
  <button class='btn btn-sm btn-outline-light' onclick="restoreOrder()">Reset</button>
</th>"""
for model in model_cols:
    html += f"""<th>{model_name(model)}<br>
      <button class='btn btn-sm btn-outline-light' onclick="sortTable('{model}', true)">↑</button>
      <button class='btn btn-sm btn-outline-light' onclick="sortTable('{model}', false)">↓</button>
    </th>"""
html += "</tr></thead>\n<tbody>\n"

for _, row in df_wide.iterrows():
    task_id = row["task_id"]
    task_name = row["task_name"]
    safe_tid = safe_id(task_id)

    html += f"<tr data-taskid='{safe_tid}'><td>{task_name}</td>"

    for model in model_cols:
        val = row[model]
        val_str = "NA" if pd.isna(val) else f"{int(val)}"
        cell_id = f"trigger-{safe_tid}-{model}"
        html += (
            f"<td id='{cell_id}' class='score-cell' title='Click to view plot' onclick=\"togglePlot('{model}', '{safe_tid}')\">"
            f"{val_str}</td>"
        )

    html += "</tr>\n"

    for model in model_cols:
        plot_id = f"plot-{safe_tid}-{model}"
        plot_cell_id = f"plot-cell-{safe_tid}-{model}"
        select_id = f"facet-select-{safe_tid}-{model}"
        plot_div_id = f"vl-plot-{safe_tid}-{model}"

        idx = model_cols.index(model)
        prev_model = model_cols[idx - 1] if idx > 0 else None
        next_model = model_cols[idx + 1] if idx + 1 < len(model_cols) else None

        nav_buttons = ""
        if prev_model:
            nav_buttons += f"""<button class="btn btn-sm btn-outline-light" onclick="navigateModel('{task_id}', '{prev_model}', 0)">&lt; {model_name(prev_model)}</button>"""
        if next_model:
            nav_buttons += f"""<button class="btn btn-sm btn-outline-light" onclick="navigateModel('{task_id}', '{next_model}', 0)">{model_name(next_model)} &gt;</button>"""

        html += f"""<tr id='{plot_id}' style='display:none;'>
        <td colspan='{len(model_cols)+1}' id='{plot_cell_id}'>
        <div class="d-flex align-items-center justify-content-between mb-2">
            <div class="d-flex align-items-center">
            <label class="form-label text-white me-2 mb-0" for="{select_id}">Select Conditioning Value</label>
            <select class="form-select form-select-sm bg-dark text-light border-secondary"
                    id="{select_id}"
                    onchange="updateFacetPlot('{task_id}', '{model}', '{safe_tid}')">
            </select>
            </div>
            <div class="d-flex gap-2">
            {nav_buttons}
            </div>
        </div>
        <div class="mb-2 text-light"><strong>Question:</strong> <em id="question-text-{safe_tid}-{model}"></em></div>
        <div id="{plot_div_id}"></div>
        </td>
        </tr>\n"""

html += "</tbody></table>\n</div>\n</section>\n"
html += open("www/template/script.js").read()
html += """
<div id="footer"></div>
<script>
  fetch("footer.html")
    .then(res => res.text())
    .then(html => document.getElementById("footer").innerHTML = html);
</script>
<script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.8/dist/umd/popper.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.min.js"></script>
"""
html += "\n</body>\n</html>"

with open("www/interactive_table.html", "w") as f:
    f.write(html)
