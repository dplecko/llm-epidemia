
from workspace.common import *

tasks = task_specs
models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", 
          "gemma3_27b_instruct", "deepseek_7b_chat"]

df, eval_map = build_eval_df(models, tasks)
df = df[["task_id", "task_name", "model", "score"]]
df_wide = df.pivot(index=["task_id", "task_name"], columns="model", values="score").reset_index()
model_cols = [col for col in df_wide.columns if col not in {"task_id", "task_name"}]

for i in range(df.shape[0]):
    task_idx = df.iloc[i]["task_id"]
    model = df.iloc[i]["model"]
    p = make_detail_plotnine(eval_map[i], task_specs[task_idx])
    p.save(f"www/img/l1_task{task_idx}_{model}.png", format='png', dpi=300, width=7, height=4, units="in", verbose=False)

# ---- Begin HTML ----
html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Qualitative Model Knowledge Inspection</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="style.css">
  <style>
    body {
      background-color: #0d1b2a;
      color: #f8f9fa;
    }

    table.table-dark {
      background-color: #0d1b2a;
    }

    .table-dark th,
    .table-dark td {
      background-color: #0d1b2a;
      color: #f8f9fa;
    }

    .table-dark thead th {
      background-color: #1a2638;
      color: #ffffff;
    }

    .plot-img {
      max-width: 100%;
      height: auto;
      margin-top: 10px;
    }

    .active-cell {
      background-color: #003366 !important; /* Deep blue */
      font-weight: bold;
    }
  </style>
</head>
<body class="text-light">
  <div id="navbar"></div>
  <script>
    fetch("navbar.html")
      .then(res => res.text())
      .then(data => document.getElementById("navbar").innerHTML = data);
  </script>

  <section class="container my-5">
    <h1 class="mb-4 text-center">Qualitative Model Knowledge Inspection</h1>
    <div class="table-responsive">
"""

# ---- Table generation ----
table = "<table class='table table-dark table-bordered text-center align-middle'>\n"
table += "<thead class='table-light text-dark'><tr><th>Task</th>"
for model in model_cols:
    table += f"<th>{model_name(model)}</th>"
table += "</tr></thead>\n<tbody>\n"

for _, row in df_wide.iterrows():
    task_id = row["task_id"]
    task_name = row["task_name"]
    table += f"<tr><td>{task_name}</td>"
    for model in model_cols:
        val = row[model]
        val_str = "NA" if pd.isna(val) else f"{int(val)}"
        cell_id = f"trigger-{task_id}-{model}"
        table += (
            f"<td id='{cell_id}' class='score-cell' onclick=\"togglePlot('{model}', {task_id})\">"
            f"{val_str}</td>"
        )
    table += "</tr>\n"
    for model in model_cols:
        plot_id = f"plot-{task_id}-{model}"
        plot_cell_id = f"plot-cell-{task_id}-{model}"
        table += f"<tr id='{plot_id}' style='display:none;'><td colspan='{len(model_cols)+1}' id='{plot_cell_id}'></td></tr>\n"


table += "</tbody></table>\n"

# ---- JS logic ----
script = """
<script>
let currentOpen = null;
let currentCell = null;

function togglePlot(model, task_id) {
  const rowId = `plot-${task_id}-${model}`;
  const cellId = `plot-cell-${task_id}-${model}`;
  const triggerId = `trigger-${task_id}-${model}`;
  
  const row = document.getElementById(rowId);
  const cell = document.getElementById(cellId);
  const triggerCell = document.getElementById(triggerId);
  const imgPath = `img/l1_task${task_id}_${model}.png`;

  // Close previous open plot
  if (currentOpen && currentOpen !== row) {
    currentOpen.style.display = "none";
    currentOpen.querySelector("td").innerHTML = "";
    if (currentCell) currentCell.classList.remove("active-cell");
  }

  const isSame = currentOpen === row;

  if (isSame && row.style.display !== "none") {
    row.style.display = "none";
    cell.innerHTML = "";
    triggerCell.classList.remove("active-cell");
    currentOpen = null;
    currentCell = null;
  } else {
    cell.innerHTML = `<img src="${imgPath}" class="plot-img">`;
    row.style.display = "table-row";
    triggerCell.classList.add("active-cell");
    currentOpen = row;
    currentCell = triggerCell;
  }
}
</script>
"""

# ---- Wrap up ----
html += table + "</div>\n</section>\n" + script + "\n</body>\n</html>"

# ---- Write to file ----
with open("www/interactive_table.html", "w") as f:
    f.write(html)

