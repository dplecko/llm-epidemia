<script>
function safe_id(x) {
  return String(x).replace(/[^a-zA-Z0-9_]/g, "_");
}

let currentOpen = null;
let currentCell = null;

function togglePlot(model, task_id) {
  console.log("togglePlot called", { model, task_id });
  const safeTid = safe_id(task_id);
  const rowId = `plot-${safeTid}-${model}`;
  const plotDivId = `vl-plot-${safeTid}-${model}`;
  const selectId = `facet-select-${safeTid}-${model}`;
  const questionId = `question-text-${safeTid}-${model}`;
  const dataPath = `data/l1_task${task_id}_${model}.json`;

  const row = document.getElementById(rowId);
  const plotDiv = document.getElementById(plotDivId);
  const select = document.getElementById(selectId);
  const questionElem = document.getElementById(questionId);

  if (currentOpen && currentOpen !== row) {
    currentOpen.style.display = "none";
    if (currentCell) currentCell.classList.remove("active-cell");
  }

  const isSame = currentOpen === row;

  if (isSame && row.style.display !== "none") {
    row.style.display = "none";
    plotDiv.innerHTML = "";
    select.innerHTML = "";
    if (questionElem) questionElem.textContent = "";
    document.getElementById(`trigger-${safeTid}-${model}`).classList.remove("active-cell");
    currentOpen = null;
    currentCell = null;
  } else {
    fetch(dataPath)
      .then(res => res.json())
      .then(data => {
        console.log("Loaded JSON data:", data);
        const facets = [...new Set(data.map(d => d.cond))];
        select.innerHTML = facets.map(f => `<option value="${f}">${f}</option>`).join("");
        select.dataset.json = JSON.stringify(data);
        const defaultFacet = facets[0];
        const record = data.find(d => d.cond === defaultFacet);
        if (record && record.prompt && questionElem) {
          questionElem.textContent = record.prompt.replace("{}", defaultFacet);
        }
        updateFacetPlot(task_id, model, safeTid);
      });

    row.style.display = "table-row";
    document.getElementById(`trigger-${safeTid}-${model}`).classList.add("active-cell");
    currentOpen = row;
    currentCell = document.getElementById(`trigger-${safeTid}-${model}`);
  }
}

function updateFacetPlot(task_id, model, safe_tid) {
  const select = document.getElementById(`facet-select-${safe_tid}-${model}`);
  const selectedFacet = select.value;
  const data = JSON.parse(select.dataset.json);
  const filtered = data.filter(d => String(d.cond).trim().toLowerCase() === selectedFacet.toLowerCase());

  // Update the question text
  const qEl = document.getElementById(`question-text-${safe_tid}-${model}`);
  const selectedFacetObj = data.find(d => String(d.cond).trim().toLowerCase() === selectedFacet.toLowerCase());
  qEl.textContent = selectedFacetObj?.prompt || "(no question)";

  const spec = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "background": "#0d1b2a",
    "data": { "values": filtered },
    "mark": {
      "type": "bar",
      "tooltip": true
    },
    "encoding": {
      "x": {
        "field": "lvl_names",
        "type": "nominal",
        "axis": {
          "labelAngle": 45,
          "title": "Answers",
          "labelColor": "#f8f9fa",
          "titleColor": "#f8f9fa"
        }
      },
      "y": {
        "field": "prop",
        "type": "quantitative",
        "stack": null,
        "axis": {
          "title": "Proportion",
          "labelColor": "#f8f9fa",
          "titleColor": "#f8f9fa"
        }
      },
      "xOffset": { "field": "type" },
      "color": {
        "field": "type",
        "type": "nominal",
        "scale": {
          "domain": ["Reality", "Model"],
          "range": ["#FFA500", "#1f77b4"]
        },
        "legend": {
          "labelColor": "#f8f9fa",
          "titleColor": "#f8f9fa"
        }
      },
      "tooltip": [
        { "field": "lvl_names", "title": "Answer" },
        { "field": "type", "title": "Source" },
        { "field": "prop", "title": "Proportion", "format": ".1%" }
      ]
    },
    "width": 500,
    "height": 300,
    "config": {
      "axis": { "labelFontSize": 13, "titleFontSize": 15 },
      "legend": { "labelFontSize": 13, "titleFontSize": 15 },
      "style": { "cell": { "stroke": "transparent" } }
    }
  };

  vegaEmbed(`#vl-plot-${safe_tid}-${model}`, spec, { actions: false });
}

function sortTable(model, ascending = true) {
  const table = document.querySelector("table tbody");
  const allRows = Array.from(table.querySelectorAll("tr"));
  const visibleRows = [];

  for (let i = 0; i < allRows.length; i++) {
    const main = allRows[i];
    if (!main.querySelector(".score-cell")) continue;
    const safeTaskId = main.getAttribute("data-taskid");
    const triggerCell = document.getElementById(`trigger-${safeTaskId}-${model}`);
    const score = parseInt(triggerCell?.textContent) || -Infinity;
    const plotRows = [];
    for (let j = i + 1; j < allRows.length; j++) {
      if (!allRows[j].id.startsWith("plot-")) break;
      plotRows.push(allRows[j]);
    }
    visibleRows.push([score, main, plotRows]);
  }

  visibleRows.sort((a, b) => ascending ? b[0] - a[0] : a[0] - b[0]);

  while (table.firstChild) table.removeChild(table.firstChild);
  visibleRows.forEach(([_, main, plots]) => {
    table.appendChild(main);
    plots.forEach(p => table.appendChild(p));
  });
}

let originalOrder = [];

document.addEventListener("DOMContentLoaded", () => {
  const tbody = document.querySelector("table tbody");
  originalOrder = Array.from(tbody.children);
});

function restoreOrder() {
  const tbody = document.querySelector("table tbody");
  while (tbody.firstChild) tbody.removeChild(tbody.firstChild);
  originalOrder.forEach(row => tbody.appendChild(row));
}

function navigateModel(task_id, newModel, dummyOffset) {
  togglePlot(newModel, task_id)
}

document.addEventListener("DOMContentLoaded", () => {
  const firstRow = document.querySelector("table tbody tr[data-taskid]");
  if (!firstRow) return;

  const task_id = firstRow.getAttribute("data-taskid");
  const firstModelCell = firstRow.querySelector(".score-cell");
  if (!firstModelCell) return;

  const modelMatch = firstModelCell.id.match(/^trigger-[^_]+-(.+)$/);
  if (!modelMatch) return;

  const model = modelMatch[1];
  togglePlot(model, task_id);
});

</script>
