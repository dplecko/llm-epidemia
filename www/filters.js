function getDatasetMeta() {
  return {
    "ACS": { link: "https://www.census.gov/programs-surveys/acs.html", desc: "American Community Survey" },
    "BLS": { link: "https://www.bls.gov/", desc: "Bureau of Labor Statistics" },
    "BRFSS": { link: "https://www.cdc.gov/brfss/annual_data/annual_2023.html", desc: "Behavioral Risk Factor Surveillance System" },
    "FBI Arrests": { link: "https://ucr.fbi.gov/crime-in-the-u.s/2019/crime-in-the-u.s.-2019/tables/table-42/table-42.xls", desc: "FBI Arrest Data" },
    "GSS": { link: "https://gss.norc.org/", desc: "General Social Survey" },
    "IPEDS": { link: "https://nces.ed.gov/ipeds/", desc: "IPEDS Education Data (Department of Education)" },
    "MEPS": { link: "https://meps.ahrq.gov/", desc: "Medical Expenditure Panel Survey" },
    "NHANES": { link: "https://www.cdc.gov/nchs/nhanes/about/index.html", desc: "National Health and Nutrition Examination Survey" },
    "NSDUH": { link: "https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-and-health/national-releases/2023", desc: "National Survey on Drug Use and Health" },
    "SCF": { link: "https://www.federalreserve.gov/econres/scfindex.htm", desc: "Survey of Consumer Finances" }
  };
}

const datasetGroups = {
  "Health": ["BRFSS", "NHANES", "NSDUH", "MEPS"],
  "Social": ["GSS", "FBI Arrests", "IPEDS"],
  "Economic": ["BLS", "ACS", "SCF"]
};

function toggleCheckboxes(className, check) {
  document.querySelectorAll('.' + className).forEach(cb => { cb.checked = check });
  updateModelList();
}

function toggleGroup(group, check) {
  const groupMap = {
    "Health": ["BRFSS", "NHANES", "NSDUH", "MEPS"],
    "Social": ["GSS", "FBI Arrests", "IPEDS"],
    "Economic": ["BLS", "ACS", "SCF"]
  }
  const targets = groupMap[group] || []
  targets.forEach(ds => {
    const el = document.getElementById(`ds-${ds}`)
    if (el) el.checked = check
  })
  updateModelList()
}

function renderFilters() {
  let dsDiv = document.getElementById("dataset-checkboxes");
  let dimDiv = document.getElementById("dimension-checkboxes");
  const dsMeta = getDatasetMeta();

  dsDiv.innerHTML = `
    <div class="filter-inner">
      <div class="d-flex gap-2 mb-2">
        <button class="btn btn-sm btn-outline-light" onclick="toggleCheckboxes('dataset-filter', true)">All</button>
        <button class="btn btn-sm btn-outline-light" onclick="toggleCheckboxes('dataset-filter', false)">None</button>
      </div>
      ${Object.entries(datasetGroups).map(([group, datasets]) => `
        <div class="d-flex align-items-center gap-2 mt-2 mb-1">
          <span class="fw-bold text-warning mb-0">${group}</span>
          <div class="btn-group btn-group-subfilter" role="group">
            <button class="btn btn-sm btn-outline-light" onclick="toggleGroup('${group}', true)">All</button>
            <button class="btn btn-sm btn-outline-light" onclick="toggleGroup('${group}', false)">None</button>
          </div>
        </div>
        ${datasets.filter(d => allDatasets.has(d)).map(d => `
          <div class="form-check">
            <input class="form-check-input dataset-filter" type="checkbox" value="${d}" id="ds-${d}" checked>
            <label class="form-check-label"
                  for="ds-${d}"
                  data-bs-toggle="popover"
                  data-bs-html="true"
                  data-bs-trigger="manual"
                  data-bs-placement="right"
                  data-bs-content="<a href='${dsMeta[d].link}' target='_blank'>${dsMeta[d].desc}</a>">
              ${d}
            </label>
          </div>
        `).join('')}
      `).join('')}      
    </div>`;

  dimDiv.innerHTML = `
    <div class="filter-inner">
      <div class="d-flex gap-2 mb-2">
        <button class="btn btn-sm btn-outline-light" onclick="toggleCheckboxes('dim-filter', true)">All</button>
        <button class="btn btn-sm btn-outline-light" onclick="toggleCheckboxes('dim-filter', false)">None</button>
      </div>
      ${[...allDims].sort().map(d => `
        <div class="form-check">
          <input class="form-check-input dim-filter" type="checkbox" value="${d}" id="dim-${d}" checked>
          <label class="form-check-label" for="dim-${d}" title="Demographic Dimension d=${d}">d=${d}</label>
        </div>`).join('')}
    </div>`;

  const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
  tooltipTriggerList.forEach(el => new bootstrap.Tooltip(el));
  document.querySelectorAll('[data-bs-toggle="popover"]').forEach(el => new bootstrap.Popover(el));
  document.querySelectorAll('input[name="prompting"]').forEach(el => el.addEventListener('change', updateModelList));
  document.querySelectorAll('.dataset-filter').forEach(el => el.addEventListener('change', updateModelList));
  document.querySelectorAll('.dim-filter').forEach(el => el.addEventListener('change', updateModelList));
  let activePopover = null;
  document.querySelectorAll('[data-bs-toggle="popover"]').forEach(el => {
    const pop = new bootstrap.Popover(el);

    el.addEventListener('mouseenter', () => {
      // Close any other open popover
      if (activePopover && activePopover !== pop) activePopover.hide();

      pop.show();
      activePopover = pop;

      const tip = document.querySelector('.popover');
      if (tip) {
        tip.addEventListener('mouseleave', () => {
          if (!el.matches(':hover')) pop.hide();
        });
      }
    });

    el.addEventListener('mouseleave', () => {
      setTimeout(() => {
        const tipHovered = document.querySelector('.popover:hover');
        if (!el.matches(':hover') && !tipHovered) {
          pop.hide();
          activePopover = null;
        }
      }, 100);
    });
  });
}

