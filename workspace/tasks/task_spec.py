
from .task_spec_census import task_specs_census
from .task_spec_labor import task_specs_labor
from .task_spec_crime import task_specs_crime
from .task_spec_edu import task_specs_edu
from .task_spec_nhanes import task_specs_nhanes
from .task_spec_gss import task_specs_gss

task_specs = (
    tasks_acs +
    tasks_labor +
    tasks_fbi +
    tasks_edu +
    tasks_nhanes +
    tasks_gss
)