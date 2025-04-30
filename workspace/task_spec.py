
import sys
import os
sys.path.append(os.path.abspath("workspace/tasks"))
from tasks_acs import tasks_acs
from tasks_labor import tasks_labor
from tasks_fbi import tasks_fbi
from tasks_edu import tasks_edu
from tasks_nhanes import tasks_nhanes
from tasks_gss import tasks_gss
from tasks_meps import tasks_meps
from tasks_scf import tasks_scf
from tasks_brfss import tasks_brfss
from tasks_nsduh import tasks_nsduh

task_specs = (
    tasks_acs +
    tasks_labor +
    tasks_fbi +
    tasks_edu +
    tasks_nhanes +
    tasks_gss #+
    # tasks_meps +
    # tasks_scf +
    # tasks_brfss +
    # tasks_nsduh
)