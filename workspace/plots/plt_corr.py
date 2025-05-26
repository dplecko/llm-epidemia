
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "workspace"))
sys.path.append(os.path.join(os.getcwd(), "workspace/utils"))
from task_spec import task_specs_hd
from utils.plot_helpers import hd_corr_plot

models = ["llama3_8b_instruct", "llama3_70b_instruct", "mistral_7b_instruct", "phi4", 
          "gemma3_27b_instruct", "deepseek_7b_chat"]

# high-dimensional tasks: correlation
plt_hdcor = hd_corr_plot(models, task_specs_hd)
plt_hdcor.save("data/plots/hd_corr_plot.png", dpi=300, width=8, height=6)