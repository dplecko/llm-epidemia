
# nohup python py/main.py > story-inference.log 2>&1 &
import sys
import os

# Get the current working directory and append "py" subdirectory
sys.path.append(os.path.join(os.getcwd(), "py"))

from py.v0.pv_model import model_pv, generate_prompt, query_model
from model_load import load_model

# models
models = ['llama3_8b', 'llama3_8b_instruct', 'gpt2', 'mistral_7b', 'mistral_7b_instruct', 
          'deepseek_7b', 'llama3_70b_instruct']


model_pv("llama3_8b_instruct", "crime", "in-context", folder = "sensitivity")
model_pv("llama3_8b_instruct", "edu", "in-context", folder = "sensitivity")
# for mode in ['story', 'logits', 'in-context', 'story-gender', 'logits-gender']:
#     model_pv("llama3_8b_instruct", "crime", mode, folder = "sensitivity")

# model_pv("mistral_7b_instruct", "edu", "story")
# 

# for model_name in models:
#     for context in ["labor", "health", "edu", "crime"]:
#         if "instruct" in model_name:
#             modes = ["story", "logits"]
#         else:
#             modes = ["logits"]
        
#         for mode in modes:
#             model_pv(model_name, context, mode)

