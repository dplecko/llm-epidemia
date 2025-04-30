
import pandas as pd

model_name_map = {
    "llama3_8b_instruct": "LLama3 8B",
    "llama3_70b_instruct": "LLama3 70B",
    "mistral_7b_instruct": "Mistral 7B",
    "deepseek_7b_chat": "DeepSeek 7B",
    "phi4": "Phi4",
    "gemma3_27b_instruct": "Gemma3 27B",
    "nhanes": "NHANES",
    "gss": "GSS"
}

model_display_map = {v: k for k, v in model_name_map.items()}

def model_name(mod):
    if isinstance(mod, pd.Series):
        return mod.map(lambda x: model_name_map.get(x, x))
    elif isinstance(mod, list):
        return [model_name_map.get(m, m) for m in mod]
    return model_name_map.get(mod, mod)

def model_unname(mod):
    if isinstance(mod, pd.Series):
        return mod.map(lambda x: model_display_map.get(x, x))
    elif isinstance(mod, list):
        return [model_display_map.get(m, m) for m in mod]
    return model_display_map.get(mod, mod)
