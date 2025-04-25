
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import models

# Model paths and instruct flags
MODEL_PATHS = {
    # ### instruct versions
    # "llama3_8b_instruct": ("/local/eb/dp3144/llama3_8b_instruct", True),  # LLaMA 3.1 8B-Instruct
    # "llama3_70b_instruct": ("/local/eb/dp3144/llama3_70b_instruct", True),  # LLaMA 3.3 70B-Instruct
    # "mistral_7b_instruct": ("/local/eb/dp3144/mistral_7b_instruct", True),  # Instruct Mistral
    # "deepseek_7b_chat": ("/local/eb/dp3144/deepseek_7b_chat", True),  # Instruct DeepSeek
    # "phi4": ("/local/eb/dp3144/phi4", True),  # Microsoft Phi-4
    # "gemma3_27b_instruct": ("/local/eb/dp3144/gemma3_27b_instruct", True),  # Microsoft Phi-4
    # ### non-instruct versions
    # "llama3_8b": ("/local/eb/dp3144/llama3_8b", False),  # Regular LLaMA 3 8B
    # "mistral_7b": ("/local/eb/dp3144/mistral_7b", False),  # Regular Mistral
    # "deepseek_7b": ("/local/eb/dp3144/deepseek_7b", False),  # Regular DeepSeek
    "gpt2": ("openai-community/gpt2", False),  # Regular GPT-2,
    
}

API_MODELS = {
    "gpt-4.1": {"model_name": "gpt-4.1",
                "is_instruct": False},  # simplest, no reasoning
    
    "o4-mini":{"model_name": "o4-mini",
               "is_instruct": False,
               "reasoning": {"effort": "low"}},
    
    "o3": {"model_name": "o3",
           "is_instruct": False,
           "reasoning": {"effort": "low"}},
    
    "gpt-4.1_web": {"model_name": "gpt-4.1",
                    "is_instruct": False,
                    "tools": [{ "type": "web_search_preview" }]},  # simplest, no reasoning
    
    # "o4-mini_web":{"model_name": "o4-mini",
    #                "is_instruct": False,
    #                "reasoning": {"effort": "low"},
    #                "tools": [{ "type": "web_search_preview" }]},
    
    # "o3_web": {"model_name": "o3",
    #            "is_instruct": False,
    #            "reasoning": {"effort": "low"},
    #            "tools": [{ "type": "web_search_preview" }]},
}


def load_hf_model(model_name):
    """Loads the specified model and tokenizer, and returns instruct flag."""
    if model_name not in MODEL_PATHS:
        return None, None, None

    model_path, is_instruct = MODEL_PATHS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if "llama" in model_name or "mistral" in model_name else torch.float16,
        device_map="auto"
    )
    hf_model = models.HuggingFaceModel(model, tokenizer)
    hf_model.is_instruct = is_instruct
    return hf_model


def load_api_model(model_name):
    kwargs = API_MODELS.get(model_name, {})
    reasoning = kwargs.get("reasoning", None)
    tools = kwargs.get("tools", [])
    api_model = models.OpenAIAPIModel(kwargs["model_name"], reasoning=reasoning, tools=tools)
    api_model.is_instruct = kwargs.get("is_instruct", False)
    return api_model
    

def load_model(model_name):
    if model_name in MODEL_PATHS:
        return load_hf_model(model_name)
    else:
        return load_api_model(model_name)
