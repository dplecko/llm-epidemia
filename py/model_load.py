
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model paths and instruct flags
MODEL_PATHS = {
    ### instruct versions
    "llama3_8b_instruct": ("/local/eb/dp3144/llama3_8b_instruct", True),  # LLaMA 3.1 8B-Instruct
    "llama3_70b_instruct": ("/local/eb/dp3144/llama3_70b_instruct", True),  # LLaMA 3.3 70B-Instruct
    "mistral_7b_instruct": ("/local/eb/dp3144/mistral_7b_instruct", True),  # Instruct Mistral
    "deepseek_7b_chat": ("/local/eb/dp3144/deepseek_7b_chat", True),  # Instruct DeepSeek
    "phi4": ("/local/eb/dp3144/phi4", True),  # Microsoft Phi-4
    "gemma3_27b_instruct": ("/local/eb/dp3144/gemma3_27b_instruct", True),  # Microsoft Phi-4
    ### non-instruct versions
    "llama3_8b": ("/local/eb/dp3144/llama3_8b", False),  # Regular LLaMA 3 8B
    "mistral_7b": ("/local/eb/dp3144/mistral_7b", False),  # Regular Mistral
    "deepseek_7b": ("/local/eb/dp3144/deepseek_7b", False),  # Regular DeepSeek
    "gpt2": ("/local/eb/dp3144/gpt2", False),  # Regular GPT-2
}

def load_model(model_name):
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
    
    return tokenizer, model, is_instruct
