import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import re


def extract_tag(text: str, tag: str = "story") -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL|re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: if the model disobeys, strip known headings and return the first paragraph
    text = re.sub(r"(?is)^#+.*?$", "", text)                # markdown headers
    text = re.sub(r"(?is)^(analyzing errors|analysis).*?$", "", text).strip()
    # take first non-empty paragraph
    for para in re.split(r"\n\s*\n", text):
        p = para.strip()
        if p:
            return p
    return text.strip()


def get_device(prefer_gpu_idx: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{prefer_gpu_idx}")
    raise RuntimeError("CUDA not available; cannot accelerate generation.")

def get_model(model_path, prefer_gpu_idx: int = 0):
    device = get_device(prefer_gpu_idx)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        cache_dir="/local/eb/shreyas/huggingface_cache",
        attn_implementation="eager",  # change to "flash_attention_2" if your stack supports it
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding is typically faster for decoder-only models in batched generate
    tokenizer.padding_side = "left"

    return model, tokenizer, device


def sample_weighted(df, n, weight_col="weight", replace=False, random_state=42):
    w = df[weight_col].astype(float).fillna(0.0)
    sampled = df.sample(n=n, replace=replace, weights=w, random_state=random_state)
    train = sampled[: int(0.8 * n)]
    val   = sampled[int(0.8 * n):]
    return train.reset_index(drop=True), val.reset_index(drop=True)

def negate(x):
    s = str(x).strip().lower()
    return "not" if s in {"no", "false", "0"} else ""


@torch.inference_mode()
def generate_brfss_data_batched(
    data: pd.DataFrame,
    model,
    tokenizer,
    device,
    prompt_template: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 16,          # tune per GPU memory
):
    # 1) Build all prompts up front (vectorized)
    prompts = [
        prompt_template.format(
            state=row["state"],
            age=row["age"],
            sex=row["sex"],
            race=row["race"],
            education=row["education"],
            income=row["income"],
            smoker=negate(row["smoker"]),
            bmi=row["bmi"],
            exercise_monthly=negate(row["exercise_monthly"]),
            poor_mental_health=negate(row["poor_mental_health"]),
            diabetes=negate(row["diabetes"]),
            high_bp=negate(row["high_bp"]),
            asthma=negate(row["asthma"]),
            cholesterol=negate(row["cholesterol"]),
            heart_attack=negate(row["heart_attack"]),
            stroke=negate(row["stroke"]),
            depression=negate(row["depression"]),
        )
        for _, row in data.iterrows()
    ]

    out_texts = []
    # 2) Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating (batched)"):
        batch_prompts = prompts[i : i + batch_size]

        # Tokenize with padding; keep per-sample lengths for slicing
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        input_lens = attention_mask.sum(dim=1)  # effective prompt lengths per sample

        # Generate
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.05, 
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", None)),
            use_cache=True,  # should be default, but make explicit
        )

        # 3) Slice out continuations per-sample and decode
        # gen: (B, seq_len = prompt + new)
        for j in range(gen.size(0)):
            ilen = int(input_lens[j].item())
            seq = gen[j, ilen:] if gen.size(1) > ilen else gen[j]
            raw = tokenizer.decode(seq, skip_special_tokens=True).strip()
            story = extract_tag(raw, tag="story")
            # hard cap ~200 words as a final guard
            words = story.split()
            if len(words) > 200:
                story = " ".join(words[:200]).rstrip()
            out_texts.append(story)
            print(out_texts[-1], flush=True)

        # (optional) free some memory sooner
        del enc, input_ids, attention_mask, gen
        torch.cuda.empty_cache()

    return out_texts


def save_datasets(train_synthetic, val_synthetic):
    train_texts = [str(t).strip() for t in train_synthetic]
    val_texts   = [str(t).strip() for t in val_synthetic]

    ds = DatasetDict({
        "train": Dataset.from_dict({"text": train_texts}),
        "validation": Dataset.from_dict({"text": val_texts}),
    })
    cache_dir = "datasets/brfss_synth_gemma"
    ds.save_to_disk(cache_dir)
    print(f"Synthetic datasets saved to {cache_dir}")


if __name__ == "__main__":
    # data
    path = "datasets/brfss/data/brfss.parquet"
    data = pd.read_parquet(path)
    train, val = sample_weighted(data, n=2500, weight_col="weight", replace=True)

    # model
    model_path = "/local/eb/dp3144/gemma3_27b_instruct"  # local model path
    model, tokenizer, device = get_model(model_path, prefer_gpu_idx=0)

    brfss_prompt_template = (
        "You are a data generator. Follow the rules strictly.\n"
        "RULES:\n"
        "1) Write a single narrative enclosed in <story>...</story>.\n"
        "2) Do NOT include headings, lists, analysis, or any text outside the tags.\n"
        "3) Mention ALL facts given below exactly once (state, age, sex, race, education, income, smoking status, BMI, exercise, mental health, diabetes, high blood pressure, asthma, cholesterol, heart attack, stroke, depression).\n"
        "4) Keep it under 200 words.\n\n"
        "FACTS:\n"
        "- state: {state}\n"
        "- age: {age}\n"
        "- sex: {sex}\n"
        "- race: {race}\n"
        "- education: {education}\n"
        "- income: {income}\n"
        "- smoking status: {smoker}\n"
        "- BMI: {bmi}\n"
        "- exercise monthly: {exercise_monthly}\n"
        "- poor mental health: {poor_mental_health}\n"
        "- diabetes: {diabetes}\n"
        "- high blood pressure: {high_bp}\n"
        "- asthma: {asthma}\n"
        "- cholesterol: {cholesterol}\n"
        "- heart attack: {heart_attack}\n"
        "- stroke: {stroke}\n"
        "- depression: {depression}\n\n"
        "OUTPUT FORMAT:\n"
        "<story>\n"
        "(your narrative here)\n"
        "</story>\n"
    )

    # Batched generation (tune batch_size to your GPU memory)
    train_synth = generate_brfss_data_batched(
        train, model, tokenizer, device, brfss_prompt_template,
        max_new_tokens=256, temperature=0.7, top_p=0.9, batch_size=16
    )
    val_synth = generate_brfss_data_batched(
        val, model, tokenizer, device, brfss_prompt_template,
        max_new_tokens=256, temperature=0.7, top_p=0.9, batch_size=16
    )

    save_datasets(train_synth, val_synth)
