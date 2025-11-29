# nohup python workspace/fgai_elicit.py > fgai_elicit.log 2>&1 &
import sys, os

sys.path.append(os.getcwd())
import re
import pandas as pd
import torch
import string
from tqdm import tqdm
from workspace.common import *
from workspace.model_load import MODEL_PATHS


# model loading utilities
def get_device(prefer_gpu_idx: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{prefer_gpu_idx}")
    raise RuntimeError("CUDA not available; cannot accelerate generation.")


def get_model(model_path, prefer_gpu_idx: int = 0):
    device = get_device(prefer_gpu_idx)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # change to "flash_attention_2" if your stack supports it
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding is typically faster for decoder-only models in batched generate
    tokenizer.padding_side = "left"

    return model, tokenizer, device


# make the variable list in prompt
def varlist_to_prompt(var_dict, var_names):

    prompt_start = (
        "You are a data generator. Follow the rules strictly.\n"
        "RULES:\n"
        "1) Write a single narrative enclosed in <story>...</story>.\n"
        "2) Do NOT include headings, lists, analysis, or any text outside the tags.\n"
        "3) Mention ALL facts given below exactly once ({}).\n"
        "4) Keep it under 200 words.\n\n"
        "FACTS to be mentioned:"
    )
    prompt_start = prompt_start.format(
        ", ".join(var_names[var] for var in var_dict.keys())
    )

    # for each variable, mention its name and give the categories
    var_lines = []
    for var, categories in var_dict.items():
        cat_str = ", ".join(categories)
        var_line = f"- {var} (possible values: {cat_str})"
        var_lines.append(var_line)

    prompt_end = "OUTPUT FORMAT:\n" "<story>\n" "(your narrative here)\n" "</story>\n"

    prompt = prompt_start + "\n" + "\n".join(var_lines) + "\n\n" + prompt_end
    return prompt


def extract_tag(text: str, tag: str = "story") -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: if the model disobeys, strip known headings and return the first paragraph
    text = re.sub(r"(?is)^#+.*?$", "", text)  # markdown headers
    text = re.sub(r"(?is)^(analyzing errors|analysis).*?$", "", text).strip()
    # take first non-empty paragraph
    for para in re.split(r"\n\s*\n", text):
        p = para.strip()
        if p:
            return p
    return text.strip()


@torch.inference_mode()
def gen_data_batched(
    nsamp: int = 100,
    model=None,
    tokenizer=None,
    device=None,
    prompt: str = "",
    max_new_tokens: int = 256,
    temperature: float = 1,
    top_p: float = 0.9,
    batch_size: int = 16,  # tune per GPU memory
):
    # 1) Build all prompts up front (vectorized)
    prompts = [prompt for _ in range(nsamp)]

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
            repetition_penalty=1.05,  # NOTE: what is this for?
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            pad_token_id=getattr(
                tokenizer,
                "pad_token_id",
                getattr(tokenizer, "eos_token_id", None),
            ),
            use_cache=True,
        )

        # 3) Slice out continuations per-sample and decode
        for j in range(gen.size(0)):
            ilen = int(input_lens[j].item())
            seq = gen[j, ilen:] if gen.size(1) > ilen else gen[j]
            raw = tokenizer.decode(seq, skip_special_tokens=True).strip()

            # check if the raw output starts with a proper <story> tag
            story = re.sub(r"<[^>]+>|{[^}]+}", "", raw)

            # hard cap ~200 words as guard
            words = story.split()
            if len(words) > 200:
                story = " ".join(words[:200]).rstrip()

            out_texts.append(story)

        # free some memory sooner
        del enc, input_ids, attention_mask, gen
        torch.cuda.empty_cache()

    return out_texts


# helper functions
def prepare_answers(levels):
    """
    Prepare the answers for the model.
    :param levels: A list of possible answers.
    :return: A list of tokenized answers.
    """
    if len(levels) > 26:
        raise ValueError("Supports up to 26 items (A-Z)")

    letters = string.ascii_uppercase  # 'A', 'B', ...
    mapping = {letters[i]: item for i, item in enumerate(levels)}
    answer_key = "\n".join(f"{k}. {v}" for k, v in mapping.items())

    return answer_key, mapping


def prepare_prompt(text, var_name, levels):
    """
    Prepare the prompt for the model.
    :param prompt: The initial prompt.
    :param levels: A list of possible answers.
    :return: The prepared prompt.
    """

    # clean from any appearance of "{" or "}" which may break format
    text = re.sub(r"{|}", "", text)

    # make this conditional with try and except with a breakpoint() if it fails
    try:
        prompt = (
            "Input: Based on the following text:\n\n"
            + text
            + "\n\ndetermine the person's {}. "
            + "Begin your answer with the capital letter corresponding to your chosen option below, followed by a period.\n"
        ).format(var_name)
    except Exception as e:
        breakpoint()
    answers, answer_mapping = prepare_answers(levels)
    prompt += answers
    prompt += "\nOutput: "
    return prompt, answer_mapping


def annotate_data(model, tokenizer, device, texts, var_dict):

    # initiate a data.frame to hold the results
    df = pd.DataFrame(columns=var_dict.keys())

    # for loop over all texts
    for var, levels in tqdm(var_dict.items(), desc="Annotating data"):

        _, answer_mapping = prepare_answers(levels)

        for i, text in enumerate(texts):
            # extract relevant variables from var_dict

            inputs = tokenizer(
                prepare_prompt(text, var, levels)[0], return_tensors="pt"
            ).to(device)
            level_ids = [
                [tokenizer.convert_tokens_to_ids(tok) for tok in ans]
                for ans in answer_mapping.keys()
            ]

            with torch.no_grad():
                outputs = model(**inputs).logits
                next_token_logits = outputs[
                    :, -1, :
                ]  # Last token in the input sequence
                probs = torch.softmax(next_token_logits, dim=-1)

            # Normalise probability mass over the provided answers
            level_probs = [
                sum(probs[0, tid].item() for tid in ids) for ids in level_ids
            ]

            # get the predicted answer from the list
            pred_idx = max(range(len(level_probs)), key=level_probs.__getitem__)
            pred_answer = answer_mapping[list(answer_mapping.keys())[pred_idx]]

            # save the answer to the data frame for the i-th
            df.loc[i, var] = pred_answer

        # ensure that the variable is categorical with correct levels
        df[var] = pd.Categorical(df[var], categories=levels)

    return df


def get_var_names(dataset):
    if dataset == "nsduh":
        return {
            "age": "age",
            "sex": "sex",
            "race": "race",
            "edu": "education",
            "income": "income",
            "alc_monthly": "alcohol last month use",
            "cig_monthly": "cigarette last month use",
            "mj_monthly": "marijuana last month use",
            "coc_ever": "cocaine ever use",
        }
    elif dataset == "brfss":
        return {
            "age": "age",
            "state": "state",
            "sex": "sex",
            "race": "race",
            "education": "education",
            "income": "income",
            "exercise_monthly": "exercise monthly",
            "bmi": "body mass index (BMI)",
            "diabetes": "diabetes",
            "high_bp": "high blood pressure",
        }


dataset = "nsduh"
df = pd.read_parquet(f"data/clean/{dataset}.parquet")
var_names = get_var_names(dataset)
var_dict = {}
for var in var_names.keys():
    # extract the variable categories
    var_dict[var] = sorted(df[var].unique())

model_path = MODEL_PATHS["llama3_8b_instruct"][0]
model, tokenizer, device = get_model(model_path, prefer_gpu_idx=0)

nsamp = 10000

texts = gen_data_batched(
    nsamp, model, tokenizer, device, varlist_to_prompt(var_dict, var_names)
)

# save the intermediate texts list to a pickle file
import pickle

with open("data/fgai/{dataset}_fgai_texts.pkl", "wb") as f:
    pickle.dump(texts, f)

# read the pickle file
with open("data/fgai/{dataset}_fgai_texts.pkl", "rb") as f:
    texts = pickle.load(f)

df_m = annotate_data(model, tokenizer, device, texts, var_dict)

# go over all columns in df, and if they are categorical, make df_m inherit their category and order
for col in df.columns:
    if col in df_m.columns:
        dt = df[col].dtype
        if isinstance(dt, pd.CategoricalDtype):
            df_m[col] = pd.Categorical(
                df_m[col], categories=df[col].cat.categories, ordered=dt.ordered
            )

# sample n samp rows from original data, with weights
df_w = (
    df[vars].sample(n=nsamp, weights=df["weight"], replace=False).reset_index(drop=True)
)

# rbind the two dataframes, and add a binary 0/1 env column
df_m["env"] = 1
df_w["env"] = 0
df_cmb = pd.concat([df_m, df_w], ignore_index=True)


# pre-processing function to remove categoricals
def clean_cats(df, X):

    # binary categoricals: map yes/no to 1/0
    for col in df.columns:
        dt = df[col].dtype
        if not isinstance(dt, pd.CategoricalDtype):
            continue
        cats = list(df[col].cat.categories)
        if len(cats) == 2:
            if set(cats) == {"yes", "no"}:
                df[col] = df[col].map({"yes": 1, "no": 0})
            else:
                df[col] = df[col].cat.codes  # map to 0/1

    # multi-level categoricals
    for col in list(df.columns):  # df may change when we add dummies
        dt = df[col].dtype
        if not isinstance(dt, pd.CategoricalDtype):
            continue

        s = df[col]
        cats = list(s.cat.categories)
        if len(cats) <= 2:
            continue

        if s.cat.ordered:
            # ordered -> integer codes (1..K)
            df[col] = s.cat.codes + 1
        else:
            # unordered, non-binary, and not in X -> one-hot
            if col in X:
                continue
            dummies = pd.get_dummies(s, prefix=col)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    return df


df_res = clean_cats(df_cmb, X=["race"])
df_res.to_parquet("data/fgai/{dataset}_envs.parquet", index=False)

# auditing the model labels
import pickle

with open("data/fgai/nsduh_fgai_texts.pkl", "rb") as f:
    texts = pickle.load(f)

df_m = pd.read_parquet("data/fgai/nsduh_envs.parquet")
df_m = df_m[df_m["env"] == 1].reset_index(drop=True)

# sample 10 random rows to inspect
import random

random.seed(42)
sample_indices = random.sample(range(len(texts)), 10)

sample_indices = [1824, 409, 4506, 4012, 3657, 2286, 1679, 8935, 1424, 9674]

print(texts[sample_indices[3]])

idx = 9674
print(texts[idx])
# need to convert the levels back to original labels
for var in df_m.columns:
    if var in ["age", "edu", "income"]:
        print(var, df[var].cat.categories[df_m.loc[idx, var] - 1])
    elif var in ["sex"]:
        print(var, df[var].cat.categories[df_m.loc[idx, var]])
    else:
        print(var, df_m.loc[idx, var])

# automated checking


# load the data from fgai/data/story-truth.csv
df_truths = pd.read_csv("../fgai/data/story-truth.csv")

# verify that each column value is in the correct levels
for var in df_truths.columns:

    if var not in var_dict:
        continue

    # correct levels
    levels = var_dict[var]

    # check if all values in df_truths[var] are in levels
    invalid_values = set(df_truths[var].unique()) - set(levels)
    if len(invalid_values) > 0:
        print(f"Variable {var} has invalid values: {invalid_values}")

# get the automatic check
for i in range(len(df_truths)):

    idx = df_truths.loc[i, "index"]

    for var in df_truths.columns:
        if var not in var_dict:
            continue

        true_value = df_truths.loc[i, var]

        if var in ["age", "edu", "income"]:
            pred_value = df[var].cat.categories[df_m.loc[idx, var] - 1]
        elif var in ["sex", "alc_monthly", "cig_monthly", "mj_monthly", "coc_ever"]:
            pred_value = df[var].cat.categories[df_m.loc[idx, var]]
        else:
            pred_value = df_m.loc[idx, var]

        if true_value != pred_value:
            print(
                f"Row {i}, Variable {var}: true value = {true_value}, predicted value = {pred_value}"
            )
