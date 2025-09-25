# minimal_llama_causal_train.py
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)
import torch, os, json, matplotlib.pyplot as plt

# --- data ---
ds = load_from_disk("datasets/nsduh_synth_gemma")  # expects splits: 'train','validation' TODO: change here for different dataset
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # TODO: change for different model

tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def tokenize(x):  # assume a 'text' column; adjust if yours is different
    return tok(x["text"], truncation=True, max_length=2048)
tok_ds = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)

# optional: simple packing for efficiency
def group_texts(examples, block_size=2048):
    ids = sum(examples["input_ids"], [])
    n = (len(ids) // block_size) * block_size
    ids = ids[:n]
    return {"input_ids": [ids[i:i+block_size] for i in range(0, n, block_size)]}
g_train = tok_ds["train"].map(group_texts, batched=True, remove_columns=tok_ds["train"].column_names)
g_eval  = tok_ds["validation"].map(group_texts, batched=True, remove_columns=tok_ds["validation"].column_names)

collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

# --- model ---
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=dtype, device_map="auto", attn_implementation="eager",
)
model.gradient_checkpointing_enable()

# --- training ---
out = "" # TODO: here for output
steps=5
args = TrainingArguments(
    output_dir=out, per_device_train_batch_size=1, per_device_eval_batch_size=1,
    gradient_accumulation_steps=8, num_train_epochs=6,
    evaluation_strategy="steps", eval_steps=steps, save_strategy="steps", save_steps=steps,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    logging_steps=steps, report_to="none", fp16=False, bf16=(dtype==torch.bfloat16),
    save_total_limit=2, ddp_find_unused_parameters=False
)

trainer = Trainer(
    model=model, args=args, data_collator=collator,
    train_dataset=g_train, eval_dataset=g_eval
)

trainer.train()

# save the best model (trainer.model is already the best due to load_best_model_at_end)
best_dir = os.path.join(out, "best")
os.makedirs(best_dir, exist_ok=True)
trainer.save_model(best_dir)
tok.save_pretrained(best_dir)

# --- collect & save logs for later plotting ---
hist = trainer.state.log_history  # list of dicts, includes {'loss': ...} and {'eval_loss': ...}
with open(os.path.join(out, "loss_history.json"), "w") as f:
    json.dump(hist, f, indent=2)

# --- quick plot ---
train_x, train_y, eval_x, eval_y = [], [], [], []
step = 0
for e in hist:
    if "loss" in e and "epoch" in e:  # training log
        step += 1
        train_x.append(step)
        train_y.append(e["loss"])
    if "eval_loss" in e and "epoch" in e:  # eval per epoch
        eval_x.append(e["epoch"])
        eval_y.append(e["eval_loss"])

plt.figure()
if train_x:
    plt.plot(train_x, train_y, label="train loss")
if eval_x:
    plt.plot(eval_x, eval_y, marker="o", label="val loss")
plt.xlabel("steps (train) / epochs (val)")
plt.ylabel("loss")
plt.legend(); plt.title("Llama CLM fine-tuning loss"); plt.tight_layout()
plt.savefig(os.path.join(out, "loss_plot.png"), dpi=200)
plt.show()

print("Best model saved to:", best_dir)
