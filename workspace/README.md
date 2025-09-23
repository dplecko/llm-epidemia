
---
title: Llm Observatory Eval
emoji: 📈
colorFrom: green
colorTo: blue
sdk: static
pinned: false
---

<div align="center">
  <h1 style="display:inline-flex; align-items:center; gap:0.5rem; white-space:nowrap; margin:0;">
    <img
      src="./img/logo-icon.png"
      alt="LLM Observatory"
      width="55"
      height="40"
      style="display:block;"
    />
    <span style="font-size:2em; margin:0;">LLM Observatory</span>
  </h1>
</div>


<p align="center">
  <a href="https://llm-observatory.org/index.html">
    <img src="https://img.shields.io/badge/Website-Visit-blue?style=flat" alt="Website">
  </a>
  &nbsp;&nbsp;
  <a href="https://github.com/dplecko/llm-epidemia">
    <img src="https://img.shields.io/badge/GitHub-Repository-black?style=flat&logo=github" alt="GitHub">
  </a>
</p>


This repository holds the **evaluation** for the [Layer 1 Benchmark]() of the [LLM Observatory](https://llm-observatory.org/index.html). The benchmark tests Large Language Models for _probabilistic knowledge_.

## FAQ

<details>
<summary><strong>Q1: What does the benchmark measure?</strong></summary>

Our Layer 1 benchmark measures <em>probabilistic knowledge</em>. Here, probabilistic is used as opposed
                to
                factual knowledge.
                For instance, answering questions with a known correct answer (e.g., "What is the capital of England? -
                London")
                corresponds to factual knowledge. Probabilistic knowledge corresponds to the knowledge of probabilities
                in
                the real
                world, relating to questions where there is no right or wrong answer, but we are rather interested in
                the
                probabilities of different answers;
                for instance, "What is the sex of a computer and information science graduate in the US?" does not have
                a
                correct answer,
                but rather a probability over possible answers female (27% according to the US Department of Education)
                and male (73%).
</details>

<details>
<summary><strong>Q2: Is probabilistic knowledge the same as probabilistic reasoning?</strong></summary>

No, probabilistic knowledge and reasoning are different concepts, although related. Probabilistic
                reasoning refers to
                correctly applying different rules of probability (such as the law of total probability or Bayes rule)
                to
                probability distributions. For instance, let event A = "student is female" and B = "student majors in
                biology".
                Given that P(A, B) = 0.015 and P(B) = 0.03, using the Bayes rule one can compute P(A | B) = 0.015 / 0.03
                =
                0.5, and such a computation would fall under probabilistic reasoning.
                Probabilistic knowledge, however, refers to knowing correct probabilities of an event P(A), or a
                conditional event P(A | B); for instance, knowing that 27%
                of computer and information science graduates in the US are female, while 73% are male. Our benchmark
                tests LLMs in this latter ability.
</details>

<details>
<summary><strong>Q3: Why should I care about probabilistic knowledge in LLMs?</strong></summary>

Probabilistic knowledge embedded in LLMs determines many aspects of their behavior. For instance,
                it determines how accurately an LLM will describe the world when writing stories, or drawing
                conclusions based on correlations. Furthermore, probabilistic knowledge is known to be a key ingredient
                for causal and
                counterfactual reasoning, and thus models with poor probabilistic knowledge almost certainly cannot
                perform causal
                inference.
</details>

<details>
<summary><strong>Q4: How do you measure probabilistic knowledge?</strong></summary>

Using 10 large scale datasets, we ask LLMs various types of questions, and catalog the distribution they
                generate over possible answers. Then, we compare this distribution to the real world. You can read more
                about this in the [Benchmark Methodology](https://llm-observatory.org/l1-description.html)
                section. Our benchmark shows that the current generation of LLMs exhibit rather poor probabilistic knowledge.
</details>


## Getting Started -- Minimal Working Example

Evaluating models on the benchmark is straightforward. Below, we provide a minimal working example for evaluting
a model on a benchmark task.

```python
import evaluate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import models

# load LLM Observatory infrastructure
llm_obs = evaluate.load("llm-observatory/llm-observatory-eval")

# prepare the model
model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)
hf_model = models.HuggingFaceModel(model, tokenizer)

# extract model answers
llm_obs.extract(model_name = "llama3_8b_instruct", model=hf_model, task=llm_obs.task_specs[0])
llm_obs.compute(models = ["llama3_8b_instruct"], tasks=llm_obs.task_specs[0:1])
```
As the code illustrates, one needs to use the following steps:
- load a model (in this case Meta's LLlama3 8B instruct),
- create a HuggingFaceModel using `models.HuggingFaceModel`,
- extract the model response using `.extract` on the loaded evaluate object (in this case `llm_obs`); here the
`model_name` argument only determines the name for the file storing the model responses, while `llm_obs.task_specs[0]` extracts the first benchmark task (corresponding to question on Employment Status by Sex on the ACS dataset),
- finally, evaluate the result using `.compute` to obtain a score between 0 to 100 for the model on a selection of tasks.

##  Citation Information
```
@techreport{llmobservatory2025layer1,
  title={Epidemiology of Large Language Models: A Benchmark for Observational Distribution Knowledge},
  author={Plecko, Drago and Okanovic, Patrik and Hoeffler, Torsten and Bareinboim, Elias},
  institution={Causal AI Lab, Columbia University},
  year={2025},
  url={\url{https://causalai.net/r136.pdf}}
}
```

## Contribute
LLM Observatory is an open-source initiative interested in your contributions.
Further details on contributing can be found at this link: [contribute](https://llm-observatory.org/contribute.html).
