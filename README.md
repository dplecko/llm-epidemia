
# Epidemiology of Large Language Models: A Benchmark for Observational Distribution Knowledge

### 0. Pull the Repository and Create the Environment

Clone the repo as usual.

Create the environment:
```bash
conda create -n llm-epidemiology python=3.10
conda activate llm-epidemiology
pip install -r requirements.txt
```

---

### 1. Data Download & Setup

- Set-up details for datasets can be found in `datasets/README.md`


- Place the `.parquet` files in the `data/clean` directory

---

### 2. Obtaining Language Model Responses

- **Set up model paths**:
  - Edit `workspace/model_load.py` to specify the correct `MODEL_PATHS` for your local setup

- **Define tasks**:
  - Tasks are specified in `workspace/tasks/`. Here's an example:

    ```json
    {
        "name": "MEPS: Expenditure by Age Group",
        "variables": ["expenditure_group", "age_group"],
        "prompt": "What is the annual healthcare expenditure for a person aged {} in the US?",
    }
    ```

    - This task looks at the MEPS data.
    - We're estimating the distribution of `V1 = expenditure_group` given `V2 = age_group`.
    - The `prompt` is the initial input to the model, with `{}` replaced by variables of the conditioning 

- **Run the model evaluator**:
  - Use `workspace/extract.py` and the `task_extract()` function
  ```python
  python workspace/evaluator.py
  ```


- **Results**:
  - Results are saved as `.json` files in `data/results/benchmark/` and `data/results/benchmark-hd/`

---

### 3. Running the Benchmark Evaluation


- After generating all JSON files, run for evaluation:

```python
python workspace/eval.py
```

- For generating plots run:
```python
python workspace/plots/plot_run.py
```
