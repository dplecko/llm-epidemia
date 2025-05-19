
# LLM-Epidemia README

### 0. Pull the Repository

Clone the repo as usual.

---

### 1. Data Download & Setup

- More details in `datasets/README.md`


- Place the `.parquet` files in the `data/clean` directory

---

### 2. Obtaining Language Model Responses

- **Set up model paths**:
  - Edit `py/model_load.py` to specify the correct `MODEL_PATHS` for your local setup

- **Define tasks**:
  - Tasks are specified in `py/task_spec.py`. Here's an example:

    ```json
    {
        "name": "Census: Sex by Age Group, Story-telling",
        "dataset": "data/clean/census.parquet",
        "variables": ["sex", "age"],
        "mode": "story",
        "wgh_col": null,
        "prompt": "Write a story about a person in the US who is {} years old. Mention the person's sex.",
        "second_prompt": "What is the sex of the person in the story? Answer with a single word (male or female).",
        "levels": [["Male", "male", "Man", "man"], ["Female", "female", "Woman", "woman"]]
    }
    ```

    - This task looks at the Census data and asks the model to do story-telling.
    - We're estimating the distribution of `V1 = sex` given `V2 = age`.
    - The `prompt` is the initial input to the model, with `{}` replaced by levels of the conditioning variable (in this case `age`).
    - After the model response, the `second_prompt` is used to extract the information.
    - `levels` list contains the possible values that are matched against the 
    answers to the `second_prompt`.
    - The `wgh_col` (optional) is used to weight ground-truth samples.

- **Run the model evaluator**:
  - Use `py/evaluator.py` and the `evaluator()` function
  - Supporting helpers are in `py/evaluator_helpers.py`

- **Results**:
  - Results are saved as `.json` files in `data/results/benchmark/`
  - Each result file contains:
    - `condition`: Value v2 of the current level of variable V2 in the conditional
    expectation E[V1|V2=v2]
    - `true_vals`: True values of the variable V1|V2=v2 from the ground truth dataset.
    - `weights`: Weights that need to be used to obtain the correct V1|V2=v2
    - `model_vals`: Model values for the distribution V1|V2=v2.
    - `model_texts`: A sample of the model texts for the distribution V1|V2=v2.
    This is useful to see how we are decoding the model responses into numeric/categorical values.

---

### 3. Running the Benchmark Evaluation

- **Launch the Shiny app**:
  - After generating all JSON files in `data/benchmark`, run:

    ```r
    shiny::runApp('shiny')
    ```

- **App functionality**:
  - It provides a benchmark evaluation of the model responses, and gives a table
  specifying how each model performs on each task
  - Clickable cells show additional info (e.g., conditional means, raw model responses)

- **Evaluation code**:
  - Scoring functions: `r/eval.R`
  - Helpers: `r/eval-helpers.R`
  - All functions are documented

- **Scoring method**:
  - Let `n_mc` be the number of Monte Carlo samples
  - For each task, we can draw `n_mc` samples from the true distribution. 
  We can then see how far in terms of some distributional distance such samples 
  are from the true distribution given by `true_vals`. If we cannot statistically
  distinguish the model responses from the true distribution, then the model is
  given a score of 100 for this task.
  - For the lower scores, we say a model gets a score of 0 if the model responses
  cannot be distinguished from guessing the answer uniformly at random.
  - Scores between 0-100 are assigned assigned as a linear interpolation of the two extremes,
  based on the value of the distributional distance between `model_vals` and `true_vals`.
