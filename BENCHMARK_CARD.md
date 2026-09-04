# LLM Epidemiology Benchmark: benchmark and data card

## Summary

The LLM Epidemiology Benchmark evaluates how closely a language model's
elicited answers match observational distributions estimated from ten public
US survey and administrative data sources. Version **1.0.0** contains 169
tasks: 75 low-dimensional conditional-distribution tasks and 94
high-dimensional tasks. This is a benchmark/data card; the repository does not
release a general-purpose trained model.

Benchmark v1.0.0 defines the paper's reported results, task definitions, prompt
templates, and survey years and data releases used to construct the ground
truth. The machine-readable source manifest is
[`benchmark_versions/v1.0.0.json`](benchmark_versions/v1.0.0.json).

## Intended uses

- Evaluate statistical fidelity of model outputs to the specified US
  population estimates under the benchmark's QA or likelihood elicitation
  protocol.
- Compare models under the same benchmark version, source-release specification, task
  set, and prompting mode.
- Study elicitation effects, temporal alignment, and performance as the number
  of conditioning variables increases.
- Extend the benchmark with explicitly versioned survey years, data releases,
  tasks, or model adapters.

The benchmark is not intended to diagnose individuals, estimate an
individual's risk, make eligibility or law-enforcement decisions, rank social
groups, attempt to re-identify or link source-data records, or substitute for
direct analysis of the source data. A high score is
descriptive agreement with a population statistic, not evidence that applying
group base rates to an individual is appropriate or fair. Scores measure a
joint outcome of model knowledge, source and year alignment, prompting,
calibration, retrieval, preprocessing, and sampling error; they do not isolate
an internal model representation.

## Data composition and provenance

| Source | Pinned year/release in v1.0.0 | Population/statistic | Weight or basis |
|---|---:|---|---|
| ACS | 2023 | US population in the ACS 1-year PUMS | `PWGTP` |
| NHANES | Aug. 2021-Aug. 2023 | US civilian noninstitutionalized adults aged 18+ covered by the released files | `mec_wgh` and `diet_wgh` retained; see weighting note below |
| BRFSS | 2023 | US adults covered by BRFSS | `_LLCPWT` |
| MEPS | 2022 | US civilian noninstitutionalized population | `PERWT22F` |
| NSDUH | 2023 | US civilian noninstitutionalized population aged 12+ | `ANALWT2_C` |
| SCF | 2022 | US families represented by the SCF | `wgt` |
| GSS | 2022 | US adults represented by the GSS cross-section | `wtssnrps`, stored as `wgh`; see weighting note below |
| IPEDS | 2022-23 | Degrees/certificates reported by Title IV degree-granting institutions | published counts |
| BLS CPS | 2023 | Annual-average employment by detailed occupation | published counts/percentages |
| FBI UCR | 2019 | Arrests reported in UCR Tables 42 and 43 | published counts/percentages |

The build scripts in `datasets/` record the transformations applied to these
releases: variable selection, category recoding, discretization of selected
continuous variables, and source-specific missing-data handling. Most builders
download their inputs; IPEDS and BLS use locally supplied spreadsheets. No new
personal data are collected.

For low-dimensional tasks, the current evaluator applies record weights when a
processed dataset exposes a column named `weight`; otherwise it uses equal
record weights. Consequently, the current GSS (`wgh`) and NHANES (`mec_wgh` and
`diet_wgh`) low-dimensional evaluations do **not** apply their retained survey
weights. The four high-dimensional sources (BRFSS, MEPS, NSDUH, and SCF) expose
a standardized `weight` column; their five-fold out-of-fold LightGBM fits and
evaluation distances use it. Bootstrap resampling represents finite-sample
uncertainty in the normalized score. This distinction should be considered
when interpreting population representativeness.

## Tasks, prompts, and scoring

Each task defines an outcome, conditioning variables, a natural-language prompt
template, and valid answers. Task code is under [`workspace/tasks/`](workspace/tasks/).
QA prompting estimates a distribution from next-token probabilities over
letter-labelled answers and averages over answer-order permutations (all
permutations up to 128, otherwise 128 sampled permutations). Likelihood
prompting selects among 22 probability levels. Ground truth and baselines are
mapped to the same likelihood grid before scoring.

The score is a normalized weighted L1 distance to the pinned ground truth. A
score of 100 falls within the benchmark's bootstrap uncertainty threshold; 0 is
set by the applicable no-knowledge baseline. Scores are meaningful only with
the benchmark version, task subset, model revision, and prompting mode.

## Models and evaluation scope

The harness supports Hugging Face causal language models and API adapters. The
paper evaluates open-weight pretrained and instruction-tuned models, plus a
restricted evaluation of closed models. Closed models are evaluated with
likelihood prompting on 63 of the 94 high-dimensional tasks because the APIs
used in the study did not expose the required next-token probabilities and
query budgets precluded the full sampling protocol. Those scores must not be
treated as a full-task, method-matched
comparison unless the same 63-task likelihood subset is used for all models.

Model providers can update mutable model aliases. Reproductions should record a
checkpoint revision or immutable model snapshot where available, the access
date for APIs, inference-library versions, and decoding parameters.

## Known limitations and responsible use

- The sources describe US populations and institutions; results should not be
  generalized to other countries or populations.
- Survey and administrative estimates reflect their sampling frames,
  nonresponse, measurement choices, category definitions, and publication
  dates. FBI arrest data measure reported arrests, not underlying criminality.


## Versioning and access

Published releases are immutable. Compatible additions of survey years or data
releases are MINOR releases; changes that break score comparability require a
MAJOR release. Each published release is retained as a Git tag and archive;
distributed processed artifacts receive a matching immutable Hugging Face
revision. Full rules are in [`VERSIONING.md`](VERSIONING.md).

When reporting a result, include: benchmark version; dataset year/release if
not the default; task subset; QA or likelihood prompting; model identifier and
revision; evaluation-code commit; and API access date when applicable.

## Licensing and citation

Software is Apache-2.0. Original tasks, prompt templates, group assignments,
version manifests, and documentation are CC BY 4.0. Source datasets retain
their providers' terms, and model weights/outputs retain model-provider terms.
See [`LICENSE.md`](LICENSE.md) for scope. Cite the accompanying paper and state
the benchmark version. The code, version manifests, and issue tracker are hosted
at [GitHub](https://github.com/dplecko/llm-epidemia); model evaluation is hosted
in the [Hugging Face Space](https://huggingface.co/spaces/llm-observatory/llm-observatory-eval),
and results are available on the [project site](https://llm-observatory.org/).
Questions and error reports can be filed through the GitHub issue tracker. The
authors accept responsibility for rights compliance for the
materials they distribute and confirm the license scope stated above.
