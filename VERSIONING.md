# Benchmark versioning

The manuscript results correspond to **LLM Epidemiology Benchmark v1.0.0**.
The survey years, data-release identifiers, source URLs, build entry points, and
results-code revision are recorded in
[`benchmark_versions/v1.0.0.json`](benchmark_versions/v1.0.0.json). A benchmark
version covers the selected data releases, preprocessing code, task definitions,
prompt templates, task and domain/group assignments, elicitation protocol, and
scoring code.

## Release identifiers

Releases use semantic versioning (`MAJOR.MINOR.PATCH`) and the Git tag
`benchmark-vMAJOR.MINOR.PATCH`.

- **MAJOR**: a change that breaks score comparability, such as changing an
  existing task, prompt, population, preprocessing rule, elicitation protocol,
  or scoring rule.
- **MINOR**: an additive, backward-compatible release, such as adding a newer
  survey year or data release, dataset, task, or model result while retaining
  the prior ones.
- **PATCH**: a correction that does not change benchmark scores, such as a
  documentation or metadata fix. Any correction that changes a score requires
  at least a MINOR release and is identified in that release's notes.

A result should report the benchmark version, dataset year/release, model
identifier and revision, prompting mode, and evaluation-code commit. Scores from different
MAJOR versions must not be placed on the same leaderboard without an explicit
compatibility analysis.

## Preservation and cross-year comparisons

Published releases are immutable. For every release, the maintainers create a
Git tag and release archive. When processed artifacts are distributed through
Hugging Face, they receive a matching immutable tag or commit revision. A new
survey year or data release is appended under a new MINOR release and never
overwrites an older release.

Raw source files are archived only when their provider permits redistribution.
For v1.0.0, the original experiment-time retrieval dates and checksums are not
available for every input. Its manifest therefore records the source URL and
release identifier, pins the results-code commit, and states this provenance
limitation rather than claiming artifact-level reproducibility. Beginning with
the next release, manifests will also record retrieval dates, raw-input
checksums where an input can be obtained, and checksums for distributed
processed artifacts.

Within a MAJOR series, existing task definitions and their ordering, category
definitions, preprocessing, prompts, elicitation, and scoring are held fixed
across survey years and data releases. New tasks are appended rather than
inserted into the existing ordering. A task is matched across years by its
dataset, outcome, conditioning variables, and any subset/range restriction. If
a source questionnaire or schema changes, the release notes either document a
harmonization that preserves the construct or treat it as a new task. A
non-comparable change triggers a MAJOR release. Cross-year tables must show the
dataset year/release separately and use only harmonized tasks.
