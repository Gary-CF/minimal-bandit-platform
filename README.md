# Minimal Bandit Platform

A small, transparent, and reproducible experimental platform for studying stochastic multi-armed bandit algorithms.

> **Status:** Bandit Platform **v1.0 release candidate**.
>
> Core implementation, automated regression tests, canonical benchmarks,
> benchmark figures, benchmark summaries, and end-to-end reproduction have
> passed. The remaining release work is repository hygiene,
> fresh-environment validation, and Git/GitHub release management.

---

## What the project provides

The current platform includes:

- Bernoulli and Gaussian bandit environments;
- Random Policy, UCB1, UCB-V, and Beta–Bernoulli Thompson Sampling;
- a common algorithm interaction interface;
- explicit environment–algorithm compatibility validation;
- JSON-based experiment configuration;
- single- and multi-horizon experiments;
- reproducible multi-seed execution;
- independent environment and algorithm RNG streams;
- step-level CSV logging;
- experiment-specific configuration snapshots;
- stale-result provenance protection;
- automated regression tests;
- canonical multi-seed benchmarks;
- benchmark figures;
- automatic benchmark summary generation;
- a one-command reproduction pipeline;
- a human-readable v1.0 benchmark report.

The project intentionally prioritizes transparency and correctness over framework size.

---

## Implemented algorithms

| Algorithm | Main idea | Current implementation scope |
|---|---|---|
| Random Policy | Uniform random action selection | Supported by both current environments |
| UCB1 | Empirical mean plus a Hoeffding-style confidence bonus | Current implementation is used with Bernoulli rewards |
| UCB-V | Empirical mean plus an empirical-Bernstein-style variance-aware bonus | Current implementation assumes bounded rewards and is used with Bernoulli rewards |
| Thompson Sampling | Posterior sampling | Current implementation uses a Beta–Bernoulli posterior and is Bernoulli-only |

The compatibility rules describe the scope of the **current code**, not the full theoretical scope of each algorithm family.

---

## Environments

### Bernoulli Bandit

Each arm \(i\) has a mean

\[
\mu_i\in[0,1].
\]

After selecting arm \(i\),

\[
X_{t,i}\sim\operatorname{Bernoulli}(\mu_i).
\]

The environment validates:

- a non-empty one-dimensional mean vector;
- finite arm means;
- arm means inside \([0,1]\);
- valid action indices.

### Gaussian Bandit

Each arm \(i\) has a configured mean \(\mu_i\) and standard deviation \(\sigma_i\):

\[
X_{t,i}\sim\mathcal N(\mu_i,\sigma_i^2).
\]

The Gaussian environment is useful for separating environment abstraction from the bounded-reward assumptions made by some current algorithms.

It validates:

- compatible mean/std shapes;
- non-empty arm definitions;
- positive standard deviations;
- valid action indices.

The existence of `GaussianBandit` does **not** imply that every current algorithm implementation supports Gaussian rewards.

---

## Current compatibility boundary

| Environment | Random | UCB1 | UCB-V | Thompson Sampling |
|---|---:|---:|---:|---:|
| Bernoulli | ✓ | ✓ | ✓ | ✓ |
| Gaussian | ✓ | ✗ | ✗ | ✗ |

This matrix is intentionally conservative.

For example, UCB algorithms and Thompson Sampling have Gaussian variants in the literature, but those variants are not implemented by the current classes.

Unsupported combinations fail early instead of silently running under invalid assumptions.

---

## Regret

The platform reports **pseudo-regret**.

Let

\[
\mu^\star=\max_i\mu_i
\]

and let \(A_t\) be the action selected at step \(t\).

Instantaneous pseudo-regret is

\[
r_t=\mu^\star-\mu_{A_t}.
\]

Cumulative pseudo-regret is

\[
R_T=\sum_{t=1}^{T}r_t.
\]

Pseudo-regret depends on the true arm means and selected actions rather than on the realized reward noise.

---

## Project structure

```text
.
├── algorithms/
│   ├── base.py
│   ├── thompson_sampling.py
│   ├── ucb1.py
│   └── ucb_v.py
├── configs/
│   ├── basic.json
│   ├── config_system_smoke.json
│   ├── different_horizon.json
│   ├── easy_gap.json
│   ├── hard_gap.json
│   └── ucb_v_smoke.json
├── envs/
│   ├── bernoulli_bandit.py
│   └── gaussian_bandit.py
├── plots/
│   └── plot_benchmarks.py
├── reports/
│   ├── benchmark_summary.csv
│   └── benchmark_v1_0.md
├── scripts/
│   └── summarize_benchmarks.py
├── tests/
│   ├── test_compatibility.py
│   ├── test_config.py
│   ├── test_environment.py
│   ├── test_gaussian_environment.py
│   ├── test_runner.py
│   ├── test_thompson.py
│   ├── test_ucb.py
│   └── test_ucb_v.py
├── CONTEXT.md
├── PROGRESS.md
├── README.md
├── reproduce.sh
├── requirements.txt
└── run.py
```

Generated local outputs include:

```text
results/
figures/
```

These directories, Python caches, virtual environments, and pytest caches should not be committed.

`reports/benchmark_summary.csv` is generated automatically from canonical benchmark results and is retained as release evidence.

---

## Installation

Python 3.10 or later is recommended.

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Current dependencies are intentionally minimal:

```text
numpy
matplotlib
pytest
```

---

## Running the automated tests

The canonical test command is:

```bash
python -m pytest -q
```

All tests collected by this command must pass before release.

The automated suite currently covers:

- Bernoulli environment validation;
- Gaussian environment validation;
- action-boundary regressions;
- UCB1 behavior;
- UCB-V initialization and statistics;
- Thompson Sampling;
- short-horizon runner behavior;
- configuration normalization;
- experiment provenance protection;
- environment–algorithm compatibility.

Use the command from the project root.

---

## Configuration

Experiments are defined by JSON configuration files.

Example:

```json
{
  "experiment_name": "easy_gap",
  "environment": {
    "name": "bernoulli",
    "arm_means": [0.2, 0.5, 0.7]
  },
  "algorithms": [
    {
      "name": "random",
      "parameters": {}
    },
    {
      "name": "ucb1",
      "parameters": {}
    },
    {
      "name": "ucb_v",
      "parameters": {
        "reward_range": 1.0
      }
    },
    {
      "name": "thompson_sampling",
      "parameters": {}
    }
  ],
  "horizon": 5000,
  "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}
```

The runner also accepts multiple horizons:

```json
"horizons": [1000, 5000, 20000]
```

instead of a single:

```json
"horizon": 5000
```

For backward compatibility, algorithms may be written as strings:

```json
"algorithms": [
  "ucb1",
  "thompson_sampling"
]
```

or as dictionaries containing parameters:

```json
"algorithms": [
  {
    "name": "ucb_v",
    "parameters": {
      "reward_range": 1.0
    }
  }
]
```

After normalization, the runner internally treats algorithms as structured dictionaries.

---

## Experiment identity and provenance

Each experiment writes into:

```text
results/<experiment_name>/
```

and records the normalized configuration in:

```text
results/<experiment_name>/config_snapshot.json
```

The snapshot acts as the configuration identity of that result directory.

Before writing results, the runner checks:

- if the result directory does not exist: allow the run;
- if the result directory is empty: allow the run;
- if an existing snapshot matches the current normalized configuration: allow the run;
- if an existing snapshot differs from the current configuration: fail;
- if the directory contains files but has no snapshot: fail.

This prevents stale CSV files from different experiment definitions from being silently mixed together.

The current mechanism protects **configuration provenance**.

It does not attempt to store complete code provenance such as Git commit hashes inside each result directory. The release process instead fixes a Git revision and regenerates the canonical benchmarks from clean generated outputs.

---

## Current experiment configurations

### Development and smoke configurations

- `configs/basic.json`: general development/demo configuration;
- `configs/config_system_smoke.json`: small config-system smoke test;
- `configs/ucb_v_smoke.json`: small UCB-V smoke test.

### Canonical v1.0 benchmarks

- `configs/easy_gap.json`;
- `configs/hard_gap.json`;
- `configs/different_horizon.json`.

The JSON configuration files are the source of truth for exact experiment parameters.

---

## Canonical v1.0 benchmarks

### Easy-gap

```text
environment = Bernoulli
arm_means = [0.20, 0.50, 0.70]
horizon = 5000
num_seeds = 10
```

Purpose:

- verify that learning algorithms concentrate on a clearly superior arm;
- inspect finite-time exploration;
- sanity-check Random regret analytically.

### Hard-gap

```text
environment = Bernoulli
arm_means = [0.45, 0.48, 0.50]
horizon = 10000
num_seeds = 10
```

Purpose:

- create a statistically harder arm-identification problem;
- inspect sustained exploration;
- inspect seed variability.

### Different-horizon

```text
environment = Bernoulli
arm_means = [0.45, 0.48, 0.50]
horizons = [1000, 5000, 20000]
num_seeds = 10
```

Purpose:

- compare final regret as \(T\) increases;
- contrast Random's approximately linear empirical growth with learning algorithms;
- check qualitative consistency with regret theory without claiming an asymptotic proof.

---

## Running an experiment

Run one configuration:

```bash
python run.py --config configs/easy_gap.json
```

Smoke-test examples:

```bash
python run.py --config configs/config_system_smoke.json
python run.py --config configs/ucb_v_smoke.json
```

---

## Output layout

Each experiment saves a normalized configuration snapshot:

```text
results/<experiment_name>/config_snapshot.json
```

and one CSV per environment–algorithm–horizon–seed combination:

```text
results/<experiment_name>/<environment>_<algorithm>_T<horizon>_seed<seed>.csv
```

Each CSV row contains:

```text
environment
algorithm
horizon
seed
step
action
reward
instant_regret
cumulative_regret
```

Canonical plotting produces:

```text
figures/easy_gap_regret.png
figures/hard_gap_regret.png
figures/horizon_comparison.png
figures/action_frequency.png
```

Canonical summary statistics are written to:

```text
reports/benchmark_summary.csv
```

---

## Reproducing the v1.0 benchmarks

From the project root:

```bash
bash reproduce.sh
```

The reproduction pipeline:

1. removes old generated `results/` and `figures/`;
2. runs the automated test suite;
3. runs the easy-gap benchmark;
4. runs the hard-gap benchmark;
5. runs the different-horizon benchmark;
6. generates benchmark figures;
7. generates `reports/benchmark_summary.csv`;
8. verifies expected snapshots, CSV counts, figures, and summary output.

A successful canonical run produces:

```text
Easy-gap runs:       40
Hard-gap runs:       40
Horizon runs:        120
```

and ends with:

```text
Reproduction completed successfully.
```

This script is the main end-to-end release contract for the current repository.

---

## Randomness and reproducibility

A root experiment seed is split using NumPy `SeedSequence`.

Conceptually:

```python
seed_sequence = np.random.SeedSequence(seed)
env_seed, algorithm_seed = seed_sequence.spawn(2)

env_rng = np.random.default_rng(env_seed)
algorithm_rng = np.random.default_rng(algorithm_seed)
```

This keeps:

- environment reward noise;
- algorithm-internal randomness

on separate RNG streams.

Therefore, adding algorithm-side random sampling does not directly consume the environment RNG sequence.

Exact CSV reproducibility assumes the same:

- code;
- normalized configuration;
- dependency behavior;
- random-call order.

---

## Benchmark summary and analysis

Machine-readable summary:

```text
reports/benchmark_summary.csv
```

Human-readable analysis:

```text
reports/benchmark_v1_0.md
```

The benchmark report is the canonical location for:

- experiment design;
- final-regret statistics;
- Random-policy sanity checks;
- hard-gap interpretation;
- UCB exploration analysis;
- UCB-V finite-time behavior;
- Thompson Sampling seed variability;
- different-horizon interpretation;
- anomaly audit;
- limitations.

Benchmark numbers should not be duplicated into project-management documents unless necessary, because duplicated numerical tables easily become stale when experiments are regenerated.

---

## Current qualitative findings

The v1.0 benchmarks support the following limited conclusions:

- Random Policy exhibits approximately linear cumulative pseudo-regret.
- Smaller arm gaps make identification substantially harder.
- UCB exploration is driven by confidence bonuses rather than explicit random exploration.
- UCB-V can benefit from variance information, but its finite-sample correction may offset that advantage.
- Thompson Sampling shows strong finite-time behavior in the current Bernoulli experiments and can exhibit environment-dependent seed variability.
- In the tested horizons, learning algorithms grow substantially more slowly in regret than Random.
- The finite benchmark does not prove a specific asymptotic regret bound.
- The finite benchmark does not establish a universal ranking among algorithms.

---

## Release status

Completed:

- core Bernoulli and Gaussian environment implementations;
- Random, UCB1, UCB-V, and Beta–Bernoulli Thompson Sampling;
- environment and action-boundary validation;
- automated regression tests;
- environment–algorithm compatibility tests;
- JSON config normalization;
- experiment provenance protection;
- canonical multi-seed benchmarks;
- canonical benchmark figures;
- automatic benchmark summary generation;
- benchmark report synchronization;
- end-to-end reproduction from clean generated outputs.

Remaining before the final `v1.0` tag:

- final documentation and repository hygiene audit;
- fresh virtual-environment installation and reproduction;
- final Git diff/status audit;
- merge of the release branch;
- annotated `v1.0` tag;
- push to GitHub;
- GitHub Release.

---

## Scope and limitations

The v1.0 target is a focused undergraduate research-training platform, not a general industrial framework.

It intentionally does not include:

- contextual bandits;
- linear bandits;
- adversarial bandits;
- non-stationary bandits;
- neural bandits;
- distributed experiment execution;
- database-backed logging;
- dashboards;
- MLflow or Weights & Biases;
- large hyperparameter sweeps;
- checkpoints and failure recovery;
- paper-scale benchmark coverage.

The goal is to make every current design decision, update rule, experiment, and regret curve understandable and reproducible before expanding the project further.