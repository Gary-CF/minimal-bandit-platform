# Minimal Bandit Platform

A small, transparent, and reproducible experimental platform for studying stochastic multi-armed bandit algorithms.

> **Status:** Bandit Platform **v1.0 is under final validation**.
> The earlier v0.1 milestone completed the minimal Bernoulli experiment loop. The current v1.0 development stage adds a second environment, a variance-aware algorithm, tests, standard benchmarks, compatibility checks, and an experiment report.

## What the project currently provides

- Bernoulli and Gaussian bandit environments;
- Random Policy, UCB1, UCB-V, and Bernoulli Thompson Sampling;
- a common algorithm interaction interface;
- JSON-based experiment configuration;
- single- or multi-horizon experiments;
- reproducible multi-seed execution;
- explicit environment–algorithm compatibility checks;
- step-level CSV logging;
- regret aggregation and benchmark figures;
- a minimal automated test suite;
- smoke-test configurations;
- a one-command reproduction entry point;
- a human-readable v1.0 benchmark report.

## Implemented algorithms

| Algorithm | Main idea | Current implementation scope |
|---|---|---|
| Random Policy | Uniform random action selection | Baseline for supported environments |
| UCB1 | Empirical mean plus a Hoeffding/sub-Gaussian confidence bonus | Use only with configurations accepted by the compatibility checker |
| UCB-V | Empirical mean plus an empirical-Bernstein-style, variance-aware bonus | Current implementation assumes bounded rewards and is used with Bernoulli rewards |
| Thompson Sampling | Beta posterior sampling | Current implementation is Beta–Bernoulli and therefore Bernoulli-only |

The compatibility checker describes the scope of the **current code**, not the full theoretical scope of each algorithm family.

## Environments

### Bernoulli Bandit

Each arm \(i\) has an unknown mean \(\mu_i \in [0,1]\). After selecting arm \(i\),

\[
X_{t,i} \sim \operatorname{Bernoulli}(\mu_i).
\]

### Gaussian Bandit

Each arm \(i\) is configured with a mean \(\mu_i\) and standard deviation \(\sigma_i\):

\[
X_{t,i} \sim \mathcal{N}(\mu_i,\sigma_i^2).
\]

The Gaussian environment is useful for separating bounded-reward assumptions from sub-Gaussian noise assumptions. Not every current algorithm implementation is compatible with it.

## Regret

The instantaneous pseudo-regret is

\[
r_t = \mu^\star-\mu_{A_t},
\]

where \(\mu^\star=\max_i\mu_i\) and \(A_t\) is the selected arm.

The cumulative pseudo-regret is

\[
R_T=\sum_{t=1}^{T}r_t.
\]

Pseudo-regret depends on the true arm means and the selected action, rather than on the realized reward noise.

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
├── figures/
├── plots/
├── reports/
│   └── benchmark_v1_0.md
├── results/
├── tests/
│   ├── test_environment.py
│   ├── test_thompson.py
│   └── test_ucb.py
├── check_compatibility.py
├── check_ucb_v.py
├── CONTEXT.md
├── PROGRESS.md
├── README.md
├── reproduce.sh
├── requirements.txt
└── run.py
```

`results/`, `figures/`, Python caches, the virtual environment, and pytest caches are generated locally and should not be committed unless a representative artifact is deliberately copied into a documentation asset directory.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The test suite requires `pytest`. During final v1.0 validation, make sure it is included in `requirements.txt`; until then it can be installed explicitly with:

```bash
python -m pip install pytest
```

## Canonical test command

Run tests from the project root with:

```bash
python -m pytest -q
```

This is the canonical command for the current repository. Directly invoking `pytest -q` may use a launcher whose import path does not include the project root, causing collection-time errors such as `No module named 'algorithms'` or `No module named 'envs'`. Those errors concern test-runner invocation, not the algorithm implementations.

At the current v1.0 development checkpoint, all tests collected by `python -m pytest -q` pass.

## Configuration

The v1.0 configuration system separates the experiment name, environment, algorithms, horizon or horizons, and seeds.

Example:

```json
{
  "experiment_name": "easy_gap",
  "environment": {
    "name": "bernoulli",
    "arm_means": [0.02, 0.05, 0.95]
  },
  "algorithms": [
    {"name": "random", "parameters": {}},
    {"name": "ucb1", "parameters": {}},
    {"name": "ucb_v", "parameters": {"reward_range": 1.0}},
    {"name": "thompson_sampling", "parameters": {}}
  ],
  "horizon": 5000,
  "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}
```

The runner also supports:

```json
"horizons": [500, 1000, 2000, 5000, 10000]
```

instead of a single `horizon`.

For backward compatibility, an algorithm may be written as a string:

```json
"algorithms": ["ucb1", "thompson_sampling"]
```

or as a dictionary with parameters:

```json
"algorithms": [
  {"name": "ucb_v", "parameters": {"reward_range": 1.0}}
]
```

## Current experiment configurations

- `configs/basic.json`: general development configuration;
- `configs/config_system_smoke.json`: small configuration-system smoke test;
- `configs/ucb_v_smoke.json`: small UCB-V smoke test;
- `configs/easy_gap.json`: clearly separated Bernoulli arms;
- `configs/hard_gap.json`: closely spaced Bernoulli arms;
- `configs/different_horizon.json`: final regret across multiple horizons.

The JSON files are the source of truth for exact local parameters.

## Running experiments

Run one configuration:

```bash
python run.py --config configs/easy_gap.json
```

Run the configuration-system smoke test:

```bash
python run.py --config configs/config_system_smoke.json
```

Run the UCB-V smoke test:

```bash
python run.py --config configs/ucb_v_smoke.json
```

Run the project reproduction entry point:

```bash
bash reproduce.sh
```

The reproduction script should be treated as part of the final release contract and must be rechecked from a clean output directory before the v1.0 tag is created.

## Output layout

A normalized run saves a configuration snapshot and step-level CSV files under an experiment-specific directory:

```text
results/<experiment_name>/config_snapshot.json
results/<experiment_name>/<environment>_<algorithm>_T<horizon>_seed<seed>.csv
```

Each CSV row records:

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

The exact figure filenames depend on the plotting or benchmark script used. Standard development outputs include easy-gap, hard-gap, horizon-comparison, action-frequency, and low-noise Gaussian figures.

## Reproducibility

Each root seed is split into independent random-number generators for:

- the environment;
- the algorithm.

The project uses NumPy's `Generator` API and `SeedSequence` rather than global random state. The same code, configuration, dependency versions, and random-call order should reproduce the same CSV output.

A simple check is:

```bash
python run.py --config configs/config_system_smoke.json
cp results/<experiment_name>/<one_file>.csv /tmp/first.csv

python run.py --config configs/config_system_smoke.json
diff results/<experiment_name>/<one_file>.csv /tmp/first.csv
```

No `diff` output means the files are identical.

## Current benchmark findings

The v1.0 benchmark work currently supports the following qualitative conclusions:

- Random Policy has approximately linear cumulative pseudo-regret because its per-step expected regret is constant.
- Easy-gap environments are statistically easy to identify, while hard-gap environments require many more observations.
- UCB methods explore strongly when an arm has been selected only a few times.
- UCB-V may be conservative at short horizons because its finite-sample range correction can dominate its variance advantage.
- Thompson Sampling shows strong finite-time performance in the current Bernoulli experiments, but its relative seed variability depends on the environment.
- The experiments are consistent with sublinear learning-algorithm regret, but they do not prove a specific asymptotic regret bound or a universal algorithm ranking.

See:

```text
reports/benchmark_v1_0.md
```

## Known development issues and release status

Completed validation:

- `python -m pytest -q`: 13 tests passed;
- `config_system_smoke.json`: 8 experiments completed successfully;
- `ucb_v_smoke.json`: completed successfully.

The following items still remain before the v1.0 release is final:

- complete a clean `reproduce.sh` run;
- ensure `pytest` is recorded as a dependency;
- complete and proofread the benchmark report and README;
- remove temporary files and inspect `git status`;
- create clear Git commits;
- create the annotated `v1.0` tag only after the working tree and release checks are clean.

## Scope and limitations

The v1.0 target is a focused undergraduate research-training platform, not a general industrial framework. It intentionally does not include:

- contextual, linear, adversarial, or non-stationary bandits;
- distributed experiment execution;
- dashboards, databases, MLflow, or Weights & Biases;
- large hyperparameter sweeps;
- checkpoints and failure recovery;
- paper-scale benchmark coverage.

The goal is to make every current design decision, update rule, experiment, and regret curve understandable and reproducible.
