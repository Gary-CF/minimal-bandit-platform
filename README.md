# Minimal Bandit Platform

A minimal and reproducible experimental platform for studying stochastic multi-armed bandit algorithms.

The project provides:

- a Bernoulli bandit environment;
- several bandit algorithms with a common interface;
- JSON-based experiment configuration;
- reproducible multi-seed experiments;
- step-level CSV logging;
- mean cumulative regret visualization;
- a one-command reproduction script.

## Implemented Algorithms

The following algorithms are currently implemented:

- Random Policy
- UCB1
- Thompson Sampling

The default experiment compares:

- UCB1
- Thompson Sampling

## Environment

The environment is a Bernoulli multi-armed bandit.

Each arm \(i\) has an unknown reward probability \(\mu_i\). After selecting arm \(i\), the environment returns:

\[
X_{t,i} \sim \operatorname{Bernoulli}(\mu_i).
\]

The instantaneous pseudo-regret is:

\[
r_t = \mu^\star - \mu_{A_t},
\]

where \(\mu^\star\) is the largest arm mean and \(A_t\) is the arm selected at step \(t\).

The cumulative pseudo-regret is:

\[
R_T = \sum_{t=1}^{T} r_t.
\]

## Project Structure

```text
.
├── algorithms/
│   ├── base.py
│   ├── thompson_sampling.py
│   └── ucb1.py
├── configs/
│   └── basic.json
├── envs/
│   └── bernoulli_bandit.py
├── plots/
│   └── plot_regret.py
├── .gitignore
├── README.md
├── reproduce.sh
├── requirements.txt
└── run.py
```

The `results/` and `figures/` directories are generated automatically when the experiments are reproduced.

## Installation

Python 3.10 or later is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or WSL:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Reproduction

From the project root directory, run:

```bash
bash reproduce.sh
```

The script will:

1. remove old generated outputs;
2. run all configured experiments;
3. save step-level results as CSV files;
4. compute the mean cumulative regret across seeds;
5. save the comparison figure.

## Configuration

The default experiment is defined in:

```text
configs/basic.json
```

The default configuration is:

```json
{
  "arm_means": [0.3, 0.5, 0.7],
  "horizon": 5000,
  "algorithms": [
    "ucb1",
    "thompson_sampling"
  ],
  "seeds": [0, 1, 2, 3, 4]
}
```

The fields have the following meanings:

- `arm_means`: reward probabilities of the Bernoulli arms;
- `horizon`: number of interaction steps in each experiment;
- `algorithms`: algorithms included in the experiment;
- `seeds`: random seeds used for repeated experiments.

To run the experiments without the reproduction script:

```bash
python run.py --config configs/basic.json
```

To generate the regret figure separately:

```bash
python plots/plot_regret.py
```

## Output Files

Each algorithm and seed pair produces one CSV file:

```text
results/ucb1_seed_0.csv
results/ucb1_seed_1.csv
results/ucb1_seed_2.csv
results/ucb1_seed_3.csv
results/ucb1_seed_4.csv
results/thompson_sampling_seed_0.csv
results/thompson_sampling_seed_1.csv
results/thompson_sampling_seed_2.csv
results/thompson_sampling_seed_3.csv
results/thompson_sampling_seed_4.csv
```

Each CSV row contains:

```text
algorithm
seed
step
action
reward
instant_regret
cumulative_regret
```

The mean cumulative regret figure is saved to:

```text
figures/regret_curve.png
```

## Reproducibility

The root seed of each experiment is used to create independent random number generators for:

- the bandit environment;
- the algorithm.

Running the same configuration with the same code and dependencies should reproduce the same CSV files.

A simple reproducibility check is:

```bash
python run.py --config configs/basic.json
cp results/ucb1_seed_0.csv /tmp/ucb1_seed_0.csv

python run.py --config configs/basic.json
diff results/ucb1_seed_0.csv /tmp/ucb1_seed_0.csv
```

If `diff` prints nothing, the two files are identical.

## Preliminary Results

The generated figure compares the mean cumulative pseudo-regret of UCB1 and Thompson Sampling across five random seeds.

When reading the curves, observe:

1. whether cumulative regret continues to increase;
2. whether its growth rate gradually decreases;
3. which algorithm obtains lower regret under the current experimental configuration.

These finite experiments do not prove a theoretical regret bound. They provide only an empirical comparison under the environment, horizon, and seeds specified in `configs/basic.json`.

## Future Work

Possible extensions include:

- standard-deviation or confidence-interval shading;
- additional bandit environments;
- epsilon-greedy and KL-UCB algorithms;
- configurable input and output paths;
- automated unit tests;
- additional evaluation metrics;
- larger-scale experiment management.
