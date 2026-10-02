# TabPFN SI/PI Experiment Pipeline

A Python project for generating synthetic coupled-microstrip design data, comparing regression models across training-set sizes, and plotting evaluation metrics.

## Source files

- `physics.py` — deterministic coupled-microstrip loss model.
- `01_generate_data.py` — generates the synthetic design dataset.
- `02_run_experiment.py` — runs the model comparison experiments.
- `03_plot_results.py` — creates plots and summary tables from experiment outputs.

## Setup

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
```

Run the pipeline from the project directory:

```bash
python 01_generate_data.py
python 02_run_experiment.py
python 03_plot_results.py
```

Generated data, results, figures, local environments, and manuscript files are excluded from version control.
