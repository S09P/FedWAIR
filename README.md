# FedWAIR

**Fair Federated Learning via Wasserstein–Fairness Regularization, Fairness-Aware Aggregation, and Multi-Round Pareto Selection**

FedWAIR is a fairness framework designed specifically for federated learning.
It treats demographic fairness as a property of the whole training pipeline
instead of a single post-hoc correction, and it acts at four connected stages:

1. **Data level:** each client balances its own label × protected-attribute
   subgroups with per-subgroup Wasserstein GANs.
2. **Optimization level:** local training adds a differentiable Wasserstein-1
   penalty, computed on the current mini-batch outputs of the two protected
   groups, to the standard cross-entropy loss.
3. **Aggregation level:** clients are weighted by accuracy and an
   exponentially smoothed (EMA) disparate-impact estimate, not by sample count.
4. **Model-selection level:** client and global snapshots from every round are
   archived, and the final model is chosen from the Pareto front over
   accuracy, balanced accuracy, and disparate impact.

This repository contains the implementation and the experiment driver used in
the empirical study, to support reproducibility and future research on fair
federated learning.

---

## Repository Contents

| Path | Description |
|---|---|
| `FedWAIR.ipynb` | Complete pipeline: WGAN balancing, local fairness-regularized training, fairness-aware aggregation, Pareto selection, and the main / sensitivity / ablation experiment driver |
| `preprocessed_data/` | Expected location of the preprocessed dataset CSVs (not tracked; see [Preparing the Data](#preparing-the-data)) |
| `results/` | Created on first run: experiment logs and ROC data (see [Outputs](#outputs)) |

The notebook has six cells, run top to bottom:

| Cell | Contents |
|---|---|
| 1 | Imports and device selection (GPU if available, otherwise CPU) |
| 2 | Configuration: datasets, paths, hyperparameters, seeds, experiment switches |
| 3 | `WGANGenerator`, `WGANCritic`, `train_wgan`, `generate_synthetic`, `GlobalMLP`, `soft_wasserstein_penalty`, `fairness_metrics`, `full_eval`, `is_pareto_nondominated` |
| 4 | `run_pipeline()`: one complete FedWAIR run for a given dataset, attribute, seed, and variant |
| 5 | Data loading, resume logic, and logging helpers |
| 6 | Master loop over all dataset–attribute configurations and experiment types |

---

## Study Setting

The study considers **horizontal federated learning** with `K = 10` clients
coordinated by a central server. Each client holds a different subset of
examples that share one feature space. Client partitions are generally non-IID
in their joint label–protected-attribute distribution. Raw data never leaves a
client; only model parameters and scalar fairness statistics are exchanged.

### Data split protocol

For every run (determined by the seed), the data is split once:

| Split | Share | Used for |
|---|---:|---|
| Train | 70% | Local client training and WGAN balancing (shuffled, then split into 10 contiguous client chunks) |
| Validation | 15% | Per-round evaluation, EMA fairness tracking, aggregation weights, candidate archiving, and model selection |
| Test | 15% | Final reporting of the selected model only |

The test split plays no role in training, aggregation, or model selection.

---

## Datasets

The experiments use **four public datasets** under **six
dataset–protected-attribute configurations**.

| Dataset | Task | Protected attribute(s) | Size | Features |
|---|---|---|---:|---:|
| COMPAS | Recidivism prediction | Sex, Race | 7,214 | 28 |
| Bank Marketing | Term-deposit subscription | Age | 45,211 | 16 |
| Default of Credit Card Clients | Payment-default prediction | Sex | 30,000 | 24 |
| Adult | Income prediction | Sex, Race | 48,842 | 14 |

The six configurations queued in the notebook are:

| # | Dataset name in code | Protected attribute | Expected CSV |
|---|---|---|---|
| 1 | `Compas` | `sex` | `compas_processed.csv` |
| 2 | `Compas` | `race` | `compas_processed.csv` |
| 3 | `Adult` | `race` | `adult_processed.csv` |
| 4 | `Adult` | `sex` | `adult_processed.csv` |
| 5 | `Bank` | `age` | `bank_processed.csv` |
| 6 | `Default` | `sex` | `default_processed.csv` |

### Original Dataset References

- **COMPAS** — ProPublica. *Data and analysis for "Machine Bias".*
  https://github.com/propublica/compas-analysis
- **Bank Marketing** — UCI Machine Learning Repository.
  https://archive.ics.uci.edu/ml/datasets/bank+marketing
- **Default of Credit Card Clients** — Yeh, I-C. UCI Machine Learning
  Repository. https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
- **Adult (Census Income)** — Kohavi, R., and Becker, B. UCI Machine Learning
  Repository. https://archive.ics.uci.edu/ml/datasets/Adult

---

## Preparing the Data

The notebook reads **preprocessed** CSV files. Raw files from the sources above
must be converted first. Each preprocessed file must satisfy the following
format.

| Requirement | Detail |
|---|---|
| File name | `<dataset name in lower case>_processed.csv`, inside `CSV_DIR` |
| Label column | Named `Probability` (change via `TARGET_COL_DEFAULT` or `TARGET_COL_OVERRIDES`). Binary, `1` = favorable outcome |
| Protected attribute column | Named exactly as in `protected_attributes` (`sex`, `race`, or `age`). Binary, `1` = privileged group, `0` = unprivileged group |
| Other columns | Numeric features only (`int64` / `float64`). Categorical variables must be encoded beforehand |
| Missing values | Must be handled before loading |

The loader (`load_dataset`) then applies min–max scaling to the numeric feature
columns, casts the label and protected attribute to integers, and returns the
frame with the label column name. The protected attribute is kept as a column
of the data but is **not** rescaled.


---

## Installation

Python 3.9 or newer is recommended. A CUDA GPU is optional but strongly
recommended because every run trains up to four WGANs per client.

```bash
git clone <repository-url>
cd FedWAIR
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch numpy pandas scikit-learn matplotlib jupyter
```

For an exact environment, install the versions you ran with and record them:

```bash
pip freeze > requirements.txt
```

---

## Quick Start

1. **Place the data.** Put the six preprocessed CSVs in `preprocessed_data/`
   (see [Preparing the Data](#preparing-the-data)).
2. **Set the paths** in Cell 2:

   ```python
   CSV_DIR     = r'./preprocessed_data'
   RESULTS_DIR = r'results'
   ```

   The notebook ships with an absolute path from the authors' machine. Replace
   it, or the loader will report `CSV not found` and skip each configuration.
3. **Launch Jupyter** and open the notebook:

   ```bash
   jupyter notebook FedWAIR.ipynb
   ```

4. **Run all cells in order.** Cell 6 runs every queued experiment and prints
   one line per completed run, for example:

   ```text
   === Compas_sex ===
     [main] seed=42  Acc=0.6420  1-DI=0.9750  AUC=0.6800
   ```

5. **Smoke test first.** Before a full run, set
   `RUN_SENSITIVITY = False`, `RUN_ABLATION = False`, and reduce
   `SEEDS_MAIN = [42]` and `NUM_ROUNDS_MAIN = 5`. Restore the original values
   for the full study.

---

## Pipeline Description

Everything in this section is implemented in `run_pipeline()` and the helper
functions of Cell 3.

| Stage | What it does | Code |
|---|---|---|
| **I. Local subgroup balancing** | For each client and each joint cell `(y, s)`, computes the deficit to the client's largest cell. Each deficient cell gets its own WGAN trained only on that cell; synthetic rows are sampled and appended with the cell's labels. Cells with fewer than 5 real rows use resampling with Gaussian noise (σ = 0.01, clipped to [0, 1]) instead. Runs once before round 1 | `train_wgan`, `generate_synthetic` |
| **II. Differentiable fairness regularization** | Local loss = BCE + `μ · max(0, Ŵ₁ − ε_fair)`, where `Ŵ₁` is a quantile-matched Wasserstein-1 distance between the model outputs of the two protected groups in the current mini-batch. A batch missing either group contributes zero penalty; batches with fewer than 2 rows are skipped (BatchNorm) | `soft_wasserstein_penalty` |
| **III. Fairness tracking and aggregation** | After local training, each client's DI is evaluated on the validation split and smoothed by an EMA (`λ`). Client score `Γ = Acc^α · exp(−β · \|DI_EMA\|)`, normalized to aggregation weights; the global model is the weighted average of client weights | `run_pipeline` (aggregation block) |
| **IV. Multi-round Pareto selection** | Each round archives two candidates: the client with the smallest raw `\|DI\|` and the new global model. After all rounds, three non-dominated sets are formed (Acc–DI, BAcc–DI, Acc–BAcc–DI), united, and the final model maximizes `Acc^α · BAcc^0.5 · exp(−β · \|DI\|)` | `is_pareto_nondominated`, `run_pipeline` (selection block) |

**Classifier.** `GlobalMLP`: `Linear(d, 64) → BatchNorm → ReLU → Dropout(0.2) →
Linear(64, 32) → ReLU → Linear(32, 1) → Sigmoid`. Predictions use a 0.5
threshold. Local training uses Adam (lr 0.001, batch size 32, 5 local epochs).

**WGAN.** Generator `latent(32) → 64 → 128 → d` with LeakyReLU(0.2) and a final
Sigmoid (features are min–max scaled to [0, 1]); critic `d → 128 → 64 → 1`.
RMSprop (lr 1e-4), 300 epochs, 5 critic steps per generator step, weight
clipping at 0.01.

---

## Configuration Reference

All settings are in Cell 2.

<details>
<summary><b>Federation and training</b></summary>

| Name | Default | Meaning |
|---|---:|---|
| `NUM_CLIENTS` | 10 | Number of clients |
| `LOCAL_EPOCHS` | 5 | Local epochs per round |
| `BATCH_SIZE` | 32 | Local mini-batch size |
| `LR` | 0.001 | Adam learning rate |
| `NUM_ROUNDS_MAIN` | 30 | Rounds for main and ablation runs |
| `NUM_ROUNDS_SWEEP` | 20 | Rounds for sensitivity runs |

The 70 / 15 / 15 split is fixed inside `run_pipeline()`. The `TEST_SIZE`
constant in Cell 2 is not used by it.

</details>

<details>
<summary><b>FedWAIR hyperparameters</b></summary>

| Name | Symbol | Default | Role |
|---|---|---:|---|
| `DEFAULT_ALPHA` | α | 1.5 | Accuracy exponent in aggregation and selection |
| `DEFAULT_BETA` | β | 3.0 | Fairness severity in aggregation and selection |
| `DEFAULT_EMA` | λ | 0.7 | EMA retention for DI smoothing |
| `DEFAULT_MU` | μ | 0.5 | Weight of the Wasserstein fairness penalty |
| `DEFAULT_EPS_FAIR` | ε_fair | 0.1 | Tolerance below which the penalty is inactive |
| `WASSERSTEIN_QUANTILES` | Q_quant | 20 | Quantile points in the Wasserstein surrogate |

The balanced-accuracy exponent γ = 0.5 is fixed in the selection step.

</details>

<details>
<summary><b>WGAN settings</b></summary>

| Name | Default |
|---|---:|
| `WGAN_EPOCHS` | 300 |
| `WGAN_LR` | 1e-4 |
| `N_CRITIC` | 5 |
| `CLIP_VALUE` | 0.01 |
| `WGAN_HIDDEN` | 64 |
| `WGAN_LATENT` | 32 |

</details>

<details>
<summary><b>Seeds, sweeps, and switches</b></summary>

| Name | Default |
|---|---|
| `SEEDS_MAIN` | `[42, 43, 44, 45, 46]` |
| `SEEDS_SWEEP` | `[42, 43, 44]` |
| `SEEDS_ABLATION` | `[42, 43, 44, 45, 46]` |
| `ALPHA_VALUES` | `[0.5, 1.0, 1.5, 2.0, 3.0]` |
| `BETA_VALUES` | `[0.5, 1.0, 2.0, 3.0, 5.0]` |
| `LAMBDA_VALUES` | `[0.3, 0.5, 0.7, 0.8, 0.9]` |
| `MU_VALUES` | `[0.1, 0.3, 0.5, 0.7, 1.0]` |
| `ABLATION_DATASET_IDX` | `[0, 1, 2, 3, 4, 5]` (indices into `dataset_names`) |
| `RUN_MAIN` / `RUN_SENSITIVITY` / `RUN_ABLATION` | `True` |
| `SAVE_ROC_DATA` | `True` |
| `SAVE_MODEL_WEIGHTS` | `False` (attaches the selected `state_dict` to the in-memory result; weights are not written to disk) |

</details>

---

## Experiments

Cell 6 loops over the six configurations and runs three experiment types.

| Experiment | What varies | Rounds | Seeds | Runs |
|---|---|---:|---|---:|
| **Main** | Default hyperparameters, variant `full` | 30 | 5 | 30 |
| **Sensitivity** | One of α, β, λ, μ over 5 values; the others stay at defaults | 20 | 3 | 360 |
| **Ablation** | 6 variants (below), default hyperparameters | 30 | 5 | 180 |
| | | | **Total** | **570** |

Run counts are per the default configuration (6 configurations × settings ×
seeds). Each run trains up to four WGANs per client, so a full study is
computationally heavy; use a GPU.

### Ablation variants

| `variant` | Behavior |
|---|---|
| `full` | Complete FedWAIR |
| `no_wgan` | Skips subgroup balancing; clients train on their raw partitions |
| `no_ema` | Uses the raw round-level DI instead of the EMA |
| `fedavg_agg` | Aggregates by client sample count (FedAvg-style weights) |
| `final_round_only` | Skips Pareto selection and returns the final-round global model |
| `no_fair_penalty` | Disables the Wasserstein penalty (plain cross-entropy locally) |

---

## Outputs

Everything is written to `RESULTS_DIR` (default `results/`).

### `master_log.csv`

One row per completed run.

| Column group | Columns |
|---|---|
| Run identity | `dataset`, `protected_attr`, `experiment_type` (`main` / `sensitivity` / `ablation`), `method`, `variant`, `param_name`, `param_value`, `seed` |
| Selection | `selected_round`, `selected_type` (`client` or `global`) |
| Primary metrics | `acc`, `bacc`, `di1` |



### `round_history_log.csv`

One row per communication round of every run, with the global model's metrics
(`global_acc`, `global_bacc`, `global_di1`, `global_spd`, `global_eod`,
`global_aod`, `global_auc`), the mean client `mean_client_di1`, and the round's
best-fairness client (`best_client_id`, `best_client_*`), together with the run
identity columns. These are per-round diagnostics, evaluated on the validation
split. They support the training-dynamics analysis.



### Disparate impact convention

The log column **`di1`** is the conventional disparate-impact **ratio**,
`P(ŷ=1 | s=0) / P(ŷ=1 | s=1)`, for which `1` means parity. The paper reports
the zero-centered form

```text
DI = 1 − P(ŷ=1 | s=0) / P(ŷ=1 | s=1) = 1 − di1
```

so `DI = 0` means demographic parity and lower `|DI|` is better. To reproduce
the paper's DI values from the log:

```python
import pandas as pd
log = pd.read_csv("results/master_log.csv")
log["DI"] = (1 - log["di1"]).abs()
```

---



## Metrics

| Metric | Definition |
|---|---|
| Accuracy (`acc`) | Fraction of correct predictions at threshold 0.5 |
| Balanced accuracy (`bacc`) | `(TPR + TNR) / 2` |
| Disparate impact | `di1` ratio as above; the paper's `DI = 1 − di1` |


---

## Reproducing the Paper

| Paper component | Source in this repository |
|---|---|
| FedWAIR results (comparison table) | `master_log.csv`, rows with `experiment_type == "main"` |
| Component ablation | `master_log.csv`, rows with `experiment_type == "ablation"`, grouped by `variant` |
| Training dynamics | `round_history_log.csv`, rows with `experiment_type == "main"` |
| Hyperparameter sensitivity | `master_log.csv`, rows with `experiment_type == "sensitivity"`, grouped by `param_name` and `param_value` |
| Candidate archive / Pareto figure | Candidate metrics are held in memory inside `run_pipeline()`; add logging there to export them |



### Baselines

The paper compares FedWAIR with FedAvg (reported separately as a fairness-agnostic reference) and six fairness-aware federated baselines: AgnosticFair, Ditto, FCFL, FairFed, FairTrade, and AFed-G. The `baselines/` folder contains the corresponding Jupyter notebooks (`.ipynb`) for all baseline implementations. Each baseline is executed using the same experimental configuration as FedWAIR, including client partitions, data splits, number of communication rounds, and classifier, ensuring a fair and consistent comparison. All methods are evaluated using the same metrics and demographic parity (DI) convention.





---

## Reproducibility Notes

- Seeds are set for Python, NumPy, and PyTorch at the start of every run, and
  each WGAN is seeded from the run seed, client index, and subgroup.
- `torch.backends.cudnn.benchmark = True` is enabled on GPU, which can make
  GPU runs non-deterministic. Results should agree with the reported ones
  within the seed-to-seed variation, but bit-exact reproduction is not
  guaranteed.
- The min–max scaler in `load_dataset` is fitted on the whole dataset before
  the train / validation / test split.

---



## Contact

Shivam Pathak and Sandeep Kumar, Indian Institute of Technology Roorkee, India.
