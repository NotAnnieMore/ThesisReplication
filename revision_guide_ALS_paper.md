# Revision Guide — AUC–Sensitivity Decoupling in Imbalanced ALS Prognosis
### What to add and how to add it to reach publication standard at *Artificial Intelligence in Medicine*

> **Context:** This guide is based on peer review analysis of the submitted manuscript. The paper has genuine contributions and strong methodology, but four gaps need to be closed before it can be accepted. Each section below explains *why* the addition matters, *exactly what* to implement, and *how* to do it in Python.

---

## Priority ranking

| # | Gap | Severity | Effort |
|---|-----|----------|--------|
| 1 | Threshold optimisation | Critical | Medium |
| 2 | Bootstrap confidence intervals | Critical | Low |
| 3 | Missing data quantification | Moderate | Low |
| 4 | Calibration analysis | Supplementary | Medium |

---

## 1. Threshold Optimisation (Critical)

### Why this matters

The paper's central finding is that the MLP collapses to sensitivity 0.029 at the default 0.5 decision threshold despite an AUC of 0.918. A reviewer will immediately ask: *"Did you try a different threshold?"* Without answering this, the paper proves a symptom but not a diagnosis. You need to show what happens when the threshold is optimised — and crucially, whether the MLP recovers or remains inferior to LightGBM.

The expected narrative after adding this is: *"Even with an optimal threshold, the MLP reaches sensitivity X and specificity Y, whereas LightGBM+ROS at its optimal threshold achieves sensitivity 0.857 and balanced accuracy 0.844, demonstrating that the decoupling is not merely a calibration artefact but reflects a fundamental difference in minority-class probability mass."*

### What to add

Three thresholds must be reported for each classifier:

- **Youden threshold**: maximises `sensitivity + specificity − 1`. This is the standard clinical threshold for binary classifiers.
- **Sensitivity-optimised threshold**: the lowest threshold that achieves sensitivity ≥ 0.90 (you choose the target; 0.90 is a reasonable clinical floor for a fatal disease).
- **F1-optimal threshold**: the threshold that maximises the F1 score on the minority class. This is the PR-curve equivalent.

**Critical methodological rule:** All three thresholds must be derived from the **validation folds** (or a separate validation partition), never from the held-out test set. If you select the threshold on the test set and then report test performance at that threshold, you have optimised on evaluation data and the reported metrics are no longer unbiased. The correct procedure is:

1. During cross-validation, compute the threshold on the out-of-fold predictions.
2. Average the threshold across folds to get a single value.
3. Apply that value to the held-out test set.

### How to implement it

```python
import numpy as np
from sklearn.metrics import roc_curve, f1_score

def find_youden_threshold(y_true, y_proba):
    """Compute Youden's J threshold from validation predictions."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx]

def find_sensitivity_threshold(y_true, y_proba, min_sensitivity=0.90):
    """Lowest threshold achieving at least min_sensitivity."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    # tpr = sensitivity; find where sensitivity >= target
    valid = np.where(tpr >= min_sensitivity)[0]
    if len(valid) == 0:
        return thresholds[-1]  # best available
    return thresholds[valid[-1]]  # highest threshold meeting criterion

def find_f1_threshold(y_true, y_proba):
    """Threshold maximising F1 on the minority class."""
    thresholds = np.linspace(0.01, 0.99, 200)
    f1_scores = [f1_score(y_true, (y_proba >= t).astype(int), pos_label=1, zero_division=0)
                 for t in thresholds]
    return thresholds[np.argmax(f1_scores)]

# --- Cross-validation threshold derivation ---
# Collect out-of-fold probabilities during your existing CV loop
# Then:
oof_proba = np.concatenate(oof_probas)   # all out-of-fold predictions
oof_true  = np.concatenate(oof_trues)

youden_thresh   = find_youden_threshold(oof_true, oof_proba)
sens_thresh     = find_sensitivity_threshold(oof_true, oof_proba, min_sensitivity=0.90)
f1_thresh       = find_f1_threshold(oof_true, oof_proba)

# --- Apply to held-out test set ---
for thresh, label in [(0.5, "Default"), (youden_thresh, "Youden"),
                      (sens_thresh, "Sensitivity-opt"), (f1_thresh, "F1-opt")]:
    y_pred = (test_proba >= thresh).astype(int)
    # compute and report sensitivity, specificity, balanced accuracy
```

### What to report

Add a new table (Table X) to the Results section showing, for the MLP and LightGBM+ROS, all four thresholds with their corresponding sensitivity, specificity, and balanced accuracy. A short paragraph should then state whether the MLP recovers adequately or remains inferior. This directly answers the reviewer's question and either strengthens or nuances the core thesis.

---

## 2. Bootstrap Confidence Intervals (Critical)

### Why this matters

The held-out test set contains only 35 Short Survivors. One misclassification shifts sensitivity by 2.86 percentage points (1/35). The paper presents LightGBM+ROS (sensitivity 0.857, 30/35) and SVM+SMOTEENN (sensitivity 0.886, 31/35) as if they are meaningfully different. They are not — the difference is a single patient. Without confidence intervals, readers and reviewers cannot tell which differences are real and which are noise. This is the easiest fix in the paper and the most damaging omission to leave open.

### What to add

Bootstrap 95% percentile confidence intervals for every metric in Table 3 (the held-out test results): AUC, balanced accuracy, sensitivity, specificity, and G-Mean.

**Important technical note on bootstrapping small minority classes:** With only 35 positive cases, some bootstrap samples will contain very few or even zero Short Survivors. You must define a handling rule and state it explicitly:
- If a bootstrap sample contains zero Short Survivors, sensitivity is undefined. Exclude that sample from the sensitivity CI calculation and report the number of such samples.
- Use the **percentile interval** (2.5th and 97.5th percentile of the bootstrap distribution), not the normal approximation interval. The normal approximation assumes a symmetric distribution, which does not hold for sensitivity under severe imbalance.

### How to implement it

```python
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.metrics import recall_score, confusion_matrix

def bootstrap_ci(y_true, y_proba, y_pred, metric_fn, n_bootstrap=1000, ci=0.95,
                 random_state=42):
    """
    Compute bootstrap percentile CI for a given metric.
    metric_fn: callable(y_true, y_proba, y_pred) -> float
    Returns (lower, upper, n_valid_samples)
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    scores = []
    n_excluded = 0

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_proba[idx]
        yc = y_pred[idx]

        # Skip samples with no positive cases (sensitivity undefined)
        if yt.sum() == 0:
            n_excluded += 1
            continue

        try:
            score = metric_fn(yt, yp, yc)
            scores.append(score)
        except Exception:
            n_excluded += 1
            continue

    alpha = (1 - ci) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)
    return lower, upper, n_excluded

# Define individual metric functions
def metric_auc(yt, yp, yc):
    return roc_auc_score(yt, yp)

def metric_sensitivity(yt, yp, yc):
    return recall_score(yt, yc, pos_label=1, zero_division=np.nan)

def metric_specificity(yt, yp, yc):
    tn, fp, fn, tp = confusion_matrix(yt, yc).ravel()
    return tn / (tn + fp)

def metric_balanced_acc(yt, yp, yc):
    return balanced_accuracy_score(yt, yc)

def metric_gmean(yt, yp, yc):
    sens = metric_sensitivity(yt, yp, yc)
    spec = metric_specificity(yt, yp, yc)
    return np.sqrt(sens * spec)

# --- Usage ---
y_true  = test_labels          # numpy array
y_proba = test_probabilities   # numpy array, minority class proba
y_pred  = test_predictions     # numpy array, binary

for label, fn in [("AUC", metric_auc), ("Sensitivity", metric_sensitivity),
                  ("Specificity", metric_specificity),
                  ("Balanced Acc", metric_balanced_acc), ("G-Mean", metric_gmean)]:
    lo, hi, excl = bootstrap_ci(y_true, y_proba, y_pred, fn, n_bootstrap=2000)
    print(f"{label}: 95% CI [{lo:.3f}, {hi:.3f}]  (excluded samples: {excl})")
```

### What to report

Update Table 3 to include a `95% CI` column for each metric, or add a companion table. In the Methods section (Evaluation Metrics paragraph), add one sentence explaining that bootstrap CIs were computed using 2,000 resamples with the percentile method, and that bootstrap samples containing no Short Survivors were excluded from sensitivity CI calculations. Report the number of excluded samples in a footnote.

---

## 3. Temporal Leakage Quantification (Moderate)

### Why this matters

Section 3.2 states that where no pre-diagnosis ALSFRS-R assessment was available, the nearest post-diagnosis visit was used to compute functional decline slopes. This is a limitation shared with the reference study, but it needs to be quantified rather than noted vaguely. Reviewers will ask how many patients are affected. If it is under 10%, it is an acceptable limitation. If it is over 30%, it significantly undermines the prognostic framing.

### What to add

A single count and percentage in the Methods or Dataset Overview section:

> *"Post-diagnosis assessments were used as proxies for X patients (Y%), of whom Z% were Short Survivors and W% were Standard Survival patients."*

If the proportion is high, you should also run a sensitivity analysis: repeat the held-out evaluation excluding the affected patients and report whether the results change materially. If they do not change, the leakage is not driving the results. If they do, it must be flagged as a major limitation.

### How to implement it

```python
# During feature engineering, log which patients required a post-diagnosis visit
post_diag_flag = []

for patient_id, visits in patient_visits.items():
    diagnosis_date = get_diagnosis_date(patient_id)
    pre_diag = [v for v in visits if v.date <= diagnosis_date]
    if len(pre_diag) == 0:
        post_diag_flag.append(patient_id)

n_affected = len(post_diag_flag)
pct_affected = n_affected / total_patients * 100
print(f"Patients using post-diagnosis visit: {n_affected} ({pct_affected:.1f}%)")

# Breakdown by class
short_survivor_ids = set(labels[labels == 1].index)
n_ss_affected = sum(1 for p in post_diag_flag if p in short_survivor_ids)
print(f"  of which Short Survivors: {n_ss_affected} ({n_ss_affected/len(short_survivor_ids)*100:.1f}%)")
```

---

## 4. Calibration Analysis (Supplementary)

### Why this matters

Calibration analysis measures whether a model's predicted probabilities match observed event frequencies — for example, does a predicted probability of 0.7 correspond to 70% of patients actually being Short Survivors? This is distinct from the AUC–sensitivity decoupling argument, so be careful not to conflate the two. Calibration analysis belongs in the paper as supporting evidence for model trustworthiness in clinical deployment, not as direct proof of the main thesis.

### What to add

For the two primary models (LightGBM+ROS and MLP), add:

- **Calibration curve** (reliability diagram): plot mean predicted probability vs. observed fraction of positives in each probability bin.
- **Brier score**: a proper scoring rule measuring overall probabilistic accuracy. Lower is better; a random classifier on an 11.6% prevalence dataset has a Brier score of approximately 0.103.
- **Expected Calibration Error (ECE)**: the weighted mean absolute difference between predicted probability and observed frequency across bins.

### How to implement it

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def calibration_analysis(y_true, y_proba, model_name, n_bins=10):
    """
    Plot calibration curve and compute Brier score and ECE.
    """
    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='uniform'
    )

    # Brier score
    brier = brier_score_loss(y_true, y_proba)

    # Expected Calibration Error
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_proba[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', label=model_name)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed fraction of positives')
    ax.set_title(f'{model_name} — Calibration curve\nBrier: {brier:.4f}  |  ECE: {ece:.4f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'calibration_{model_name.replace("+", "_").replace(" ", "_")}.png', dpi=150)
    plt.close()

    print(f"{model_name}: Brier score = {brier:.4f}, ECE = {ece:.4f}")
    return brier, ece

# Usage
calibration_analysis(y_test, lgbm_proba, "LightGBM+ROS")
calibration_analysis(y_test, mlp_proba,  "MLP+None")
```

### What to report

Add a short paragraph in the Results section and a figure showing both calibration curves. Frame it as follows: *"Calibration analysis complements the threshold analysis by showing whether the models' probability estimates are reliable, independent of the chosen operating point."* Do not claim calibration analysis proves the AUC–sensitivity decoupling argument — it does not. It answers a different, complementary question.

---

## Summary of changes to each section of the paper

| Section | What changes |
|---------|-------------|
| **Methods — Evaluation Metrics** | Add threshold derivation procedure; add bootstrap CI method with exclusion rule |
| **Methods — Data Preprocessing** | Add count of patients requiring post-diagnosis visit |
| **Results — Held-Out Test** | Update Table 3 with bootstrap CIs; add threshold comparison table |
| **Results — Feature Importance** | No change needed |
| **Discussion — MLP Generalisation** | Expand with threshold analysis results; state whether MLP recovers |
| **Discussion — Limitations** | Replace vague leakage statement with quantified percentage |
| **Supplementary material** | Add calibration curves and Brier scores |

---

## Recommended order of implementation

1. **Bootstrap CIs first** — least effort, highest reviewer impact. Two hours of work.
2. **Threshold optimisation** — medium effort, directly strengthens the core thesis. Half a day.
3. **Temporal leakage count** — requires going back to the feature engineering notebook. One hour.
4. **Calibration curves** — lowest priority, add as supplementary if time is limited.

---

*Guide prepared by Bruno Oliveira (supervisor) based on peer review analysis, May 2026.*
