# BehaviorGraph-NUTS Supporting Implementation

This supporting material accompanies the manuscript:

**BehaviorGraph-NUTS: A Graph-Derived Bayesian Posterior Fusion Framework for Uncertainty-Aware Anti-Money Laundering Detection in IoT-Enabled Financial Systems**

## Files

- `BehaviorGraph_NUTS_Supplementary_Implementation_Clean.ipynb`  
  Main no-output notebook for reproducing the complete experimental pipeline.

- `BehaviorGraph_NUTS_Supplementary_Implementation_Clean.py`  
  Optional plain-Python export of the same notebook content.

- `BehaviorGraph_NUTS_requirements.txt`  
  Python package list for setting up the environment.

## Dataset

The notebook expects the HI-Small AML transaction CSV with the following columns:

`Timestamp`, `From Bank`, `Account`, `To Bank`, `Account.1`, `Amount Received`, `Receiving Currency`, `Amount Paid`, `Payment Currency`, `Payment Format`, `Is Laundering`.

Set the dataset path in Cell 2 by editing `CSV_PATH`, or define:

```bash
export BEHAVIORGRAPH_NUTS_CSV_PATH="/path/to/HI-Small_Trans.csv"
```

Outputs can be redirected with:

```bash
export BEHAVIORGRAPH_NUTS_OUTPUT_DIR="/path/to/output_directory"
```

## What the notebook reproduces

The notebook includes:

1. Environment setup and reproducibility controls.
2. Dataset loading and canonical column validation.
3. Leakage-controlled train/validation/test partitioning.
4. Graph-derived behavioural and structural-risk feature construction.
5. Tabular XGBoost, BehaviorGraph-XGBoost, RUS-XGBoost, Isolation Forest, and rule-based AML evidence generation.
6. Probability calibration for evidence streams.
7. Deterministic logistic fusion baseline.
8. Bayesian posterior fusion using NUTS.
9. Posterior coefficient, uncertainty, R-hat, ESS, trace, density, and subset-size diagnostics.
10. Classification, ranking, calibration, threshold-policy, and top-k alert-prioritization analyses.
11. Five-seed robustness evaluation and paired statistical comparisons.
12. Runtime and memory inference transparency.
13. Export of tables, figures, models, preprocessors, diagnostics, and configuration files.

## Notes

The notebook is intentionally distributed without outputs. Running the full pipeline, especially NUTS and five-seed evaluation, can be computationally expensive. The original experiments were designed for a high-memory Google Colab/A100-style runtime.
