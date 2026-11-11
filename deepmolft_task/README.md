# DeepMOLFT on Sjögren’s salivary-gland RNA

A separate module that applies the **DeepMOLFT** late-fusion deep network and its SHAP biomarker analysis
([priyashdhakal/DeepMOLFT](https://github.com/priyashdhakal/DeepMOLFT)) to the seven salivary-gland cohorts prepared
by the main pipeline. It reads the pipeline's stage-2 matrices; its SHAP table feeds the model-importance layer and the
consensus in stage 7.

```
python run_deepmolft.py            # full run  -> outputs/   (~1 h on CPU)
python run_deepmolft.py --quick    # smoke test -> outputs_quick/
```

## What was kept from DeepMOLFT

| DeepMOLFT (`src/`) | Here |
|---|---|
| One dense branch per modality: 128 → 64 → 32, ReLU, L1 0.001, dropout 0.5 / 0.3 / 0.2 | identical (`deepmolft_sjs/model.py`) |
| Concatenate → 3 fused dense layers (tuned) → sigmoid output | identical; the logit is returned and sigmoid applied for probabilities |
| Glorot-uniform weights, zero bias | identical |
| Adam, (balanced) binary cross-entropy + L1, batch 32, ≤ 500 epochs, early stopping on val loss (patience 50, best weights restored) | identical (`train.py`) |
| Optuna over lr 1e-5–1e-1 (log), fused sizes 64–256 / 32–128 / 16–64, dropouts 0.1–0.5 | identical search space; 30 trials per outer fold (original 10,000) |
| Correlation filter (|r| > 0.85) and StandardScaler fitted on training data only | identical (vectorised version of `correlation()`) |
| `shap.DeepExplainer` on the multi-input model, SHAP on held-out samples, beeswarm summary | identical idea (`explain.py`, figure 3) |

## What had to change, and why

1. **Framework: TensorFlow 2.12 → PyTorch.** TensorFlow has no build for this environment’s Python 3.15. The network,
   regulariser, loss, optimiser and early-stopping rules were ported one to one; SHAP DeepExplainer supports both.
2. **Modalities.** DeepMOLFT fuses methylation / miRNA / mRNA measured on the same patients. Here there is one data type
   (RNA), so the three late-fusion branches are three **gene groups by cell of origin** in the HPA salivary-gland
   single-cell atlas: **immune-cell genes**, **gland (epithelial + stromal) genes** and **broadly expressed genes**.
   This keeps the late-fusion design and makes SHAP answer a biological question: does the model rely on infiltrate or on
   the gland itself?
3. **No test-set leakage.** In the original `main.py` the test set is passed as `validation_data`, so early stopping and
   the Optuna objective both see the test data, which inflates the reported accuracy. Here early stopping and Optuna
   use an inner split of the training cohorts (stratified, grouped by patient so the two glands of one GSE173808
   patient never straddle the split).
4. **Validation scheme.** Instead of one fixed split: **leave-one-cohort-out** over the seven cohorts. Every reported
   metric and every SHAP value comes from a cohort that was never used for training, tuning or early stopping.
5. **Cross-platform input.** Genes measured in all seven cohorts (7,616), z-scored within each cohort (label-free) to
   remove platform and scale differences; then, per fold and on training cohorts only, the 3,000 most variable genes
   (mean within-cohort variance), the 0.85 correlation filter and the scaler.
6. **Stability.** The final model of each fold is a 5-seed ensemble (different inner splits); SHAP is averaged over it.
   SHAP uses 100 training samples as background (the original used the test set itself).
7. **Baselines.** Elastic-net logistic regression and random forest are trained on exactly the same fold features, so
   DeepMOLFT can be judged against simpler models.
8. **Ranking genes.** Late fusion squeezes every branch into 32 units, so per-gene SHAP scales inversely with branch
   size (smaller branch → larger per-gene values). Genes are therefore ranked **within their branch**, and each branch’s
   overall share of SHAP is reported separately.

## Outputs (`outputs/`)

| file | content |
|---|---|
| `loco_metrics.csv`, `loco_summary.csv` | AUROC, AUPRC, balanced accuracy, sensitivity, specificity, F1, MCC, Brier per held-out cohort and pooled, for DeepMOLFT and the baselines |
| `loco_predictions.csv` | out-of-cohort probabilities per sample |
| `optuna_best_params.csv` | tuned hyper-parameters per fold (inner-validation AUROC) |
| `shap_by_fold.csv`, `shap_heldout_<GSE>.npz` | per-gene SHAP on every held-out cohort (raw matrices in the .npz) |
| `deepmolft_shap_targets.csv` | consensus SHAP table: branch, within-branch rank, fold stability, direction, joined to the main pipeline (meta-analysis, target rank) |
| `deepmolft_top50_<branch>_branch.csv` | top 50 genes per branch |
| `deepmolft_supported_targets.csv` | branch top-50 genes that are also meta-analysis candidates with the same direction |
| `branch_contribution.csv` | share of |SHAP| per branch per held-out cohort |
| `deletion_test.csv` | SHAP faithfulness: AUROC after removing the top-k SHAP genes vs random genes |
| `comparison_with_pipeline.json` | overlap with the main pipeline’s ranking |
| `figures/` | 4 figures (PNG 300 dpi + PDF) |

## Results (full run, 7 leave-one-cohort-out folds)

**Classification on held-out cohorts** (`loco_summary.csv`, `loco_metrics.csv`)

| model | mean AUROC | n-weighted AUROC | pooled AUROC | pooled MCC | pooled Brier |
|---|---|---|---|---|---|
| Random forest | 0.885 | 0.831 | 0.840 | 0.51 | 0.157 |
| Elastic net | 0.879 | 0.831 | 0.812 | 0.42 | 0.175 |
| DeepMOLFT | 0.851 | 0.808 | 0.806 | 0.39 | 0.210 |

DeepMOLFT per held-out cohort: GSE40611 0.97 · GSE23117 0.95 · GSE127952 0.94 · GSE157159 0.92 · GSE173808 0.81 ·
GSE40568 0.80 · GSE154926 0.58. With 252 samples the late-fusion DNN does **not** beat the simpler baselines; its
value here is the SHAP explanation, not extra accuracy.

**SHAP** (`deepmolft_shap_targets.csv`, figures 2–4)

* The model relies mostly on immune-cell genes (61% of |SHAP|), then gland genes (23%) and broad genes (16%).
* Top gland-branch genes: IFI27 (main-pipeline rank #1), SLC1A1, IFI44L, CALML5, IFI6, TSPAN7, GBP3 (up) and TPM2, RHOB,
  PGF (down). Top immune-branch genes: BMP6 (7/7 folds), LAMP3, TNFRSF17 (BCMA), SLC7A5, HERC5, HLA-DRA, MS4A1 (CD20).
* 45 of the top-50 SHAP genes push in the same direction as the meta-analysis; 35 branch-top-50 genes are also
  meta-analysis candidates (`deepmolft_supported_targets.csv`), including IFI27, ISG15, MX1, IFI6, CCL19, CXCL9, BMP6.
* Faithfulness: removing the top-20 SHAP genes lowers mean held-out AUROC from 0.851 to 0.824, removing 20 random genes
  leaves it at 0.850. The attributions are real but modest, because many genes carry redundant signal.
