# Figure legends

Figures are in `figures/` (PNG, 300 dpi). They are produced by `python run_pipeline.py --stages 9` (pipeline figures, `fig01`–`fig15`, PNG + PDF in `outputs/figures/`) and `deepmolft_task/run_deepmolft.py` (`deepmolft_fig1`–`fig4`), then copied with `python tools/collect_results.py`.

## Results

**Fig. 1 | Cohorts.** Seven salivary-gland transcriptome cohorts in the meta-analysis (bars: cases, blue; controls, orange; healthy controls except GSE173808, where controls are symptomatic non-SS sicca patients). GSE272409 (minor-gland scRNA-seq, 7 SjD vs 6 sicca) is used only for validation.

**Fig. 2 | Quality of the differential-expression step.** (a) Within-batch label permutations of GSE40611 (n = 100): fraction of genes with p < 0.05 per permutation; mean 0.061 vs the nominal 0.05. (b) Deconvolved immune fraction (NNLS, HPA salivary-gland reference) against histological %CD45+ cells in paired parotid and labial biopsies of GSE173808. (c) Genes at FDR < 0.05 in each cohort without (blue) and with (orange) adjustment for the immune fraction (symlog axis).

**Fig. 3 | Meta-analysis volcano.** DerSimonian-Laird random-effects meta-analysis over the seven cohorts (case vs control log2 fold change, equivalent-z standard errors). Red / blue: FDR < 0.05 and |log2FC| ≥ 0.585; dashed lines mark the thresholds; circled genes are the 12 highest-priority targets.

**Fig. 4 | Cross-cohort consistency of the top-30 targets.** Per-cohort log2 fold change (blue-grey-red diverging scale) and the random-effects estimate (right column). • P < 0.05, •• P < 0.001; * clinical-stage Sjögren target.

**Fig. 5 | Forest plots for the six highest-priority targets.** Per-cohort log2FC ± 95% CI (blue) and random-effects estimate (orange diamond), with between-cohort heterogeneity (I²) and meta FDR.

**Fig. 6 | Hallmark pathways.** GSEA prerank (gseapy, 1,000 permutations) on the meta z statistic, before (blue) and after (orange) immune-fraction adjustment. Filled: FDR < 0.05; open: not significant. Interferon-γ and -α responses remain the strongest signals after adjustment.

**Fig. 7 | Composition of the target priority score.** Top 30 genes in target mode; segments are weighted evidence layers after the abundance-marker penalty. Right: compartment of origin in the HPA salivary-gland single-cell reference ("broad" = tau < 0.8; "low in reference" = max CPM < 10 in healthy tissue).

**Fig. 8 | Robustness of the ranking.** (a) AUROC with which each layer, and the combined score, recovers the 17 clinical-stage Sjögren drug targets present among the 609 candidates (held out: never used for scoring). (b) Rank under the configured weights (orange) and median rank with interquartile range over 1,000 Dirichlet-perturbed weight vectors (blue); right, probability of staying in the top 20.

**Fig. 9 | Independent single-cell validation (GSE272409).** (a) Cell of origin of the top-20 targets: dot size, % of cells expressing; colour, mean log CP10K (cell types assigned with HPA salivary-gland markers). (b) Pseudobulk limma-voom log2FC SjD vs sicca within the epithelial, stromal and immune compartments; • P < 0.05, •• P < 0.01. Interferon-stimulated genes (ISG15, IFI6, STAT1, XAF1) rise inside epithelial cells, i.e. not only through immune-cell influx.

## Models (supporting evidence)

**Fig. 10 | Nested cross-validation metrics.** GSE40611 pSS vs control (n = 35), 5 × repeated 5-fold outer CV with 3-fold inner grid search; feature selection and scaling inside each training fold. Out-of-fold AUROC, AUPRC, balanced accuracy, sensitivity, specificity, F1, Matthews correlation and Brier score (mean ± SD over repeats). Threshold: probability 0.5 or decision value 0. The linear SVM has no probability output (no Brier score). Numbers: `results/models/model_metrics.csv`.

**Fig. 11 | ROC and precision-recall curves.** Out-of-fold curves averaged over the five CV repeats; dashed grey, prior-only baseline.

**Fig. 12 | Model robustness.** (a) Label-permutation null for the best model (gradient boosting; hyper-parameters fixed at the modal nested-CV values, so the observed 0.983 is slightly optimistic relative to the nested 0.961). (b) Leave-one-technical-batch-out AUROC. (c) Batch handling: nested-CV AUROC and batch read-out accuracy per matrix; the matrix was chosen without diagnosis labels (lowest batch read-out). (d) Pooled confusion matrix of the best model.

**Fig. 13 | External validation.** Models trained on GSE40611 and applied to every other cohort after per-dataset gene standardisation. (a) AUROC for each model × cohort. (b) Best model with 95% bootstrap CI. Interpret with care: GSE23117 has batch confounded with diagnosis, GSE40568 and GSE157159 are small, and GSE173808 uses symptomatic sicca controls.

**Fig. 14 | Explainability.** (a-c) Deletion curves: out-of-fold AUROC as the top-k genes (by model attribution, training F-score or at random) are set to the training mean; a faithful attribution falls faster than random. Integrated gradients for the autoencoder are not more faithful than random. (d) Top-20 genes of the most faithful model (gradient boosting, TreeSHAP): mean |SHAP| over 25 outer folds, coloured by meta-analysis direction; labels give the % of folds in which the gene passed feature selection.

## Consensus (added after the DeepMOLFT integration)

**Fig. 15 | Consensus of the multi-evidence ranking and DeepMOLFT.** (a) Pipeline rank (x, log) against DeepMOLFT SHAP rank within its input branch (y, log; leave-one-cohort-out). Shaded: pipeline top 100; dashed: DeepMOLFT branch top 50. Blue: tier-1 consensus genes (both methods, same direction, ≥ 1.5-fold, robust to leaving out any cohort, infiltration-independent); orange: tier 2 (smaller effects). (b) Meta-analysis fold change of the 18 tier-1 genes, grouped by biological theme; #n = pipeline rank. Table: `results/targets/target_consensus.csv`.

## DeepMOLFT (deepmolft_task/)

**DeepMOLFT Fig. 1 | Leave-one-cohort-out validation.** AUROC on each held-out cohort for DeepMOLFT, elastic net and random forest trained on the other six cohorts (same features per fold).

**DeepMOLFT Fig. 2 | Top SHAP genes per input branch.** Share of the branch's |SHAP| on held-out cohorts for the gland, immune and broadly expressed gene branches; ↑/↓ direction in Sjögren's, k/7 folds in the branch top 20, #n main-pipeline rank.

**DeepMOLFT Fig. 3 | SHAP summary (beeswarm)** of the top genes per branch on all held-out samples.

**DeepMOLFT Fig. 4 | Branch contribution and faithfulness.** Share of |SHAP| per branch per held-out cohort; mean held-out AUROC after removing the top-k SHAP genes versus random genes.
