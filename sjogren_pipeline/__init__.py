"""Sjogren's syndrome RNA target-prioritization pipeline.

Stages (each module exposes ``run(ctx)``):

    stage01_download     GEO metadata, raw CEL -> RMA, RNA-seq matrices, rule-based labels
    stage02_preprocess   QC, filtering, normalisation, label-free batch handling
    stage03_diffexpr     limma per cohort (R), deconvolution, random-effects meta-analysis
    stage04_pathways     GO / Reactome / KEGG / Hallmark over-representation and GSEA
    stage05_models       classical ML with nested CV and external cohort tests
    stage06_explain      SHAP, integrated gradients, deletion and ablation tests
    stage07_ranking      multi-evidence target ranking (uses DeepMOLFT SHAP) + consensus
    stage08_validation   replication, gland single-cell validation, RT-qPCR panel design
    stage09_figures      report figures and model metric tables

The DeepMOLFT late-fusion model lives in ``deepmolft_task/`` and runs between stages 6 and 7.
"""

__version__ = "2.0.0"
