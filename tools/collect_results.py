#!/usr/bin/env python
"""Copy the curated result tables and figures into results/ and figures/ (the folders kept in version control).

    python tools/collect_results.py

Run after the pipeline (stages 1-9) and deepmolft_task/run_deepmolft.py. Everything under outputs/ and
deepmolft_task/outputs/ stays local (large, regenerable); only the files listed here are published.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TABLES = {
    # differential expression and meta-analysis
    "outputs/stage03_diffexpr/de_all_genes.csv": "meta_analysis/de_all_genes.csv",
    "outputs/stage03_diffexpr/candidate_genes.csv": "meta_analysis/candidate_genes.csv",
    "outputs/stage03_diffexpr/de_summary.csv": "meta_analysis/de_summary_per_cohort.csv",
    "outputs/stage03_diffexpr/calibration_permutation.csv": "meta_analysis/calibration_permutation.csv",
    "outputs/stage03_diffexpr/deconvolution_validation.csv": "meta_analysis/deconvolution_vs_histology.csv",
    # classical models
    "outputs/stage05_models/nested_cv_summary.csv": "models/nested_cv_summary.csv",
    "outputs/stage05_models/external_validation.csv": "models/external_validation.csv",
    "outputs/stage05_models/leave_one_batch_out.csv": "models/leave_one_batch_out.csv",
    "outputs/figures/model_metrics.csv": "models/model_metrics.csv",
    # DeepMOLFT
    "deepmolft_task/outputs/loco_summary.csv": "deepmolft/loco_summary.csv",
    "deepmolft_task/outputs/loco_metrics.csv": "deepmolft/loco_metrics.csv",
    "deepmolft_task/outputs/optuna_best_params.csv": "deepmolft/optuna_best_params.csv",
    "deepmolft_task/outputs/deepmolft_shap_targets.csv": "deepmolft/deepmolft_shap_targets.csv",
    "deepmolft_task/outputs/deepmolft_top50_gland_branch.csv": "deepmolft/top50_gland_branch.csv",
    "deepmolft_task/outputs/deepmolft_top50_immune_branch.csv": "deepmolft/top50_immune_branch.csv",
    "deepmolft_task/outputs/deepmolft_top50_broad_branch.csv": "deepmolft/top50_broad_branch.csv",
    "deepmolft_task/outputs/branch_contribution.csv": "deepmolft/branch_contribution.csv",
    "deepmolft_task/outputs/deletion_test.csv": "deepmolft/deletion_test.csv",
    # target ranking and validation
    "outputs/stage07_ranking/target_ranking_all.csv": "targets/target_ranking_all.csv",
    "outputs/stage07_ranking/target_consensus.csv": "targets/target_consensus.csv",
    "outputs/stage07_ranking/target_ranking_gland_intrinsic.csv": "targets/target_ranking_gland_intrinsic.csv",
    "outputs/stage07_ranking/target_ranking_immune.csv": "targets/target_ranking_immune.csv",
    "outputs/stage07_ranking/ranking_evaluation.csv": "targets/ranking_evaluation_clinical_targets.csv",
    "outputs/stage07_ranking/weight_sensitivity.csv": "targets/weight_sensitivity.csv",
    "outputs/stage07_ranking/literature_pubmed_counts.csv": "targets/literature_pubmed_counts.csv",
    "outputs/stage08_validation/scrna_target_summary.csv": "validation/scrna_target_summary.csv",
    "outputs/stage08_validation/replication_tests.csv": "validation/replication_tests.csv",
    "outputs/stage08_validation/qpcr_panel.csv": "validation/qpcr_panel.csv",
}


def main() -> None:
    res, fig = ROOT / "results", ROOT / "figures"
    n = 0
    for src, dst in TABLES.items():
        s = ROOT / src
        if not s.exists():
            print(f"missing (run the pipeline first): {src}")
            continue
        d = res / dst
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        n += 1
    fig.mkdir(exist_ok=True)
    pngs = sorted((ROOT / "outputs/figures").glob("fig*.png")) + sorted((ROOT / "deepmolft_task/outputs/figures").glob("*.png"))
    for p in pngs:
        shutil.copy2(p, fig / p.name)
    print(f"copied {n} tables to results/ and {len(pngs)} figures to figures/")


if __name__ == "__main__":
    main()
