"""Stage 2: QC, gene filtering, normalisation and batch-effect handling for every bulk cohort.

Per cohort (labels listed in ``drop_labels`` are removed first):

  microarray_cel     RMA (stage 1) -> GMM expression filter -> probe->gene (highest mean probe)
  microarray_matrix  mas5: log2(x+1) + quantile normalisation; log2: quantile normalisation;
                     log_ratio (Agilent two-colour): used as deposited, probes with missing values dropped
  rnaseq_counts      genes with CPM >= rnaseq_min_cpm in >= smallest-group samples; raw counts kept for voom,
                     log2-CPM (TMM-free, library-size scaled) for everything else
  rnaseq_tpm         log2(TPM + 1), same expression filter on TPM

Discovery cohort only: array QC by within-stratum RLE (noisy arrays inside a batch -> technical sub-batch),
and the batch handling for machine learning, chosen WITHOUT diagnosis labels: the candidate whose batch is
least readable from the expression (batch-prediction accuracy), ties -> simplest. Batch correction is never
applied when batches are confounded with the diagnosis. (limma in stage 3 always models batch as a
covariate on uncorrected data.)

Outputs (``outputs/stage02_preprocess``):
    <GSE>_gene_expr.parquet     samples x genes, log2 scale (all cohorts)
    <GSE>_gene_counts.parquet   samples x genes, raw counts (count cohorts)
    <GSE>_samples_final.csv     labels, batch, parsed characteristics
    <GSE>_cohort_bc.parquet     discovery case-vs-control matrix after the chosen batch handling
    batch_method_comparison.csv, preprocess_summary.json, figures/*.png
"""
from __future__ import annotations

from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .context import Context
from .preprocessing import (batch_predictability, batches_confounded, collapse_to_genes, combat, drop_control_probes,
                            expression_filter, expression_threshold, log2_transform, pca_diagnostics,
                            quantile_normalize, remove_batch_effect, rle_iqr_within_strata,
                            split_noisy_subbatch)
from .utils import dump_json, read_table, write_table

STAGE = "stage02_preprocess"
PREV = "stage01_download"


def _apply_method(df: pd.DataFrame, method: str, batch: np.ndarray) -> pd.DataFrame:
    if method == "none":
        return df
    if method == "remove_batch_effect":
        return remove_batch_effect(df, batch)
    if method == "combat":
        return combat(df, batch)
    raise ValueError(method)


def _compare_methods(acc: str, cohort: pd.DataFrame, meta: pd.DataFrame, cfg: dict):
    """PCA / batch-readability diagnostics per candidate. Group R2 is reported but NOT used for the choice."""
    batch, group = meta["batch"].astype(str).to_numpy(), meta["group"].astype(str).to_numpy()
    rows, matrices = [], {}
    for name in cfg["batch_candidates"]:
        mat = _apply_method(cohort, name, batch)
        _, summ, _ = pca_diagnostics(mat, group, batch, cfg["pca_top_variable"])
        rows.append({"dataset": acc, "method": name, **summ, "batch_pred_acc": batch_predictability(mat, batch)})
        matrices[name] = mat
    return pd.DataFrame(rows), matrices


def _label_free_choice(cmp_df: pd.DataFrame, order: list, tol: float = 0.05) -> str:
    best = cmp_df["batch_pred_acc"].min()
    return next(m for m in order if cmp_df.loc[cmp_df.method == m, "batch_pred_acc"].iloc[0] <= best + tol)


def _pca_figure(path, mats: Dict[str, pd.DataFrame], meta: pd.DataFrame, title: str, top_var: int):
    fig, axes = plt.subplots(1, len(mats), figsize=(4.6 * len(mats), 4.2), squeeze=False)
    batch, group = meta["batch"].astype(str).to_numpy(), meta["group"].astype(str).to_numpy()
    colors = {b: c for b, c in zip(sorted(set(batch)), ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"])}
    markers = {g: m for g, m in zip(sorted(set(group)), ["o", "s", "^", "D"])}
    for ax, (name, mat) in zip(axes[0], mats.items()):
        tab, summ, sc = pca_diagnostics(mat, group, batch, top_var)
        for g in markers:
            for b in colors:
                m = (group == g) & (batch == b)
                ax.scatter(sc.loc[m, "PC1"], sc.loc[m, "PC2"], c=colors[b], marker=markers[g], s=38,
                           edgecolors="k", linewidths=0.4, label=f"{g}/b{b}")
        ax.set_title(f"{name}\nbatch R2(w)={summ['batch_R2_w']:.2f} | group R2(w)={summ['group_R2_w']:.2f}", fontsize=9)
        ax.set_xlabel(f"PC1 ({tab.loc[0, 'var_explained'] * 100:.0f}%)")
        ax.set_ylabel(f"PC2 ({tab.loc[1, 'var_explained'] * 100:.0f}%)")
    axes[0][-1].legend(fontsize=6, ncol=1, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _array_genes(ctx: Context, acc: str, d: dict, raw: pd.DataFrame, cfg: dict, min_s: int):
    platform = pd.read_csv(ctx.stage_dir(PREV) / f"{acc}_platform.csv.gz", dtype=str)
    scale = "rma" if d["type"] == "microarray_cel" else d.get("scale", "mas5")
    if scale == "log_ratio":
        X = raw.dropna(axis=1)
        sym = d.get("symbol_column", "GENE_NAME")
        if "CONTROL_TYPE" in platform:
            platform = platform[platform["CONTROL_TYPE"].fillna("FALSE").str.upper().isin(["FALSE", "0"])]
        platform = platform.rename(columns={sym: "Gene Symbol"})
        thr, how = np.nan, "none (log ratios)"
        Xf = X.loc[:, X.var(axis=0) > 0]
    else:
        if scale == "mas5":
            X = quantile_normalize(log2_transform(raw, cfg["log2_offset"]))
        elif scale == "log2":
            X = quantile_normalize(raw)
        else:                                   # rma: already log2 + quantile normalised
            X = raw
        if cfg["drop_control_probes"]:
            X = drop_control_probes(X)
        thr, how = expression_threshold(X, cfg["filter"]["method"], cfg["filter"]["quantile"], ctx.seed)
        Xf = expression_filter(X, min_s, thr)
    probes, pmap = collapse_to_genes(Xf, platform, cfg["multi_gene_probes"], cfg["collapse_method"])
    genes = Xf[probes].copy()
    genes.columns = pmap["gene"].tolist()
    ctx.log.info("  [%s] %s: features %d -> %d after filter (thr %s via %s) -> %d genes", acc, scale, X.shape[1],
                 Xf.shape[1], f"{thr:.2f}" if np.isfinite(thr) else "-", how, genes.shape[1])
    return genes, {"scale": scale, "filter_threshold": thr, "filter_method": how}


def _rnaseq_genes(ctx: Context, acc: str, d: dict, raw: pd.DataFrame, cfg: dict, min_s: int):
    raw = raw.fillna(0.0)
    if d["type"] == "rnaseq_counts":
        lib = raw.sum(axis=1)
        cpm = raw.div(lib, axis=0) * 1e6
        keep = (cpm >= cfg["rnaseq_min_cpm"]).sum(axis=0) >= min_s
        counts = raw.loc[:, keep].round().astype(int)
        logexpr = np.log2(cpm.loc[:, keep] + 1.0)
    else:
        keep = (raw >= cfg["rnaseq_min_cpm"]).sum(axis=0) >= min_s
        counts = None
        logexpr = np.log2(raw.loc[:, keep] + 1.0)
    ctx.log.info("  [%s] %s: %d -> %d genes (>= %.1f in >= %d samples)", acc, d["type"], raw.shape[1],
                 int(keep.sum()), cfg["rnaseq_min_cpm"], min_s)
    return logexpr, counts, {"scale": "log2CPM" if counts is not None else "log2TPM"}


def run(ctx: Context) -> None:
    cfg = ctx.cfg["preprocess"]
    out, src = ctx.stage_dir(STAGE), ctx.stage_dir(PREV)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    disc = ctx.cfg["analysis"]["discovery_dataset"]
    summary, comparison = {}, []

    for acc, d in ctx.datasets():
        if d["type"] == "scrna_10x":
            continue
        ctx.log.info("[%s] preprocessing", acc)
        raw = read_table(src / f"{acc}_expression_raw.parquet")
        meta = pd.read_csv(src / f"{acc}_samples.csv").set_index("gsm")
        meta["batch"] = meta["batch"].astype(str)
        keep = ~meta["group"].isin(d.get("drop_labels", []))
        if (~keep).any():
            ctx.log.info("  excluded %d samples (labels %s)", int((~keep).sum()), d.get("drop_labels"))
        meta = meta[keep]
        raw = raw.loc[meta.index]
        min_s = int(meta["group"].value_counts().min()) if cfg["filter"]["min_samples"] == "smallest_group" \
            else int(cfg["filter"]["min_samples"])

        counts = None
        if d["type"].startswith("microarray"):
            genes, info = _array_genes(ctx, acc, d, raw, cfg, min_s)
        else:
            genes, counts, info = _rnaseq_genes(ctx, acc, d, raw, cfg, min_s)

        # ---- array QC (discovery): within-stratum RLE -> technical sub-batches -------------
        if acc == disc:
            rle = rle_iqr_within_strata(genes, meta["batch"], meta["group"])
            thr_rle = cfg["noisy_rle_iqr"]
            if thr_rle == "auto":
                thr_rle = float(rle.median() + 3 * 1.4826 * (rle - rle.median()).abs().median())
            noisy = rle > thr_rle
            meta["batch_original"] = meta["batch"]
            meta["batch"] = split_noisy_subbatch(meta["batch_original"], noisy, min_size=3)
            meta["rle_iqr"], meta["noisy_array"] = rle, noisy
            ctx.log.info("  RLE IQR median %.3f (max %.3f); %d noisy arrays (> %.2f) -> technical batches %s",
                         rle.median(), rle.max(), int(noisy.sum()), thr_rle, sorted(set(meta["batch"])))
            info.update({"rle_iqr_median": float(rle.median()), "noisy_arrays": list(meta.index[noisy]),
                         "technical_batches": sorted(set(meta["batch"]))})

        write_table(genes, out / f"{acc}_gene_expr.parquet")
        if counts is not None:
            write_table(counts, out / f"{acc}_gene_counts.parquet")
        meta.to_csv(out / f"{acc}_samples_final.csv")
        info.update({"samples": int(len(meta)), "genes": int(genes.shape[1]),
                     "groups": meta["group"].value_counts().to_dict(), "min_samples": min_s})

        # ---- discovery cohort: label-free batch handling for ML ------------------------------
        if acc == disc:
            case, ctrl = d["case"], d["control"]
            cm = meta[meta["group"].isin([case, ctrl])]
            cg = genes.loc[cm.index]
            confounded = batches_confounded(cm["batch"], cm["group"])
            cmp_df, mats = _compare_methods(acc, cg, cm, cfg)
            comparison.append(cmp_df)
            if confounded:
                chosen, reason = "none", "batches confounded with diagnosis -> no batch correction"
            elif cfg["batch_method"] == "auto":
                chosen = _label_free_choice(cmp_df, cfg["batch_candidates"])
                reason = "label-free: lowest batch-prediction accuracy (ties within 0.05 -> simplest)"
            else:
                chosen, reason = cfg["batch_method"], "set in config"
            ctx.log.info("  batch handling for ML -> %s (%s)\n%s", chosen, reason,
                         cmp_df[["method", "gene_R2_batch", "gene_R2_group", "batch_pred_acc"]].round(3).to_string(index=False))
            write_table(mats[chosen], out / f"{acc}_cohort_bc.parquet")
            cm.assign(batch_method=chosen).to_csv(out / f"{acc}_cohort_samples.csv")
            _pca_figure(fig_dir / f"{acc}_pca_batch_methods.png", mats, cm, f"{acc}: {case} vs {ctrl}, PCA by method",
                        cfg["pca_top_variable"])
            info.update({"batch_confounded": confounded, "batch_method": chosen, "batch_reason": reason})

        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.boxplot([genes.loc[i].values for i in genes.index], showfliers=False)
        ax.set_title(f"{acc}: gene-level expression ({info['scale']})", fontsize=9)
        ax.set_xticks([])
        fig.tight_layout()
        fig.savefig(fig_dir / f"{acc}_boxplots.png", dpi=110)
        plt.close(fig)
        summary[acc] = info

    if comparison:
        pd.concat(comparison).to_csv(out / "batch_method_comparison.csv", index=False)
    dump_json(summary, out / "preprocess_summary.json")
    ctx.log.info("stage 2 done -> %s", out)
