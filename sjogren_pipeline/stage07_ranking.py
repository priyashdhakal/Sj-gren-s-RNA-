"""Stage 7: multi-evidence target ranking (mode ``target`` or ``biomarker``, see config ``ranking``).

Candidate universe = meta-analysis candidates (stage 3) U top model-importance genes (DeepMOLFT branch top-50) U curated
pSS GWAS genes measured in the cohorts. Every candidate gets layer scores in [0, 1]:

  differential_expression  random-effects meta P, |meta logFC|, fraction of cohorts nominally significant in the
                           meta direction
  gland_intrinsic          effect survives deconvolution adjustment (meta P after adjustment, retained fraction
                           of the effect) and the gene is not mainly expressed by immune cells in the HPA
                           salivary-gland single-cell reference
  genetics                 Open Targets genetic-association score for Sjogren + curated GWAS loci
  tractability             Open Targets small-molecule (structure / pocket / druggable family), antibody
                           (surface / secreted) or PROTAC evidence -- clinical-precedence buckets excluded
  network                  personalised PageRank from genetic seed genes on the STRING high-confidence network
  model_importance         DeepMOLFT leave-one-cohort-out SHAP (deepmolft_task/), within-branch percentile x
                           fold stability; falls back to stage-6 attributions when DeepMOLFT has not been run
  safety                   1 - liabilities (core-essential gene, Open Targets safety events)
  cell_type_specificity    tau over HPA salivary-gland cell types (biomarker mode)

priority = weighted mean x (1 - infiltrate_penalty x abundance_marker), where abundance_marker = immune share of
expression x fraction of the effect lost after deconvolution adjustment. Immune-cell genes with effects beyond
cell abundance, genetics or druggability are therefore kept; pure "more lymphocytes" markers are pushed down.

Evaluation (held out): clinical-stage Sjogren drug targets from Open Targets are never used in any layer;
AUROC / top-N enrichment of the priority score and of each layer for recovering them.

Outputs (``outputs/stage07_ranking``): target_ranking_all.csv, target_ranking_gland_intrinsic.csv,
target_ranking_immune.csv, target_consensus.csv (pipeline top-100 AND DeepMOLFT branch top-50), layer_scores.csv,
ranking_evaluation.csv, weight_sensitivity.csv, figures/.
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from .context import Context
from .external_data import (celltype_profile, hpa_salivary_reference, ot_disease_associations,
                            ot_disease_clinical_targets, ot_target_annotations, string_network)
from .stage01_download import ensembl_to_symbol
from .utils import rank_normalize

warnings.filterwarnings("ignore")
STAGE, S3, S6 = "stage07_ranking", "stage03_diffexpr", "stage06_explain"


def _safe(fn, ctx: Context, what: str, default):
    try:
        return fn()
    except Exception as exc:  # offline or API change: layer falls back to neutral values
        ctx.log.warning("  %s unavailable (%s: %s) -> neutral layer", what, exc.__class__.__name__, str(exc)[:120])
        return default


def network_layer(net: pd.DataFrame, universe: list, seeds: list) -> pd.Series:
    G = nx.Graph()
    G.add_nodes_from(universe)
    for a, b, s in net[["a", "b", "combined_score"]].itertuples(index=False):
        G.add_edge(a, b, weight=s / 1000.0)
    seeds = [s for s in seeds if s in G and G.degree(s) > 0]
    if not seeds or G.number_of_edges() == 0:
        return pd.Series(0.0, index=universe)
    pers = {n: (1.0 if n in seeds else 0.0) for n in G.nodes}
    pr = pd.Series(nx.pagerank(G, alpha=0.85, personalization=pers, weight="weight"))
    deg = pd.Series(dict(G.degree())).reindex(universe).fillna(0)
    pr = pr.reindex(universe).fillna(0.0)
    return (0.7 * rank_normalize(pr) + 0.3 * rank_normalize(deg)).where(deg > 0, 0.0)


def model_importance_source(ctx: Context, rcfg: dict, s6):
    """Model-importance layer: DeepMOLFT leave-one-cohort-out SHAP when available, else stage-6 attributions.

    DeepMOLFT (deepmolft_task/) is trained across cohorts and explained on cohorts it never saw, so it is the
    preferred source. Per gene: within-branch SHAP percentile (late fusion makes per-gene SHAP comparable only
    within a branch) x fold stability (0.5 + 0.5 x share of folds in the branch top 20), halved when the SHAP
    direction disagrees with the meta-analysis. Returns (score, top genes added to the universe, label, table).
    """
    mcfg = rcfg.get("model_importance", {})
    if mcfg.get("source", "stage6") == "deepmolft":
        path = ctx.root / mcfg.get("deepmolft_dir", "deepmolft_task/outputs") / "deepmolft_shap_targets.csv"
        if path.exists():
            dm = pd.read_csv(path, index_col=0)
            n_folds = max(int(dm["folds_selected"].max()), 1)
            pct = dm.groupby("branch")["within_branch_importance"].rank(pct=True)
            stab = 0.5 + 0.5 * dm["folds_in_branch_top20"] / n_folds
            agree = dm["direction_agrees_with_meta"].fillna(False).astype(bool)
            score = pct * stab * np.where(agree, 1.0, 0.5)
            top = dm.index[dm["branch_rank"] <= int(mcfg.get("deepmolft_branch_top", 50))]
            return score, list(top), "DeepMOLFT SHAP (leave-one-cohort-out)", dm
        ctx.log.warning("  DeepMOLFT results not found at %s -> run deepmolft_task/run_deepmolft.py; using stage 6", path)
    p = s6 / "gene_importance_consensus.csv"
    if not p.exists():
        return pd.Series(dtype=float), [], "none", pd.DataFrame()
    imp = pd.read_csv(p, index_col=0)
    score = imp["consensus_score"] * (0.5 + 0.5 * imp["stability"].fillna(0.0))
    return score, list(imp.head(100).index), "stage-6 single-cohort models", pd.DataFrame()


def run(ctx: Context) -> None:
    rcfg = ctx.cfg["ranking"]
    mode = rcfg["mode"]
    mcfg = rcfg["modes"][mode]
    out = ctx.stage_dir(STAGE)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    s3, s6 = ctx.stage_dir(S3), ctx.stage_dir(S6)
    seed, topn = ctx.seed, rcfg["top_n"]
    disease = rcfg["opentargets_disease"]

    de = pd.read_csv(s3 / "de_all_genes.csv", index_col=0)
    cand = pd.read_csv(s3 / "candidate_genes.csv", index_col=0)
    mi_score, mi_top, mi_label, dm = model_importance_source(ctx, rcfg, s6)
    kg = pd.read_csv(ctx.external / rcfg["known_genes_file"])
    gwas = kg[kg["category"].str.startswith("GWAS")]
    known_present = sorted(set(gwas["gene"]) & set(de.index))
    universe = sorted(set(cand.index) | set(mi_top) | set(known_present))
    universe = [g for g in universe if g in de.index]
    U = de.loc[universe]
    ctx.log.info("mode = %s | model importance from %s | candidate universe: %d genes (%d meta candidates, %d top model "
                 "genes, %d curated GWAS genes)", mode, mi_label, len(universe), len(cand), len(mi_top), len(known_present))

    # ---- external evidence ------------------------------------------------------------------
    ref = hpa_salivary_reference(ctx)
    ct = celltype_profile(ref, universe)
    assoc = _safe(lambda: ot_disease_associations(ctx, disease), ctx, "Open Targets associations", pd.DataFrame())
    clin = _safe(lambda: ot_disease_clinical_targets(ctx, disease), ctx, "Open Targets clinical targets", pd.DataFrame())
    sym2ens = {v: k for k, v in ensembl_to_symbol(ctx).items()}
    if len(assoc):
        sym2ens.update(dict(zip(assoc["gene"], assoc["ensembl"])))
    ann = _safe(lambda: ot_target_annotations(ctx, {g: sym2ens.get(g) for g in universe}), ctx,
                "Open Targets target annotations", pd.DataFrame(index=universe))
    ann = ann.reindex(universe)
    gen_ot = (assoc.groupby("gene")["ot_genetic_association"].max() if "ot_genetic_association" in assoc
              else pd.Series(dtype=float))
    seeds = sorted(set(gen_ot[gen_ot >= 0.1].index) | set(gwas.loc[gwas["tier"] == 1, "gene"]))
    net = _safe(lambda: string_network(ctx, universe + seeds), ctx, "STRING", pd.DataFrame(columns=["a", "b", "combined_score"]))

    # ---- layers --------------------------------------------------------------------------------
    L = pd.DataFrame(index=universe)
    frac_nom = (U["n_nominal_same_direction"] / U["n_cohorts"]).fillna(0.0)
    L["differential_expression"] = (0.5 * rank_normalize(-np.log10(U["meta_P"].clip(lower=1e-300))) +
                                    0.3 * rank_normalize(U["meta_logFC"].abs()) + 0.2 * frac_nom)

    same = np.sign(U["adj_meta_logFC"]) == np.sign(U["meta_logFC"])
    adj_p = rank_normalize(-np.log10(U["adj_meta_P"].clip(lower=1e-300))).where(same, 0.0).fillna(0.0)
    retained = (U["adj_meta_logFC"] / U["meta_logFC"]).clip(0, 1).where(same, 0.0).fillna(0.0)
    immune_share = ct["immune_share"].reindex(universe).fillna(0.5)
    L["gland_intrinsic"] = (adj_p + retained + (1.0 - immune_share)) / 3.0

    kw = gwas.groupby("gene")["evidence_weight"].max().reindex(universe).fillna(0.0)
    L["genetics"] = np.maximum(gen_ot.reindex(universe).fillna(0.0).to_numpy(), kw.to_numpy())

    tr = ann.reindex(columns=["tract_small_molecule", "tract_antibody", "tract_protac"]).fillna(False).astype(bool)
    L["tractability"] = np.maximum.reduce([tr["tract_small_molecule"] * 1.0, tr["tract_antibody"] * 0.8,
                                           tr["tract_protac"] * 0.5])
    L["network"] = network_layer(net, universe, seeds)
    mi = mi_score.reindex(universe).fillna(0.0)
    L["model_importance"] = mi / max(mi.max(), 1e-9)
    ess = ann.get("is_essential", pd.Series(index=universe, dtype=float)).astype(float)
    liab = ann.get("n_safety_liabilities", pd.Series(index=universe, dtype=float)).astype(float)
    L["safety"] = (1.0 - 0.5 * ess.fillna(0.5) - 0.5 * (liab / 5.0).clip(upper=1.0).fillna(0.5)).clip(0, 1)
    L["cell_type_specificity"] = ct["tau"].reindex(universe).fillna(0.0)

    w = pd.Series(mcfg["weights"], dtype=float)
    w = w[w > 0] / w[w > 0].sum()
    abundance_marker = immune_share * (1.0 - retained)
    penalty = 1.0 - float(mcfg.get("infiltrate_penalty", 0.0)) * abundance_marker
    L["abundance_marker"] = abundance_marker
    L["priority_score"] = (L[w.index] * w).sum(axis=1) * penalty
    L["rank"] = L["priority_score"].rank(ascending=False, method="first").astype(int)

    # ---- weight sensitivity -------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    draws = rng.dirichlet(w.to_numpy() * 20.0, size=rcfg["weight_sensitivity_draws"])
    M = L[w.index].to_numpy() * penalty.to_numpy()[:, None]
    ranks = np.column_stack([(-(M @ d)).argsort().argsort() + 1 for d in draws])
    L["rank_median_sens"] = np.median(ranks, axis=1)
    L["rank_iqr_sens"] = stats.iqr(ranks, axis=1)
    L["P_top20_sens"] = (ranks <= 20).mean(axis=1)
    L[f"P_top{topn}_sens"] = (ranks <= topn).mean(axis=1)

    # ---- held-out evaluation: clinical-stage Sjogren targets ---------------------------------------
    clin_genes = set(clin["gene"]) if len(clin) else set()
    y = pd.Series([g in clin_genes for g in universe], index=universe).astype(int)
    ev = []
    if y.nunique() == 2:
        scores = {"priority_score": L["priority_score"], "priority_without_tractability":
                  (L[w.drop("tractability", errors="ignore").index] * w.drop("tractability", errors="ignore")).sum(axis=1) * penalty}
        scores.update({c: L[c] for c in list(w.index) + ["cell_type_specificity"] if c not in scores})
        K, N = int(y.sum()), len(y)
        for name, sc in scores.items():
            k = int(((sc.rank(ascending=False, method="first") <= topn) & (y == 1)).sum())
            ev.append({"ranking": name, "AUROC_clinical_targets": roc_auc_score(y, sc), f"clinical_in_top{topn}": k,
                       "expected_by_chance": K * topn / N, "clinical_targets_in_universe": K, "universe": N,
                       "enrichment_P": stats.hypergeom.sf(k - 1, N, K, topn)})
    ev = pd.DataFrame(ev).sort_values("AUROC_clinical_targets", ascending=False) if ev else pd.DataFrame()
    ev.to_csv(out / "ranking_evaluation.csv", index=False)
    if len(ev):
        ctx.log.info("held-out check: recovery of %d clinical-stage Sjogren targets in the universe (not used in scoring):\n%s",
                     int(y.sum()), ev.round(4).to_string(index=False))

    # ---- assemble tables ----------------------------------------------------------------------
    keep_de = ["meta_logFC", "meta_FDR", "loo_max_P", "rep_meta_FDR", "n_cohorts", "n_same_direction",
               "n_nominal_same_direction", "I2", "adj_meta_logFC", "adj_meta_FDR", "disc_logFC", "disc_adj.P.Val"]
    res = L.join(U[[c for c in keep_de if c in U]]).join(ct).join(ann.drop(columns=["ensembl"], errors="ignore"))
    res["direction"] = np.where(res["meta_logFC"] > 0, "up_in_pSS", "down_in_pSS")
    res["infiltration_robust"] = (res["adj_meta_FDR"] < 0.05) & same.reindex(res.index)
    res["clinical_sjogren_target"] = res.index.isin(clin_genes)
    res["genetic_seed"] = res.index.isin(seeds)
    res = res.join(kg.groupby("gene")["category"].agg(lambda s: ";".join(sorted(set(s)))).rename("curated_category"))
    if len(dm):
        res = res.join(dm[["branch", "branch_rank", "folds_in_branch_top20", "within_branch_importance"]]
                       .add_prefix("deepmolft_"), how="left")
    res = res.sort_values("priority_score", ascending=False)
    res.index.name = "gene"
    res.to_csv(out / "target_ranking_all.csv")

    # ---- consensus: high priority here AND among DeepMOLFT's top SHAP genes (independent method) ----------
    if len(dm):
        cn = int(rcfg.get("model_importance", {}).get("consensus_top", 100))
        top_dm = set(dm.index[dm["branch_rank"] <= int(rcfg.get("model_importance", {}).get("deepmolft_branch_top", 50))])
        cons = res[(res["rank"] <= cn) & res.index.isin(top_dm) & (res["meta_FDR"] < 0.05) & (res["loo_max_P"] < 0.05)].copy()
        cons["deepmolft_direction_agrees"] = np.sign(dm["direction_corr"].reindex(cons.index)) == np.sign(cons["meta_logFC"])
        cons = cons[cons["deepmolft_direction_agrees"]]
        # tier 1: robust >= 1.5-fold meta candidate that survives the infiltration adjustment; tier 2: the rest
        cons["consensus_tier"] = np.where(cons.index.isin(cand.index) & cons["infiltration_robust"], 1, 2)
        cons.sort_values(["consensus_tier", "rank"]).to_csv(out / "target_consensus.csv")
        res["consensus_tier"] = cons["consensus_tier"].reindex(res.index)
        res.to_csv(out / "target_ranking_all.csv")
        ctx.log.info("CONSENSUS (pipeline top-%d AND DeepMOLFT branch top-50, same direction, robust): tier 1 (%d): %s | "
                     "tier 2 (%d): %s", cn, int((cons["consensus_tier"] == 1).sum()),
                     list(cons.index[cons["consensus_tier"] == 1]), int((cons["consensus_tier"] == 2).sum()),
                     list(cons.index[cons["consensus_tier"] == 2]))
    gi = res[(res["top_compartment"] != "immune") & res["infiltration_robust"]]
    gi.assign(rank_within=np.arange(1, len(gi) + 1)).to_csv(out / "target_ranking_gland_intrinsic.csv")
    im = res[res["top_compartment"] == "immune"]
    im.assign(rank_within=np.arange(1, len(im) + 1)).to_csv(out / "target_ranking_immune.csv")
    L.to_csv(out / "layer_scores.csv")
    res[["rank", "rank_median_sens", "rank_iqr_sens", "P_top20_sens", f"P_top{topn}_sens"]].to_csv(out / "weight_sensitivity.csv")

    show = ["priority_score", "meta_logFC", "meta_FDR", "top_cell_type", "infiltration_robust", "genetics",
            "tractability", "clinical_sjogren_target", "P_top20_sens"]
    ctx.log.info("TOP 25 (%s mode):\n%s", mode, res.head(25)[show].round(3).to_string())
    ctx.log.info("top 25 compartments: %s | infiltration-robust in top 25: %d | gland-intrinsic list: %d genes",
                 res.head(25)["top_compartment"].value_counts().to_dict(), int(res.head(25)["infiltration_robust"].sum()), len(gi))

    # ---- figure ---------------------------------------------------------------------------------
    top = res.head(30)
    contrib = (top[w.index] * w).mul(penalty.reindex(top.index), axis=0).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 8))
    left = np.zeros(len(contrib))
    palette = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd", "#937860", "#da8bc3"]
    for col, color in zip(w.index, palette):
        ax.barh(range(len(contrib)), contrib[col], left=left, color=color, label=col)
        left += contrib[col].to_numpy()
    ax.set_yticks(range(len(contrib)))
    ax.set_yticklabels([f"{g}{' *' if c else ''}" for g, c in zip(contrib.index, top["clinical_sjogren_target"].iloc[::-1])],
                       fontsize=8)
    ax.set_xlabel(f"weighted priority score, {mode} mode (* = clinical-stage Sjogren target, held out)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "top30_layer_contributions.png", dpi=150)
    plt.close(fig)
    ctx.log.info("stage 7 done -> %s", out)
