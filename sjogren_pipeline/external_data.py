"""External evidence for target ranking, each cached under ``data/external`` (cache -> online -> empty fallback).

* Open Targets Platform (GraphQL):
    - disease associations for Sjogren syndrome (overall + per-datatype scores; ``genetic_association`` is used)
    - per-target tractability, essentiality, safety liabilities, number of clinical-stage drugs (any indication)
    - clinical-stage Sjogren drug targets (held out: used only to *evaluate* the ranking)
* STRING (high-confidence PPI, combined score >= 700) among candidate genes and seed genes
* Human Protein Atlas salivary-gland single-cell clusters (gene x cell type expression)

Tractability is summarised from structural / localisation evidence only (no "Approved Drug" / "Clinical"
buckets), so the clinical Sjogren targets remain a fair benchmark.
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .context import Context

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
STRING_URL = "https://string-db.org/api/tsv/network"
HPA_CLUSTER_URL = "https://www.proteinatlas.org/download/tsv/rna_single_cell_cluster.tsv.zip"

IMMUNE = {"t-cells", "b-cells", "plasma cells", "macrophages", "monocytes", "neutrophils", "nk-cells", "mast cells",
          "innate lymphoid cells", "dendritic cells"}
EPITHELIAL = {"salivary acinar cells", "salivary duct cells", "salivary basal cells", "salivary myoepithelial cells",
              "salivary ionocytes"}
BROAD_TAU = 0.8        # tau below this over the 20 salivary-gland cell types -> "broadly expressed" (ACTB: 0.74)
MIN_REF_CPM = 10.0     # max CPM below this -> not reliably detected in the (healthy) reference, e.g. IFN-induced CXCL9/10
# tractability buckets that do not depend on clinical precedence
SM_STRUCT = {"Structure with Ligand", "High-Quality Ligand", "High-Quality Pocket", "Med-Quality Pocket",
             "Druggable Family"}
AB_LOC = {"UniProt loc high conf", "GO CC high conf", "UniProt loc med conf", "UniProt SigP or TMHMM",
          "GO CC med conf", "Human Protein Atlas loc"}
CLINICAL = {"Approved Drug", "Advanced Clinical", "Phase 1 Clinical"}


def _ot(query: str, variables: dict = None, retries: int = 3) -> dict:
    import requests

    for i in range(retries):
        try:
            r = requests.post(OT_URL, json={"query": query, "variables": variables or {}}, timeout=120)
            js = r.json()
            if "errors" in js:
                raise RuntimeError(js["errors"][:1])
            return js["data"]
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))


def _cache_dir(ctx: Context, name: str) -> Path:
    p = ctx.external / name
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------
# HPA salivary-gland single-cell reference
# --------------------------------------------------------------------------
def hpa_salivary_reference(ctx: Context) -> pd.DataFrame:
    """Gene-symbol x cell-type CPM (read counts pooled over clusters of the same cell type)."""
    cache = _cache_dir(ctx, "hpa") / "salivary_celltype_cpm.csv.gz"
    if cache.exists():
        return pd.read_csv(cache, index_col=0)
    src = ctx.external / "hpa" / "rna_single_cell_cluster.tsv.zip"
    if not src.exists():
        import urllib.request

        ctx.log.info("  downloading HPA single-cell clusters")
        urllib.request.urlretrieve(HPA_CLUSTER_URL, src)
    rows = []
    with zipfile.ZipFile(src) as z, z.open(z.namelist()[0]) as fh:
        for chunk in pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8"), sep="\t", chunksize=500_000):
            rows.append(chunk[chunk["Tissue"] == "salivary gland"])
    d = pd.concat(rows)
    d = d[~d["Cell type"].isin(["mixed cell types"]) & ~d["Cell type"].str.startswith("cycling")]
    rc = d.groupby(["Gene name", "Cell type"])["Read count"].sum().unstack(fill_value=0)
    cpm = rc / rc.sum(axis=0) * 1e6
    cpm.index.name = "gene"
    cpm.to_csv(cache)
    return cpm


def compartment_of(cell_type: str) -> str:
    if cell_type in IMMUNE:
        return "immune"
    if cell_type in EPITHELIAL:
        return "epithelial"
    return "stromal"


def celltype_profile(ref: pd.DataFrame, genes: List[str]) -> pd.DataFrame:
    """Per gene: top cell type, compartment, tau specificity, immune share of expression."""
    e = ref.reindex(genes)
    mx = e.max(axis=1)
    tau = (1.0 - e.div(mx.replace(0, np.nan), axis=0)).sum(axis=1) / (e.shape[1] - 1)
    comp = pd.Series({c: compartment_of(c) for c in e.columns})
    by_comp = e.T.groupby(comp).max().T                 # max CPM within each compartment
    share = by_comp.div(by_comp.sum(axis=1).replace(0, np.nan), axis=0)
    top = e.fillna(-1.0).idxmax(axis=1)                 # genes absent from the reference -> NaN below
    out = pd.DataFrame({"top_cell_type": top, "tau": tau, "max_cpm": mx,
                        "immune_share": share.get("immune"), "epithelial_share": share.get("epithelial"),
                        "stromal_share": share.get("stromal")}, index=genes)
    out["top_compartment"] = out["top_cell_type"].map(lambda c: compartment_of(c) if isinstance(c, str) else np.nan)
    broad = out["tau"] < BROAD_TAU                      # ubiquitous genes (e.g. interferon-stimulated genes)
    out.loc[broad, "top_compartment"] = "broad"
    out.loc[broad, "top_cell_type"] = "broad (" + out.loc[broad, "top_cell_type"].astype(str) + " highest)"
    low = out["max_cpm"].fillna(0) < MIN_REF_CPM        # cell of origin unknown -> neutral immune share
    out.loc[low, "top_compartment"] = "low_in_reference"
    out.loc[low, "top_cell_type"] = "low in healthy reference"
    out.loc[low, ["immune_share", "epithelial_share", "stromal_share"]] = np.nan
    out.loc[mx.isna() | (mx <= 0), ["top_cell_type", "top_compartment"]] = np.nan
    return out


# --------------------------------------------------------------------------
# Open Targets
# --------------------------------------------------------------------------
def ot_disease_associations(ctx: Context, disease: str) -> pd.DataFrame:
    cache = _cache_dir(ctx, "opentargets") / f"associations_{disease}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    q = """query($id:String!, $i:Int!){ disease(efoId:$id){ associatedTargets(page:{index:$i,size:500}){ count
             rows{ target{ id approvedSymbol } score datatypeScores{ id score } } } } }"""
    rows, i = [], 0
    while True:
        a = _ot(q, {"id": disease, "i": i})["disease"]["associatedTargets"]
        for r in a["rows"]:
            row = {"ensembl": r["target"]["id"], "gene": r["target"]["approvedSymbol"], "ot_overall": r["score"]}
            row.update({f"ot_{x['id']}": x["score"] for x in r["datatypeScores"]})
            rows.append(row)
        i += 1
        if i * 500 >= a["count"]:
            break
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    ctx.log.info("  Open Targets: %d associated targets for %s", len(df), disease)
    return df


def ot_disease_clinical_targets(ctx: Context, disease: str) -> pd.DataFrame:
    cache = _cache_dir(ctx, "opentargets") / f"clinical_targets_{disease}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    q = """query($id:String!){ disease(efoId:$id){ drugAndClinicalCandidates{ rows{ maxClinicalStage
             drug{ name mechanismsOfAction{ rows{ targets{ approvedSymbol } } } } } } } }"""
    rows = []
    for r in _ot(q, {"id": disease})["disease"]["drugAndClinicalCandidates"]["rows"]:
        drug = r.get("drug") or {}
        for m in ((drug.get("mechanismsOfAction") or {}).get("rows") or []):
            for t in m.get("targets") or []:
                rows.append({"gene": t["approvedSymbol"], "drug": drug.get("name"), "stage": r["maxClinicalStage"]})
    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(cache, index=False)
    return df


def ot_target_annotations(ctx: Context, ensembl: Dict[str, str]) -> pd.DataFrame:
    """symbol -> ensembl map in, per-gene annotation table out (cached per Ensembl ID)."""
    cache = _cache_dir(ctx, "opentargets") / "targets.jsonl"
    have = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            have[r["id"]] = r
    need = sorted({e for e in ensembl.values() if isinstance(e, str)} - set(have))
    q = """query($ids:[String!]!){ targets(ensemblIds:$ids){ id approvedSymbol isEssential
             tractability{ label modality value } safetyLiabilities{ event }
             drugAndClinicalCandidates{ count rows{ maxClinicalStage } } } }"""
    with open(cache, "a", encoding="utf-8") as fh:
        for i in range(0, len(need), 100):
            for r in _ot(q, {"ids": need[i:i + 100]})["targets"] or []:
                have[r["id"]] = r
                fh.write(json.dumps(r) + "\n")
    if need:
        ctx.log.info("  Open Targets: annotated %d new targets", len(need))
    rows = []
    for sym, e in ensembl.items():
        r = have.get(e)
        if r is None:
            rows.append({"gene": sym})
            continue
        tr = [(t["label"], t["modality"]) for t in r.get("tractability") or [] if t["value"]]
        sm = any(m == "SM" and lab in SM_STRUCT for lab, m in tr)
        ab = any(m == "AB" and lab in AB_LOC for lab, m in tr)
        pr = any(m == "PR" and lab not in CLINICAL for lab, m in tr)
        stages = [x["maxClinicalStage"] for x in ((r.get("drugAndClinicalCandidates") or {}).get("rows") or [])]
        rows.append({"gene": sym, "ensembl": e, "tract_small_molecule": sm, "tract_antibody": ab, "tract_protac": pr,
                     "tract_clinical_any_indication": any(lab in CLINICAL for lab, _ in tr),
                     "n_clinical_drugs_any_indication": len(stages),
                     "approved_drug_any_indication": "APPROVAL" in stages,
                     "is_essential": bool(r.get("isEssential")),
                     "n_safety_liabilities": len({s["event"] for s in r.get("safetyLiabilities") or []})})
    return pd.DataFrame(rows).set_index("gene")


# --------------------------------------------------------------------------
# STRING
# --------------------------------------------------------------------------
def string_network(ctx: Context, genes: List[str], min_score: int = 700) -> pd.DataFrame:
    cache = _cache_dir(ctx, "string") / "network.tsv"
    genes = sorted(set(genes))
    if cache.exists():
        net = pd.read_csv(cache, sep="\t")
        done = set(pd.read_csv(cache.with_suffix(".genes.txt"), header=None)[0])
        if set(genes) <= done:
            return net
    import requests

    rows = []
    for i in range(0, len(genes), 1500):        # STRING accepts ~2000 identifiers per call
        chunk = genes[i:i + 1500]
        r = requests.post(STRING_URL, data={"identifiers": "\r".join(chunk), "species": 9606,
                                            "required_score": min_score, "caller_identity": "sjogren_pipeline"},
                          timeout=300)
        r.raise_for_status()
        rows.append(pd.read_csv(io.StringIO(r.text), sep="\t"))
    net = pd.concat(rows).rename(columns={"preferredName_A": "a", "preferredName_B": "b", "score": "combined_score"})
    net = net[["a", "b", "combined_score"]].drop_duplicates()
    net.to_csv(cache, sep="\t", index=False)
    pd.Series(genes).to_csv(cache.with_suffix(".genes.txt"), index=False, header=False)
    ctx.log.info("  STRING: %d edges (score >= %d) among %d genes", len(net), min_score, len(genes))
    return net
