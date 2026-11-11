"""Synthetic checks for the new v2 components (no network, no R).

    python tests/test_pipeline_synthetic.py

  1. single-cell path (scrna.py): per-sample 10x files -> QC -> HPA-marker annotation -> pseudobulk
  2. deconvolution: NNLS recovers planted immune fractions
  3. random-effects meta-analysis: recovers a planted effect, keeps null genes calibrated, tau2 > 0 under
     heterogeneity
"""
from __future__ import annotations

import gzip
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sjogren_pipeline import scrna  # noqa: E402
from sjogren_pipeline.deconvolution import deconvolve  # noqa: E402
from sjogren_pipeline.stage03_diffexpr import dl_meta  # noqa: E402

TYPES = ["b-cells", "salivary acinar cells", "fibroblasts"]


def make_reference(rng, n_genes=600) -> pd.DataFrame:
    genes = [f"G{i}" for i in range(n_genes)]
    ref = pd.DataFrame(rng.gamma(1.0, 5.0, (n_genes, len(TYPES))), index=genes, columns=TYPES)
    for j, t in enumerate(TYPES):                     # 60 strong markers per type
        ref.iloc[j * 60:(j + 1) * 60, j] = rng.uniform(400, 900, 60)
    return ref / ref.sum() * 1e6


def test_single_cell(rng, ref: pd.DataFrame) -> bool:
    tmp = Path(tempfile.mkdtemp())
    genes = list(ref.index) + ["MT-CO1", "PLANTED"]
    truth = []
    for s in range(6):
        cells, ct = [], rng.choice(len(TYPES), 300)
        for c in ct:
            lam = ref.iloc[:, c].to_numpy() / 1e6 * 3000
            lam = np.r_[lam, 20.0, 5.0 if (s >= 3 and c == 1) else 0.2]   # PLANTED up in acinar cells of cases
            cells.append(rng.poisson(lam))
        truth.append(np.array(TYPES)[ct])
        M = sp.csr_matrix(np.column_stack(cells))
        stem = tmp / f"GSM{s}_S{s}"
        with gzip.open(f"{stem}_matrix.mtx.gz", "wb") as fh:
            mmwrite(fh, M)
        with gzip.open(f"{stem}_features.tsv.gz", "wt") as fh:
            fh.write("".join(f"ENSG{i}\t{g}\tGene Expression\n" for i, g in enumerate(genes)))
        with gzip.open(f"{stem}_barcodes.tsv.gz", "wt") as fh:
            fh.write("".join(f"C{i}-1\n" for i in range(M.shape[1])))
    files = scrna.sample_files(tmp)
    accs, pb = [], {}
    for k, (gsm, (mtx, feat, _)) in enumerate(sorted(files.items())):
        X, g = scrna.load_sample(mtx, feat, min_genes=50)
        ct = scrna.annotate(scrna.lognorm(X), g, ref, n_markers=30).to_numpy()
        accs.append((ct == truth[k][:X.shape[1]]).mean())
        pb[gsm] = scrna.pseudobulk(X, g, ct == "salivary acinar cells")
    P = pd.DataFrame(pb)
    lfc = np.log2(P.loc["PLANTED", ["GSM3", "GSM4", "GSM5"]].mean() / P.loc["PLANTED", ["GSM0", "GSM1", "GSM2"]].mean())
    ok = len(files) == 6 and np.mean(accs) > 0.9 and lfc > 2
    print(f"{'OK  ' if ok else 'FAIL'} single-cell: {len(files)} samples, annotation accuracy {np.mean(accs):.1%}, "
          f"planted acinar pseudobulk log2FC {lfc:.2f}")
    return ok


def test_deconvolution(rng, ref: pd.DataFrame) -> bool:
    W = rng.dirichlet([1, 1, 1], 40)                  # true proportions
    bulk = (ref.to_numpy() @ W.T) * rng.lognormal(0, 0.1, (len(ref), 40))
    expr = pd.DataFrame(np.log2(bulk + 1).T, columns=ref.index)
    est = deconvolve(expr, ref, n_per_type=40, min_spec=3.0)
    rho = spearmanr(est["b-cells"], W[:, 0])[0]
    ok = rho > 0.9 and np.isclose(est["immune_fraction"], est["b-cells"]).all()
    print(f"{'OK  ' if ok else 'FAIL'} deconvolution: Spearman(estimated, true B-cell fraction) = {rho:.3f}")
    return ok


def test_meta(rng) -> bool:
    n, k = 4000, 6
    se = rng.uniform(0.2, 0.5, (n, k))
    beta = np.zeros(n)
    beta[:200] = 0.8
    b = beta[:, None] + rng.normal(0, se)
    b[200:300] += rng.normal(0, 1.0, (100, k))         # heterogeneous null genes
    m = dl_meta(pd.DataFrame(b), pd.DataFrame(se), 4)
    power = (m["meta_FDR"].iloc[:200] < 0.05).mean()
    fpr = (m["meta_P"].iloc[300:] < 0.05).mean()
    het = m["tau2"].iloc[200:300].median() > m["tau2"].iloc[300:].median()
    ok = power > 0.9 and abs(fpr - 0.05) < 0.02 and het
    print(f"{'OK  ' if ok else 'FAIL'} meta-analysis: power {power:.2f}, null P<0.05 rate {fpr:.3f}, tau2 detects heterogeneity: {het}")
    return ok


def main() -> int:
    rng = np.random.default_rng(0)
    ref = make_reference(rng)
    ok = test_single_cell(rng, ref) & test_deconvolution(rng, ref) & test_meta(rng)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
