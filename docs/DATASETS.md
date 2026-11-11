# Datasets

All data are public (NCBI GEO, Human Protein Atlas, Open Targets, STRING). Nothing is redistributed in this
repository: `python run_pipeline.py --stages 1` downloads the expression data into `data/raw/` (~1 GB) and the
external resources are cached in `data/external/` on first use.

## Bulk salivary-gland transcriptomes (meta-analysis)

| GEO series | Tissue | Platform / data used | Cases | Controls | Role | Notes |
|---|---|---|---|---|---|---|
| [GSE40611](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40611) | Parotid gland | Affymetrix HG-U133 Plus 2.0 (GPL570), raw CEL → RMA | 17 pSS | 18 healthy (+14 non-SS sicca) | Discovery, classical ML | 3 processing batches; noisy arrays form a technical sub-batch |
| [GSE154926](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE154926) | Minor salivary gland | RNA-seq, raw gene counts | 43 pSS | 7 healthy | Meta-analysis | Ensembl IDs mapped to symbols with HPA |
| [GSE173808](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE173808) | Parotid + labial gland (paired) | RNA-seq, raw gene counts | 75 pSS samples | 39 non-SS sicca samples | Meta-analysis; checks deconvolution | Two glands per patient (`duplicateCorrelation`); histological %CD45 per gland |
| [GSE157159](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE157159) | Minor salivary gland | RNA-seq, TPM | 9 SS | 8 normal | Meta-analysis | Columns matched to GEO titles by patient prefix |
| [GSE23117](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE23117) | Minor salivary gland | GPL570, series-matrix (MAS5) values | 10 SS | 4 non-SS | Meta-analysis | Batch 1 has only SS samples (batch confounded with diagnosis); 1 ambiguous sample removed |
| [GSE40568](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40568) | Labial salivary gland | GPL570, raw CEL → RMA | 5 SS | 3 healthy | Meta-analysis | 5 IgG4-related disease samples removed |
| [GSE127952](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE127952) | Minor salivary gland, low focus score | Agilent two-colour (GPL20995), log ratios | 8 pSS | 6 healthy | Meta-analysis | All pSS have focus score ≤ 2 (low infiltration) |

**Total:** 252 bulk samples in the case-vs-control comparisons, three gland types, four technologies.
Groups are assigned by regular expressions on the GEO sample metadata (`config.yaml → datasets → labels`); a sample
matching no rule stops the run.

## Single-cell validation (never used for ranking)

| GEO series | Tissue | Data | Cases | Controls | Use |
|---|---|---|---|---|---|
| [GSE272409](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272409) | Minor labial salivary gland | 10x scRNA-seq, per-sample matrices | 7 SjD | 6 non-SjD sicca | Cell of origin of targets; pseudobulk SjD vs sicca within immune, epithelial and stromal compartments |

Cells are kept with ≥ 500 detected genes and < 20% mitochondrial reads, and annotated with HPA salivary-gland
markers (1,863–4,953 cells per sample after QC, 42,577 in total).

## Excluded

* **GSE7451** – whole saliva, not gland tissue.
* **GSE157278** – peripheral blood mononuclear cells (used in an earlier version; replaced by gland single-cell data).

## External resources

| Resource | Used for | Version / access |
|---|---|---|
| Human Protein Atlas, single-cell clusters (salivary gland, 20 cell types) | Deconvolution reference, cell of origin, gene-symbol mapping | `rna_single_cell_cluster.tsv.zip`, downloaded on first use |
| Open Targets Platform (GraphQL API), disease `MONDO_0010030` | Genetic association, tractability, essentiality, safety; clinical-stage Sjögren's targets as a held-out benchmark | Queried live, cached in `data/external/opentargets/` |
| STRING (combined score ≥ 700) | Protein-interaction network layer | API, cached in `data/external/string/` |
| Enrichr / MSigDB libraries | Pathway analysis (GO BP 2023, Reactome 2022, KEGG 2021, Hallmark 2020) | Via gseapy, cached as GMT |
| `data/external/pss_known_genes.csv` | Curated Sjögren's GWAS loci (genetics layer) and annotation | In repository, with sources |
| `data/external/cell_type_markers.csv` | Compact canonical marker panel | In repository |

## Gene universe per analysis

| Analysis | Genes |
|---|---|
| Per-cohort differential expression | 12,556–20,697 after filtering (cohort dependent) |
| Meta-analysis | 12,953 genes measured in ≥ 4 cohorts |
| Classical models (GSE40611) | 5,000 most variable genes; 20–1,000 selected inside each training fold |
| DeepMOLFT | 7,616 genes measured in all 7 cohorts → 3,000 most variable → ~2,500 after the 0.85 correlation filter |
| Target ranking | 671 candidates = 527 meta-analysis candidates + 115 DeepMOLFT top genes + 29 GWAS genes |
