# data/external

External resources. Each has a cache -> online -> fallback path; caches are created on first use (stage 3 / 7).

| file / folder | used by | notes |
|---|---|---|
| `hpa/rna_single_cell_cluster.tsv.zip` | stages 3, 7, 8 | Human Protein Atlas single-cell clusters. The salivary-gland clusters (acinar, duct, basal, myoepithelial, ionocyte, fibroblast, endothelial, T/B/plasma/NK/myeloid ...) form the deconvolution reference and the cell-type specificity layer. Cached as `hpa/salivary_celltype_cpm.csv.gz`. Note: healthy tissue, so IFN-induced genes (CXCL9/10, UBD) are barely detected and are flagged `low_in_reference`. |
| `hpa/rna_single_cell_type.tsv.zip` | stage 1 | Ensembl -> gene symbol mapping for RNA-seq count matrices. |
| `opentargets/` | stage 7 | Open Targets Platform (GraphQL) for Sjogren syndrome `MONDO_0010030`: associations (`genetic_association` datatype used), per-target tractability / essentiality / safety, and clinical-stage Sjogren drug targets (held out for evaluation only). Delete the folder to refresh. |
| `string/network.tsv` | stage 7 | STRING high-confidence PPI (combined score >= 700) among candidates and genetic seed genes. |
| `genesets/*.gmt` | stage 4 | Enrichr libraries (GO BP 2023, Reactome 2022, KEGG 2021, MSigDB Hallmark 2020); GO-BP is also built offline from the GPL570 annotation. |
| `cell_type_markers.csv` | stages 3, 6 | Compact canonical marker panel: infiltration score for log-ratio arrays (GSE127952) and group ablation in stage 6. |
| `pss_known_genes.csv` | stage 7 | Curated pSS genes. Only `GWAS*` rows feed the genetics layer; `therapeutic_target` rows are annotations (not scored, to keep the clinical-target evaluation fair). |
