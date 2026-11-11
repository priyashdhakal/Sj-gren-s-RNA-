# Run a batch of limma differential-expression jobs described in a JSON file (written by stage 3).
#
#   Rscript R/limma_jobs.R <jobs.json>
#
# Each job:
#   name, kind ("array" | "counts" | "logexpr"), expr (csv[.gz], first column = gene, then samples),
#   meta (csv, first column = sample id), formula (e.g. "~0 + group + batch + infil"),
#   group_col, levels, contrasts {name: [case, control]}, block (optional meta column for
#   duplicateCorrelation), array_weights (bool), out_dir,
#   permutations (optional): {n, seed, within (meta column or null)} -> label-permutation calibration.
#
# Arrays / log-expression: [arrayWeights] -> lmFit -> eBayes(trend = TRUE, robust = TRUE).
# Counts: TMM (edgeR) -> voomWithQualityWeights -> lmFit -> eBayes(robust = TRUE).
# Output per contrast: <out_dir>/<name>__<contrast>.csv with gene, logFC, SE, t, P.Value, adj.P.Val, AveExpr, df.
suppressPackageStartupMessages({
  library(limma)
  library(edgeR)
  library(jsonlite)
})

read_expr <- function(path) {
  x <- read.csv(path, check.names = FALSE, row.names = 1)
  as.matrix(x)
}

build_design <- function(meta, formula, group_col, levels) {
  meta[[group_col]] <- factor(meta[[group_col]], levels = levels)
  d <- model.matrix(as.formula(formula), data = meta)
  colnames(d) <- sub(paste0("^", group_col), "", colnames(d))
  colnames(d) <- make.names(colnames(d))
  d
}

fit_job <- function(job, E, meta, design, quiet = FALSE) {
  block <- if (!is.null(job$block) && nzchar(job$block)) meta[[job$block]] else NULL
  kind <- job$kind
  if (kind == "counts") {
    dge <- calcNormFactors(DGEList(E))
    v <- voomWithQualityWeights(dge, design, plot = FALSE)
    if (!is.null(block)) {
      cor <- duplicateCorrelation(v, design, block = block)$consensus.correlation
      v <- voomWithQualityWeights(dge, design, plot = FALSE, block = block, correlation = cor)
      cor <- duplicateCorrelation(v, design, block = block)$consensus.correlation
      fit <- lmFit(v, design, block = block, correlation = cor)
    } else {
      fit <- lmFit(v, design)
    }
    return(list(fit = fit, trend = FALSE))
  }
  w <- NULL
  if (isTRUE(job$array_weights)) w <- arrayWeights(E, design)
  if (!is.null(block)) {
    cor <- duplicateCorrelation(E, design, block = block, weights = w)$consensus.correlation
    fit <- lmFit(E, design, block = block, correlation = cor, weights = w)
  } else {
    fit <- lmFit(E, design, weights = w)
  }
  list(fit = fit, trend = TRUE)
}

contrast_tables <- function(job, ft, design) {
  cs <- job$contrasts
  cm <- sapply(names(cs), function(n) {
    v <- setNames(rep(0, ncol(design)), colnames(design))
    v[make.names(cs[[n]][[1]])] <- 1
    v[make.names(cs[[n]][[2]])] <- -1
    v
  })
  cm <- matrix(cm, nrow = ncol(design), dimnames = list(colnames(design), names(cs)))
  f2 <- eBayes(contrasts.fit(ft$fit, cm), trend = ft$trend, robust = TRUE)
  out <- list()
  for (j in seq_len(ncol(cm))) {
    tt <- topTable(f2, coef = j, number = Inf, sort.by = "none")
    tt$SE <- sqrt(f2$s2.post) * f2$stdev.unscaled[, j]
    tt$df <- f2$df.total
    tt$gene <- rownames(tt)
    out[[colnames(cm)[j]]] <- tt[, c("gene", "logFC", "SE", "t", "P.Value", "adj.P.Val", "AveExpr", "df")]
  }
  out
}

run_job <- function(job) {
  E <- read_expr(job$expr)
  meta <- read.csv(job$meta, row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)
  meta <- meta[colnames(E), , drop = FALSE]
  design <- build_design(meta, job$formula, job$group_col, unlist(job$levels))
  if (qr(design)$rank < ncol(design)) stop(job$name, ": design not full rank")
  ft <- fit_job(job, E, meta, design)
  res <- contrast_tables(job, ft, design)
  dir.create(job$out_dir, showWarnings = FALSE, recursive = TRUE)
  for (n in names(res)) write.csv(res[[n]], file.path(job$out_dir, paste0(job$name, "__", n, ".csv")), row.names = FALSE)
  cat(sprintf("[%s] %s: %d genes, %s\n", job$name, job$kind, nrow(E),
              paste(sprintf("%s FDR<0.05 = %d", names(res), sapply(res, function(t) sum(t$adj.P.Val < 0.05))), collapse = "; ")))

  p <- job$permutations
  if (!is.null(p) && p$n > 0) {
    set.seed(p$seed)
    first <- names(job$contrasts)[1]
    pair <- unlist(job$contrasts[[first]])
    sel <- which(meta[[job$group_col]] %in% pair)
    rows <- list()
    for (r in seq_len(p$n)) {
      m2 <- meta
      strata <- if (!is.null(p$within) && nzchar(p$within)) m2[[p$within]] else rep("all", nrow(m2))
      for (s in unique(strata[sel])) {
        ids <- sel[strata[sel] == s]
        m2[[job$group_col]][ids] <- sample(m2[[job$group_col]][ids])
      }
      d2 <- build_design(m2, job$formula, job$group_col, unlist(job$levels))
      j2 <- job
      j2$contrasts <- job$contrasts[first]
      tt <- contrast_tables(j2, fit_job(job, E, m2, d2), d2)[[1]]
      rows[[r]] <- data.frame(perm = r, n_fdr = sum(tt$adj.P.Val < 0.05), frac_p05 = mean(tt$P.Value < 0.05),
                              min_p = min(tt$P.Value))
    }
    write.csv(do.call(rbind, rows), file.path(job$out_dir, paste0(job$name, "__permutations.csv")), row.names = FALSE)
  }
}

jobs <- fromJSON(commandArgs(trailingOnly = TRUE)[1], simplifyVector = FALSE)
for (job in jobs) run_job(job)
