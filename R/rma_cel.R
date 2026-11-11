# RMA (background correction + quantile normalisation + median polish) from Affymetrix CEL files.
#
#   Rscript R/rma_cel.R <cel_dir> <out.csv.gz>
#
# Output: probe-set x sample log2 expression; sample names are the GSM accession at the start of each file name.
suppressPackageStartupMessages({
  library(affy)
})
args <- commandArgs(trailingOnly = TRUE)
cel_dir <- args[1]
out <- args[2]
files <- list.files(cel_dir, pattern = "\\.cel(\\.gz)?$", ignore.case = TRUE, full.names = TRUE)
if (length(files) == 0) stop("no CEL files in ", cel_dir)
eset <- justRMA(filenames = basename(files), celfile.path = cel_dir)
m <- exprs(eset)
colnames(m) <- sub("^(GSM[0-9]+).*$", "\\1", colnames(m))
con <- gzfile(out, "w")
write.csv(data.frame(probe = rownames(m), m, check.names = FALSE), con, row.names = FALSE)
close(con)
cat(sprintf("RMA: %d probe sets x %d arrays -> %s\n", nrow(m), ncol(m), out))
