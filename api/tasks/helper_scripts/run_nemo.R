#devtools::install_github('Shamir-Lab/NEMO/NEMO')

args = commandArgs(trailingOnly=TRUE)
is_automatically_estimation_required <- as.integer(args[1])
num_subtype_user_defined <- as.integer(args[2])
source_path <- args[3]
target_path <- args[4]
output_path <- args[5]

library(NEMO)
library(SNFtool)

source <- read.csv(source_path, row.names = 1)
target <- read.csv(target_path, row.names = 1)
target_info <- target[, c("Batch", "domain_idx")]

target <- subset(target, select = -c(Batch, domain_idx))

source_target <- rbind(source, target)
source_target <- t(source_target)
omics.list = list(source_target)

if (is_automatically_estimation_required == 1){
  clustering <- nemo.clustering(omics.list)
} else {
  clustering <- nemo.clustering(omics.list, num.clusters = num_subtype_user_defined)
}

clustering_df <- as.data.frame(clustering)
clustering_df$clustering <- paste("Cluster", clustering_df$clustering)
colnames(clustering_df) = "Subtype"

write.csv(clustering_df, output_path, row.names = TRUE, quote = FALSE)
