# R --slave --no-restore --file=STAARpipeline_Gene_Centric_Coding.R --args 22 0.01 ../../data/STAARpipeline/ ../../results/STAARpipeline 
library(gdsfmt) 
library(SeqArray) 
library(SeqVarTools) 
library(dplyr)
library(STAAR)  
library(GENESIS)
library(TxDb.Hsapiens.UCSC.hg38.knownGene) 
library(Matrix)  
library(SCANG)
library(STAARpipeline)  
library(glmnet)
library(caret)
library(SuperLearner)

source("./Burden_scores.R")


set.seed(1330)

args <- commandArgs(trailingOnly = TRUE)
chr <- as.integer(args[1])    
rare_maf_cutoff <- as.numeric(args[2])
gds_file <- as.character(args[3]) 
out_dir <- as.character(args[4]) 

cat("chr:", chr, "\n")
cat("rare_maf_cutoff:", rare_maf_cutoff, "\n")
cat("gds_file:", gds_file, "\n")
cat("out_dir:", out_dir, "\n")


ignore_gene <- data.frame(chr = character(), gene_name = character(), category=character(), stringsAsFactors = FALSE)
col_name = NULL
genes_info_chr_subset <- genes_info[genes_info[,2]==chr,] # 18445

## QC_label
QC_label <- "annotation/info/QC_label"
## variant_type
variant_type <- "variant"  # variant
# category <- "plof"  # all_categories
## geno_missing_imputation
geno_missing_imputation <- "mean"
## Annotation_dir
Annotation_dir <- "annotation/info/FunctionalAnnotation"
## Annotation channel
Annotation_name_catalog <- read.csv(paste0(out_dir, "/Step_0/Annotation_name_catalog.csv"))
## Annotation name
# Annotation_name <- c("CADD","LINSIGHT","FATHMM.XF","aPC.EpigeneticActive","aPC.EpigeneticRepressed","aPC.EpigeneticTranscription",
#                     "aPC.Conservation","aPC.LocalDiversity","aPC.Mappability","aPC.TF","aPC.Protein")


## log_dir
dir.create(paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Coding/chr", chr), recursive = TRUE)
log_file=paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Coding/chr", chr, "/chr", chr, "_", variant_type, ".log")
print(paste("Deal with chr", chr, "with", nrow(genes_info_chr_subset), "genes!"))
cat(paste("Deal with chr", chr, "with", nrow(genes_info_chr_subset), "genes!\n"), file=log_file)


### Significant Rare Variant Set's Burden Scores
genofile <- seqOpen(paste0(gds_file, "/Q0_unre_Caucasian_chr", chr, ".gds"))
sampleids <- seqGetData(genofile,"sample.id")  


execution_time <- system.time({       
    Sig_Burdens <- NULL
    ### Loop over significant rare variant sets
    ### Using IDs = sampleids to build burden scores for all individuals.
    for(i in 1:nrow(genes_info_chr_subset)){
        chr <- genes_info_chr_subset$chromosome_name[i]
        gene_name <- genes_info_chr_subset$hgnc_symbol[i]
        
        a_s <- Burden_Scores(region = "Coding",chr = chr, gene_name = gene_name,category = c("all"),
                    genofile = genofile,IDs = sampleids,rare_maf_cutoff=rare_maf_cutoff,rv_num_cutoff=2,
                    QC_label=QC_label,variant_type=variant_type,geno_missing_imputation=geno_missing_imputation,
                    Annotation_dir=Annotation_dir,Annotation_name_catalog=Annotation_name_catalog,silent = TRUE, log_file=log_file) 

        if(is.null(a_s)){ 
                ignore_gene <- rbind(ignore_gene, data.frame(chr = chr, gene_name = gene_name, category="all"))
        }else{
            for(category in c("plof","plof_ds","missense","disruptive_missense","synonymous","ptv","ptv_ds","all_categories","all_categories_incl_ptv")){
                a <- a_s[[category]]
                col_name <- c(col_name, paste0(gene_name, ":", category))

                if(length(a) == 1){  
                    ignore_gene <- rbind(ignore_gene, data.frame(chr = chr, gene_name = gene_name, category=category))
                    Sig_Burdens <- cbind(Sig_Burdens,matrix(0,nrow=length(sampleids),ncol = 1))
                }else{  
                    Sig_Burdens <- cbind(Sig_Burdens,a)
                }
            }
        }
    }
})

cat(paste("Total execution time:", execution_time["elapsed"], "seconds\n\n"), file=log_file, append = TRUE)
seqClose(genofile)

rownames(Sig_Burdens) = sampleids
colnames(Sig_Burdens) = col_name

ignore_gene_final = ignore_gene[ignore_gene$chr==chr,]
save(Sig_Burdens, file = paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Coding/chr", chr, "/chr_", chr, "_all_categories_", variant_type, ".RData"))
save(ignore_gene_final, file = paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Coding/ignore_gene_chr", chr, ".RData")) 



## Convert to npz files
library(reticulate)
np <- import("numpy")

config <- py_config()
config$numpy
print(config)

data_array <- as.matrix(Sig_Burdens)
np$savez(paste0(out_dir, "/Step_3/cutoff_", rare_maf_cutoff, "/Coding/chr", chr, "/chr_", chr, "_all_categories_", variant_type, ".npz"),
        data = data_array,
        colnames = colnames(Sig_Burdens),
        sampleid = rownames(Sig_Burdens))
