#get differential expressed gene and transcript from IsoSeq count
#File should be like ..(GeneID, PBID, count_s1, count_s2)
#usage Rscript --vanilla diff_exp.R count_file,
#written by Hyejin Cho

#! /usr/bin/Rscript
#args = commandArgs(trailingOnly=TRUE)

#if(length(args) < 1) {
#    stop("Arguments are not matched.")
#}

#count = read.table(args[1], header=TRUE, sep='\t')
##wd <- dir

count <- read.table(paste0(wd,'/','tmp_DF.txt'), header=TRUE, sep='\t')
count[is.na(count)] <- 0
#count[is.nan(count)] <- 0
##sample_count <- no

col_no = 3

calc_fisher_si <- function(x, sample_no, col1_start, t_col, conf) {
    intv = 1
    ijc_col = col1_start + 1
    diff = col_no * sample_no
    type=x[t_col]
    
    no <- seq(ijc_col, ijc_col+col_no * (sample_no-1), by=col_no)
    si_list1 <- vector()
    si_list2 <- vector()
    ic_list1 <- vector()
    ec_list1 <- vector()
    ic_list2 <- vector()
    ec_list2 <- vector()
    ord = 1
    for(i in no){
        ic_val1 =  as.numeric(x[i])
        ec_val1 = as.numeric(x[i+intv])
        ic_val2 = as.numeric(x[i+diff])
        ec_val2 = as.numeric(x[i+intv+diff])
        #print(c(ic_val1, ec_val1, ic_val2, ec_val2))
        ic_list1[ord] <- ic_val1
        ec_list1[ord] <- ec_val1
        ic_list2[ord] <- ic_val2
        ec_list2[ord] <- ec_val2
        si1 = 0.0
        #if (ic_val1 > 0 & ec_val1 > 0){
        if (ic_val1 > 0){
            si1 = ic_val1/(ic_val1+ec_val1)
        }
        si_list1[ord] = si1
        si2 = 0.0
        if (ic_val2 > 0){
            si2 = ic_val2/(ic_val2+ec_val2)
        }
        si_list2[ord] = si2
        ord = ord+1
    }
    
    ic_mean1 = round(mean(ic_list1))
    ec_mean1 = round(mean(ec_list1))
    ic_mean2 = round(mean(ic_list2))
    ec_mean2 = round(mean(ec_list2))
    
    #print(c(ic_mean1,ec_mean1,ic_mean2,ec_mean2))
    log2_fc = log2(mean(si_list1))-log2(mean(si_list2))
    mx = matrix(c(ic_mean1,ec_mean1,ic_mean2,ec_mean2), nrow=2)
    #print(mx)
    rlt = fisher.test(as.matrix(mx), conf.level=conf, workspace = 10000000, simulate.p.value=TRUE)
    list(mean(si_list1), mean(si_list2), log2_fc, rlt$p.value, rlt$estimate)
}


#get si and fisher's exact test results
tmp <- apply(count, 1, calc_fisher_si, sample_no=sample_count, col1_start=5, t_col=3, conf=0.95)
tmp.df = do.call(rbind, tmp)
tmp.dt = apply(tmp.df, 2, unlist)
df = cbind(count, tmp.dt)
colnames(df) = c(colnames(count), 'SI1','SI2', 'log2FC','p.value','odd_ratio')
df$q.value <- p.adjust(df$p.value, "fdr")
out.file = paste0(wd,'/', 'Splicing_DE.txt')
write.table(df, out.file, sep='\t', row.names=FALSE, col.names=TRUE, quote=FALSE)

