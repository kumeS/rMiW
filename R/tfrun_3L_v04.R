## tfruns
rm(list=ls())
#install.packages("tfruns")
library(tfruns)
########################################################################
getwd()
#dir()
########################################################################
#setwd(paste0("run_results"))
tuning_run("../R/tfrun_3L_3D_Analysis_v04.R",
           runs_dir="runs_3L_004",
           flags = list(size = c(256),
                        #EPOCHS = c(5),
                        EPOCHS = c(5000),
                        BATCH_SIZE = c(1, 1, 1),
                        IL = c(0.005)),
           sample=1,
           confirm = F)

##確認
#ls_runs()
#View(ls_runs())
#compare_runs()
#dir()


