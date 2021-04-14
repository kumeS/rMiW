rm(list=ls())
#packageVersion("keras")
#packageVersion("tensorflow")
#browseURL("https://keras.rstudio.com/reference/index.html")

FLAGS <- flags(
  flag_numeric("size", 256),
  flag_numeric("EPOCHS", 10),
  flag_numeric("BATCH_SIZE", 24),
  flag_numeric("IL", 0.001)
)

library(keras)
library(tfruns)
library(abind)
library(tidyverse)
library(EBImage); options(EBImage.display = "raster")
library(mmand)
library(reticulate)
#reticulate::use_python("/usr/local/bin/python", required =T)
#reticulate::py_config()
#setwd("~/Rstudio_Keras_test/rMiW_200710")
#dir("./R")

source("../R/Model_3L_v1.R", local = T)
source("../R/DL_model_plot_modi_v01.R", local = T)
source("../R/ImageDataImport_v02.R", local = T)
source("../R/ImageProcessing_v01.R", local = T)
source("../R/ImageView_v01.R", local = T)
source("../R/LearningRateSchedule_v01.R", local = T)
source("../R/list2tensor_v01.R", local = T)
source("../R/LossFunction_v01.R", local = T)

NNN <- FLAGS$size
TrainImgDataset <- Demo3dTRAINDataImport( WIDTH  = NNN, HEIGHT = NNN, Z=24 )
#str(TrainImgDataset)

#ImageView3Ddataset(ImgArray=TrainImgDataset, Sample=1, ImgSection = 1, XYsize = NNN, random=T)

# Save all
#ImageView3Ddataset(ImgArray=TrainImgDataset, Sample=1, ImgSection = 1, XYsize = 256, SaveAll=T)
#system("convert -delay 20 -loop 1 ./3Dimg_000/*.png 3Dimg_000.gif")

## Model 作成
model <- create3DModel_3L_v2(inputImageSize = c( NNN, NNN, 24, 1))
summary(model)

## plot the DL model
#model %>% plot_model_modi(width=4, height=1.2)
py_plot_model <- reticulate::import(module = "tensorflow")$keras$utils$plot_model
py_plot_model(model, to_file=paste0(NNN, '_Model.png'), show_shapes=T, show_layer_names=T)

## Compile
model <- model %>%
  compile(optimizer = optimizer_rmsprop(lr =  FLAGS$IL),
          loss = bce_dice_loss,
          metrics = custom_metric( "dice_coef", dice_coef ))

## Fit parameters
EPOCHS <- FLAGS$EPOCHS
BATCH_SIZE <- FLAGS$BATCH_SIZE

## Run fit
history <- model %>%
  fit(TrainImgDataset$Training,
      TrainImgDataset$TrainingGT,
      batch_size = BATCH_SIZE,
      epochs = EPOCHS,
      verbose = 1, validation_split = 0,
      callbacks = list(callback_early_stopping(monitor = "loss", min_delta = 0, patience = 500, mode = "auto")))
#callbacks = list( callback_learning_rate_scheduler(lr_schedule_1000))

#Fit 図
png(filename = paste0(formatC(NNN, width = 3, flag = "0"), '_Rplot.png'), width = 480, height = 480); plot(history); dev.off()

#モデル評価
score01 <- model %>%
  evaluate(TrainImgDataset$Training,
           TrainImgDataset$TrainingGT, verbose = 1)
print(score01$Singlelabel_3d_dice_coef)
#C[m,n] <- score01$Singlelabel_3d_dice_coef

save_model_hdf5(model, filepath=paste0(NNN, "_create3DModel_2L_v2.h5"))
save_model_weights_hdf5(model, filepath=paste0(NNN, "_create3DModel_2L_v2_w.h5"))

score01 <- model %>%
  evaluate(TrainImgDataset$Training,
           TrainImgDataset$TrainingGT, verbose = 1)

print(score01)
Y_hat <- predict(model, x = TrainImgDataset$Training, verbose=1)
str(Y_hat)

Data.pred <- list(Training=Y_hat, TrainingGT=TrainImgDataset$TrainingGT)
ImageView3Dresult(ImgArray=Data.pred,
                  ImgSection = 1, XYsize = 256, random=T, SaveAll=T)

#Transfer learning
#https://www.r-bloggers.com/2017/08/transfer-learning-with-keras-in-r/

#Test data







