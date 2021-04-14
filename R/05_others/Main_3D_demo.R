rm(list=ls())
.rs.restartR()

packageVersion("keras")
packageVersion("tensorflow")

if(T){
library(keras)
library(tfruns)
library(abind)
library(tidyverse)
#install.packages("BiocManager")
#BiocManager::install("EBImage")
library(EBImage); options(EBImage.display = "raster")
#install.packages("mmand")
library(mmand)
library(reticulate)
reticulate::use_python("/usr/local/bin/python", required =T)

#setwd("~/Rstudio_Keras_test/rMiW200229")
#dir("./R")

source("./R/createUNET.R", local = T)
source("./R/DL_model_plot_modi.R", local = T)
source("./R/ImageDataImport.R", local = T)
source("./R/ImageProcessing.R", local = T)
source("./R/ImageView.R", local = T)
source("./R/LearningRateSchedule.R", local = T)
source("./R/list2tensor.R", local = T)
source("./R/LossFunction.R", local = T)
}

#TrainImgDataset <- Demo3dTRAINDataImport()

TrainImgDataset <- Demo3dTRAINDataImport(WIDTH  = 256, HEIGHT = 256, Z=16, CHANNELS = 1)
str(TrainImgDataset)

ImageView3Ddataset(ImgArray=TrainImgDataset, Sample=1, ImgSection = 1, XYsize = 256, random=T)

# Save all
#ImageView3Ddataset(ImgArray=TrainImgDataset, Sample=1, ImgSection = 1, XYsize = 256, SaveAll=T)

## Model 作成
#rm("model")
model <- createUnetModel3D(inputImageSize = c( 256, 256, 20, 1),
                           numberOfOutputs = 1,
                           numberOfLayers = 2,
                           nfilters = 1,
                           convolutionKernelSize = c( 3, 3, 3 ),
                           deconvolutionKernelSize = c( 3, 3, 3 ),
                           poolSize = c( 2, 2, 2 ),
                           strides = c( 2, 2, 2 ))
summary(model)

modelRes <- createResUnetModel3D(inputImageSize= c( 256, 256, 16, 1),
                                numberOfOutputs = 1,
                                numberOfFiltersAtBaseLayer = 1,
                                bottleNeckBlockDepthSchedule = c( 3, 4 ),
                                convolutionKernelSize = c( 3, 3, 3 ),
                                deconvolutionKernelSize = c( 2, 2, 2 ),
                                dropoutRate = 0.1,
                                weightDecay = 0.001)

model <- modelRes
summary(model)

## plot the DL model
model %>% plot_model_modi(width=4, height=1.2)

## Compile
model <- model %>% compile(optimizer = optimizer_rmsprop(lr = 0.2),
                           #optimizer = optimizer_adam( lr = 0.1 ),
                           #optimizer = optimizer_adamax(lr = 0.1),
                           loss = loss_dice,
                           metrics = custom_metric( "multilabel_dice_coefficientR", multilabel_dice_coefficientR ))

model <- model %>% compile(optimizer = optimizer_rmsprop(lr = 0.2),
                           #optimizer = optimizer_adam( lr = 0.1 ),
                           #optimizer = optimizer_adamax(lr = 0.1),
                           loss = Singlelabel_3d_bce_dice_loss,
                           metrics = custom_metric("Singlelabel_3d_dice_coef", Singlelabel_3d_dice_coef))

## Fit parameters
EPOCHS <- 100; BATCH_SIZE <- 1

## Run fit
history <- model %>%
  fit(TrainImgDataset$Training,
      TrainImgDataset$TrainingGT,
      batch_size = BATCH_SIZE,
      epochs = EPOCHS,
      verbose = 1)

history

score01 <- model %>%
      evaluate(TrainImgDataset$Training,
               TrainImgDataset$TrainingGT, verbose = 1)

Y_hat <- predict(model, x = TrainImgDataset$Training, verbose=1)

str(Y_hat)
table(Y_hat[1, , , 1:16, ])
Data.pred <- list(Training=Y_hat, TrainingGT=TrainImgDataset$TrainingGT)
ImageView3Ddataset(ImgArray=Data.pred,
                   ImgSection = 1, XYsize = 256, random=T, SaveAll=T)


# mmand::threshold
Y_hat5 <- map(array_branch(Y_hat, 1),
                  .f = function(z) {mmand::threshold(z, 0.5, binarise = TRUE)})


str(Y_hat5)
table(Y_hat5[[1]][,,6,])


xTensor <- simplify2array(Y_hat5)
str(xTensor)
a <- aperm(xTensor, c(5, 1, 2, 3, 4))
str(a)

TrainImgDataset$Prediction <- list4tensor(Y_hat5)
str(TrainImgDataset)











