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
#setwd("/Users/skume/Research/02_DL_Study/00_Package_rMiW/rMiW_201227R/run_results")
#dir("./R")
#getwd()

source("../R/Model_3L_v1.R", local = T)
source("../R/DL_model_plot_modi_v01.R", local = T)
source("../R/ImageDataImport_v02.R", local = T)
source("../R/ImageProcessing_v01.R", local = T)
source("../R/ImageView_v01.R", local = T)
source("../R/LearningRateSchedule_v01.R", local = T)
source("../R/list2tensor_v01.R", local = T)
source("../R/LossFunction_v01.R", local = T)

NNN <- FLAGS$size
TrainImgDataset <- TrakEM2_TrainDataImport( WIDTH  = NNN, HEIGHT = NNN, Z=24 )
#str(TrainImgDataset)

#ImageView3Ddataset(ImgArray=TrainImgDataset, Sample=1, ImgSection = 1, XYsize = NNN, random=T)

# Save all
#ImageView3Ddataset(ImgArray=TrainImgDataset, Sample=1, ImgSection = 1, XYsize = 256, SaveAll=T)
#system("convert -delay 20 -loop 1 ./3Dimg_000/*.png 3Dimg_000.gif")

## Model 作成
model <- create3DModel_3L_v2RR(inputImageSize = c( NNN, NNN, 24, 1))
summary(model)

## plot the DL model
#model %>% plot_model_modi(width=4, height=1.2)
py_plot_model <- reticulate::import(module = "tensorflow")$keras$utils$plot_model
py_plot_model(model, to_file=paste0(NNN, '_Model.png'), show_shapes=T, show_layer_names=T)

model <- multi_gpu_model(model, gpus = 2)
summary(model)

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
  fit(TrainImgDataset$Original,
      TrainImgDataset$GroundTruth,
      batch_size = BATCH_SIZE,
      epochs = EPOCHS,
      verbose = 1, validation_split = 0,
      callbacks = list(callback_early_stopping(monitor = "loss", min_delta = 0, patience = 500, mode = "auto")))
#callbacks = list( callback_learning_rate_scheduler(lr_schedule_1000))

#Fit 図
png(filename = paste0(formatC(NNN, width = 3, flag = "0"), '_Rplot.png'), width = 480, height = 480); plot(history); dev.off()

#Model 保存
save_model_hdf5(model, filepath=paste0(NNN, "_create3DModel_3L.h5"))
save_model_weights_hdf5(model, filepath=paste0(NNN, "_create3DModel_w.h5"))

#モデル評価
score01 <- model %>%
  evaluate(TrainImgDataset$Original,
           TrainImgDataset$GroundTruth, verbose = 1)
print(score01$Singlelabel_3d_dice_coef)
#C[m,n] <- score01$Singlelabel_3d_dice_coef

print(score01)
Y_hat <- predict(model, x = TrainImgDataset$Original, verbose=1)
str(Y_hat)

Data.pred <- list(Original=Y_hat, GroundTruth=TrainImgDataset$GroundTruth)
ImageView3Dresult(ImgArray=Data.pred,
                  ImgSection = 1, XYsize = 256, random=T, SaveAll=T)

#Transfer learning
#https://www.r-bloggers.com/2017/08/transfer-learning-with-keras-in-r/

###############################################################################################
###############################################################################################
#Resize Input Size
#model00 <- load_model_hdf5(filepath=paste0(NNN, "_create3DModel_3L.h5"), compile = F)
#model00 <- load_model_hdf5(filepath="/Users/skume/Research/02_DL_Study/00_Package_rMiW/Results_201227_3L/runs_3L_003_07/2020-12-28T06-25-08Z/256_create3DModel_3L.h5", compile = F)
str(model00)

#Non-trainable
for (layer in model00$layers){
layer$trainable <- FALSE
}

str(model00)
model01 <- keras_model(inputs = get_layer(model00, "conv3d")$input,
                       outputs = get_layer(model00, "conv3d_21")$input)
model01

#Model再作成
inputs <- layer_input( shape = c(256,256,6,1) )
outputs <- inputs %>%
    layer_conv_3d( filters = 2 , kernel_size = 3, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d_transpose( filters = 1 , kernel_size = 3, strides = c(1,1,4), padding = 'same', activation = 'relu') %>%
    model01 %>%
    layer_max_pooling_3d( pool_size = c(1,1,4) ) %>%
    layer_conv_3d( filters = 2 , kernel_size = 3, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = 1, kernel_size = 3, padding = 'same' )

Model02 <- keras_model( inputs = inputs, outputs = outputs )
Model02

## plot the DL model
#model %>% plot_model_modi(width=4, height=1.2)
py_plot_model <- reticulate::import(module = "tensorflow")$keras$utils$plot_model
py_plot_model(Model02, to_file=paste0(NNN, '_Model.png'), show_shapes=T, show_layer_names=T)

#Test data
#Model 再Fit
str(Model02)

TrainImgDataset2 <- TrakEM2_TrainDataImport2( WIDTH  = NNN, HEIGHT = NNN, Z=6 )
str(TrainImgDataset2)

## Compile
source("../R/LossFunction_v01.R", local = T)
Model02 <- Model02 %>%
  compile(optimizer = optimizer_rmsprop(lr =  FLAGS$IL),
          loss = bce_dice_loss,
          metrics = custom_metric( "dice_coef", dice_coef ))

## Fit parameters
EPOCHS <- FLAGS$EPOCHS
BATCH_SIZE <- FLAGS$BATCH_SIZE

## Run fit
history02 <- Model02 %>%
  fit(TrainImgDataset2$Original,
      TrainImgDataset2$GroundTruth,
      batch_size = BATCH_SIZE,
      epochs = EPOCHS,
      verbose = 1, validation_split = 0,
      callbacks = list(callback_early_stopping(monitor = "loss", min_delta = 0, patience = 500, mode = "auto")))

#Trainable
for (layer in Model02$layers){
layer$trainable <- TRUE
}

#fine-turing
history03 <- Model02 %>%
  fit(TrainImgDataset2$Original,
      TrainImgDataset2$GroundTruth,
      batch_size = BATCH_SIZE,
      epochs = 100,
      verbose = 1, validation_split = 0,
      callbacks = list(callback_early_stopping(monitor = "loss", min_delta = 0, patience = 500, mode = "auto")))

#モデル評価
score02 <- Model02 %>%
  evaluate(TrainImgDataset2$Original,
           TrainImgDataset2$GroundTruth, verbose = 1)
print(score02$Singlelabel_3d_dice_coef)

print(score02)
Y_hat2 <- predict(Model02, x = TrainImgDataset2$Original, verbose=1)
str(Y_hat2)

Data.pred2 <- list(Original=Y_hat2, GroundTruth=TrainImgDataset2$GroundTruth)
ImageView3Dresult(ImgArray=Data.pred2,
                  ImgSection = 1, XYsize = 256, random=T, SaveAll=T)

#Prediction for Test dataset
TestImgDataset <- TrakEM2_TestDataImport( WIDTH  = NNN, HEIGHT = NNN, Z=6 )
str(TestImgDataset)

Y_hat3 <- predict(Model02, x = TestImgDataset$Original, verbose=1)
str(Y_hat3)

# mmand::threshold
try(Y_lab <- map(array_branch(Y_hat3, 1),
               .f = function(z) {mmand::threshold(z, 0.5, binarise = TRUE)}) , silent=TRUE)
str(Y_lab)
Y_lab01 <- aperm(simplify2array(Y_lab), c(5, 1, 2, 3, 4))
str(Y_lab01)

## Model 評価
try(score03 <- Model02 %>% evaluate(
 TestImgDataset$Original, TestImgDataset$GroundTruth, verbose = 0), silent=TRUE)

try(cat('Test loss:', score03[[1]], '\n'), silent=TRUE)
try(cat('Test accuracy:', score03[[2]], '\n'), silent=TRUE)

str(TestImgDataset)
#head()
PATH <- "/Users/skume/Research/02_DL_Study/00_Package_rMiW/rMiW_201228/data/02_TrakEM2_tifs/02_TestData/OriginalData"
AVE <- rep(NA, length(dir(PATH)))
for(ABC in 1:length(dir(PATH))){
#ABC <- 1
 intersection <- sum( matrix(Y_lab01[ABC,,,,]) * matrix(TestImgDataset$GroundTruth[ABC,,,,]))
 union <- sum( matrix(Y_lab01[ABC,,,,]) + matrix(TestImgDataset$GroundTruth[ABC,,,,]) ) - intersection
 result <- (intersection + 1) / ( union + 1)
 AVE[ABC] <- result
}
mean(AVE)
sd(AVE)

str(Y_lab01)
str(TestImgDataset$GroundTruth)

Data.pred3 <- list(Original=Y_lab01, GroundTruth=TestImgDataset$GroundTruth)
ImageView3Dresult(ImgArray=Data.pred3,
                  ImgSection = 1, XYsize = 256, random=T, SaveAll=T)






