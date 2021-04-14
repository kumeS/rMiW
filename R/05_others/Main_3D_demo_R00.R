packageVersion("keras")
packageVersion("tensorflow")

library(keras)
library(tfruns)
library(abind)
library(tidyverse)
library(EBImage); options(EBImage.display = "raster")
library(mmand)
library(reticulate)
reticulate::use_python("/usr/local/bin/python", required =T)
reticulate::py_config()
#setwd("~/Rstudio_Keras_test/rMiW_200710")
#dir("./R")

source("./R/createSimpleUNET_v1.R", local = T)
source("./R/DL_model_plot_modi.R", local = T)
source("./R/ImageDataImport.R", local = T)
source("./R/ImageProcessing.R", local = T)
source("./R/ImageView.R", local = T)
source("./R/LearningRateSchedule.R", local = T)
source("./R/list2tensor.R", local = T)
source("./R/LossFunction.R", local = T)

#A <- c()
#B <- c()
#C <- matrix(NA, 30, 20)

#for(n in 1:20){
#for(m in 1:30){
#rm(list=ls())
#.rs.restartR()
# m <- 1;n <- 1
# m <- 10;n <- 1
TrainImgDataset <- Demo3dTRAINDataImport( Z=24)
str(TrainImgDataset)

NNN <- 256
ImageView3Ddataset(ImgArray=TrainImgDataset, Sample=1, ImgSection = 1, XYsize = NNN, random=T)

# Save all
#ImageView3Ddataset(ImgArray=TrainImgDataset, Sample=1, ImgSection = 1, XYsize = 256, SaveAll=T)

## Model 作成
model <- create3DModel(inputImageSize3d = c( NNN, NNN, 24, 1), nFilters=2)
summary(model)

## plot the DL model
#model %>% plot_model_modi(width=4, height=1.2)
py_plot_model <- reticulate::import(module = "tensorflow")$keras$utils$plot_model
py_plot_model(model, to_file='Model.png', show_shapes=T, show_layer_names=T)
system("open Model.png")

## Compile
model <- model %>%
  compile(optimizer = optimizer_rmsprop(lr = 0.01),
          #optimizer = optimizer_adam(lr = 0.1*n),
  loss = Singlelabel_3d_bce_dice_loss,
  metrics = custom_metric("Singlelabel_3d_dice_coef", Singlelabel_3d_dice_coef))

## Fit parameters
EPOCHS <- 10000; BATCH_SIZE <- 8

## Run fit
history <- model %>%
  fit(TrainImgDataset$Training,
  TrainImgDataset$TrainingGT,
  batch_size = BATCH_SIZE,
  epochs = EPOCHS,
  verbose = 1, validation_split = 0,
  #callbacks = list( callback_learning_rate_scheduler(lr_schedule_1000))
  )

score01 <- model %>%
  evaluate(TrainImgDataset$Training,
   TrainImgDataset$TrainingGT, verbose = 1)
print(score01$Singlelabel_3d_dice_coef)
#C[m,n] <- score01$Singlelabel_3d_dice_coef

save_model_hdf5(model, filepath="model_v1.h5")
save_model_weights_hdf5(model, filepath="modelw_v1.h5")


plot(A)
max(A)

plot(B)
max(B)

image(C)
max(C)

C00 <- C
C00[C < 0.6] <- 0
C00[C >= 0.6] <- 1
image(C00)

a <- apply(C00, 2, sum)
barplot(a)

b <- apply(C, 2, mean)
barplot(b)

model <- model %>%
  compile(#optimizer = optimizer_rmsprop(lr = 0.15),
  #optimizer = optimizer_adam(lr = 0.15),
  #optimizer = optimizer_sgd(lr = 0.15),
  #optimizer = optimizer_adagrad(lr = 0.15),
  optimizer = optimizer_adamax(lr = 0.1),
loss = Singlelabel_3d_bce_dice_loss,
metrics = custom_metric("Singlelabel_3d_dice_coef", Singlelabel_3d_dice_coef))

#hit
history <- model %>%
  fit(TrainImgDataset$Training,
  TrainImgDataset$TrainingGT,
  batch_size = BATCH_SIZE,
  epochs = EPOCHS,
  verbose = 0,
  callbacks = list( callback_learning_rate_scheduler(lr_schedule_1000)))

png(filename = "Rplot_002d.png", width = 480, height = 480); plot(history); dev.off()

score01 <- model %>%
  evaluate(TrainImgDataset$Training,
   TrainImgDataset$TrainingGT, verbose = 1)

Y_hat <- predict(model, x = TrainImgDataset$Training, verbose=1)

str(Y_hat)
table(Y_hat[1, , , 1:20, ])

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











