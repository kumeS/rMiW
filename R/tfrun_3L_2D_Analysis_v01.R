rm(list=ls())
#.rs.restartR()

library(keras)
library(tidyverse)
library(reticulate)
#library(deepviz)
library(magrittr)
#install.packages("BiocManager")
#BiocManager::install("EBImage")
library(EBImage)
options(EBImage.display = "raster")
#install.packages("mmand")
library(mmand)
#devtools::install_github( "ANTsX/ANTsRNet" )
#library(ANTsRNet)
#####

## DL segmentationのやり方
getwd()


FLAGS <- flags(
 flag_numeric("kernel_size", 3),
 flag_numeric("nlevels", 4),
 flag_numeric("nfilters", 128),
 flag_numeric("BatchSize", 1),
 flag_numeric("dropout1", 0.025),
 flag_numeric("dropout2", 0.025),
 flag_numeric("dropout3", 0.05)
)

getwd()
setwd("/home/rch/Rstudio_Keras_test/")
dir()
dir("DL_Segmentation_Dataset")
dir('./DL_Segmentation_Dataset/D05_NB4_wholeALL2/')
TRAIN_PATH = './IMAGE/D05_NB4_wholeALL2/1_Train'
Teacher_PATH = './IMAGE/D05_NB4_wholeALL2/2_Teacher'
Test_PATH = './IMAGE/D05_NB4_wholeALL2/3_Test'
Test_GT_PATH = './IMAGE/D05_NB4_wholeALL2/4_Test_GT'

HEIGHT = 128
WIDTH  = 128

#HEIGHT = 512
#WIDTH = 512

CHANNELS = 1
SHAPE = c(WIDTH, HEIGHT, CHANNELS)
BATCH_SIZE = FLAGS$BatchSize
EPOCHS = 300

invert <- function(x) {
 if(mean(x) > .5)
  x <- 1 - x
 x
}

to_gray_scale <- function(x) {
 y <- rgbImage(red = getFrame(x, 1),
               green = getFrame(x, 2),
               blue = getFrame(x, 3))
 y <- channel(y, mode="luminance")
 dim(y) <- c(dim(y), 1)
 y
}

## Preprocess original images
preprocess_image01 <- function(file, shape){
 # shape = SHAPE
 # file = paste(TRAIN_PATH, "/", dir(TRAIN_PATH)[1], sep="")
 image <- readImage(file, type="png")
 #image <- to_gray_scale(image)                       ## convert to gray scale
 image <- resize(image, w = shape[1], h = shape[2], filter = "none")  ## none or bilinear
 image <- normalize(image)
 image <- clahe(image)
 image <- image^1.2
 #range(image)
 #image <- invert(image)                              ## invert brightfield
 #EBImage::display(image)
 array(image, dim=c(shape[1], shape[2], shape[3]))                                    ## return as array
}
preprocess_image02 <- function(file, shape){
 # shape = SHAPE
 #file = paste(TRAIN_PATH, "/", dir(Teacher_PATH)[1], sep="")
 image <- readImage(file, type="png")
 #image <- to_gray_scale(image)                       ## convert to gray scale
 image <- resize(image, w = shape[1], h = shape[2], filter = "none")
 #image <- clahe(image)                               ## local adaptive contrast enhancement
 #image <- normalize(image)                           ## standardize between [0, 1]
 #image <- invert(image)                              ## invert brightfield
 array(image, dim=c(shape[1], shape[2], 1))                                    ## return as array
}
preprocess_image03 <- function(file, shape){
 # shape = SHAPE
 #file = paste(TRAIN_PATH, "/", dir(Teacher_PATH)[1], sep="")
 image <- readImage(file, type="png")
 #image <- to_gray_scale(image)                       ## convert to gray scale
 image <- resize(image, w = shape[1], h = shape[2], filter = "none")  ## make all images of dimensions
 #image <- clahe(image)                               ## local adaptive contrast enhancement
 #image <- normalize(image)                           ## standardize between [0, 1]
 #image <- invert(image)                              ## invert brightfield
 array(image, dim=c(shape[1], shape[2], 3))                                    ## return as array
}
preprocess_image04 <- function(file, shape){
 # shape = SHAPE
 # file = paste(TRAIN_PATH, "/", dir(TRAIN_PATH)[1], sep="")
 image <- readImage(file, type="png")
 #image <- to_gray_scale(image)                       ## convert to gray scale
 image <- resize(image, w = shape[1], h = shape[2], filter = "none")  ## none or bilinear
 image <- normalize(image)
 image <- clahe(image)
 image <- image^0.9
 #range(image)
 #image <- invert(image)                              ## invert brightfield
 #EBImage::display(image)
 array(image, dim=c(shape[1], shape[2], shape[3]))                                    ## return as array
}

ImageFile = paste(TRAIN_PATH, "/", dir(TRAIN_PATH), sep="")
X = map(ImageFile, preprocess_image01, shape = SHAPE)
#str(X)
ImageFile = paste(Teacher_PATH, "/", dir(Teacher_PATH), sep="")
Y = map(ImageFile, preprocess_image02, shape = SHAPE)
#str(Y)

## リストの結合
XY <- list(X=X,Y=Y)
XYG <- XY

#dim(XYG$X[[1]])[1]/4
#128/8
#c((dim(XYG$X[[1]])[1]/8*5+1):(dim(XYG$X[[1]])[1]))
#c(1:(dim(XYG$X[[1]])[1]/8*5))
#display(XYG$X[[1]][c(c((dim(XYG$X[[1]])[1]/8*5+1):(dim(XYG$X[[1]])[1])), c(1:(dim(XYG$X[[1]])[1]/8*5))),,])
#display(AA[c(c((dim(AA)[1]/8*5+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*5))),,])
#display(BB[c(c((dim(BB)[1]/8*5+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*5))),,])
#display(AA[,c(c((dim(AA)[1]/8*5+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*5))),])
#display(BB[,c(c((dim(BB)[1]/8*5+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*5))),])

## データを増やす
for(j in 1:length(XY$X)){
 #j <- 1
 AA <- rotate(XYG$X[[j]], 0)
 BB <- rotate(XYG$Y[[j]], 0)
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*1+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*1))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*1+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*1))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*2+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*2))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*2+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*2))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*3+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*3))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*3+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*3))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*4+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*4))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*4+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*4))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*5+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*5))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*5+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*5))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*6+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*6))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*6+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*6))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*7))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*7))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*1+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*1))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*1+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*1))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*2+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*2+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*3+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*3))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*3+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*3))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*4+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*4))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*4+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*4))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*5+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*5))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*5+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*5))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*6+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*6))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*6+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*6))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/7*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/7*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))

  AA <- rotate(XYG$X[[j]], 90)
  BB <- rotate(XYG$Y[[j]], 90)
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*1+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*1))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*1+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*1))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*2+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*2))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*2+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*2))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*3+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*3))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*3+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*3))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*4+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*4))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*4+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*4))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*5+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*5))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*5+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*5))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*6+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*6))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*6+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*6))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*7))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*7))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*1+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*1))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*1+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*1))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*2+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*2+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*3+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*3))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*3+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*3))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*4+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*4))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*4+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*4))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*5+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*5))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*5+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*5))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*6+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*6))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*6+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*6))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/7*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/7*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))

  AA <- rotate(XYG$X[[j]], 180)
  BB <- rotate(XYG$Y[[j]], 180)
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*1+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*1))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*1+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*1))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*2+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*2))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*2+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*2))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*3+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*3))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*3+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*3))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*4+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*4))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*4+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*4))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*5+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*5))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*5+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*5))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*6+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*6))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*6+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*6))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*7))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*7))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*1+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*1))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*1+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*1))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*2+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*2+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*3+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*3))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*3+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*3))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*4+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*4))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*4+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*4))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*5+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*5))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*5+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*5))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*6+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*6))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*6+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*6))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/7*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/7*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))

  AA <- rotate(XYG$X[[j]], 270)
  BB <- rotate(XYG$Y[[j]], 270)
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*1+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*1))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*1+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*1))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*2+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*2))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*2+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*2))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*3+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*3))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*3+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*3))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*4+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*4))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*4+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*4))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*5+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*5))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*5+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*5))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*6+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*6))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*6+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*6))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*7))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*7))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*1+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*1))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*1+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*1))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*2+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*2+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*3+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*3))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*3+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*3))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*4+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*4))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*4+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*4))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*5+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*5))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*5+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*5))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*6+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*6))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*6+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*6))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/7*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/7*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))

  AA <- array(rotate(flip(XYG$X[[j]]), 0), dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))
  BB <- array(rotate(flip(XYG$Y[[j]]), 0), dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*1+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*1))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*1+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*1))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*2+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*2))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*2+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*2))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*3+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*3))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*3+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*3))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*4+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*4))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*4+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*4))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*5+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*5))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*5+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*5))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*6+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*6))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*6+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*6))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*7))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*7))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*1+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*1))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*1+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*1))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*2+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*2+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*3+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*3))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*3+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*3))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*4+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*4))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*4+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*4))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*5+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*5))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*5+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*5))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*6+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*6))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*6+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*6))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/7*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/7*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))

  AA <- array(rotate(flip(XYG$X[[j]]), 90), dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))
  BB <- array(rotate(flip(XYG$Y[[j]]), 90), dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*1+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*1))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*1+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*1))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*2+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*2))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*2+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*2))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*3+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*3))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*3+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*3))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*4+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*4))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*4+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*4))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*5+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*5))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*5+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*5))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*6+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*6))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*6+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*6))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*7))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*7))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*1+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*1))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*1+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*1))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*2+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*2+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*3+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*3))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*3+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*3))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*4+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*4))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*4+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*4))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*5+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*5))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*5+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*5))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*6+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*6))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*6+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*6))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/7*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/7*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))

  AA <- array(rotate(flip(XYG$X[[j]]), 180), dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))
  BB <- array(rotate(flip(XYG$Y[[j]]), 180), dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*1+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*1))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*1+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*1))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*2+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*2))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*2+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*2))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*3+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*3))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*3+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*3))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*4+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*4))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*4+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*4))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*5+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*5))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*5+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*5))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*6+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*6))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*6+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*6))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*7))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*7))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*1+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*1))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*1+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*1))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*2+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*2+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*3+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*3))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*3+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*3))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*4+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*4))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*4+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*4))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*5+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*5))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*5+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*5))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*6+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*6))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*6+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*6))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/7*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/7*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))

  AA <- array(rotate(flip(XYG$X[[j]]), 270), dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))
  BB <- array(rotate(flip(XYG$Y[[j]]), 270), dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*1+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*1))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*1+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*1))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*2+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*2))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*2+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*2))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*3+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*3))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*3+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*3))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*4+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*4))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*4+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*4))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*5+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*5))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*5+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*5))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*6+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*6))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*6+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*6))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*7))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*7))),,], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*1+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*1))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*1+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*1))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*2+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*2+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*3+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*3))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*3+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*3))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*4+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*4))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*4+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*4))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*5+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*5))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*5+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*5))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*6+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*6))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*6+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*6))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/7*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
  XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/7*2))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
}

###############################
## データを元に戻し
XY <- XYG
rm(XYG)
###############################
str(XY)

## アレイに変換
list2tensor <- function(xList) {
 xTensor <- simplify2array(xList)
 aperm(xTensor, c(4, 1, 2, 3))
}

##binarize
#W <- map(XY$Y, function(x) {bw <- x > 0; 1*bw})

#head(XY$X)
XL <- list2tensor(XY$X)
YL <- list2tensor(XY$Y)
#Ybw <- list2tensor(W)
#dim(XL)
#dim(YL)
#table(YL[1,,,])
#table(YL)
#dim(Ybw)

## dice_coef
K <- backend()

dice_coef <- function(y_true, y_pred, smooth = 1.0) {
 y_true_f <- k_flatten(y_true)
 y_pred_f <- k_flatten(y_pred)
 intersection <- k_sum(y_true_f * y_pred_f)
 result <- (2 * intersection + smooth) /
  (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
 return(result)
}

bce_dice_loss <- function(y_true, y_pred) {
 result <- loss_binary_crossentropy(y_true, y_pred) +
  (1 - dice_coef(y_true, y_pred))
 return(result)
}

iou <- function(y_true, y_pred, smooth = 1.0){
 y_true_f <- k_flatten(y_true)
 y_pred_f <- k_flatten(y_pred)
 intersection <- k_sum( y_true_f * y_pred_f)
 union <- k_sum( y_true_f + y_pred_f ) - intersection
 result <- (intersection + smooth) / ( union + smooth)
 return(result)
}

iou_loss <- function(y_true, y_pred) {
 result <- loss_binary_crossentropy(y_true, y_pred) + (1 - iou(y_true, y_pred))
 return(result)
}

## U-Net
## unet 2x2 2DConv layer
unet_layer <- function(object, filters, kernel_size = c(FLAGS$kernel_size, FLAGS$kernel_size),
                       padding = "same", kernel_initializer = "he_normal",
                       dropout = 0.1, activation="relu"){
 object %>%
  layer_conv_2d(filters = filters, kernel_size = kernel_size, padding = padding) %>%
  layer_batch_normalization() %>%
  layer_activation(activation) %>%
  layer_conv_2d(filters = filters, kernel_size = kernel_size, padding = padding) %>%
  layer_batch_normalization() %>%
  layer_activation(activation) %>%
  layer_spatial_dropout_2d(rate = dropout) %>%
  layer_conv_2d(filters = filters, kernel_size = kernel_size, padding = padding) %>%
  layer_batch_normalization() %>%
  layer_activation(activation)
}

unet <- function(shape, nlevels = 3, nfilters = 16, dropouts = c(0.1, 0.1, 0.3, 0.3)){

 message("Constructing U-Net with ", nlevels, " levels initial number of filters is: ", nfilters)

 filter_sizes <- nfilters*2^seq.int(0, nlevels)

 ## Loop over contracting layers
 clayers <- clayers_pooled <- list()

 ## inputs
 clayers_pooled[[1]] <- layer_input(shape = shape)

 for(i in 2:(nlevels+1)) {
  clayers[[i]] <- unet_layer(clayers_pooled[[i - 1]],
                             filters = filter_sizes[i - 1],
                             dropout = dropouts[i-1])

  clayers_pooled[[i]] <- layer_max_pooling_2d(clayers[[i]],
                                              pool_size = c(2, 2),
                                              strides = c(2, 2))
 }

 ## Loop over expanding layers
 elayers <- list()

 ## center
 elayers[[nlevels + 1]] <- unet_layer(clayers_pooled[[nlevels + 1]],
                                      filters = filter_sizes[nlevels + 1],
                                      dropout = dropouts[nlevels + 1])

 for(i in nlevels:1) {
  elayers[[i]] <- layer_conv_2d_transpose(elayers[[i+1]],
                                          filters = filter_sizes[i],
                                          kernel_size = c(2, 2),
                                          strides = c(2, 2),
                                          padding = "same")

  elayers[[i]] <- layer_concatenate(list(elayers[[i]], clayers[[i + 1]]), axis = 3)
  elayers[[i]] <- unet_layer(elayers[[i]], filters = filter_sizes[i], dropout = dropouts[i])

 }

 ## Output layer
 outputs <- layer_conv_2d(elayers[[1]], filters = 1, kernel_size = c(1, 1), activation = "sigmoid")

 return(keras_model(inputs = clayers_pooled[[1]], outputs = outputs))
}

## Model 作成
try(model <- unet(shape = SHAPE, nlevels = FLAGS$nlevels, nfilters = FLAGS$nfilters,
               dropouts = c(FLAGS$dropout1, FLAGS$dropout1,
                            FLAGS$dropout2, FLAGS$dropout2, FLAGS$dropout3)), silent=TRUE)

#summary(model)
## compile
try(model <- model %>%
     compile(
      #optimizer = optimizer_rmsprop(lr = 0.01),
      optimizer_sgd(lr = 0.001, momentum = 0.9, decay = 1e-4, nesterov=T, clipnorm = 1, clipvalue = 0.5),
      loss = bce_dice_loss,
      metrics = custom_metric("dice_coef", dice_coef)
      #loss = iou_loss,
      #metrics = custom_metric("iou", iou)
     ), silent=TRUE)

lr_schedule <- function(epoch, lr) {
 if(epoch <= 25) {
  0.01
 } else if(epoch > 25 && epoch <= 50){
  0.01
 } else if(epoch > 50 && epoch <= 75){
  0.01
 } else if(epoch > 75 && epoch <= 100){
  0.01
 } else if(epoch > 100 && epoch <= 200){
  0.01
 } else {
  0.001
 }}

lr_reducer <- callback_learning_rate_scheduler(lr_schedule)

#summary(model)
#count_params(model)
#get_weights(model)
#####################################################################################
################################     Fit          ###################################
#####################################################################################
# EPOCHS <- 3
## tensorboard 設定
DIR01 <- "Results_NB4/UNET_Test_01/Cell_LOG002"
if(!file.exists(DIR01)){dir.create(DIR01, showWarnings = F)}
TENS <- callback_tensorboard(DIR01)
tensorboard(DIR01, launch_browser=F)

try(history <- model %>% fit(XL, YL,
                             batch_size = BATCH_SIZE,
                             epochs = EPOCHS,
                             validation_split = 1,
                             verbose = 1,
                             callbacks = list( lr_reducer, TENS))
    , silent=TRUE)

dir(DIR01)
tensorboard(DIR01)

history
#max(history$metrics$iou)
## Model 評価
try(score01 <- model %>% evaluate(
 XL, YL, verbose = 0), silent=TRUE)

try(cat('Train loss:', score01[[1]], '\n'), silent=TRUE)
try(cat('Train accuracy:', score01[[2]], '\n'), silent=TRUE)

## 前のモデルを適用
#setwd("~/Rstudio_Keras_test/190513_dl")
#DIR01 <- "~/Rstudio_Keras_test/190513_dl/Results_NB4/UNET_Test_01/LOG001/190612-1813_TrainAcc1_TestAcc0.9801"
#model <- load_model_hdf5(paste(DIR01, "/Model.h5", sep=""), custom_objects = NULL, compile = F)

try(Y_hat <- predict(model, x = XL, verbose=1), silent=TRUE)
# mmand::threshold
try(Y_hat5 <- map(array_branch(Y_hat, 1),
               .f = function(z) {mmand::threshold(z, 0.5, binarise = TRUE)}), silent=TRUE)
try(Y_hat <- list2tensor(Y_hat5), silent=TRUE)
#str(Y_hat)
#str(YL)
#str(XY)
#setwd("/home/sas/Rstudio_Keras_test/190513_dl/")
try(TAVE <- rep(NA, length(dir(TRAIN_PATH))), silent=TRUE)
try(for(ABC in 1:length(dir(TRAIN_PATH))){
 intersection <- sum( matrix(XY$Y[[ABC]]) * matrix(Y_hat[ABC,,,]))
 union <- sum( matrix(XY$Y[[ABC]]) + matrix(Y_hat[ABC,,,]) ) - intersection
 result <- (intersection + 1) / ( union + 1)
 TAVE[ABC] <- result
}, silent=TRUE)
TAVE
mean(TAVE)
sd(TAVE)
#########################################################
#########################################################
## Perform predictions on the test images
#########################################################
#########################################################
#length(dir(Test_PATH))
ImageFile = paste(Test_PATH, "/", dir(Test_PATH), sep="")
TX = map(ImageFile, preprocess_image04, shape = SHAPE)
#str(TX)

ImageFile = paste(Test_GT_PATH, "/", dir(Test_GT_PATH), sep="")
TY = map(ImageFile, preprocess_image02, shape = SHAPE)
#str(TY)

## リストの結合
XYT <- list(X=X,Y=Y,TX=TX, TY=TY)
TXL <- list2tensor(XYT$TX)
TYL <- list2tensor(XYT$TY)

try(TX_hat <- predict(model, x = TXL, verbose=0), silent=TRUE)

# mmand::threshold
try(TX_lab5 <- map(array_branch(TX_hat, 1),
               .f = function(z) {mmand::threshold(z, 0.5, binarise = TRUE)}) , silent=TRUE)
try(TX_lab <- list2tensor(TX_lab5), silent=TRUE)

##########################################################
## Model 評価
try(score02 <- model %>% evaluate(
 TXL, TYL, verbose = 0), silent=TRUE)

try(cat('Test loss:', score02[[1]], '\n'), silent=TRUE)
try(cat('Test accuracy:', score02[[2]], '\n'), silent=TRUE)

#library(pROC)
#AAA <- rep(NA, dim(TYL)[1])
#for(n in 1:dim(TYL)[1]){
# Pred <- data.frame(Pred=c(TX_hat[n,,,]), GT=c(TYL[n,,,]))
# head(Pred)
# Pred.roc <- roc(formula = GT ~ Pred, data = Pred)
# AAA[n] <- c(Pred.roc$auc)
#}
#mean(AAA)

try(AVE <- rep(NA, length(dir(Test_PATH))), silent=TRUE)
try(for(ABC in 1:length(dir(Test_PATH))){
 intersection <- sum( matrix(XYT$TY[[ABC]]) * matrix(TX_lab[ABC,,,]))
 union <- sum( matrix(XYT$TY[[ABC]]) + matrix(TX_lab[ABC,,,]) ) - intersection
 result <- (intersection + 1) / ( union + 1)
 AVE[ABC] <- result
}, silent=TRUE)
mean(AVE)
sd(AVE)

## モデル保存
#dir("Results_NB4")
if(!file.exists(DIR01)){dir.create(DIR01, showWarnings = F)}
try(DIR02 <- paste("/", format(Sys.time(), "%y%m%d-%H%M"),
               "_TrainAcc", round(mean(score01[[2]]), 4),
               "_TestAcc", round(mean(AVE), 4), sep=""), silent=TRUE)
#, "_Model_Para", count_params(model)
try(dir.create(paste(DIR01, DIR02, sep=""), showWarnings = F), silent=TRUE)

try(model %>%
 save_model_hdf5(paste(DIR01, DIR02, "/Model.h5", sep=""), overwrite = TRUE, include_optimizer = TRUE), silent=TRUE)
try(model %>%
 save_model_weights_hdf5(paste(DIR01, DIR02, "/Model_weights.h5", sep="")), silent=TRUE)

##########################################################
## Testの結果グラフ保存(1)
##########################################################
try(dir.create(paste(DIR01, DIR02, "/4_Test_Results01", sep=""), showWarnings = F), silent=TRUE)

try(for(ABC in 1:length(dir(Test_PATH))){
 # ABC <- 1
 Image_color01 <- paintObjects(XYT$TY[[ABC]]/2, toRGB(XYT$TX[[ABC]])/2,opac=c(0.25, 0.25),
                               col=c("red","red"), thick=T, closed=F)
 Image_color02 <- paintObjects(TX_lab[ABC,,,]/2, toRGB(XYT$TX[[ABC]])/2,opac=c(0.25, 0.25),
                               col=c("blue","blue"), thick=T, closed=F)
 try(dev.off(), silent=T)
 png(paste(DIR01, DIR02, "/4_Test_Results01/Test_Res_", formatC(ABC, width = 4, flag = "0"), ".png", sep=""),
     width = 400, height = 400)
 par(bg = 'grey')
 EBImage::display(EBImage::combine(Image_color01, toRGB(XYT$TY[[ABC]]),
                                   Image_color02, toRGB(TX_lab[ABC,,,])),
                  nx=2, all=TRUE, spacing = 0.01, margin = 70)
 #A <- table(c(XYT$TY[[ABC]]), c(TX_lab[ABC,,,]))
 intersection <- sum( matrix(XYT$TY[[ABC]]) * matrix(TX_lab[ABC,,,]))
 union <- sum( matrix(XYT$TY[[ABC]]) + matrix(TX_lab[ABC,,,]) ) - intersection
 result <- (intersection + 1) / ( union + 1)

 text(x = -50/4, y = 250/4, label = "Ground truth",
      adj = c(0,1), col = "black", cex = 1, pos=1, srt=90)
 text(x = 250/4, y = -70/4, label = "Overlay",
      adj = c(0,1), col = "black", cex = 1, pos=1, srt=0)
 text(x = -50/4, y = 750/4, label = "Prediction",
      adj = c(0,1), col = "black", cex = 1, pos=1, srt=90)
 text(x = 770/4, y = -70/4, label = "Binary",
      adj = c(0,1), col = "black", cex = 1, pos=1, srt=0)
 text(x = 770/6, y = 1050/4,
      label = paste("IOU:", round(result, 3),
                    "\nIOU ave:", round(mean(AVE),3), "±", round(sd(AVE),3)),
      adj = c(0,1), col = "black", cex = 1.25, pos=1, srt=0)
 try(dev.off(), silent=T)
}, silent=TRUE)

#########################################################
#########################################################
## Perform predictions on others
#########################################################
preprocess_image05 <- function(file, shape){
 # shape = SHAPE
 # file = paste(TRAIN_PATH, "/", dir(TRAIN_PATH)[1], sep="")
 image <- readImage(file, type="png")
 #image <- to_gray_scale(image)                       ## convert to gray scale
 image <- resize(image, w = shape[1], h = shape[2], filter = "none")  ## none or bilinear
 image <- normalize(image)
 image <- clahe(image)
 image <- image^0.8
 #range(image)
 #image <- invert(image)                              ## invert brightfield
 #EBImage::display(image)
 array(image, dim=c(shape[1], shape[2], shape[3]))                                    ## return as array
}
preprocess_image06 <- function(file, shape){
 # shape = SHAPE
 # file = paste(TRAIN_PATH, "/", dir(TRAIN_PATH)[1], sep="")
 image <- readImage(file, type="png")
 #image <- to_gray_scale(image)                       ## convert to gray scale
 image <- resize(image, w = shape[1], h = shape[2], filter = "none")  ## none or bilinear
 image <- normalize(image)
 image <- clahe(image)
 image <- image^0.7
 #range(image)
 #image <- invert(image)                              ## invert brightfield
 #EBImage::display(image)
 array(image, dim=c(shape[1], shape[2], shape[3]))                                    ## return as array
}
#########################################################
Test_WIDE1024_PATH01 = './IMAGE/D05_NB4_WideEM/crop2_x1024_01'
#Test_WIDE1024_PATH01 = './IMAGE/D05_NB4_WideEM/crop2_x1024'

#Test_WIDE2048_PATH01 = './IMAGE/D05_NB4_WideEM/crop3_x2048_01'
Test_WIDE2048_PATH01 = './IMAGE/D05_NB4_WideEM/crop3_x2048'

ImageFile = paste(Test_WIDE1024_PATH01, "/", dir(Test_WIDE1024_PATH01), sep="")
WX1 = map(ImageFile, preprocess_image06, shape = SHAPE)

ImageFile = paste(Test_WIDE2048_PATH01, "/", dir(Test_WIDE2048_PATH01), sep="")
WX2 = map(ImageFile, preprocess_image06, shape = SHAPE)

WX <- list(WX1=WX1,WX2=WX2)
str(WX)

WX11 <- list2tensor(WX$WX1)
WX22 <- list2tensor(WX$WX2)

WX11p <- predict(model, x = WX11, verbose=1)
WX22p <- predict(model, x = WX22, verbose=1)

## 画像変更
ImageFile = paste(Test_WIDE1024_PATH01, "/", dir(Test_WIDE1024_PATH01), sep="")
WX1 = map(ImageFile, preprocess_image02, shape = SHAPE)
ImageFile = paste(Test_WIDE2048_PATH01, "/", dir(Test_WIDE2048_PATH01), sep="")
WX2 = map(ImageFile, preprocess_image02, shape = SHAPE)
WX <- list(WX1=WX1,WX2=WX2)
str(WX)
WX11 <- list2tensor(WX$WX1)
WX22 <- list2tensor(WX$WX2)

########################################################################################
## 1024x1024バージョン
########################################################################################
try(WX11pa <- map(array_branch(WX11p, 1),
                   .f = function(z) {mmand::threshold(z, 0.5, binarise = TRUE)}) , silent=TRUE)
str(WX11pa)
try(WX11pal <- list2tensor(WX11pa), silent=TRUE)
str(WX11pal)

try(dir.create(paste(DIR01, DIR02, "/5_WD11", sep=""), showWarnings = F), silent=TRUE)
try(for(ABC in 1:length(dir(Test_WIDE1024_PATH01))){
 # ABC <- 1
 #WX
 Image_color01 <- paintObjects(WX11pal[ABC,,,]/2, toRGB(WX$WX1[[ABC]])/2,opac=c(0.25, 0.25),
                               col=c("blue","blue"), thick=T, closed=F)
 try(dev.off(), silent=T)
 png(paste(DIR01, DIR02, "/5_WD11/WD_Res_", formatC(ABC, width = 5, flag = "0"), ".png", sep=""),
     width = 400, height = 400)
 #par(bg = 'grey')
 EBImage::display(Image_color01,
                  nx=1, all=TRUE, spacing = 0, margin = 0)
 try(dev.off(), silent=T)
}, silent=TRUE)

########################################################################################

ImageFile = paste(DIR01, DIR02, "/5_WD11/", dir(paste(DIR01, DIR02, "/5_WD11", sep="")), sep="")
WX1RGB = map(ImageFile, preprocess_image03, shape = SHAPE)
str(WX1RGB)

print(WX1RGB[[1]])

par(bg = 'grey', mfrow = c(1,1), mar = c(0, 0, 0, 0))
paste("Image(WX1RGB[[", 1:400, "]], colormode=Color)", sep="", collapse=",")

EBImage::display(EBImage::combine(
 Image(WX1RGB[[1]], colormode=Color),Image(WX1RGB[[2]], colormode=Color),Image(WX1RGB[[3]], colormode=Color),Image(WX1RGB[[4]], colormode=Color),Image(WX1RGB[[5]], colormode=Color),Image(WX1RGB[[6]], colormode=Color),Image(WX1RGB[[7]], colormode=Color),Image(WX1RGB[[8]], colormode=Color),Image(WX1RGB[[9]], colormode=Color),Image(WX1RGB[[10]], colormode=Color),Image(WX1RGB[[11]], colormode=Color),Image(WX1RGB[[12]], colormode=Color),Image(WX1RGB[[13]], colormode=Color),Image(WX1RGB[[14]], colormode=Color),Image(WX1RGB[[15]], colormode=Color),Image(WX1RGB[[16]], colormode=Color),Image(WX1RGB[[17]], colormode=Color),Image(WX1RGB[[18]], colormode=Color),Image(WX1RGB[[19]], colormode=Color),Image(WX1RGB[[20]], colormode=Color),Image(WX1RGB[[21]], colormode=Color),Image(WX1RGB[[22]], colormode=Color),Image(WX1RGB[[23]], colormode=Color),Image(WX1RGB[[24]], colormode=Color),Image(WX1RGB[[25]], colormode=Color),Image(WX1RGB[[26]], colormode=Color),Image(WX1RGB[[27]], colormode=Color),Image(WX1RGB[[28]], colormode=Color),Image(WX1RGB[[29]], colormode=Color),Image(WX1RGB[[30]], colormode=Color),Image(WX1RGB[[31]], colormode=Color),Image(WX1RGB[[32]], colormode=Color),Image(WX1RGB[[33]], colormode=Color),Image(WX1RGB[[34]], colormode=Color),Image(WX1RGB[[35]], colormode=Color),Image(WX1RGB[[36]], colormode=Color),Image(WX1RGB[[37]], colormode=Color),Image(WX1RGB[[38]], colormode=Color),Image(WX1RGB[[39]], colormode=Color),Image(WX1RGB[[40]], colormode=Color),Image(WX1RGB[[41]], colormode=Color),Image(WX1RGB[[42]], colormode=Color),Image(WX1RGB[[43]], colormode=Color),Image(WX1RGB[[44]], colormode=Color),Image(WX1RGB[[45]], colormode=Color),Image(WX1RGB[[46]], colormode=Color),Image(WX1RGB[[47]], colormode=Color),Image(WX1RGB[[48]], colormode=Color),Image(WX1RGB[[49]], colormode=Color),Image(WX1RGB[[50]], colormode=Color),Image(WX1RGB[[51]], colormode=Color),Image(WX1RGB[[52]], colormode=Color),Image(WX1RGB[[53]], colormode=Color),Image(WX1RGB[[54]], colormode=Color),Image(WX1RGB[[55]], colormode=Color),Image(WX1RGB[[56]], colormode=Color),Image(WX1RGB[[57]], colormode=Color),Image(WX1RGB[[58]], colormode=Color),Image(WX1RGB[[59]], colormode=Color),Image(WX1RGB[[60]], colormode=Color),Image(WX1RGB[[61]], colormode=Color),Image(WX1RGB[[62]], colormode=Color),Image(WX1RGB[[63]], colormode=Color),Image(WX1RGB[[64]], colormode=Color),Image(WX1RGB[[65]], colormode=Color),Image(WX1RGB[[66]], colormode=Color),Image(WX1RGB[[67]], colormode=Color),Image(WX1RGB[[68]], colormode=Color),Image(WX1RGB[[69]], colormode=Color),Image(WX1RGB[[70]], colormode=Color),Image(WX1RGB[[71]], colormode=Color),Image(WX1RGB[[72]], colormode=Color),Image(WX1RGB[[73]], colormode=Color),Image(WX1RGB[[74]], colormode=Color),Image(WX1RGB[[75]], colormode=Color),Image(WX1RGB[[76]], colormode=Color),Image(WX1RGB[[77]], colormode=Color),Image(WX1RGB[[78]], colormode=Color),Image(WX1RGB[[79]], colormode=Color),Image(WX1RGB[[80]], colormode=Color),Image(WX1RGB[[81]], colormode=Color),Image(WX1RGB[[82]], colormode=Color),Image(WX1RGB[[83]], colormode=Color),Image(WX1RGB[[84]], colormode=Color),Image(WX1RGB[[85]], colormode=Color),Image(WX1RGB[[86]], colormode=Color),Image(WX1RGB[[87]], colormode=Color),Image(WX1RGB[[88]], colormode=Color),Image(WX1RGB[[89]], colormode=Color),Image(WX1RGB[[90]], colormode=Color),Image(WX1RGB[[91]], colormode=Color),Image(WX1RGB[[92]], colormode=Color),Image(WX1RGB[[93]], colormode=Color),Image(WX1RGB[[94]], colormode=Color),Image(WX1RGB[[95]], colormode=Color),Image(WX1RGB[[96]], colormode=Color),Image(WX1RGB[[97]], colormode=Color),Image(WX1RGB[[98]], colormode=Color),Image(WX1RGB[[99]], colormode=Color),Image(WX1RGB[[100]], colormode=Color),
 Image(WX1RGB[[101]], colormode=Color),Image(WX1RGB[[102]], colormode=Color),Image(WX1RGB[[103]], colormode=Color),Image(WX1RGB[[104]], colormode=Color),Image(WX1RGB[[105]], colormode=Color),Image(WX1RGB[[106]], colormode=Color),Image(WX1RGB[[107]], colormode=Color),Image(WX1RGB[[108]], colormode=Color),Image(WX1RGB[[109]], colormode=Color),Image(WX1RGB[[110]], colormode=Color),Image(WX1RGB[[111]], colormode=Color),Image(WX1RGB[[112]], colormode=Color),Image(WX1RGB[[113]], colormode=Color),Image(WX1RGB[[114]], colormode=Color),Image(WX1RGB[[115]], colormode=Color),Image(WX1RGB[[116]], colormode=Color),Image(WX1RGB[[117]], colormode=Color),Image(WX1RGB[[118]], colormode=Color),Image(WX1RGB[[119]], colormode=Color),Image(WX1RGB[[120]], colormode=Color),Image(WX1RGB[[121]], colormode=Color),Image(WX1RGB[[122]], colormode=Color),Image(WX1RGB[[123]], colormode=Color),Image(WX1RGB[[124]], colormode=Color),Image(WX1RGB[[125]], colormode=Color),Image(WX1RGB[[126]], colormode=Color),Image(WX1RGB[[127]], colormode=Color),Image(WX1RGB[[128]], colormode=Color),Image(WX1RGB[[129]], colormode=Color),Image(WX1RGB[[130]], colormode=Color),Image(WX1RGB[[131]], colormode=Color),Image(WX1RGB[[132]], colormode=Color),Image(WX1RGB[[133]], colormode=Color),Image(WX1RGB[[134]], colormode=Color),Image(WX1RGB[[135]], colormode=Color),Image(WX1RGB[[136]], colormode=Color),Image(WX1RGB[[137]], colormode=Color),Image(WX1RGB[[138]], colormode=Color),Image(WX1RGB[[139]], colormode=Color),Image(WX1RGB[[140]], colormode=Color),Image(WX1RGB[[141]], colormode=Color),Image(WX1RGB[[142]], colormode=Color),Image(WX1RGB[[143]], colormode=Color),Image(WX1RGB[[144]], colormode=Color),Image(WX1RGB[[145]], colormode=Color),Image(WX1RGB[[146]], colormode=Color),Image(WX1RGB[[147]], colormode=Color),Image(WX1RGB[[148]], colormode=Color),Image(WX1RGB[[149]], colormode=Color),Image(WX1RGB[[150]], colormode=Color),Image(WX1RGB[[151]], colormode=Color),Image(WX1RGB[[152]], colormode=Color),Image(WX1RGB[[153]], colormode=Color),Image(WX1RGB[[154]], colormode=Color),Image(WX1RGB[[155]], colormode=Color),Image(WX1RGB[[156]], colormode=Color),Image(WX1RGB[[157]], colormode=Color),Image(WX1RGB[[158]], colormode=Color),Image(WX1RGB[[159]], colormode=Color),Image(WX1RGB[[160]], colormode=Color),Image(WX1RGB[[161]], colormode=Color),Image(WX1RGB[[162]], colormode=Color),Image(WX1RGB[[163]], colormode=Color),Image(WX1RGB[[164]], colormode=Color),Image(WX1RGB[[165]], colormode=Color),Image(WX1RGB[[166]], colormode=Color),Image(WX1RGB[[167]], colormode=Color),Image(WX1RGB[[168]], colormode=Color),Image(WX1RGB[[169]], colormode=Color),Image(WX1RGB[[170]], colormode=Color),Image(WX1RGB[[171]], colormode=Color),Image(WX1RGB[[172]], colormode=Color),Image(WX1RGB[[173]], colormode=Color),Image(WX1RGB[[174]], colormode=Color),Image(WX1RGB[[175]], colormode=Color),Image(WX1RGB[[176]], colormode=Color),Image(WX1RGB[[177]], colormode=Color),Image(WX1RGB[[178]], colormode=Color),Image(WX1RGB[[179]], colormode=Color),Image(WX1RGB[[180]], colormode=Color),Image(WX1RGB[[181]], colormode=Color),Image(WX1RGB[[182]], colormode=Color),Image(WX1RGB[[183]], colormode=Color),Image(WX1RGB[[184]], colormode=Color),Image(WX1RGB[[185]], colormode=Color),Image(WX1RGB[[186]], colormode=Color),Image(WX1RGB[[187]], colormode=Color),Image(WX1RGB[[188]], colormode=Color),Image(WX1RGB[[189]], colormode=Color),Image(WX1RGB[[190]], colormode=Color),Image(WX1RGB[[191]], colormode=Color),Image(WX1RGB[[192]], colormode=Color),Image(WX1RGB[[193]], colormode=Color),Image(WX1RGB[[194]], colormode=Color),Image(WX1RGB[[195]], colormode=Color),Image(WX1RGB[[196]], colormode=Color),Image(WX1RGB[[197]], colormode=Color),Image(WX1RGB[[198]], colormode=Color),Image(WX1RGB[[199]], colormode=Color),Image(WX1RGB[[200]], colormode=Color)
),nx=40, all=T, spacing = 0.0, margin = 0, drawGrid=F)

EBImage::display(EBImage::combine(
Image(WX1RGB[[1]], colormode=Color),Image(WX1RGB[[2]], colormode=Color),Image(WX1RGB[[3]], colormode=Color),Image(WX1RGB[[4]], colormode=Color),Image(WX1RGB[[5]], colormode=Color),Image(WX1RGB[[6]], colormode=Color),Image(WX1RGB[[7]], colormode=Color),Image(WX1RGB[[8]], colormode=Color),Image(WX1RGB[[9]], colormode=Color),Image(WX1RGB[[10]], colormode=Color),Image(WX1RGB[[11]], colormode=Color),Image(WX1RGB[[12]], colormode=Color),Image(WX1RGB[[13]], colormode=Color),Image(WX1RGB[[14]], colormode=Color),Image(WX1RGB[[15]], colormode=Color),Image(WX1RGB[[16]], colormode=Color),Image(WX1RGB[[17]], colormode=Color),Image(WX1RGB[[18]], colormode=Color),Image(WX1RGB[[19]], colormode=Color),Image(WX1RGB[[20]], colormode=Color),Image(WX1RGB[[21]], colormode=Color),Image(WX1RGB[[22]], colormode=Color),Image(WX1RGB[[23]], colormode=Color),Image(WX1RGB[[24]], colormode=Color),Image(WX1RGB[[25]], colormode=Color),Image(WX1RGB[[26]], colormode=Color),Image(WX1RGB[[27]], colormode=Color),Image(WX1RGB[[28]], colormode=Color),Image(WX1RGB[[29]], colormode=Color),Image(WX1RGB[[30]], colormode=Color),Image(WX1RGB[[31]], colormode=Color),Image(WX1RGB[[32]], colormode=Color),Image(WX1RGB[[33]], colormode=Color),Image(WX1RGB[[34]], colormode=Color),Image(WX1RGB[[35]], colormode=Color),Image(WX1RGB[[36]], colormode=Color),Image(WX1RGB[[37]], colormode=Color),Image(WX1RGB[[38]], colormode=Color),Image(WX1RGB[[39]], colormode=Color),Image(WX1RGB[[40]], colormode=Color),Image(WX1RGB[[41]], colormode=Color),Image(WX1RGB[[42]], colormode=Color),Image(WX1RGB[[43]], colormode=Color),Image(WX1RGB[[44]], colormode=Color),Image(WX1RGB[[45]], colormode=Color),Image(WX1RGB[[46]], colormode=Color),Image(WX1RGB[[47]], colormode=Color),Image(WX1RGB[[48]], colormode=Color),Image(WX1RGB[[49]], colormode=Color),Image(WX1RGB[[50]], colormode=Color),Image(WX1RGB[[51]], colormode=Color),Image(WX1RGB[[52]], colormode=Color),Image(WX1RGB[[53]], colormode=Color),Image(WX1RGB[[54]], colormode=Color),Image(WX1RGB[[55]], colormode=Color),Image(WX1RGB[[56]], colormode=Color),Image(WX1RGB[[57]], colormode=Color),Image(WX1RGB[[58]], colormode=Color),Image(WX1RGB[[59]], colormode=Color),Image(WX1RGB[[60]], colormode=Color),Image(WX1RGB[[61]], colormode=Color),Image(WX1RGB[[62]], colormode=Color),Image(WX1RGB[[63]], colormode=Color),Image(WX1RGB[[64]], colormode=Color),Image(WX1RGB[[65]], colormode=Color),Image(WX1RGB[[66]], colormode=Color),Image(WX1RGB[[67]], colormode=Color),Image(WX1RGB[[68]], colormode=Color),Image(WX1RGB[[69]], colormode=Color),Image(WX1RGB[[70]], colormode=Color),Image(WX1RGB[[71]], colormode=Color),Image(WX1RGB[[72]], colormode=Color),Image(WX1RGB[[73]], colormode=Color),Image(WX1RGB[[74]], colormode=Color),Image(WX1RGB[[75]], colormode=Color),Image(WX1RGB[[76]], colormode=Color),Image(WX1RGB[[77]], colormode=Color),Image(WX1RGB[[78]], colormode=Color),Image(WX1RGB[[79]], colormode=Color),Image(WX1RGB[[80]], colormode=Color),Image(WX1RGB[[81]], colormode=Color),Image(WX1RGB[[82]], colormode=Color),Image(WX1RGB[[83]], colormode=Color),Image(WX1RGB[[84]], colormode=Color),Image(WX1RGB[[85]], colormode=Color),Image(WX1RGB[[86]], colormode=Color),Image(WX1RGB[[87]], colormode=Color),Image(WX1RGB[[88]], colormode=Color),Image(WX1RGB[[89]], colormode=Color),Image(WX1RGB[[90]], colormode=Color),Image(WX1RGB[[91]], colormode=Color),Image(WX1RGB[[92]], colormode=Color),Image(WX1RGB[[93]], colormode=Color),Image(WX1RGB[[94]], colormode=Color),Image(WX1RGB[[95]], colormode=Color),Image(WX1RGB[[96]], colormode=Color),Image(WX1RGB[[97]], colormode=Color),Image(WX1RGB[[98]], colormode=Color),Image(WX1RGB[[99]], colormode=Color),Image(WX1RGB[[100]], colormode=Color),Image(WX1RGB[[101]], colormode=Color)
,Image(WX1RGB[[102]], colormode=Color),Image(WX1RGB[[103]], colormode=Color),Image(WX1RGB[[104]], colormode=Color),Image(WX1RGB[[105]], colormode=Color),Image(WX1RGB[[106]], colormode=Color),Image(WX1RGB[[107]], colormode=Color),Image(WX1RGB[[108]], colormode=Color),Image(WX1RGB[[109]], colormode=Color),Image(WX1RGB[[110]], colormode=Color),Image(WX1RGB[[111]], colormode=Color),Image(WX1RGB[[112]], colormode=Color),Image(WX1RGB[[113]], colormode=Color),Image(WX1RGB[[114]], colormode=Color),Image(WX1RGB[[115]], colormode=Color),Image(WX1RGB[[116]], colormode=Color),Image(WX1RGB[[117]], colormode=Color),Image(WX1RGB[[118]], colormode=Color),Image(WX1RGB[[119]], colormode=Color),Image(WX1RGB[[120]], colormode=Color),Image(WX1RGB[[121]], colormode=Color),Image(WX1RGB[[122]], colormode=Color),Image(WX1RGB[[123]], colormode=Color),Image(WX1RGB[[124]], colormode=Color),Image(WX1RGB[[125]], colormode=Color),Image(WX1RGB[[126]], colormode=Color),Image(WX1RGB[[127]], colormode=Color),Image(WX1RGB[[128]], colormode=Color),Image(WX1RGB[[129]], colormode=Color),Image(WX1RGB[[130]], colormode=Color),Image(WX1RGB[[131]], colormode=Color),Image(WX1RGB[[132]], colormode=Color),Image(WX1RGB[[133]], colormode=Color),Image(WX1RGB[[134]], colormode=Color),Image(WX1RGB[[135]], colormode=Color),Image(WX1RGB[[136]], colormode=Color),Image(WX1RGB[[137]], colormode=Color),Image(WX1RGB[[138]], colormode=Color),Image(WX1RGB[[139]], colormode=Color),Image(WX1RGB[[140]], colormode=Color),Image(WX1RGB[[141]], colormode=Color),Image(WX1RGB[[142]], colormode=Color),Image(WX1RGB[[143]], colormode=Color),Image(WX1RGB[[144]], colormode=Color),Image(WX1RGB[[145]], colormode=Color),Image(WX1RGB[[146]], colormode=Color),Image(WX1RGB[[147]], colormode=Color),Image(WX1RGB[[148]], colormode=Color),Image(WX1RGB[[149]], colormode=Color),Image(WX1RGB[[150]], colormode=Color),Image(WX1RGB[[151]], colormode=Color),Image(WX1RGB[[152]], colormode=Color),Image(WX1RGB[[153]], colormode=Color),Image(WX1RGB[[154]], colormode=Color),Image(WX1RGB[[155]], colormode=Color),Image(WX1RGB[[156]], colormode=Color),Image(WX1RGB[[157]], colormode=Color),Image(WX1RGB[[158]], colormode=Color),Image(WX1RGB[[159]], colormode=Color),Image(WX1RGB[[160]], colormode=Color),Image(WX1RGB[[161]], colormode=Color),Image(WX1RGB[[162]], colormode=Color),Image(WX1RGB[[163]], colormode=Color),Image(WX1RGB[[164]], colormode=Color),Image(WX1RGB[[165]], colormode=Color),Image(WX1RGB[[166]], colormode=Color),Image(WX1RGB[[167]], colormode=Color),Image(WX1RGB[[168]], colormode=Color),Image(WX1RGB[[169]], colormode=Color),Image(WX1RGB[[170]], colormode=Color),Image(WX1RGB[[171]], colormode=Color),Image(WX1RGB[[172]], colormode=Color),Image(WX1RGB[[173]], colormode=Color),Image(WX1RGB[[174]], colormode=Color),Image(WX1RGB[[175]], colormode=Color),Image(WX1RGB[[176]], colormode=Color),Image(WX1RGB[[177]], colormode=Color),Image(WX1RGB[[178]], colormode=Color),Image(WX1RGB[[179]], colormode=Color),Image(WX1RGB[[180]], colormode=Color),Image(WX1RGB[[181]], colormode=Color),Image(WX1RGB[[182]], colormode=Color),Image(WX1RGB[[183]], colormode=Color),Image(WX1RGB[[184]], colormode=Color),Image(WX1RGB[[185]], colormode=Color),Image(WX1RGB[[186]], colormode=Color),Image(WX1RGB[[187]], colormode=Color),Image(WX1RGB[[188]], colormode=Color),Image(WX1RGB[[189]], colormode=Color),Image(WX1RGB[[190]], colormode=Color),Image(WX1RGB[[191]], colormode=Color),Image(WX1RGB[[192]], colormode=Color),Image(WX1RGB[[193]], colormode=Color),Image(WX1RGB[[194]], colormode=Color),Image(WX1RGB[[195]], colormode=Color),Image(WX1RGB[[196]], colormode=Color),Image(WX1RGB[[197]], colormode=Color),Image(WX1RGB[[198]], colormode=Color),Image(WX1RGB[[199]], colormode=Color),Image(WX1RGB[[200]], colormode=Color)
,Image(WX1RGB[[201]], colormode=Color),Image(WX1RGB[[202]], colormode=Color),Image(WX1RGB[[203]], colormode=Color),Image(WX1RGB[[204]], colormode=Color),Image(WX1RGB[[205]], colormode=Color),Image(WX1RGB[[206]], colormode=Color),Image(WX1RGB[[207]], colormode=Color),Image(WX1RGB[[208]], colormode=Color),Image(WX1RGB[[209]], colormode=Color),Image(WX1RGB[[210]], colormode=Color),Image(WX1RGB[[211]], colormode=Color),Image(WX1RGB[[212]], colormode=Color),Image(WX1RGB[[213]], colormode=Color),Image(WX1RGB[[214]], colormode=Color),Image(WX1RGB[[215]], colormode=Color),Image(WX1RGB[[216]], colormode=Color),Image(WX1RGB[[217]], colormode=Color),Image(WX1RGB[[218]], colormode=Color),Image(WX1RGB[[219]], colormode=Color),Image(WX1RGB[[220]], colormode=Color),Image(WX1RGB[[221]], colormode=Color),Image(WX1RGB[[222]], colormode=Color),Image(WX1RGB[[223]], colormode=Color),Image(WX1RGB[[224]], colormode=Color),Image(WX1RGB[[225]], colormode=Color),Image(WX1RGB[[226]], colormode=Color),Image(WX1RGB[[227]], colormode=Color),Image(WX1RGB[[228]], colormode=Color),Image(WX1RGB[[229]], colormode=Color),Image(WX1RGB[[230]], colormode=Color),Image(WX1RGB[[231]], colormode=Color),Image(WX1RGB[[232]], colormode=Color),Image(WX1RGB[[233]], colormode=Color),Image(WX1RGB[[234]], colormode=Color),Image(WX1RGB[[235]], colormode=Color),Image(WX1RGB[[236]], colormode=Color),Image(WX1RGB[[237]], colormode=Color),Image(WX1RGB[[238]], colormode=Color),Image(WX1RGB[[239]], colormode=Color),Image(WX1RGB[[240]], colormode=Color),Image(WX1RGB[[241]], colormode=Color),Image(WX1RGB[[242]], colormode=Color),Image(WX1RGB[[243]], colormode=Color),Image(WX1RGB[[244]], colormode=Color),Image(WX1RGB[[245]], colormode=Color),Image(WX1RGB[[246]], colormode=Color),Image(WX1RGB[[247]], colormode=Color),Image(WX1RGB[[248]], colormode=Color),Image(WX1RGB[[249]], colormode=Color),Image(WX1RGB[[250]], colormode=Color),Image(WX1RGB[[251]], colormode=Color),Image(WX1RGB[[252]], colormode=Color),Image(WX1RGB[[253]], colormode=Color),Image(WX1RGB[[254]], colormode=Color),Image(WX1RGB[[255]], colormode=Color),Image(WX1RGB[[256]], colormode=Color),Image(WX1RGB[[257]], colormode=Color),Image(WX1RGB[[258]], colormode=Color),Image(WX1RGB[[259]], colormode=Color),Image(WX1RGB[[260]], colormode=Color),Image(WX1RGB[[261]], colormode=Color),Image(WX1RGB[[262]], colormode=Color),Image(WX1RGB[[263]], colormode=Color),Image(WX1RGB[[264]], colormode=Color),Image(WX1RGB[[265]], colormode=Color),Image(WX1RGB[[266]], colormode=Color),Image(WX1RGB[[267]], colormode=Color),Image(WX1RGB[[268]], colormode=Color),Image(WX1RGB[[269]], colormode=Color),Image(WX1RGB[[270]], colormode=Color),Image(WX1RGB[[271]], colormode=Color),Image(WX1RGB[[272]], colormode=Color),Image(WX1RGB[[273]], colormode=Color),Image(WX1RGB[[274]], colormode=Color),Image(WX1RGB[[275]], colormode=Color),Image(WX1RGB[[276]], colormode=Color),Image(WX1RGB[[277]], colormode=Color),Image(WX1RGB[[278]], colormode=Color),Image(WX1RGB[[279]], colormode=Color),Image(WX1RGB[[280]], colormode=Color),Image(WX1RGB[[281]], colormode=Color),Image(WX1RGB[[282]], colormode=Color),Image(WX1RGB[[283]], colormode=Color),Image(WX1RGB[[284]], colormode=Color),Image(WX1RGB[[285]], colormode=Color),Image(WX1RGB[[286]], colormode=Color),Image(WX1RGB[[287]], colormode=Color),Image(WX1RGB[[288]], colormode=Color),Image(WX1RGB[[289]], colormode=Color),Image(WX1RGB[[290]], colormode=Color),Image(WX1RGB[[291]], colormode=Color),Image(WX1RGB[[292]], colormode=Color),Image(WX1RGB[[293]], colormode=Color),Image(WX1RGB[[294]], colormode=Color),Image(WX1RGB[[295]], colormode=Color),Image(WX1RGB[[296]], colormode=Color),Image(WX1RGB[[297]], colormode=Color),Image(WX1RGB[[298]], colormode=Color),Image(WX1RGB[[299]], colormode=Color),Image(WX1RGB[[300]], colormode=Color)
,Image(WX1RGB[[301]], colormode=Color),Image(WX1RGB[[302]], colormode=Color),Image(WX1RGB[[303]], colormode=Color),Image(WX1RGB[[304]], colormode=Color),Image(WX1RGB[[305]], colormode=Color),Image(WX1RGB[[306]], colormode=Color),Image(WX1RGB[[307]], colormode=Color),Image(WX1RGB[[308]], colormode=Color),Image(WX1RGB[[309]], colormode=Color),Image(WX1RGB[[310]], colormode=Color),Image(WX1RGB[[311]], colormode=Color),Image(WX1RGB[[312]], colormode=Color),Image(WX1RGB[[313]], colormode=Color),Image(WX1RGB[[314]], colormode=Color),Image(WX1RGB[[315]], colormode=Color),Image(WX1RGB[[316]], colormode=Color),Image(WX1RGB[[317]], colormode=Color),Image(WX1RGB[[318]], colormode=Color),Image(WX1RGB[[319]], colormode=Color),Image(WX1RGB[[320]], colormode=Color),Image(WX1RGB[[321]], colormode=Color),Image(WX1RGB[[322]], colormode=Color),Image(WX1RGB[[323]], colormode=Color),Image(WX1RGB[[324]], colormode=Color),Image(WX1RGB[[325]], colormode=Color),Image(WX1RGB[[326]], colormode=Color),Image(WX1RGB[[327]], colormode=Color),Image(WX1RGB[[328]], colormode=Color),Image(WX1RGB[[329]], colormode=Color),Image(WX1RGB[[330]], colormode=Color),Image(WX1RGB[[331]], colormode=Color),Image(WX1RGB[[332]], colormode=Color),Image(WX1RGB[[333]], colormode=Color),Image(WX1RGB[[334]], colormode=Color),Image(WX1RGB[[335]], colormode=Color),Image(WX1RGB[[336]], colormode=Color),Image(WX1RGB[[337]], colormode=Color),Image(WX1RGB[[338]], colormode=Color),Image(WX1RGB[[339]], colormode=Color),Image(WX1RGB[[340]], colormode=Color),Image(WX1RGB[[341]], colormode=Color),Image(WX1RGB[[342]], colormode=Color),Image(WX1RGB[[343]], colormode=Color),Image(WX1RGB[[344]], colormode=Color),Image(WX1RGB[[345]], colormode=Color),Image(WX1RGB[[346]], colormode=Color),Image(WX1RGB[[347]], colormode=Color),Image(WX1RGB[[348]], colormode=Color),Image(WX1RGB[[349]], colormode=Color),Image(WX1RGB[[350]], colormode=Color),Image(WX1RGB[[351]], colormode=Color),Image(WX1RGB[[352]], colormode=Color),Image(WX1RGB[[353]], colormode=Color),Image(WX1RGB[[354]], colormode=Color),Image(WX1RGB[[355]], colormode=Color),Image(WX1RGB[[356]], colormode=Color),Image(WX1RGB[[357]], colormode=Color),Image(WX1RGB[[358]], colormode=Color),Image(WX1RGB[[359]], colormode=Color),Image(WX1RGB[[360]], colormode=Color),Image(WX1RGB[[361]], colormode=Color),Image(WX1RGB[[362]], colormode=Color),Image(WX1RGB[[363]], colormode=Color),Image(WX1RGB[[364]], colormode=Color),Image(WX1RGB[[365]], colormode=Color),Image(WX1RGB[[366]], colormode=Color),Image(WX1RGB[[367]], colormode=Color),Image(WX1RGB[[368]], colormode=Color),Image(WX1RGB[[369]], colormode=Color),Image(WX1RGB[[370]], colormode=Color),Image(WX1RGB[[371]], colormode=Color),Image(WX1RGB[[372]], colormode=Color),Image(WX1RGB[[373]], colormode=Color),Image(WX1RGB[[374]], colormode=Color),Image(WX1RGB[[375]], colormode=Color),Image(WX1RGB[[376]], colormode=Color),Image(WX1RGB[[377]], colormode=Color),Image(WX1RGB[[378]], colormode=Color),Image(WX1RGB[[379]], colormode=Color),Image(WX1RGB[[380]], colormode=Color),Image(WX1RGB[[381]], colormode=Color),Image(WX1RGB[[382]], colormode=Color),Image(WX1RGB[[383]], colormode=Color),Image(WX1RGB[[384]], colormode=Color),Image(WX1RGB[[385]], colormode=Color),Image(WX1RGB[[386]], colormode=Color),Image(WX1RGB[[387]], colormode=Color),Image(WX1RGB[[388]], colormode=Color),Image(WX1RGB[[389]], colormode=Color),Image(WX1RGB[[390]], colormode=Color),Image(WX1RGB[[391]], colormode=Color),Image(WX1RGB[[392]], colormode=Color),Image(WX1RGB[[393]], colormode=Color),Image(WX1RGB[[394]], colormode=Color),Image(WX1RGB[[395]], colormode=Color),Image(WX1RGB[[396]], colormode=Color),Image(WX1RGB[[397]], colormode=Color),Image(WX1RGB[[398]], colormode=Color),Image(WX1RGB[[399]], colormode=Color),Image(WX1RGB[[400]], colormode=Color)
),nx=40, all=T, spacing = 0.0, margin = 0, drawGrid=F)

########################################################################################
########################################################################################
## 2048x2048バージョン
########################################################################################
########################################################################################
try(WX22pa <- map(array_branch(WX22p, 1),
                  .f = function(z) {mmand::threshold(z, 0.5, binarise = TRUE)}) , silent=TRUE)
str(WX22pa)
try(WX22pal <- list2tensor(WX22pa), silent=TRUE)
str(WX22pal)

# DIR01 <- "~/Rstudio_Keras_test/190513_dl"
# DIR01 <- "~/Rstudio_Keras_test/190513_dl/Results_NB4/UNET_Test_01/LOG001"
# DIR02 <- "/190612-1813_TrainAcc1_TestAcc0.9801"
try(dir.create(paste(DIR01, DIR02, "/5_WD22", sep=""), showWarnings = F), silent=TRUE)
try(dir.create(paste(DIR01, DIR02, "/5_WD22_Bi", sep=""), showWarnings = F), silent=TRUE)

try(for(ABC in 1:length(dir(Test_WIDE2048_PATH01))){
 # ABC <- 1
 #WX
 Image_color01 <- paintObjects(WX22pal[ABC,,,]/2, toRGB(WX$WX2[[ABC]])/1.25,opac=c(0.3, 0.3),
                               col=c("blue","blue"), thick=T, closed=F)
 try(dev.off(), silent=T)
 png(paste(DIR01, DIR02, "/5_WD22/WD_Res_", formatC(ABC, width = 5, flag = "0"), ".png", sep=""),
     width = 400, height = 400)
 #par(bg = 'grey')
 EBImage::display(Image_color01,
                  nx=1, all=TRUE, spacing = 0, margin = 0)
 try(dev.off(), silent=T)
 png(paste(DIR01, DIR02, "/5_WD22_Bi/WD_ResB_", formatC(ABC, width = 5, flag = "0"), ".png", sep=""),
     width = 400, height = 400)
 #par(bg = 'grey')
 EBImage::display(WX22pal[ABC,,,],
                  nx=1, all=TRUE, spacing = 0, margin = 0)
 try(dev.off(), silent=T)
}, silent=TRUE)

########################################################################################
ImageFile = paste(DIR01, DIR02, "/5_WD22/", dir(paste(DIR01, DIR02, "/5_WD22", sep="")), sep="")
RGB = map(ImageFile, preprocess_image03, shape = SHAPE)
str(RGB)

print(RGB[[1]])
EBImage::display(Image(RGB[[1]], colormode=Color), all=T)

#par(bg = 'grey', mfrow = c(1,2), mar = c(0, 0, 0, 0))
paste("Image(RGB[[", 1:300, "]], colormode=Color)", sep="", collapse=",")

EBImage::display(EBImage::combine(
 Image(RGB[[1]], colormode=Color),Image(RGB[[2]], colormode=Color),Image(RGB[[3]], colormode=Color),Image(RGB[[4]], colormode=Color),Image(RGB[[5]], colormode=Color),Image(RGB[[6]], colormode=Color),Image(RGB[[7]], colormode=Color),Image(RGB[[8]], colormode=Color),Image(RGB[[9]], colormode=Color),Image(RGB[[10]], colormode=Color),Image(RGB[[11]], colormode=Color),Image(RGB[[12]], colormode=Color),Image(RGB[[13]], colormode=Color),Image(RGB[[14]], colormode=Color),Image(RGB[[15]], colormode=Color),Image(RGB[[16]], colormode=Color),Image(RGB[[17]], colormode=Color),Image(RGB[[18]], colormode=Color),Image(RGB[[19]], colormode=Color),Image(RGB[[20]], colormode=Color),Image(RGB[[21]], colormode=Color),Image(RGB[[22]], colormode=Color),Image(RGB[[23]], colormode=Color),Image(RGB[[24]], colormode=Color),Image(RGB[[25]], colormode=Color),Image(RGB[[26]], colormode=Color),Image(RGB[[27]], colormode=Color),Image(RGB[[28]], colormode=Color),Image(RGB[[29]], colormode=Color),Image(RGB[[30]], colormode=Color),Image(RGB[[31]], colormode=Color),Image(RGB[[32]], colormode=Color),Image(RGB[[33]], colormode=Color),Image(RGB[[34]], colormode=Color),Image(RGB[[35]], colormode=Color),Image(RGB[[36]], colormode=Color),Image(RGB[[37]], colormode=Color),Image(RGB[[38]], colormode=Color),Image(RGB[[39]], colormode=Color),Image(RGB[[40]], colormode=Color),Image(RGB[[41]], colormode=Color),Image(RGB[[42]], colormode=Color),Image(RGB[[43]], colormode=Color),Image(RGB[[44]], colormode=Color),Image(RGB[[45]], colormode=Color),Image(RGB[[46]], colormode=Color),Image(RGB[[47]], colormode=Color),Image(RGB[[48]], colormode=Color),Image(RGB[[49]], colormode=Color),Image(RGB[[50]], colormode=Color)
 ,Image(RGB[[51]], colormode=Color),Image(RGB[[52]], colormode=Color),Image(RGB[[53]], colormode=Color),Image(RGB[[54]], colormode=Color),Image(RGB[[55]], colormode=Color),Image(RGB[[56]], colormode=Color),Image(RGB[[57]], colormode=Color),Image(RGB[[58]], colormode=Color),Image(RGB[[59]], colormode=Color),Image(RGB[[60]], colormode=Color),Image(RGB[[61]], colormode=Color),Image(RGB[[62]], colormode=Color),Image(RGB[[63]], colormode=Color),Image(RGB[[64]], colormode=Color),Image(RGB[[65]], colormode=Color),Image(RGB[[66]], colormode=Color),Image(RGB[[67]], colormode=Color),Image(RGB[[68]], colormode=Color),Image(RGB[[69]], colormode=Color),Image(RGB[[70]], colormode=Color),Image(RGB[[71]], colormode=Color),Image(RGB[[72]], colormode=Color),Image(RGB[[73]], colormode=Color),Image(RGB[[74]], colormode=Color),Image(RGB[[75]], colormode=Color),Image(RGB[[76]], colormode=Color),Image(RGB[[77]], colormode=Color),Image(RGB[[78]], colormode=Color),Image(RGB[[79]], colormode=Color),Image(RGB[[80]], colormode=Color),Image(RGB[[81]], colormode=Color),Image(RGB[[82]], colormode=Color),Image(RGB[[83]], colormode=Color),Image(RGB[[84]], colormode=Color),Image(RGB[[85]], colormode=Color),Image(RGB[[86]], colormode=Color),Image(RGB[[87]], colormode=Color),Image(RGB[[88]], colormode=Color),Image(RGB[[89]], colormode=Color),Image(RGB[[90]], colormode=Color),Image(RGB[[91]], colormode=Color),Image(RGB[[92]], colormode=Color),Image(RGB[[93]], colormode=Color),Image(RGB[[94]], colormode=Color),Image(RGB[[95]], colormode=Color),Image(RGB[[96]], colormode=Color),Image(RGB[[97]], colormode=Color),Image(RGB[[98]], colormode=Color),Image(RGB[[99]], colormode=Color),Image(RGB[[100]], colormode=Color)
),nx=20, all=T, spacing = 0.0, margin = 0, drawGrid=F)

EBImage::display(EBImage::combine(
 Image(RGB[[1]], colormode=Color),Image(RGB[[2]], colormode=Color),Image(RGB[[3]], colormode=Color),Image(RGB[[4]], colormode=Color),Image(RGB[[5]], colormode=Color),Image(RGB[[6]], colormode=Color),Image(RGB[[7]], colormode=Color),Image(RGB[[8]], colormode=Color),Image(RGB[[9]], colormode=Color),Image(RGB[[10]], colormode=Color),Image(RGB[[11]], colormode=Color),Image(RGB[[12]], colormode=Color),Image(RGB[[13]], colormode=Color),Image(RGB[[14]], colormode=Color),Image(RGB[[15]], colormode=Color),Image(RGB[[16]], colormode=Color),Image(RGB[[17]], colormode=Color),Image(RGB[[18]], colormode=Color),Image(RGB[[19]], colormode=Color),Image(RGB[[20]], colormode=Color),Image(RGB[[21]], colormode=Color),Image(RGB[[22]], colormode=Color),Image(RGB[[23]], colormode=Color),Image(RGB[[24]], colormode=Color),Image(RGB[[25]], colormode=Color)
 ,Image(RGB[[26]], colormode=Color),Image(RGB[[27]], colormode=Color),Image(RGB[[28]], colormode=Color),Image(RGB[[29]], colormode=Color),Image(RGB[[30]], colormode=Color),Image(RGB[[31]], colormode=Color),Image(RGB[[32]], colormode=Color),Image(RGB[[33]], colormode=Color),Image(RGB[[34]], colormode=Color),Image(RGB[[35]], colormode=Color),Image(RGB[[36]], colormode=Color),Image(RGB[[37]], colormode=Color),Image(RGB[[38]], colormode=Color),
 Image(RGB[[39]], colormode=Color),Image(RGB[[40]], colormode=Color),Image(RGB[[41]], colormode=Color),Image(RGB[[42]], colormode=Color),Image(RGB[[43]], colormode=Color),Image(RGB[[44]], colormode=Color),Image(RGB[[45]], colormode=Color),Image(RGB[[46]], colormode=Color),Image(RGB[[47]], colormode=Color),Image(RGB[[48]], colormode=Color),Image(RGB[[49]], colormode=Color),Image(RGB[[50]], colormode=Color),Image(RGB[[51]], colormode=Color),Image(RGB[[52]], colormode=Color),Image(RGB[[53]], colormode=Color),Image(RGB[[54]], colormode=Color),Image(RGB[[55]], colormode=Color),
 Image(RGB[[56]], colormode=Color),Image(RGB[[57]], colormode=Color),Image(RGB[[58]], colormode=Color),Image(RGB[[59]], colormode=Color),Image(RGB[[60]], colormode=Color),Image(RGB[[61]], colormode=Color),Image(RGB[[62]], colormode=Color),Image(RGB[[63]], colormode=Color),Image(RGB[[64]], colormode=Color),Image(RGB[[65]], colormode=Color),Image(RGB[[66]], colormode=Color),Image(RGB[[67]], colormode=Color),Image(RGB[[68]], colormode=Color),Image(RGB[[69]], colormode=Color),Image(RGB[[70]], colormode=Color),Image(RGB[[71]], colormode=Color),Image(RGB[[72]], colormode=Color),
 Image(RGB[[73]], colormode=Color),Image(RGB[[74]], colormode=Color),Image(RGB[[75]], colormode=Color),Image(RGB[[76]], colormode=Color),Image(RGB[[77]], colormode=Color),Image(RGB[[78]], colormode=Color),Image(RGB[[79]], colormode=Color),Image(RGB[[80]], colormode=Color),Image(RGB[[81]], colormode=Color),Image(RGB[[82]], colormode=Color),Image(RGB[[83]], colormode=Color),Image(RGB[[84]], colormode=Color),Image(RGB[[85]], colormode=Color),Image(RGB[[86]], colormode=Color),Image(RGB[[87]], colormode=Color),Image(RGB[[88]], colormode=Color),Image(RGB[[89]], colormode=Color)
 ,Image(RGB[[90]], colormode=Color),Image(RGB[[91]], colormode=Color),Image(RGB[[92]], colormode=Color),Image(RGB[[93]], colormode=Color),Image(RGB[[94]], colormode=Color),Image(RGB[[95]], colormode=Color),Image(RGB[[96]], colormode=Color),Image(RGB[[97]], colormode=Color),Image(RGB[[98]], colormode=Color),Image(RGB[[99]], colormode=Color),Image(RGB[[100]], colormode=Color),Image(RGB[[101]], colormode=Color),Image(RGB[[102]], colormode=Color),Image(RGB[[103]], colormode=Color),Image(RGB[[104]], colormode=Color),Image(RGB[[105]], colormode=Color),Image(RGB[[106]], colormode=Color),Image(RGB[[107]], colormode=Color),Image(RGB[[108]], colormode=Color),Image(RGB[[109]], colormode=Color),Image(RGB[[110]], colormode=Color),Image(RGB[[111]], colormode=Color),Image(RGB[[112]], colormode=Color),Image(RGB[[113]], colormode=Color),Image(RGB[[114]], colormode=Color),Image(RGB[[115]], colormode=Color)
 ,Image(RGB[[116]], colormode=Color),Image(RGB[[117]], colormode=Color),Image(RGB[[118]], colormode=Color),Image(RGB[[119]], colormode=Color),Image(RGB[[120]], colormode=Color),Image(RGB[[121]], colormode=Color),Image(RGB[[122]], colormode=Color),Image(RGB[[123]], colormode=Color),Image(RGB[[124]], colormode=Color),Image(RGB[[125]], colormode=Color),Image(RGB[[126]], colormode=Color),Image(RGB[[127]], colormode=Color),Image(RGB[[128]], colormode=Color),Image(RGB[[129]], colormode=Color),Image(RGB[[130]], colormode=Color),Image(RGB[[131]], colormode=Color),Image(RGB[[132]], colormode=Color),Image(RGB[[133]], colormode=Color),Image(RGB[[134]], colormode=Color),Image(RGB[[135]], colormode=Color),Image(RGB[[136]], colormode=Color),Image(RGB[[137]], colormode=Color)
 ,Image(RGB[[138]], colormode=Color),Image(RGB[[139]], colormode=Color),Image(RGB[[140]], colormode=Color),Image(RGB[[141]], colormode=Color),Image(RGB[[142]], colormode=Color),Image(RGB[[143]], colormode=Color),Image(RGB[[144]], colormode=Color),Image(RGB[[145]], colormode=Color),Image(RGB[[146]], colormode=Color),Image(RGB[[147]], colormode=Color),Image(RGB[[148]], colormode=Color),Image(RGB[[149]], colormode=Color),Image(RGB[[150]], colormode=Color),Image(RGB[[151]], colormode=Color),Image(RGB[[152]], colormode=Color),Image(RGB[[153]], colormode=Color),Image(RGB[[154]], colormode=Color),Image(RGB[[155]], colormode=Color),Image(RGB[[156]], colormode=Color),Image(RGB[[157]], colormode=Color),Image(RGB[[158]], colormode=Color),Image(RGB[[159]], colormode=Color),Image(RGB[[160]], colormode=Color),Image(RGB[[161]], colormode=Color),Image(RGB[[162]], colormode=Color),Image(RGB[[163]], colormode=Color),Image(RGB[[164]], colormode=Color),Image(RGB[[165]], colormode=Color),Image(RGB[[166]], colormode=Color),
 Image(RGB[[167]], colormode=Color),Image(RGB[[168]], colormode=Color),Image(RGB[[169]], colormode=Color),Image(RGB[[170]], colormode=Color),Image(RGB[[171]], colormode=Color),Image(RGB[[172]], colormode=Color),Image(RGB[[173]], colormode=Color),Image(RGB[[174]], colormode=Color),Image(RGB[[175]], colormode=Color),Image(RGB[[176]], colormode=Color),Image(RGB[[177]], colormode=Color),Image(RGB[[178]], colormode=Color),Image(RGB[[179]], colormode=Color),Image(RGB[[180]], colormode=Color),Image(RGB[[181]], colormode=Color),Image(RGB[[182]], colormode=Color),Image(RGB[[183]], colormode=Color),Image(RGB[[184]], colormode=Color),Image(RGB[[185]], colormode=Color),Image(RGB[[186]], colormode=Color),Image(RGB[[187]], colormode=Color),Image(RGB[[188]], colormode=Color),Image(RGB[[189]], colormode=Color),Image(RGB[[190]], colormode=Color),Image(RGB[[191]], colormode=Color),Image(RGB[[192]], colormode=Color),Image(RGB[[193]], colormode=Color),Image(RGB[[194]], colormode=Color),Image(RGB[[195]], colormode=Color),Image(RGB[[196]], colormode=Color)
 ,Image(RGB[[197]], colormode=Color),Image(RGB[[198]], colormode=Color),Image(RGB[[199]], colormode=Color),Image(RGB[[200]], colormode=Color))
 ,nx=20, all=T, spacing = 0.0, margin = 0, drawGrid=F)

EBImage::display(EBImage::combine(
Image(RGB[[1]], colormode=Color),Image(RGB[[2]], colormode=Color),Image(RGB[[3]], colormode=Color),Image(RGB[[4]], colormode=Color),Image(RGB[[5]], colormode=Color),Image(RGB[[6]], colormode=Color),Image(RGB[[7]], colormode=Color),Image(RGB[[8]], colormode=Color),Image(RGB[[9]], colormode=Color),Image(RGB[[10]], colormode=Color),Image(RGB[[11]], colormode=Color),Image(RGB[[12]], colormode=Color),Image(RGB[[13]], colormode=Color),Image(RGB[[14]], colormode=Color),Image(RGB[[15]], colormode=Color),Image(RGB[[16]], colormode=Color),Image(RGB[[17]], colormode=Color),Image(RGB[[18]], colormode=Color),Image(RGB[[19]], colormode=Color),Image(RGB[[20]], colormode=Color),Image(RGB[[21]], colormode=Color),Image(RGB[[22]], colormode=Color),Image(RGB[[23]], colormode=Color),Image(RGB[[24]], colormode=Color),Image(RGB[[25]], colormode=Color)
,Image(RGB[[26]], colormode=Color),Image(RGB[[27]], colormode=Color),Image(RGB[[28]], colormode=Color),Image(RGB[[29]], colormode=Color),Image(RGB[[30]], colormode=Color),Image(RGB[[31]], colormode=Color),Image(RGB[[32]], colormode=Color),Image(RGB[[33]], colormode=Color),Image(RGB[[34]], colormode=Color),Image(RGB[[35]], colormode=Color),Image(RGB[[36]], colormode=Color),Image(RGB[[37]], colormode=Color),Image(RGB[[38]], colormode=Color),
Image(RGB[[39]], colormode=Color),Image(RGB[[40]], colormode=Color),Image(RGB[[41]], colormode=Color),Image(RGB[[42]], colormode=Color),Image(RGB[[43]], colormode=Color),Image(RGB[[44]], colormode=Color),Image(RGB[[45]], colormode=Color),Image(RGB[[46]], colormode=Color),Image(RGB[[47]], colormode=Color),Image(RGB[[48]], colormode=Color),Image(RGB[[49]], colormode=Color),Image(RGB[[50]], colormode=Color),Image(RGB[[51]], colormode=Color),Image(RGB[[52]], colormode=Color),Image(RGB[[53]], colormode=Color),Image(RGB[[54]], colormode=Color),Image(RGB[[55]], colormode=Color),
Image(RGB[[56]], colormode=Color),Image(RGB[[57]], colormode=Color),Image(RGB[[58]], colormode=Color),Image(RGB[[59]], colormode=Color),Image(RGB[[60]], colormode=Color),Image(RGB[[61]], colormode=Color),Image(RGB[[62]], colormode=Color),Image(RGB[[63]], colormode=Color),Image(RGB[[64]], colormode=Color),Image(RGB[[65]], colormode=Color),Image(RGB[[66]], colormode=Color),Image(RGB[[67]], colormode=Color),Image(RGB[[68]], colormode=Color),Image(RGB[[69]], colormode=Color),Image(RGB[[70]], colormode=Color),Image(RGB[[71]], colormode=Color),Image(RGB[[72]], colormode=Color),
Image(RGB[[73]], colormode=Color),Image(RGB[[74]], colormode=Color),Image(RGB[[75]], colormode=Color),Image(RGB[[76]], colormode=Color),Image(RGB[[77]], colormode=Color),Image(RGB[[78]], colormode=Color),Image(RGB[[79]], colormode=Color),Image(RGB[[80]], colormode=Color),Image(RGB[[81]], colormode=Color),Image(RGB[[82]], colormode=Color),Image(RGB[[83]], colormode=Color),Image(RGB[[84]], colormode=Color),Image(RGB[[85]], colormode=Color),Image(RGB[[86]], colormode=Color),Image(RGB[[87]], colormode=Color),Image(RGB[[88]], colormode=Color),Image(RGB[[89]], colormode=Color)
,Image(RGB[[90]], colormode=Color),Image(RGB[[91]], colormode=Color),Image(RGB[[92]], colormode=Color),Image(RGB[[93]], colormode=Color),Image(RGB[[94]], colormode=Color),Image(RGB[[95]], colormode=Color),Image(RGB[[96]], colormode=Color),Image(RGB[[97]], colormode=Color),Image(RGB[[98]], colormode=Color),Image(RGB[[99]], colormode=Color),Image(RGB[[100]], colormode=Color),Image(RGB[[101]], colormode=Color),Image(RGB[[102]], colormode=Color),Image(RGB[[103]], colormode=Color),Image(RGB[[104]], colormode=Color),Image(RGB[[105]], colormode=Color),Image(RGB[[106]], colormode=Color),Image(RGB[[107]], colormode=Color),Image(RGB[[108]], colormode=Color),Image(RGB[[109]], colormode=Color),Image(RGB[[110]], colormode=Color),Image(RGB[[111]], colormode=Color),Image(RGB[[112]], colormode=Color),Image(RGB[[113]], colormode=Color),Image(RGB[[114]], colormode=Color),Image(RGB[[115]], colormode=Color)
,Image(RGB[[116]], colormode=Color),Image(RGB[[117]], colormode=Color),Image(RGB[[118]], colormode=Color),Image(RGB[[119]], colormode=Color),Image(RGB[[120]], colormode=Color),Image(RGB[[121]], colormode=Color),Image(RGB[[122]], colormode=Color),Image(RGB[[123]], colormode=Color),Image(RGB[[124]], colormode=Color),Image(RGB[[125]], colormode=Color),Image(RGB[[126]], colormode=Color),Image(RGB[[127]], colormode=Color),Image(RGB[[128]], colormode=Color),Image(RGB[[129]], colormode=Color),Image(RGB[[130]], colormode=Color),Image(RGB[[131]], colormode=Color),Image(RGB[[132]], colormode=Color),Image(RGB[[133]], colormode=Color),Image(RGB[[134]], colormode=Color),Image(RGB[[135]], colormode=Color),Image(RGB[[136]], colormode=Color),Image(RGB[[137]], colormode=Color)
,Image(RGB[[138]], colormode=Color),Image(RGB[[139]], colormode=Color),Image(RGB[[140]], colormode=Color),Image(RGB[[141]], colormode=Color),Image(RGB[[142]], colormode=Color),Image(RGB[[143]], colormode=Color),Image(RGB[[144]], colormode=Color),Image(RGB[[145]], colormode=Color),Image(RGB[[146]], colormode=Color),Image(RGB[[147]], colormode=Color),Image(RGB[[148]], colormode=Color),Image(RGB[[149]], colormode=Color),Image(RGB[[150]], colormode=Color),Image(RGB[[151]], colormode=Color),Image(RGB[[152]], colormode=Color),Image(RGB[[153]], colormode=Color),Image(RGB[[154]], colormode=Color),Image(RGB[[155]], colormode=Color),Image(RGB[[156]], colormode=Color),Image(RGB[[157]], colormode=Color),Image(RGB[[158]], colormode=Color),Image(RGB[[159]], colormode=Color),Image(RGB[[160]], colormode=Color),Image(RGB[[161]], colormode=Color),Image(RGB[[162]], colormode=Color),Image(RGB[[163]], colormode=Color),Image(RGB[[164]], colormode=Color),Image(RGB[[165]], colormode=Color),Image(RGB[[166]], colormode=Color),
Image(RGB[[167]], colormode=Color),Image(RGB[[168]], colormode=Color),Image(RGB[[169]], colormode=Color),Image(RGB[[170]], colormode=Color),Image(RGB[[171]], colormode=Color),Image(RGB[[172]], colormode=Color),Image(RGB[[173]], colormode=Color),Image(RGB[[174]], colormode=Color),Image(RGB[[175]], colormode=Color),Image(RGB[[176]], colormode=Color),Image(RGB[[177]], colormode=Color),Image(RGB[[178]], colormode=Color),Image(RGB[[179]], colormode=Color),Image(RGB[[180]], colormode=Color),Image(RGB[[181]], colormode=Color),Image(RGB[[182]], colormode=Color),Image(RGB[[183]], colormode=Color),Image(RGB[[184]], colormode=Color),Image(RGB[[185]], colormode=Color),Image(RGB[[186]], colormode=Color),Image(RGB[[187]], colormode=Color),Image(RGB[[188]], colormode=Color),Image(RGB[[189]], colormode=Color),Image(RGB[[190]], colormode=Color),Image(RGB[[191]], colormode=Color),Image(RGB[[192]], colormode=Color),Image(RGB[[193]], colormode=Color),Image(RGB[[194]], colormode=Color),Image(RGB[[195]], colormode=Color),Image(RGB[[196]], colormode=Color)
,Image(RGB[[197]], colormode=Color),Image(RGB[[198]], colormode=Color),Image(RGB[[199]], colormode=Color),Image(RGB[[200]], colormode=Color),Image(RGB[[201]], colormode=Color),Image(RGB[[202]], colormode=Color),Image(RGB[[203]], colormode=Color),Image(RGB[[204]], colormode=Color),Image(RGB[[205]], colormode=Color),Image(RGB[[206]], colormode=Color),Image(RGB[[207]], colormode=Color),Image(RGB[[208]], colormode=Color),Image(RGB[[209]], colormode=Color),Image(RGB[[210]], colormode=Color),Image(RGB[[211]], colormode=Color),Image(RGB[[212]], colormode=Color),Image(RGB[[213]], colormode=Color),Image(RGB[[214]], colormode=Color),Image(RGB[[215]], colormode=Color),Image(RGB[[216]], colormode=Color),Image(RGB[[217]], colormode=Color),Image(RGB[[218]], colormode=Color),Image(RGB[[219]], colormode=Color),Image(RGB[[220]], colormode=Color),Image(RGB[[221]], colormode=Color),Image(RGB[[222]], colormode=Color),Image(RGB[[223]], colormode=Color),Image(RGB[[224]], colormode=Color),Image(RGB[[225]], colormode=Color),Image(RGB[[226]], colormode=Color),Image(RGB[[227]], colormode=Color)
,Image(RGB[[228]], colormode=Color),Image(RGB[[229]], colormode=Color),Image(RGB[[230]], colormode=Color),Image(RGB[[231]], colormode=Color),Image(RGB[[232]], colormode=Color),Image(RGB[[233]], colormode=Color),Image(RGB[[234]], colormode=Color),Image(RGB[[235]], colormode=Color),Image(RGB[[236]], colormode=Color),Image(RGB[[237]], colormode=Color),Image(RGB[[238]], colormode=Color),Image(RGB[[239]], colormode=Color),Image(RGB[[240]], colormode=Color),Image(RGB[[241]], colormode=Color),Image(RGB[[242]], colormode=Color),Image(RGB[[243]], colormode=Color),Image(RGB[[244]], colormode=Color),Image(RGB[[245]], colormode=Color),Image(RGB[[246]], colormode=Color),Image(RGB[[247]], colormode=Color),Image(RGB[[248]], colormode=Color),Image(RGB[[249]], colormode=Color),Image(RGB[[250]], colormode=Color),Image(RGB[[251]], colormode=Color),Image(RGB[[252]], colormode=Color),Image(RGB[[253]], colormode=Color),Image(RGB[[254]], colormode=Color),Image(RGB[[255]], colormode=Color),Image(RGB[[256]], colormode=Color),Image(RGB[[257]], colormode=Color),Image(RGB[[258]], colormode=Color),Image(RGB[[259]], colormode=Color)
,Image(RGB[[260]], colormode=Color),Image(RGB[[261]], colormode=Color),Image(RGB[[262]], colormode=Color),Image(RGB[[263]], colormode=Color),Image(RGB[[264]], colormode=Color),Image(RGB[[265]], colormode=Color),Image(RGB[[266]], colormode=Color),Image(RGB[[267]], colormode=Color),Image(RGB[[268]], colormode=Color),Image(RGB[[269]], colormode=Color),Image(RGB[[270]], colormode=Color),Image(RGB[[271]], colormode=Color),Image(RGB[[272]], colormode=Color),Image(RGB[[273]], colormode=Color),Image(RGB[[274]], colormode=Color),Image(RGB[[275]], colormode=Color),Image(RGB[[276]], colormode=Color),Image(RGB[[277]], colormode=Color),Image(RGB[[278]], colormode=Color),Image(RGB[[279]], colormode=Color),Image(RGB[[280]], colormode=Color),Image(RGB[[281]], colormode=Color),Image(RGB[[282]], colormode=Color),Image(RGB[[283]], colormode=Color),Image(RGB[[284]], colormode=Color),Image(RGB[[285]], colormode=Color),Image(RGB[[286]], colormode=Color),Image(RGB[[287]], colormode=Color),Image(RGB[[288]], colormode=Color),Image(RGB[[289]], colormode=Color),Image(RGB[[290]], colormode=Color),Image(RGB[[291]], colormode=Color),Image(RGB[[292]], colormode=Color)
,Image(RGB[[293]], colormode=Color),Image(RGB[[294]], colormode=Color),Image(RGB[[295]], colormode=Color),Image(RGB[[296]], colormode=Color),Image(RGB[[297]], colormode=Color),Image(RGB[[298]], colormode=Color),Image(RGB[[299]], colormode=Color),Image(RGB[[300]], colormode=Color)
),nx=20, all=T, spacing = 0.0, margin = 0, drawGrid=F)







