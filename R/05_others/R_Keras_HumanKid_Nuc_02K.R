rm(list=ls())
#.rs.restartR()
#system("nvidia-smi")
#system("cat /etc/nginx/nginx.conf")
#system("cat /etc/nginx/conf.d/default.conf")

#system("CUDA_ROOT/nvcc -V")

#system("nvcc -V")

library(keras)
library(tidyverse)
library(reticulate)
library(devtools)
library(magrittr)
library(EBImage)
options(EBImage.display = "raster")
library(mmand)

K <- backend()

FLAGS <- flags(
 flag_numeric("kernel_size", 3),
 flag_numeric("nlevels", 3),
 flag_numeric("nfilters", 128),
 flag_numeric("BatchSize", 16),
 flag_numeric("dropout1", 0.1),
 flag_numeric("dropout2", 0.1),
 flag_numeric("dropout3", 0.1)
)

#####
#getwd()
#setwd("../")
setwd("~/Rstudio_Keras_test/")
dir("./DL_Segmentation_Dataset")
dir("./DL_Segmentation_Dataset/01_HumanANCA_KidNuc_All_ver190815")

PATH <- "./DL_Segmentation_Dataset/01_HumanANCA_KidNuc_All_ver190815"
TRAIN_PATH = paste(PATH, "/1_Train", sep="")
Teacher_PATH = paste(PATH, "/2_Teacher", sep="")
Test_PATH = paste(PATH, "/3_Test", sep="")
Test_GT_PATH = paste(PATH, "/4_Test_GT", sep="")

#HEIGHT = 128
#WIDTH  = 128

#HEIGHT = 256
#WIDTH  = 256

HEIGHT = 512
WIDTH = 512

#HEIGHT = 800
#WIDTH = 800

CHANNELS = 1
SHAPE = c(WIDTH, HEIGHT, CHANNELS)
BATCH_SIZE = FLAGS$BatchSize
EPOCHS = 150

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
 image <- image^1.0
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
reprocess_image04 <- function(file, shape){
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
str(X)
ImageFile = paste(Teacher_PATH, "/", dir(Teacher_PATH), sep="")
Y = map(ImageFile, preprocess_image02, shape = SHAPE)
str(Y)
#table(Y[[1]])

## リストの結合
XY <- list(X=X,Y=Y)
XYG <- XY

#display(XYG$X[[1]][c(c((dim(XYG$X[[1]])[1]/8*5+1):(dim(XYG$X[[1]])[1])), c(1:(dim(XYG$X[[1]])[1]/8*5))),,])
#display(XYG$Y[[1]][c(c((dim(XYG$Y[[1]])[1]/8*5+1):(dim(XYG$Y[[1]])[1])), c(1:(dim(XYG$Y[[1]])[1]/8*5))),,])

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
 XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*7))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
 XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*7))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
 
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
 XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*7))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
 XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*7))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
 
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
 XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*7))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
 XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*7))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
 
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
 XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*7))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
 XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*7))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
 
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
 XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*7))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
 XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*7))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
 
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
 XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*7))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
 XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*7))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
 
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
 XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*7))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
 XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*7))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
 
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
 XYG$X <- c(XYG$X, list(array(AA[,c(c((dim(AA)[1]/8*7+1):(dim(AA)[1])), c(1:(dim(AA)[1]/8*7))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
 XYG$Y <- c(XYG$Y, list(array(BB[,c(c((dim(BB)[1]/8*7+1):(dim(BB)[1])), c(1:(dim(BB)[1]/8*7))),], dim=c(SHAPE[1], SHAPE[2], SHAPE[3]))))
}

###############################
## データを元に戻し
###############################
#install.packages("random")
library(random)
str(XYG)
str(XYG$X)
length(XYG$X)
Ran <- c(randomSequence(min=1,max=length(XYG$X),col=1))

## データ順番入れ替え
XYG$X <- XYG$X[Ran]
XYG$Y <- XYG$Y[Ran]

XY <- XYG
rm(XYG)
str(XY)
###############################
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
#K <- backend()

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
  layer_spatial_dropout_2d(rate = dropout) %>%
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

#####################################################################
## Model 作成
#####################################################################
FLAGS$nfilters
try(model1 <- unet(shape = SHAPE, nlevels = FLAGS$nlevels, nfilters = FLAGS$nfilters, 
                  dropouts = c(FLAGS$dropout1, FLAGS$dropout1, 
                               FLAGS$dropout2, FLAGS$dropout2, FLAGS$dropout3)), silent=TRUE)

summary(model1)
model <- multi_gpu_model(model1, gpus = 2)
summary(model)

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
  0.001
 } else if(epoch > 100 && epoch <= 200){
  0.001
 } else {
  0.03
 }}

lr_reducer <- callback_learning_rate_scheduler(lr_schedule)

#summary(model)
#count_params(model)
#get_weights(model)
#source("./R_Script/PLOTmodel/DL_plot_modi_KK.R", local =T)
#model1 %>% deepviz::plot_model()
#model %>% deepviz::plot_model()
#model %>% plot_model_modi(width=4.5, height=1)
#####################################################################################
################################     Fit          ###################################
#####################################################################################
## tensorboard 設定
DIR01 <- "Results/Results_HumanKid/UNET_03_HumanANCA_Nuc"
if(!file.exists(DIR01)){dir.create(DIR01, showWarnings = F)}
DIR01 <- "Results/Results_HumanKid/UNET_03_HumanANCA_Nuc/LOG003"
if(!file.exists(DIR01)){dir.create(DIR01, showWarnings = F)}
TENS <- callback_tensorboard(DIR01)
tensorboard(DIR01, launch_browser=F)

#BATCH_SIZE <- 4
#EPOCHS <- 150

try(history <- model %>% fit(XL, YL,
                             batch_size = BATCH_SIZE,
                             epochs = EPOCHS,
                             validation_split = 1,
                             verbose = 1,
                             callbacks = list( lr_reducer, TENS))
    , silent=TRUE)


### ### ### ### ### ### ### ###
#getwd()
#system("df")
dir(DIR01)
summary(model)
#tensorboard(DIR01)
history

#max(history$metrics$iou)
## Model 評価
try(score01 <- model %>% evaluate(
 XL, YL, verbose = 1), silent=TRUE)

try(cat('Train loss:', score01[[1]], '\n'), silent=TRUE)
try(cat('Train accuracy:', score01[[2]], '\n'), silent=TRUE)

## 前のモデルを適用
#setwd("~/Rstudio_Keras_test/190513_dl")
#DIR01 <- "~/Rstudio_Keras_test/190513_dl/Results_NB4/UNET_Test_01/Nuc_LOG003/190613-0010_TrainAcc0.9998_TestAcc0.9622"
#model <- load_model_hdf5(paste(DIR01, "/Model.h5", sep=""), custom_objects = NULL, compile = F)
#model <- load_model_hdf5(paste(DIR01, DIR02, "/Model.h5", sep=""), custom_objects = NULL, compile = F)

try(Y_hat <- predict(model, x = XL, verbose=1), silent=TRUE)
# mmand::threshold
# str(Y_hat)
try(Y_hat5 <- map(array_branch(Y_hat, 1), 
                  .f = function(z) {mmand::threshold(z, 0.5, binarise = TRUE)}), silent=TRUE)
# str(Y_hat5)
try(Y_hat <- list2tensor(Y_hat5), silent=TRUE)
#str(Y_hat)
#str(YL)
#str(XY)
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

##########################################################
## Trainの結果グラフ保存(1)
##########################################################
try(dir.create(paste(DIR01, "/1_Train_Results", sep=""), showWarnings = F), silent=TRUE)

XYsize <- 256
for(ABC in 1:length(dir(TRAIN_PATH))){
 # ABC <- 1
 Image_color01 <- paintObjects(fillHull(XY$Y[[ABC]])/2, toRGB(XY$X[[ABC]])/2,opac=c(0.25, 0.25),
                               col=c("red","red"), thick=T, closed=F)
 Image_color02 <- paintObjects(fillHull(Y_hat[ABC,,,])/2, toRGB(XY$X[[ABC]])/2,opac=c(0.25, 0.25),
                               col=c("blue","blue"), thick=T, closed=F)
 try(dev.off(), silent=T)
 png(paste(DIR01, "/1_Train_Results/Train_Res_", 
           formatC(ABC, width = 4, flag = "0"), ".png", sep=""), 
     width = 1000, height = 1000)
 par(bg = 'grey')
 EBImage::display(EBImage::combine(resize(Image_color01, XYsize, XYsize), resize(toRGB(XY$Y[[ABC]]), XYsize, XYsize),
                                   resize(Image_color02, XYsize, XYsize), resize(toRGB(Y_hat[ABC,,,]), XYsize, XYsize)), 
                  nx=2, all=TRUE, spacing = 0.01, margin = 70)
 
 intersection <- sum( matrix(XY$Y[[ABC]]) * matrix(Y_hat[ABC,,,]))
 union <- sum( matrix(XY$Y[[ABC]]) + matrix(Y_hat[ABC,,,]) ) - intersection
 result <- (intersection + 1) / ( union + 1)
 
 #256x256
 text(x = -50/2, y = 250/2, label = "Ground truth", adj = c(0,1), col = "black", cex = 3.5, pos=1, srt=90)
 text(x = 250/2, y = -70/2, label = "Overlay", adj = c(0,1), col = "black", cex = 3.5, pos=1, srt=0)
 text(x = -50/2, y = 750/2, label = "Prediction", adj = c(0,1), col = "black", cex = 3.5, pos=1, srt=90)
 text(x = 770/2, y = -70/2, label = "Binary", adj = c(0,1), col = "black", cex = 3.5, pos=1, srt=0)
 text(x = 770/3, y = 1050/2, label = paste("IOU:", round(result, 3),
                                           "\nIOU ave:", round(mean(TAVE),3), "±", round(sd(TAVE),3)), 
      adj = c(0,1), col = "black", cex = 3.5, pos=1, srt=0)
 try(dev.off(), silent=T)
 
 ## 重ね合わせ
 png(paste(DIR01, "/1_Train_Results/Train_Res_", 
           formatC(ABC, width = 4, flag = "0"), "_ov.png", sep=""), 
     width = 1000, height = 1000)
 par(bg = 'grey')
 Image_color01 <- paintObjects(XY$Y[[ABC]], toRGB(XY$X[[ABC]])/0.9,opac=c(0, 0.5),
                               col=c("red","red"), thick=T, closed=F)
 Image_color02 <- paintObjects(Y_hat[ABC,,,], toRGB(Image_color01)/0.9,opac=c(0, 0.5),
                               col=c("blue","blue"), thick=T, closed=F)
 EBImage::display(Image_color02, nx=1, all=TRUE, spacing = 0.01, margin = 70)
 text(x = 770/3, y = 1050/2, label = paste("IOU:", round(result, 3),
                                           "\nIOU ave:", round(mean(TAVE),3), "±", round(sd(TAVE),3)), 
      adj = c(0,1), col = "black", cex = 3.5, pos=1, srt=0)
 try(dev.off(), silent=T)
 
 ## DL_Pred_Binary
 png(paste(DIR01, "/1_Train_Results/Train_Res_", 
           formatC(ABC, width = 4, flag = "0"), "_DL_Pred_Binary.png", sep=""), 
     width = 1000, height = 1000)
 #par(bg = 'grey')
 EBImage::display(Y_hat[ABC,,,], nx=1, all=TRUE, spacing = 0, margin = 0)
 try(dev.off(), silent=T)
 
 ## Original
 png(paste(DIR01, "/1_Train_Results/Train_Res_", 
           formatC(ABC, width = 4, flag = "0"), "_ori.png", sep=""), 
     width = 1000, height = 1000)
 #par(bg = 'grey')
 EBImage::display(XY$X[[ABC]], nx=1, all=TRUE, spacing = 0, margin = 0)
 try(dev.off(), silent=T)
 
 ## Original * Pred
 png(paste(DIR01, "/1_Train_Results/Train_Res_", 
           formatC(ABC, width = 4, flag = "0"), "_Extract.png", sep=""), 
     width = 1000, height = 1000)
 #par(bg = 'grey')
 EBImage::display(XY$X[[ABC]][,,1]*Y_hat[ABC,,,], nx=1, all=TRUE, spacing = 0, margin = 0)
 try(dev.off(), silent=T)
}

############################################################################
## Perform predictions on the test images
############################################################################
## ガンマ値を変えた時にどう最適化されるか
#length(dir(Test_PATH))
MMM <- seq(0.7, 1.3, by=0.01)
Dat1 <- rep(NA, length(MMM))
Dat2 <- rep(NA, length(MMM))

for(n in 1:length(MMM)){
 preprocess_image01T <- function(file, shape){
  # shape = SHAPE
  #n <- 1
  # file = paste(TRAIN_PATH, "/", dir(TRAIN_PATH)[1], sep="")
  image <- readImage(file, type="png")
  #image <- to_gray_scale(image)                       ## convert to gray scale  
  image <- resize(image, w = shape[1], h = shape[2], filter = "none")  ## none or bilinear
  image <- normalize(image)
  image <- clahe(image)
  image <- image^MMM[n]
  #range(image)
  #image <- invert(image)                              ## invert brightfield
  #EBImage::display(image)
  array(image, dim=c(shape[1], shape[2], shape[3]))                                    ## return as array
 }
 
 ImageFile = paste(Test_PATH, "/", dir(Test_PATH), sep="")
 TX = map(ImageFile, preprocess_image01T, shape = SHAPE)
 #str(TX)
 
 ImageFile = paste(Test_GT_PATH, "/", dir(Test_GT_PATH), sep="")
 TY = map(ImageFile, preprocess_image02, shape = SHAPE)
 #str(TY)
 
 ## リストの結合
 XYT <- list(X=X,Y=Y,TX=TX, TY=TY)
 TXL <- list2tensor(XYT$TX)
 TYL <- list2tensor(XYT$TY)
 
 try(TX_hat <- predict(model, x = TXL, verbose=1), silent=TRUE)
 
 # mmand::threshold
 try(TX_lab5 <- map(array_branch(TX_hat, 1), 
                    .f = function(z) {mmand::threshold(z, 0.5, binarise = TRUE)}) , silent=TRUE)
 try(TX_lab <- list2tensor(TX_lab5), silent=TRUE)
 
 ##########################################################
 ## Model 評価
 try(score02 <- model %>% evaluate(
  TXL, TYL, verbose = 1), silent=TRUE)
 
 print(MMM[n])
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
 Dat1[n] <- mean(AVE)
 Dat2[n] <- sd(AVE)
}

Dat3 <- data.frame(SEQ=MMM, Dat1, Dat2)
Dat3[order(-Dat3[,2]),][1,1]
plot(Dat3[,1:2], type="b")

##################################################################################################################
## Perform predictions on the test images
##################################################################################################################
## 再実行
preprocess_image02T <- function(file, shape){
 # shape = SHAPE
 #n <- 1
 # file = paste(TRAIN_PATH, "/", dir(TRAIN_PATH)[1], sep="")
 image <- readImage(file, type="png")
 #image <- to_gray_scale(image)                       ## convert to gray scale  
 image <- resize(image, w = shape[1], h = shape[2], filter = "none")  ## none or bilinear
 image <- normalize(image)
 image <- clahe(image)
 image <- image^Dat3[order(-Dat3[,2]),][1,1]
 #image <- image^0.87
 #range(image)
 #image <- invert(image)                              ## invert brightfield
 #EBImage::display(image)
 array(image, dim=c(shape[1], shape[2], shape[3]))                                    ## return as array
}

ImageFile = paste(Test_PATH, "/", dir(Test_PATH), sep="")
TX = map(ImageFile, preprocess_image02T, shape = SHAPE)
str(TX)

ImageFile = paste(Test_GT_PATH, "/", dir(Test_GT_PATH), sep="")
TY = map(ImageFile, preprocess_image02, shape = SHAPE)
str(TY)

## リストの結合
XYT <- list(X=X,Y=Y,TX=TX, TY=TY)
TXL <- list2tensor(XYT$TX)
TYL <- list2tensor(XYT$TY)

try(TX_hat <- predict(model, x = TXL, verbose=1), silent=TRUE)

# mmand::threshold
try(TX_lab5 <- map(array_branch(TX_hat, 1), 
                   .f = function(z) {mmand::threshold(z, 0.5, binarise = TRUE)}) , silent=TRUE)
try(TX_lab <- list2tensor(TX_lab5), silent=TRUE)

##########################################################
## Model 評価
try(score02 <- model %>% evaluate(
 TXL, TYL, verbose = 1), silent=TRUE)

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
AVE
mean(AVE)
sd(AVE)

## モデル保存
#dir("Results_NB4")

if(!file.exists(DIR01)){dir.create(DIR01, showWarnings = F)}
try(DIR02 <- paste("/", format(Sys.time(), "%y%m%d-%H%M"), 
                   "_TrainAcc", round(mean(TAVE), 4), 
                   "_TestAcc", round(mean(AVE), 4), sep=""), silent=TRUE)
#, "_Model_Para", count_params(model)
#DIR02 <- "/LOG"
try(dir.create(paste(DIR01, DIR02, sep=""), showWarnings = F), silent=TRUE)
try(model %>% 
     save_model_hdf5(paste(DIR01, DIR02, "/Model.h5", sep=""), overwrite = TRUE, include_optimizer = TRUE), silent=TRUE)
try(model %>% 
     save_model_weights_hdf5(paste(DIR01, DIR02, "/Model_weights.h5", sep="")), silent=TRUE)

##########################################################
## Testの結果グラフ保存(1)
##########################################################
try(dir.create(paste(DIR01, "/2_Test_Results01", sep=""), showWarnings = F), silent=TRUE)

#getwd()

XYsize <- 256
try(for(ABC in 1:length(dir(Test_PATH))){
 # ABC <- 1
 Image_color01 <- paintObjects(XYT$TY[[ABC]], toRGB(XYT$TX[[ABC]])/0.95,opac=c(0.25, 0.25),
                               col=c("red","red"), thick=T, closed=F)
 TX_lab1 <- TX_lab; TX_lab1[ABC,,,] <- fillHull(TX_lab1[ABC,,,])
 Image_color02 <- paintObjects(TX_lab1[ABC,,,], toRGB(XYT$TX[[ABC]])/0.95,opac=c(0.25, 0.25),
                               col=c("blue","blue"), thick=T, closed=F)
 try(dev.off(), silent=T)
 png(paste(DIR01, "/2_Test_Results01/Test_Res_", formatC(ABC, width = 4, flag = "0"), ".png", sep=""), 
     width = 1000, height = 1000)
 par(bg = 'grey')
 EBImage::display(EBImage::combine(resize(Image_color01, XYsize, XYsize), resize(toRGB(XYT$TY[[ABC]]), XYsize, XYsize),
                                   resize(Image_color02, XYsize, XYsize), resize(toRGB(TX_lab1[ABC,,,]), XYsize, XYsize)), 
                  nx=2, all=TRUE, spacing = 0.01, margin = 70)
 #A <- table(c(XYT$TY[[ABC]]), c(TX_lab[ABC,,,]))
 intersection <- sum( matrix(XYT$TY[[ABC]]) * matrix(TX_lab[ABC,,,]))
 union <- sum( matrix(XYT$TY[[ABC]]) + matrix(TX_lab[ABC,,,]) ) - intersection
 result <- (intersection + 1) / ( union + 1)
 
 #256x256
 text(x = -50/2, y = 250/2, label = "Ground truth", adj = c(0,1), col = "black", cex = 3.5, pos=1, srt=90)
 text(x = 250/2, y = -70/2, label = "Overlay", adj = c(0,1), col = "black", cex = 3.5, pos=1, srt=0)
 text(x = -50/2, y = 750/2, label = "Prediction", adj = c(0,1), col = "black", cex = 3.5, pos=1, srt=90)
 text(x = 770/2, y = -70/2, label = "Binary", adj = c(0,1), col = "black", cex = 3.5, pos=1, srt=0)
 text(x = 770/3, y = 1050/2, label = paste("IOU:", round(result, 3),
                                           "\nIOU ave:", round(mean(AVE),3), "±", round(sd(AVE),3)), 
      adj = c(0,1), col = "black", cex = 3.5, pos=1, srt=0)
 try(dev.off(), silent=T)
 
 ## 重ね合わせ
 png(paste(DIR01, "/2_Test_Results01/Test_Res_", formatC(ABC, width = 4, flag = "0"), "_ov.png", sep=""), 
     width = 1000, height = 1000)
 par(bg = 'grey')
 Image_color01 <- paintObjects(XYT$TY[[ABC]], toRGB(XYT$TX[[ABC]])/0.9,opac=c(0, 0.5),
                               col=c("red","red"), thick=T, closed=F)
 Image_color02 <- paintObjects(TX_lab1[ABC,,,], toRGB(Image_color01)/0.9,opac=c(0, 0.5),
                               col=c("blue","blue"), thick=T, closed=F)
 EBImage::display(Image_color02, nx=1, all=TRUE, spacing = 0.01, margin = 70)
 
 text(x = 770/3, y = 1050/2, label = paste("IOU:", round(result, 3),
                                           "\nIOU ave:", round(mean(AVE),3), "±", round(sd(AVE),3)), 
      adj = c(0,1), col = "black", cex = 3.5, pos=1, srt=0)
 try(dev.off(), silent=T)
 
 ## DL_Pred_Binary
 png(paste(DIR01, "/2_Test_Results01/Test_Res_", 
           formatC(ABC, width = 4, flag = "0"), "_DL_Pred_Binary.png", sep=""), 
     width = 1000, height = 1000)
 #par(bg = 'grey')
 EBImage::display(TX_lab1[ABC,,,], nx=1, all=TRUE, spacing = 0, margin = 0)
 try(dev.off(), silent=T)
 
 ## Original
 png(paste(DIR01, "/2_Test_Results01/Test_Res_", 
           formatC(ABC, width = 4, flag = "0"), "_ori.png", sep=""), 
     width = 1000, height = 1000)
 #par(bg = 'grey')
 EBImage::display(XYT$TX[[ABC]], nx=1, all=TRUE, spacing = 0, margin = 0)
 try(dev.off(), silent=T)
 
 ## Original * Pred
 png(paste(DIR01, "/2_Test_Results01/Test_Res_", 
           formatC(ABC, width = 4, flag = "0"), "_Extract.png", sep=""), 
     width = 1000, height = 1000)
 #par(bg = 'grey')
 EBImage::display(XYT$TX[[ABC]][,,1]*TX_lab1[ABC,,,], nx=1, all=TRUE, spacing = 0, margin = 0)
 try(dev.off(), silent=T)

 }, silent=TRUE)

##################################################################################################################
## Perform predictions on others
##################################################################################################################
Dat3[order(-Dat3[,2]),][1,1]
preprocess_image05 <- function(file, shape){
 # shape = SHAPE
 # file = paste(TRAIN_PATH, "/", dir(TRAIN_PATH)[1], sep="")
 image <- readImage(file, type="png")
 #image <- to_gray_scale(image)                       ## convert to gray scale  
 image <- resize(image, w = shape[1], h = shape[2], filter = "none")  ## none or bilinear
 image <- normalize(image)
 image <- clahe(image)
 #image <- image^0.87
 image <- image^Dat3[order(-Dat3[,2]),][1,1]
 #range(image)
 #image <- invert(image)                              ## invert brightfield
 #EBImage::display(image)
 array(image, dim=c(shape[1], shape[2], shape[3]))                                    ## return as array
}

##getwd()
#dir()
#dir('./DL_Segmentation_Dataset/99_WideEM/ANCA/')
if(TRUE){
WideEM_PATH01 = './DL_Segmentation_Dataset/99_WideEM/ANCA/Kidney_10nm_糸球体のみ/crop_x512'
WideEM_PATH02 = './DL_Segmentation_Dataset/99_WideEM/ANCA/Kidney_10nm_糸球体のみ/crop_x1024'
WideEM_PATH03 = './DL_Segmentation_Dataset/99_WideEM/ANCA/Kidney_10nm_糸球体のみ/crop_x2048'

WideEM_PATH04 = './DL_Segmentation_Dataset/99_WideEM/ANCA/Kidney_10nm_尿細管のみ/crop_x512'
WideEM_PATH05 = './DL_Segmentation_Dataset/99_WideEM/ANCA/Kidney_10nm_尿細管のみ/crop_x1024'
WideEM_PATH06 = './DL_Segmentation_Dataset/99_WideEM/ANCA/Kidney_10nm_尿細管のみ/crop_x2048'

WideEM_PATH07 = './DL_Segmentation_Dataset/99_WideEM/ANCA/Kidney_10nm_cut/crop_x512'
WideEM_PATH08 = './DL_Segmentation_Dataset/99_WideEM/ANCA/Kidney_10nm_cut/crop_x1024'
WideEM_PATH09 = './DL_Segmentation_Dataset/99_WideEM/ANCA/Kidney_10nm_cut/crop_x2048'
#"crop_x1024" "crop_x2048" "crop_x512"

ImageFile = paste(WideEM_PATH01, "/", dir(WideEM_PATH01), sep="")
WX1 = map(ImageFile, preprocess_image05, shape = SHAPE)

ImageFile = paste(WideEM_PATH02, "/", dir(WideEM_PATH02), sep="")
WX2 = map(ImageFile, preprocess_image05, shape = SHAPE)

ImageFile = paste(WideEM_PATH03, "/", dir(WideEM_PATH03), sep="")
WX3 = map(ImageFile, preprocess_image05, shape = SHAPE)

ImageFile = paste(WideEM_PATH04, "/", dir(WideEM_PATH04), sep="")
WX4 = map(ImageFile, preprocess_image05, shape = SHAPE)

ImageFile = paste(WideEM_PATH05, "/", dir(WideEM_PATH05), sep="")
WX5 = map(ImageFile, preprocess_image05, shape = SHAPE)

ImageFile = paste(WideEM_PATH06, "/", dir(WideEM_PATH06), sep="")
WX6 = map(ImageFile, preprocess_image05, shape = SHAPE)

ImageFile = paste(WideEM_PATH07, "/", dir(WideEM_PATH07), sep="")
WX7 = map(ImageFile, preprocess_image05, shape = SHAPE)

ImageFile = paste(WideEM_PATH08, "/", dir(WideEM_PATH08), sep="")
WX8 = map(ImageFile, preprocess_image05, shape = SHAPE)

ImageFile = paste(WideEM_PATH09, "/", dir(WideEM_PATH09), sep="")
WX9 = map(ImageFile, preprocess_image05, shape = SHAPE)

WX <- list(WX1=WX1, WX2=WX2,
           WX3=WX3, WX4=WX4,
           WX5=WX5, WX6=WX6,
           WX7=WX7, WX8=WX8,
           WX9=WX9)
str(WX)

WX1lt <- list2tensor(WX$WX1)
WX2lt <- list2tensor(WX$WX2)
WX3lt <- list2tensor(WX$WX3)
WX4lt <- list2tensor(WX$WX4)
WX5lt <- list2tensor(WX$WX5)
WX6lt <- list2tensor(WX$WX6)
WX7lt <- list2tensor(WX$WX7)
WX8lt <- list2tensor(WX$WX8)
WX9lt <- list2tensor(WX$WX9)

#rm(WX)
rm(WX1); rm(WX2); rm(WX3); rm(WX4); rm(WX5); rm(WX6); rm(WX7); rm(WX8); rm(WX9)

WX1lt.p <- predict(model, x = WX1lt, verbose=1)
WX2lt.p <- predict(model, x = WX2lt, verbose=1)
WX3lt.p <- predict(model, x = WX3lt, verbose=1)
WX4lt.p <- predict(model, x = WX4lt, verbose=1)
WX5lt.p <- predict(model, x = WX5lt, verbose=1)
WX6lt.p <- predict(model, x = WX6lt, verbose=1)
WX7lt.p <- predict(model, x = WX7lt, verbose=1)
WX8lt.p <- predict(model, x = WX8lt, verbose=1)
WX9lt.p <- predict(model, x = WX9lt, verbose=1)

rm(WX1lt); rm(WX2lt); rm(WX3lt); rm(WX4lt); rm(WX5lt); rm(WX6lt); rm(WX7lt); rm(WX8lt); rm(WX9lt)
}
########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################

## データの入れ替え
#WX.p <- WX1lt.p; WX.cont <- WX$WX1; DIR03 <- "Kidney_10nm_糸球体のみ_crop_x512"
#WX.p <- WX2lt.p; WX.cont <- WX$WX2; DIR03 <- "Kidney_10nm_糸球体のみ_crop_x1024"
#WX.p <- WX3lt.p; WX.cont <- WX$WX3; DIR03 <- "Kidney_10nm_糸球体のみ_crop_x2048"
#WX.p <- WX4lt.p; WX.cont <- WX$WX4; DIR03 <- "Kidney_10nm_尿細管のみ_crop_x512"
#WX.p <- WX5lt.p; WX.cont <- WX$WX5; DIR03 <- "Kidney_10nm_尿細管のみ_crop_x1024"
#WX.p <- WX6lt.p; WX.cont <- WX$WX6; DIR03 <- "Kidney_10nm_尿細管のみ_crop_x2048"
#WX.p <- WX7lt.p; WX.cont <- WX$WX7; DIR03 <- "Kidney_10nm_cut_crop_x512"
#WX.p <- WX8lt.p; WX.cont <- WX$WX8; DIR03 <- "Kidney_10nm_cut_crop_x1024"
#WX.p <- WX9lt.p; WX.cont <- WX$WX9; DIR03 <- "Kidney_10nm_cut_crop_x2048"

## 全部実行
if(TRUE){
  try(WX.pa <- map(array_branch(WX.p, 1), 
                   .f = function(z) {mmand::threshold(z, 0.5, binarise = TRUE)}) , silent=TRUE)
  #str(WX.pa)
  try(WX.pal <- list2tensor(WX.pa), silent=TRUE)
  #str(WX.pal)
  #str(WX.cont)
  
  try(dir.create(paste(DIR01, "/5_", DIR03, sep=""), showWarnings = F), silent=TRUE)
  try(dir.create(paste(DIR01, "/5_", DIR03, "_Bi", sep=""), showWarnings = F), silent=TRUE)
  try(dir.create(paste(DIR01, "/5_", DIR03, "_fillHull", sep=""), showWarnings = F), silent=TRUE)

  try(for(ABC in 1:dim(WX.pal)[1]){
    # ABC <- 25
    #WX
    Image_color01 <- paintObjects(WX.pal[ABC,,,], toRGB(WX.cont[[ABC]])/0.95,opac=c(0.5, 0.25),
                                  col=c("red","red"), thick=T, closed=F)
    try(dev.off(), silent=T)
    png(paste(DIR01, "/5_", DIR03, "/WD_Res_", formatC(ABC, width = 5, flag = "0"), ".png", sep=""), 
        width = 500, height = 500)
    #par(bg = 'grey')
    EBImage::display(Image_color01,
                     nx=1, all=TRUE, spacing = 0, margin = 0)
    try(dev.off(), silent=T)
    
    png(paste(DIR01, "/5_", DIR03, "_Bi", "/WD_Bi", formatC(ABC, width = 5, flag = "0"), ".png", sep=""), 
        width = 500, height = 500)
    #par(bg = 'grey')
    EBImage::display(WX.pal[ABC,,,],
                     nx=1, all=TRUE, spacing = 0, margin = 0)
    try(dev.off(), silent=T)
    
    Image_color01 <- paintObjects(fillHull(WX.pal[ABC,,,]), 
                                  toRGB(WX.cont[[ABC]])/0.95,opac=c(0.5, 0.25),
                                  col=c("red","red"), thick=T, closed=F)
    try(dev.off(), silent=T)
    png(paste(DIR01, "/5_", DIR03, "_fillHull/WD_fillHull_", formatC(ABC, width = 5, flag = "0"), ".png", sep=""), 
        width = 500, height = 500)
    #par(bg = 'grey')
    EBImage::display(Image_color01,
                     nx=1, all=TRUE, spacing = 0, margin = 0)
    try(dev.off(), silent=T)
    }, silent=TRUE)}


#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
## 画像結合
#getwd()
#length(dir())
#744/2/2/2/3

if(TRUE){
setwd("~/Rstudio_Keras_test/Results/Results_HumanKid/UNET_03_HumanANCA_Nuc/LOG003/5_Kidney_10nm_cut_crop_x1024")
system(paste("montage ", paste("WD_Res_", formatC(1:744, width = 5, flag = "0"), ".png ", collapse="", sep=""), 
      "-tile 31x24 -geometry +0+0 ", "out.png", sep=""))
system("convert -resize 2500x out.png out_resize.png")

setwd("~/Rstudio_Keras_test/Results/Results_HumanKid/UNET_03_HumanANCA_Nuc/LOG003/5_Kidney_10nm_cut_crop_x1024_Bi")
system(paste("montage ", paste("WD_Bi", formatC(1:744, width = 5, flag = "0"), ".png ", collapse="", sep=""), 
       "-tile 31x24 -geometry +0+0 ", "out.png", sep=""))
system("convert -resize 3000x out.png out_resize.png")

setwd("~/Rstudio_Keras_test/Results/Results_HumanKid/UNET_03_HumanANCA_Nuc/LOG003/5_Kidney_10nm_cut_crop_x1024_fillHull")
system(paste("montage ", paste("WD_fillHull_", formatC(1:744, width = 5, flag = "0"), ".png ", collapse="", sep=""), 
             "-tile 31x24 -geometry +0+0 ", "out.png", sep=""))
system("convert -resize 3000x out.png out_resize.png")

#99/3/3
setwd("~/Rstudio_Keras_test/Results/Results_HumanKid/UNET_03_HumanANCA_Nuc/LOG003/5_Kidney_10nm_糸球体のみ_crop_x1024")
system(paste("montage ", paste("WD_Res_", formatC(1:99, width = 5, flag = "0"), ".png ", collapse="", sep=""), 
             "-tile 11x9 -geometry +0+0 ", "out.png", sep=""))
system("convert -resize 2500x out.png out_resize.png")

setwd("~/Rstudio_Keras_test/Results/Results_HumanKid/UNET_03_HumanANCA_Nuc/LOG003/5_Kidney_10nm_糸球体のみ_crop_x1024_fillHull")
system(paste("montage ", paste("WD_fillHull_", formatC(1:99, width = 5, flag = "0"), ".png ", collapse="", sep=""), 
             "-tile 11x9 -geometry +0+0 ", "out.png", sep=""))
system("convert -resize 2500x out.png out_resize.png")

#20/2/2
#setwd("~/Rstudio_Keras_test/Results/Results_HumanKid/UNET_03_HumanANCA_Nuc/LOG003/5_Kidney_10nm_糸球体のみ_crop_x2048")
#system(paste("montage ", paste("WD_Res_", formatC(1:20, width = 5, flag = "0"), ".png ", collapse="", sep=""), 
#             "-tile 5x4 -geometry +0+0 ", "out.png", sep=""))
#system("convert -resize 2500x out.png out_resize.png")

#418/2/11
#setwd("~/Rstudio_Keras_test/Results/Results_HumanKid/UNET_03_HumanANCA_Nuc/LOG003/5_Kidney_10nm_糸球体のみ_crop_x512")
#system(paste("montage ", paste("WD_Res_", formatC(1:418, width = 5, flag = "0"), ".png ", collapse="", sep=""), 
#             "-tile 22x19 -geometry +0+0 ", "out.png", sep=""))
#system("convert -resize 2500x out.png out_resize.png")

#99/3/3
setwd("~/Rstudio_Keras_test/Results/Results_HumanKid/UNET_03_HumanANCA_Nuc/LOG003/5_Kidney_10nm_尿細管のみ_crop_x1024")
system(paste("montage ", paste("WD_Res_", formatC(1:99, width = 5, flag = "0"), ".png ", collapse="", sep=""), 
             "-tile 11x9 -geometry +0+0 ", "out.png", sep=""))
system("convert -resize 2500x out.png out_resize.png")

setwd("~/Rstudio_Keras_test/Results/Results_HumanKid/UNET_03_HumanANCA_Nuc/LOG003/5_Kidney_10nm_尿細管のみ_crop_x1024_fillHull")
system(paste("montage ", paste("WD_fillHull_", formatC(1:99, width = 5, flag = "0"), ".png ", collapse="", sep=""), 
             "-tile 11x9 -geometry +0+0 ", "out.png", sep=""))
system("convert -resize 2500x out.png out_resize.png")


#setwd("~/Rstudio_Keras_test/Results/Results_HumanKid/UNET_03_HumanANCA_Nuc/LOG003/5_Kidney_10nm_尿細管のみ_crop_x1024_Bi")
#system(paste("montage ", paste("WD_Bi", formatC(1:99, width = 5, flag = "0"), ".png ", collapse="", sep=""), 
#             "-tile 11x9 -geometry +0+0 ", "out.png", sep=""))
#system("convert -resize 2500x out.png out_resize.png")
}

#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################
########################################################################################
########################################################################################
# 細胞 Negativeは一度白くする
# 細胞と核の0と1のかけ算をする
# 0 1 データも保存する
########################################################################################
########################################################################################
HEIGHT = 256
WIDTH  = 256
#HEIGHT = 512
#WIDTH = 512
CHANNELS = 1
SHAPE = c(WIDTH, HEIGHT, CHANNELS)

preprocess_image02 <- function(file, shape){
 # shape = SHAPE
 #file = paste(TRAIN_PATH, "/", dir(Teacher_PATH)[1], sep="")
 image <- readImage(file, type="png")
 #image <- to_gray_scale(image)                ## convert to gray scale  
 image <- resize(image, w = shape[1], h = shape[2], filter = "none")
 #image <- clahe(image)                        ## local adaptive contrast enhancement
 #image <- normalize(image)                    ## standardize between [0, 1]
 #image <- invert(image)                       ## invert brightfield
 array(image, dim=c(shape[1], shape[2], 1))    ## return as array
}

## オリジナル
#getwd()
setwd("/home/sas/Rstudio_Keras_test/190513_dl/")
Test_WIDE2048_PATH01 = './IMAGE/D05_NB4_WideEM/crop3_x2048'
ImageFile = paste(Test_WIDE2048_PATH01, "/", dir(Test_WIDE2048_PATH01), sep="")
ORI = map(ImageFile, preprocess_image02, shape = SHAPE)

## 細胞
DIR_CB <- "~/Rstudio_Keras_test/190513_dl/Results_NB4/UNET_Test_01/LOG003/190612-1813_TrainAcc1_TestAcc0.9801/5_WD22_Bi"
ImageFile = paste(DIR_CB, "/", dir(DIR_CB), sep="")
CB = map(ImageFile, preprocess_image02, shape = SHAPE)

## 核
DIR_NB <- "~/Rstudio_Keras_test/190513_dl/Results_NB4/UNET_Test_01/Nuc_LOG003/190613-0010_TrainAcc0.9998_TestAcc0.9622/5_WD22_Bi"
ImageFile = paste(DIR_NB, "/", dir(DIR_NB), sep="")
NB = map(ImageFile, preprocess_image02, shape = SHAPE)

############## ############## ############## ##############
setwd("~/Rstudio_Keras_test/190513_dl/Results_NB4/UNET_Test_01/Cell_Nuc_Res")
str(ORI)
str(CB)
str(NB)

EBImage::display(ORI[[1]],
                 nx=1, all=TRUE, spacing = 0, margin = 0)
EBImage::display(CB[[1]],
                 nx=1, all=TRUE, spacing = 0, margin = 0)
EBImage::display(NB[[1]],
                 nx=1, all=TRUE, spacing = 0, margin = 0)

##核
EBImage::display(CB[[1]]*NB[[1]],
                 nx=1, all=TRUE, spacing = 0, margin = 0)
## 細胞
EBImage::display(CB[[1]] - NB[[1]],
                 nx=1, all=TRUE, spacing = 0, margin = 0)

### ### ### ### ### ### ###
### 重ね合わせ
### ### ### ### ### ### ###
Image_color01 <- paintObjects((CB[[1]] - NB[[1]])/2, toRGB(ORI[[1]]/1.5),opac=c(0.3, 0.3),
                              col=c("blue","blue"), thick=T, closed=F)
EBImage::display(Image_color01,
                 nx=1, all=TRUE, spacing = 0, margin = 0)

Image_color02 <- paintObjects((CB[[1]]*NB[[1]])/2, toRGB(Image_color01/1),opac=c(0.3, 0.3),
                              col=c("red","red"), thick=T, closed=F)
EBImage::display(Image_color02,
                 nx=1, all=TRUE, spacing = 0, margin = 0)

setwd("~/Rstudio_Keras_test/190513_dl/Results_NB4/UNET_Test_01/Cell_Nuc_Res")
CN_Nam <- "Cell_Nuc_"

Test_WIDE2048 = '/home/sas/Rstudio_Keras_test/190513_dl/IMAGE/D05_NB4_WideEM/crop3_x2048'

try(for(ABC in 1:length(dir(Test_WIDE2048))){
 # ABC <- 1
 Image_color01 <- paintObjects((CB[[ABC]] - NB[[ABC]])/2, toRGB(ORI[[ABC]]/1.5),opac=c(0.3, 0.3),
                               col=c("blue","blue"), thick=T, closed=F)
 Image_color02 <- paintObjects((CB[[ABC]]*NB[[ABC]])/2, toRGB(Image_color01/1),opac=c(0.3, 0.3),
                               col=c("red","red"), thick=T, closed=F)
 
 try(dev.off(), silent=T)
 png(paste(CN_Nam, formatC(ABC, width = 5, flag = "0"), ".png", sep=""), width = 400, height = 400)
 #par(bg = 'grey')
 EBImage::display(Image_color02,
                  nx=1, all=TRUE, spacing = 0, margin = 0)
 try(dev.off(), silent=T)
}, silent=TRUE)

################################ ################################ ################################