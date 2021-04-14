#install.packages("tidyverse")
library(tidyverse)
#install.packages("BiocManager")
#BiocManager::install("EBImage")
library(EBImage)
options(EBImage.display = "raster")
library(png)

#setwd("./Mask2")
dir()
#setwd("../")

PATH01 = './1_Raw/1_Image'
PATH02 = './1_Raw/2_GroundTruth'
PATH03 = './2_Proc/1_Train'
PATH04 = './2_Proc/2_Teacher'

#AAA <- "_Mito_Binary.png"
#AAA <- "_Nuc_Binary.png"
AAA <- "_mito1.png"

#BBB <- 512
BBB <- 256

X <- dir(PATH01); length(X)
Y <- dir(PATH02); length(Y)
for(n in 1:length(X)){
  # n <- 1
  X1 <- sub(".tif", "", X[n]); X1 <- sub(".png", "", X1)
  X2 <- paste(X1, ".png", sep="")
  X3 <- paste(X1, AAA, sep="")
  
  X4 <- paste(PATH01, "/", X[n], sep="")
  X5 <- paste(PATH01, "/", X2, sep="")
  X6 <- paste(PATH02, "/", X3, sep="")
  X7 <- paste(PATH02, "/", X2, sep="")
  
  X8 <- paste(PATH03, "/", X2, sep="")
  X9 <- paste(PATH04, "/", X2, sep="")
  
  system(paste("convert -quality 100 -depth 8 -type Grayscale ", X4, " " , X5, sep=""))
  system(paste("mv ", X5, " " , X8, sep=""))
  
  system(paste("convert -quality 100 -depth 8 -type Grayscale  ", X6, " " , X7, sep=""))
  system(paste("mv ", X7, " " , X9, sep=""))
  
  system(paste("convert -quality 100 -depth 8 -type Grayscale -crop ", BBB, "x", BBB, " ",  X8, " ", sub(".png", "", X8),"-%03d.png", sep=""))
  file.remove(X8)
  system(paste("convert -quality 100 -depth 8 -type Grayscale -crop ", BBB, "x", BBB, " ", X9, " ", sub(".png", "", X9),"-%03d.png", sep=""))
  file.remove(X9)
}

A01 <- dir(PATH03); length(A01)
B01 <- dir(PATH04); length(B01)

for(n in 1:length(A01)){
  #  n <- 8
  image1 <- readImage(paste(PATH03, "/", A01[n], sep=""))
  image2 <- readImage(paste(PATH04, "/", B01[n], sep=""))
  if(A01[n] == B01[n]){
    print("OK")
  }
  if(!(dim(image1)[1] == BBB && dim(image1)[2] == BBB)){
    file.remove(paste(PATH03, "/", A01[n], sep=""))
    file.remove(paste(PATH04, "/", B01[n], sep=""))
  }else{}
  #table(image2)
  #if(sum(image2) == 0){file.remove(paste(PATH01, "/",  X[n], sep=""))
  # file.remove(paste(PATH02, "/",  Y[n], sep=""))}
}




