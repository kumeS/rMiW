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
AAA <- "_BMem_Binary.png"


X <- dir(PATH01); length(X)
Y <- dir(PATH02); length(Y)
for(n in 1:length(X)){
  # n <- 1
  X1 <- sub(".tif", "", X[n], )
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
  
  system(paste("convert -quality 100 -depth 8 -type Grayscale ", X8, " ", sub(".png", "", X8),"-%03d.png", sep=""))
  file.remove(X8)
  system(paste("convert -quality 100 -depth 8 -type Grayscale ", X9, " ", sub(".png", "", X9),"-%03d.png", sep=""))
  file.remove(X9)
}





