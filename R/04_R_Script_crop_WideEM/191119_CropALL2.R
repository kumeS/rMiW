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

PATH01 = './1_Train'
PATH02 = './2_Teacher'

#AAA <- "_Mito_Binary.png"
#AAA <- "_Nuc_Binary.png"
#AAA <- "_BMem_Binary.png"
#AAA <- "_Mem_Binary.png"
#AAA <- "_Glom_Binary.png"
#AAA <- "_UTubule_Binary.tif"
#AAA <- "_Fibro_Binary.png"
#AAA <- "_Podo_Binary.png"

#AAA <- "＿Gap_Binary.png"
#AAA <- "_Mito_Binary.png"

#AAA <- "_Nuc_Binary.png"

AAA <- "_Nuc_Binary.png"



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
  
  system(paste("convert -quality 100 -depth 8 -type Grayscale ", X4, " " , X5, sep=""))
  system(paste("convert -quality 100 -depth 8 -type Grayscale  ", X6, " " , X7, sep=""))

  file.remove(X4)
  file.remove(X6)
}





