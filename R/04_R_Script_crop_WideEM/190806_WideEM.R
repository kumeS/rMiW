## Rscript [R file] [Directory] 
library(EBImage)
options(EBImage.display = "raster")
library(png)

argv = commandArgs(T)
## 絶対パス
Argv01 = argv[1]
#Argv01 = "/Users/skume/Desktop/R_keras_DL/02_Dataset/Mouse_Kidney/NEP4/Tiled_img"

#getwd()
setwd(Argv01)

len <- list.files(pattern=".tif")
len1 <- sub(".tif", "", len)

dir.create("99_WideEM", showWarnings = F)

for(n in 1:length(len1)){
  dir.create(paste("./99_WideEM/", len1[n], sep=""))
}

#getwd()

for( n in 1:length(len1) ){
  # n <- 1
  setwd(paste("./99_WideEM/", len1[n], sep=""))

  dir.create("crop_x512")
  dir.create("crop_x1024")
  dir.create("crop_x2048")
  
  system(paste("convert -quality 100 -depth 8 -type Grayscale -crop 512x512 ../../",
               len[n], " ",
               "./crop_x512/WideEM_x512-%04d.png", sep=""))
  
  A01 <- dir("./crop_x512"); length(A01)
  for(m in 1:length(A01)){
    #  m <- 1
    image1 <- readImage(paste("./crop_x512/", A01[m], sep=""))
    if(!(dim(image1)[1] == 512 && dim(image1)[2] == 512)){
      file.remove(paste("./crop_x512/", A01[m], sep=""))
    }else{}
  }
  
  system(paste("convert -quality 100 -depth 8 -type Grayscale -crop 1024x1024 ../../",
               len[n], " ",
               "./crop_x1024/WideEM_x1024-%04d.png", sep=""))
  
  A01 <- dir("./crop_x1024"); length(A01)
  for(m in 1:length(A01)){
    #  m <- 1
    image1 <- readImage(paste("./crop_x1024/", A01[m], sep=""))
    if(!(dim(image1)[1] == 1024 && dim(image1)[2] == 1024)){
      file.remove(paste("./crop_x1024/", A01[m], sep=""))
    }else{}
  }
  
  system(paste("convert -quality 100 -depth 8 -type Grayscale -crop 2048x2048 ../../",
               len[n], " ",
               "./crop_x2048/WideEM_x2048-%04d.png", sep=""))
  
  A01 <- dir("./crop_x2048"); length(A01)
  for(m in 1:length(A01)){
    #  m <- 1
    image1 <- readImage(paste("./crop_x2048/", A01[m], sep=""))
    if(!(dim(image1)[1] == 2048 && dim(image1)[2] == 2048)){
      file.remove(paste("./crop_x2048/", A01[m], sep=""))
    }else{}
  }
  
  setwd("../../")
  }






