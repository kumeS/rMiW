##' @title crop2D: Crop image to array
##'
##' @param x
##' @param XY
##' @param Crop a numeric: ex. 2^n
##'
##' @importFrom EBImage resize
##' @export crop2D
##' @author Satoshi Kume
##'

crop2D <- function(x, XY=1024, Crop=2^3) {
Img00 <- EBImage::resize(x, w = XY, h = XY, filter="bilinear")

#Parameter set
k <- XY/Crop
m <- XY[1]/k
m1 <- c(1:m)*k - k + 1
m2 <- c(1:m)*k

#X-axis setting
w1 <- rep(m1, times = m)
w2 <- rep(m2, times = m)

#Y-axis setting
h1 <- rep(m1, each = length(m1))
h2 <- rep(m2, each = length(m2))

#Convert to array
ImgDatCrop <- c()
for(n in 1:c(m*m)){
  Crop01 <- Img00[c(h1[n]:h2[n]),c(w1[n]:w2[n]),]
  ImgDatCrop[[n]] <- Crop01@.Data
}

ImgDatCrop00 <- rMiW::list2tensor(ImgDatCrop)
return(ImgDatCrop00)

}









