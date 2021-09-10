##' @title Img2DClustering
##'
##' @param x
##' @param Cluster a number of cluster
##' @param XY pixels values of width and height
##' @param OriginalScale
##'
##' @export Img2DClustering
##' @importFrom EBImage resize
##' @importFrom stats kmeans
##' @importFrom EBImage Image
##'
##' @author Satoshi Kume
##'

Img2DClustering <- function(x, Cluster=3, XY=1024, OriginalScale=TRUE){

Img00 <- EBImage::resize(x, w = XY, h = XY, filter="bilinear")

#create RGB verctors
a <- matrix(unlist(Img00), nrow = XY*XY)
b <- matrix(a[,3], nrow=XY)

y <- stats::kmeans(a, centers=Cluster, iter.max = 1000,
       nstart = 1, algorithm = "Hartigan-Wong")

d <- matrix(y$cluster/max(y$cluster), nrow=XY)

if(OriginalScale){
#str(d)
d0 <- EBImage::resize(EBImage::Image(d, colormode = "Grayscale"),
                     w = dim(x@.Data )[1], h = dim(x@.Data )[2], filter="none")
#str(d0)
d1 <- matrix(d0@.Data, nrow=dim(d0@.Data)[1],  ncol=dim(d0@.Data)[2])
#str(d1)
return(d1)
}else{
return(d)
}
}
