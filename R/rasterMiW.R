##' @title rasterMiW: display as a raster image
##'
##' @param x
##' @param c chroma value in the HCL color description.
##' @param
##'
##' @importFrom EBImage resize
##' @importFrom grDevices col2rgb
##' @importFrom colorspace rainbow_hcl
##' @export rasterMiW
##' @author Satoshi Kume
##'

rasterMiW <- function(x, c=75) {

#x <- ImgClus; str(x)
x1 <- x
a <- table(x)
a1 <- order(-a)
for(n in 1:length(a1)){x1[x == names(a)[n]] <- formatC(a1[n], flag = "0", width = 3)}
a <- names(table(x1))
b <- length(a)

img = channel(matrix(1, nrow=dim(x)[1], ncol=dim(x)[2]), 'rgb')@.Data
for(n in 1:b){
  #n <- 3
  v <- grDevices::col2rgb(colorspace::rainbow_hcl(b, c=c)[n]) / 255
  img[,,1][x1 == a[n]] <- img[,,1][x1 == a[n]]*v[1]
  img[,,2][x1 == a[n]] <- img[,,2][x1 == a[n]]*v[2]
  img[,,3][x1 == a[n]] <- img[,,3][x1 == a[n]]*v[3]
}

return(EBImage::display(EBImage::Image(img, colormode = "Color"), method = "browser"))

}









