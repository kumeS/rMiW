##' @title rasterMiW: display as a raster image
##'
##' @param res a result of Img2DClustering
##' @param c chroma value in the HCL color description.
##' @param method select "browser" or "raster"
##'
##' @importFrom EBImage resize
##' @importFrom grDevices col2rgb
##' @importFrom colorspace rainbow_hcl
##' @export rasterMiW
##' @author Satoshi Kume
##'

rasterMiW <- function(res, c=75, method="raster") {

if(!all(names(res) == c("Original", "Cluster", "ClusterNumber"))){
  return(message("Warning: not proper value of res"))}
#x <- ImgClus; str(x); str(res)
x1 <- res$Cluster
x <- res$Cluster
y <- res$Original
a <- table(res$Cluster)
a1 <- order(-a)
for(n in 1:length(a1)){x1[x == names(a)[n]] <- formatC(a1[n], flag = "0", width = 3)}
a <- names(table(x1))
b <- length(a)

img = EBImage::channel(matrix(1, nrow=dim(x)[1], ncol=dim(x)[2]), 'rgb')@.Data
for(n in 1:b){
  #n <- 3
  v <- grDevices::col2rgb(colorspace::rainbow_hcl(b, c=c)[n]) / 255
  img[,,1][x1 == a[n]] <- img[,,1][x1 == a[n]]*v[1]
  img[,,2][x1 == a[n]] <- img[,,2][x1 == a[n]]*v[2]
  img[,,3][x1 == a[n]] <- img[,,3][x1 == a[n]]*v[3]
}

if(method == "browser"){
EBImage::display(EBImage::combine(EBImage::Image(y, colormode = "Color"),
                                  EBImage::Image(img, colormode = "Color")),
                 nx=2, all=T, spacing = 0, margin = 0, drawGrid=F, method = "browser")
}

if(method == "raster"){
EBImage::display(EBImage::combine(EBImage::Image(y, colormode = "Color"),
                                  EBImage::Image(img, colormode = "Color")),
                 nx=2, all=T, spacing = 0, margin = 0, drawGrid=F, method = "raster")
}
}









