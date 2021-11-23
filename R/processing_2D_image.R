##' @title Create an array data of 3D images
##'
##' @param file a file path
##' @param shape a vector of width and height of the resized image or Null
##' @param type an image type (i.e. png, tiff)
##' @param filter a filtering method (default method: bilinear)
##' @param normalize a logical; Intensity values linear scaling
##' @param clahe a logical; Contrast Limited Adaptive Histogram Equalization
##' @param GammaVal a numeric of gamma
##'
##' @importFrom EBImage readImage
##' @importFrom EBImage resize
##' @importFrom EBImage normalize
##' @importFrom EBImage clahe
##'
##' @export processing_2D_image
##' @author Satoshi Kume
##'

processing_2D_image <- function(file, shape, type="png", filter="bilinear",
                                normalize=FALSE, clahe=FALSE, GammaVal=1.0){
  image <- EBImage::readImage(file, type=type)
  if(!is.null(shape)){ image <- EBImage::resize(image, w = shape[1], h = shape[2], filter = filter) }
  if(normalize){ image <- EBImage::normalize(image) }
  if(clahe){ image <- EBImage::clahe(image) }
  if(is.numeric(GammaVal)){ image <- image^GammaVal }

  if(length(dim(image)) == 2){
    shapeNew <- c(dim(image), 1)
  }else{
    shapeNew <- c(dim(image))
  }

  return(array(image, dim=c(shapeNew[1], shapeNew[2], shapeNew[3])))

}


