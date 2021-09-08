##' @title convert to the gray scaled image
##'
##' @param x an Image object
##' @param mode A character value; luminance or gray.
##'
##' @export toGrayScale
##' @importFrom EBImage rgbImage
##' @importFrom EBImage channel
##' @importFrom EBImage getFrame
##'
##' @author Satoshi Kume
##'


toGrayScale <- function(x, mode){
y <- EBImage::rgbImage(red = EBImage::getFrame(x, 1),
                       green = EBImage::getFrame(x, 2),
                       blue = EBImage::getFrame(x, 3))

switch(mode,
      "luminance" = y <- EBImage::channel(y, mode="luminance"),
      "grey" = y <- EBImage::channel(y, mode="grey"),
      "gray" = y <- EBImage::channel(y, mode="gray"),
      return("Warning: no proper value of mode")
       )

dim(y) <- c(dim(y), 1)

return(y)

}
