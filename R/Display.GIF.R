##' @title Visualization of GIF animation.
##'
##'
##' @description Display the GIF animation file.
##'
##' @usage Display.GIF(GifFileName, View=TRUE)
##' @param GifFileName a file path of GIF file.
##' @param View logical; if TRUE, display the file now.
##'
##' @return Nothing files
##' @author Satoshi Kume
##'
##' @importFrom magick image_read
##' @export Display.gif
##'
##' @examples \dontrun{
##'
##' Img <- Display.GIF(GifFileName="./animation.gif", View=TRUE)
##'
##' }

Display.gif <- function(GifFileName, View=TRUE){
if(!grepl(".gif$", GifFileName)){return(print("No GIF file!!"))}
a <- magick::image_read(GifFileName)
if(View){print(a); return(a)}else{return(a)}
}

