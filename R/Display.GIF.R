##' @description Display the GIF animation file.
##'
##' @title Visualization of GIF animation.
##' @param GifFileName a file path of GIF file.
##' @param View logical; if TRUE, display the file now.
##' @return Nothing files
##' @author Satoshi Kume
##' @export Display.GIF
##' @examples \dontrun{
##'
##' Img <- Display.GIF(GifFileName="./animation.gif", View=TRUE)
##'
##' }

Display.GIF <- function(GifFileName, View=TRUE){
if(!grepl(".gif$", GifFileName)){return(print("No GIF file!!"))}
a <- magick::image_read(GifFileName)
if(View){print(a); return(a)}else{return(a)}
}

