##' @title invert the image
##'
##' @param x
##'
##' @export invert
##' @author Satoshi Kume
##'

invert <- function(x) {
 if(mean(x) > .5)
  x <- 1 - x
 x
}

