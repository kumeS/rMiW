##' @title A new symbol
##'
##' @param x an array.
##' @param y an array.
##'
##' @return x
##' @author Satoshi Kume
##' @export %||%
##'

`%||%` <- function(x, y) {
  if (is.null(x)) {
    y
  } else {
    x
  }
}


