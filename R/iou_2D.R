##' @title 2D IoU (Intersection-Over-Union) for single label
##'
##' @usage iou(y_true, y_pred, smooth = 1.0)
##' @usage iou_loss(y_true, y_pred)
##'
##' @param y_true XXX
##' @param y_pred XXX
##' @param smooth XXX
##'
##' @importFrom keras k_flatten
##' @importFrom keras k_sum
##' @importFrom keras loss_binary_crossentropy
##'
##' @export iou
##' @export iou_loss
##' @author Satoshi Kume
##'

iou <- function(y_true, y_pred, smooth = 1.0){
  y_true_f <- keras::k_flatten(y_true)
  y_pred_f <- keras::k_flatten(y_pred)
  intersection <- keras::k_sum( y_true_f * y_pred_f)
  union <- keras::k_sum( y_true_f + y_pred_f ) - intersection
  result <- (intersection + smooth) / ( union + smooth)
  return(result)
}

iou_loss <- function(y_true, y_pred) {
  result <- keras::loss_binary_crossentropy(y_true, y_pred) + (1 - iou(y_true, y_pred))
  return(result)
}
