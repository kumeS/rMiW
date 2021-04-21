##' @description visualize a model in Keras
##'
##' @title Create network view for the keras model object.
##' @param model the model object in Keras.
##' @param width a plot size of width.
##' @param height a plot size of height.
##' @return A plot of the model network.
##' @author Satoshi Kume
##' @export
##' @examples
##' \dontrun{
##' plot_model_modi(model)
##' }


############################################
## 2D IoU (Intersection-Over-Union) for single label
############################################
iou <- function(y_true, y_pred, smooth = 1.0){
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  intersection <- k_sum( y_true_f * y_pred_f)
  union <- k_sum( y_true_f + y_pred_f ) - intersection
  result <- (intersection + smooth) / ( union + smooth)
  return(result)
}

iou_loss <- function(y_true, y_pred) {
  result <- loss_binary_crossentropy(y_true, y_pred) + (1 - iou(y_true, y_pred))
  return(result)
}


