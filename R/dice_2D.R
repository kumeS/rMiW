##' @title 2D Dice coefficient (F1 score) for single label.
##'
##' @description
##' Dice coefficient and loss function for the compile of model.
##'
##' @usage dice_coef(y_true, y_pred, smooth = 1.0)
##' @usage bce_dice_loss(y_true, y_pred)
##'
##' @param y_true XXX
##' @param y_pred XXX
##' @param smooth XXX
##'
##' @author Satoshi Kume
##'
##' @importFrom  keras k_flatten
##' @importFrom  keras k_sum
##' @importFrom  keras loss_binary_crossentropy
##'
##' @export dice_coef
##' @export bce_dice_loss
##'
##' @examples
##' \dontrun{
##' # Compile
##' model <- model %>%
##'   compile(optimizer = optimizer_rmsprop( lr = 0.01 ),
##'   loss = bce_dice_loss,
##'   metrics = custom_metric( "dice_coef", dice_coef ))
##' }

dice_coef <- function(y_true, y_pred, smooth = 1.0) {
  y_true_f <- keras::k_flatten(y_true)
  y_pred_f <- keras::k_flatten(y_pred)
  intersection <- keras::k_sum(y_true_f * y_pred_f)
  result <- (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
  return(result)
}

bce_dice_loss <- function(y_true, y_pred) {
  result <- keras::loss_binary_crossentropy(y_true, y_pred) + (1 - dice_coef(y_true, y_pred))
  return(result)
}
