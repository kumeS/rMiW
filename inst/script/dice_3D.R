##' @title Create network view for the keras model object.
##'
##' @description visualize a model in Keras
##'
##' @usage Singlelabel_3d_dice_coef(y_true, y_pred, axis = c(1:3), smooth = 1.0)
##' @usage Singlelabel_3d_bce_dice_loss(y_true, y_pred)
##' @usage multilabel_dice_coefficientR(y_true, y_pred, smooth = 1.0)
##'
##' @importFrom keras k_sum
##' @importFrom keras k_int_shape
##' @importFrom keras k_permute_dimensions
##' @importFrom keras k_gather
##' @importFrom keras k_flatten
##'
##' @author Satoshi Kume
##'
##' @export Singlelabel_3d_dice_coef
##' @export Singlelabel_3d_bce_dice_loss
##' @export multilabel_dice_coefficientR
##' @export metric_multilabel_dice_coefficient
##'


############################################
## 3D Dice Coefficient for single label
############################################
#K <- keras::backend()

Singlelabel_3d_dice_coef <- function( y_true, y_pred, axis = c(1:3), smooth = 1.0 ){
  prediction <- y_pred
  y_true_r <- y_true

  intersection <- keras::k_sum(y_true_r * prediction, axis=axis)
  union <- keras::k_sum(y_true_r + prediction, axis=axis)

  numerator <- 2 * intersection + smooth
  denominator = union + smooth
  result <- numerator / denominator

  return( result )
}

Singlelabel_3d_bce_dice_loss <- function( y_true, y_pred ) {
  1 - Singlelabel_3d_dice_coef(y_true, y_pred, axis = c(1:3), smooth = 1.0)
}

multilabel_dice_coefficientR <- function( y_true, y_pred, smooth = 1.0 ){

  y_dims <- unlist( keras::k_int_shape( y_pred ) )
  numberOfLabels <- y_dims[length(y_dims)]

  # Unlike native R, indexing starts at 0.  However, we are
  # assuming the background is 0 so we skip index 0.

  if( length(y_dims) == 4 ){
    # 2-D image
    y_true_permuted <- keras::k_permute_dimensions(y_true, pattern = c( 4L, 1L, 2L, 3L ) )
    y_pred_permuted <- keras::k_permute_dimensions(y_pred, pattern = c( 4L, 1L, 2L, 3L ) )
  } else {
    # 3-D image
    y_true_permuted <- keras::k_permute_dimensions(y_true, pattern = c( 5L, 1L, 2L, 3L, 4L ) )
    y_pred_permuted <- keras::k_permute_dimensions(y_pred, pattern = c( 5L, 1L, 2L, 3L, 4L ) )
  }

  y_true_label <- keras::k_gather( y_true_permuted, indices = c( 1L ) )
  y_pred_label <- keras::k_gather( y_pred_permuted, indices = c( 1L ) )

  y_true_label_f <- keras::k_flatten( y_true_label )
  y_pred_label_f <- keras::k_flatten( y_pred_label )
  intersection <- y_true_label_f * y_pred_label_f
  union <- y_true_label_f + y_pred_label_f - intersection

  numerator <- keras::k_sum( intersection )
  denominator <- keras::k_sum( union )

  #numberOfLabels = 3
  if( numberOfLabels > 2 ){
    for( j in 2L:( numberOfLabels ) ){
      y_true_label <- keras::k_gather( y_true_permuted, indices = c( j ) )
      y_pred_label <- keras::k_gather( y_pred_permuted, indices = c( j ) )
      y_true_label_f <- keras::k_flatten( y_true_label )
      y_pred_label_f <- keras::k_flatten( y_pred_label )
      intersection <- y_true_label_f * y_pred_label_f
      union <- y_true_label_f + y_pred_label_f - intersection

      numerator <- numerator + keras::k_sum( intersection )
      denominator <- denominator + keras::k_sum( union )
    }
  }

  unionOverlap <- numerator / denominator
  result <- ( 2.0 * unionOverlap + smooth ) / ( 1.0 + unionOverlap + smooth )

  return ( result )
}

metric_multilabel_dice_coefficient <-
  keras::custom_metric( "multilabel_dice_coefficientR", multilabel_dice_coefficientR )

loss_dice <- function( y_true, y_pred ) {1 - multilabel_dice_coefficientR(y_true, y_pred)}

