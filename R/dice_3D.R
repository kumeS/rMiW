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
## 3D Dice Coefficient for single label
############################################
#K <- keras::backend()

Singlelabel_3d_dice_coef <- function( y_true, y_pred, axis = c(1:3), smooth = 1.0 ){
  prediction <- y_pred
  y_true_r <- y_true

  intersection <- k_sum(y_true_r * prediction, axis=axis)
  union <- k_sum(y_true_r + prediction, axis=axis)

  numerator <- 2 * intersection + smooth
  denominator = union + smooth
  result <- numerator / denominator

  return( result )
}

Singlelabel_3d_bce_dice_loss <- function( y_true, y_pred ) {
  1 - Singlelabel_3d_dice_coef(y_true, y_pred, axis = c(1:3), smooth = 1.0)
}

###
#str(TrainImgDataset)
#y_true <- TrainImgDataset$Training
#y_pred <- TrainImgDataset$TrainingGT

multilabel_dice_coefficientR <- function( y_true, y_pred, smooth = 1.0 ){

  y_dims <- unlist( k_int_shape( y_pred ) )
  numberOfLabels <- y_dims[length(y_dims)]

  # Unlike native R, indexing starts at 0.  However, we are
  # assuming the background is 0 so we skip index 0.

  if( length(y_dims) == 4 ){
    # 2-D image
    y_true_permuted <- k_permute_dimensions(y_true, pattern = c( 4L, 1L, 2L, 3L ) )
    y_pred_permuted <- k_permute_dimensions(y_pred, pattern = c( 4L, 1L, 2L, 3L ) )
  } else {
    # 3-D image
    y_true_permuted <- k_permute_dimensions(y_true, pattern = c( 5L, 1L, 2L, 3L, 4L ) )
    y_pred_permuted <- k_permute_dimensions(y_pred, pattern = c( 5L, 1L, 2L, 3L, 4L ) )
  }

  y_true_label <- k_gather( y_true_permuted, indices = c( 1L ) )
  y_pred_label <- k_gather( y_pred_permuted, indices = c( 1L ) )

  y_true_label_f <- k_flatten( y_true_label )
  y_pred_label_f <- k_flatten( y_pred_label )
  intersection <- y_true_label_f * y_pred_label_f
  union <- y_true_label_f + y_pred_label_f - intersection

  numerator <- k_sum( intersection )
  denominator <- k_sum( union )

  #numberOfLabels = 3
  if( numberOfLabels > 2 ){
    for( j in 2L:( numberOfLabels ) ){
      y_true_label <- k_gather( y_true_permuted, indices = c( j ) )
      y_pred_label <- k_gather( y_pred_permuted, indices = c( j ) )
      y_true_label_f <- k_flatten( y_true_label )
      y_pred_label_f <- k_flatten( y_pred_label )
      intersection <- y_true_label_f * y_pred_label_f
      union <- y_true_label_f + y_pred_label_f - intersection

      numerator <- numerator + k_sum( intersection )
      denominator <- denominator + k_sum( union )
    }
  }

  unionOverlap <- numerator / denominator
  result <- ( 2.0 * unionOverlap + smooth ) / ( 1.0 + unionOverlap + smooth )

  return ( result )
}

metric_multilabel_dice_coefficient <-
   keras::custom_metric( "multilabel_dice_coefficientR",
     multilabel_dice_coefficientR )

loss_dice <- function( y_true, y_pred ) {1 - multilabel_dice_coefficientR(y_true, y_pred)}

#attr(loss_dice, "py_function_name") <- "multilabel_dice_coefficientR"

