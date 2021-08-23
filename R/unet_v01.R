##' @title Provide a layer of 2D U-net model (v01)
##'
##' @usage unet_2Dlayer_v01(object,
##'                         filters,
##'                         dropout)
##' @param object
##' @param filters
##' @param dropout
##'
##' @import keras
##' @export unet_2Dlayer_v01
##' @author Satoshi Kume
##'

unet_2Dlayer_v01 <- function(object, filters, dropout){

#Parameters
Kernel_size <- c(3, 3)
Padding <- "same"
Kernel_initializer <- "he_normal"
Activation <- "relu"

object %>%
  keras::layer_conv_2d(filters = filters, kernel_size = Kernel_size, padding = Padding, kernel_initializer = Kernel_initializer) %>%
  keras::layer_batch_normalization() %>%
  keras::layer_activation(activation=Activation) %>%
  keras::layer_conv_2d(filters = filters, kernel_size = Kernel_size, padding = Padding, kernel_initializer = Kernel_initializer) %>%
  keras::layer_batch_normalization() %>%
  keras::layer_activation(activation=Activation) %>%
  keras::layer_spatial_dropout_2d(rate = dropout) %>%
  keras::layer_conv_2d(filters = filters, kernel_size = Kernel_size, padding = Padding, kernel_initializer = Kernel_initializer) %>%
  keras::layer_batch_normalization() %>%
  keras::layer_activation(activation=Activation)
}

##' @title Cretae a model of 2D U-net (v01)
##'
##' @usage unet2D_v01(shape, nlevels = 3, nfilters = 16, dropouts = c(0.1, 0.1, 0.3, 0.3))
##' @param shape a 3-element vector: Width, Height, Channels.
##'
##' @import keras
##' @export unet2D_v01
##' @author Satoshi Kume
##' @example {
##'
##' model <- unet(shape=c(512,512,1), nlevels = 3, nfilters = 16)
##'
##' }
##'

unet2D_v01 <- function(shape){

nlevels <- 3
nfilters <- 16
dropouts <- c(0.1, 0.1, 0.3, 0.3)

message(paste0("Constructing U-Net v01 with ", nlevels, " Levels\n",
               "Initial number of filters is ", nfilters, "\n",
               "Dropout rates: ", paste(dropouts, collapse = ", ")))

filter_sizes <- nfilters*2^seq.int(0, nlevels)

## Loop over contracting layers
clayers <- clayers_pooled <- list()

## inputs
clayers_pooled[[1]] <- keras::layer_input(shape = shape)

for(i in 2:(nlevels+1)) {
clayers[[i]] <- unet_layer(clayers_pooled[[i - 1]],
                             filters = filter_sizes[i - 1],
                             dropout = dropouts[i-1])

clayers_pooled[[i]] <- keras::layer_max_pooling_2d(clayers[[i]],
                                              pool_size = c(2, 2),
                                              strides = c(2, 2))
}

## Loop over expanding layers
elayers <- list()

## center
elayers[[nlevels + 1]] <- unet_layer(clayers_pooled[[nlevels + 1]],
                                      filters = filter_sizes[nlevels + 1],
                                      dropout = dropouts[nlevels + 1])

for(i in nlevels:1) {
elayers[[i]] <- keras::layer_conv_2d_transpose(elayers[[i+1]],
                                          filters = filter_sizes[i],
                                          kernel_size = c(2, 2),
                                          strides = c(2, 2),
                                          padding = "same")

elayers[[i]] <- keras::layer_concatenate(list(elayers[[i]], clayers[[i + 1]]), axis = 3)
elayers[[i]] <- unet_layer(elayers[[i]], filters = filter_sizes[i], dropout = dropouts[i])

}

## Output layer
outputs <- keras::layer_conv_2d(elayers[[1]], filters = 1, kernel_size = c(1, 1), activation = "sigmoid")

return(keras::keras_model(inputs = clayers_pooled[[1]], outputs = outputs))

}
