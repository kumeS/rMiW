#########################
## create 2D U-NET
## unet 3x3 2DConv layer
#########################
unet_layer <- function(object, filters, kernel_size = c(FLAGS$kernel_size, FLAGS$kernel_size),
                       padding = "same", kernel_initializer = "he_normal",
                       dropout = 0.1, activation="relu"){
  object %>%
    layer_conv_2d(filters = filters, kernel_size = kernel_size, padding = padding) %>%
    layer_batch_normalization() %>%
    layer_activation(activation) %>%
    layer_spatial_dropout_2d(rate = dropout) %>%
    layer_conv_2d(filters = filters, kernel_size = kernel_size, padding = padding) %>%
    layer_batch_normalization() %>%
    layer_activation(activation) %>%
    layer_conv_2d(filters = filters, kernel_size = kernel_size, padding = padding) %>%
    layer_batch_normalization() %>%
    layer_activation(activation) %>%
    layer_spatial_dropout_2d(rate = dropout) %>%
    layer_conv_2d(filters = filters, kernel_size = kernel_size, padding = padding) %>%
    layer_batch_normalization() %>%
    layer_activation(activation)
}

unet <- function(shape, nlevels = 3, nfilters = 16, dropouts = c(0.1, 0.1, 0.3, 0.3)){
  message("Constructing U-Net with ", nlevels, " levels initial number of filters is: ", nfilters)
  filter_sizes <- nfilters*2^seq.int(0, nlevels)
  ## Loop over contracting layers
  clayers <- clayers_pooled <- list()

  ## inputs
  clayers_pooled[[1]] <- layer_input(shape = shape)

  for(i in 2:(nlevels+1)) {
    clayers[[i]] <- unet_layer(clayers_pooled[[i - 1]],
                               filters = filter_sizes[i - 1],
                               dropout = dropouts[i-1])

    clayers_pooled[[i]] <- layer_max_pooling_2d(clayers[[i]],
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
    elayers[[i]] <- layer_conv_2d_transpose(elayers[[i+1]],
                                            filters = filter_sizes[i],
                                            kernel_size = c(2, 2),
                                            strides = c(2, 2),
                                            padding = "same")

    elayers[[i]] <- layer_concatenate(list(elayers[[i]], clayers[[i + 1]]), axis = 3)
    elayers[[i]] <- unet_layer(elayers[[i]], filters = filter_sizes[i], dropout = dropouts[i])

  }

  ## Output layer
  outputs <- layer_conv_2d(elayers[[1]], filters = 1, kernel_size = c(1, 1), activation = "sigmoid")

  return(keras_model(inputs = clayers_pooled[[1]], outputs = outputs))
}

#########################
## create 3D U-NET
#########################
if(F){
inputImageSize = c( 256, 256, 20, 1)
numberOfOutputs = 1
numberOfLayers = 3
nfilters = 32
convolutionKernelSize = c( 3, 3, 3 )
deconvolutionKernelSize = c( 2, 2, 2 )
poolSize = c( 2, 2, 2 )
strides = c( 2, 2, 2 )
dropouts = 0.0
weightDecay = 0.0
mode = 'classification'
}

unet_3dlayer <- function(object, numberOfFilters=numberOfFilters,
                         convolutionKernelSize=convolutionKernelSize,
                       padding = "same", dropout = 0.1, activation="relu",
                       weightDecay=weightDecay){
object %>%
    layer_conv_3d( filters = numberOfFilters,
                   kernel_size = convolutionKernelSize,
                   activation = activation,
                   padding = padding,
                   kernel_regularizer = regularizer_l2( weightDecay )) %>%
    layer_conv_3d( filters = numberOfFilters,
                   kernel_size = convolutionKernelSize,
                   activation = activation,
                   padding = padding,
                   kernel_regularizer = regularizer_l2( weightDecay )) %>%
    layer_conv_3d( filters = numberOfFilters,
                   kernel_size = convolutionKernelSize,
                   activation = activation,
                   padding = padding,
                   kernel_regularizer = regularizer_l2( weightDecay )) %>%
    #layer_dropout( rate = dropouts ) %>%
    layer_conv_3d( filters = numberOfFilters,
                   kernel_size = convolutionKernelSize,
                   activation = activation,
                   padding = padding,
                   kernel_regularizer = regularizer_l2( weightDecay ))
}


createUnetModel3D <- function( inputImageSize,
                               numberOfOutputs = 1,
                               numberOfLayers = 3,
                               nfilters = 4,
                               convolutionKernelSize = c( 3, 3, 3 ),
                               deconvolutionKernelSize = c( 2, 2, 2 ),
                               poolSize = c( 2, 2, 2 ),
                               strides = c( 2, 2, 2 ),
                               dropouts = 0.1,
                               weightDecay = 0.0,
                               mode = 'classification'){

inputs <- layer_input( shape = inputImageSize )

# Encoding path
encodingConvolutionLayers <- list()

#nlevels = 3
#nfilters*2^seq.int(0, nlevels)
for( i in seq_len( numberOfLayers ) ){
numberOfFilters <- nfilters * 2 ^ ( i - 1 )

if( i == 1 ){
conv <- unet_3dlayer(inputs, numberOfFilters=numberOfFilters,
                     convolutionKernelSize=convolutionKernelSize,
                     padding = "same", dropout = 0.1, activation="relu",
                     weightDecay=weightDecay)
} else {
conv <- unet_3dlayer(inputs, numberOfFilters=numberOfFilters,
                     convolutionKernelSize=convolutionKernelSize,
                     padding = "same", dropout = 0.1, activation="relu",
                     weightDecay=weightDecay)
}

encodingConvolutionLayers[[i]] <- conv

if( i < numberOfLayers ){
conv <- conv %>%
        layer_max_pooling_3d( pool_size = poolSize, strides = strides, padding = 'same' )
}
}

#Decoding path
outputs <- encodingConvolutionLayers[[numberOfLayers]]

for( i in 2:numberOfLayers ){
#i <- 2
numberOfFilters <- nfilters * 2 ^ ( numberOfLayers - i )
deconv <- outputs %>%
  layer_conv_3d_transpose( filters = numberOfFilters,
                           kernel_size = deconvolutionKernelSize,
                           padding = 'same', kernel_regularizer = regularizer_l2( weightDecay ) )
deconv <- deconv %>% layer_upsampling_3d( size = poolSize )

outputs <- layer_concatenate( list( deconv, encodingConvolutionLayers[[numberOfLayers - i + 1]] ), axis = 4 )

outputs <- unet_3dlayer(deconv, numberOfFilters=numberOfFilters,
                     convolutionKernelSize=convolutionKernelSize,
                     padding = "same", dropout = 0.1, activation="relu",
                     weightDecay=weightDecay)
}

convActivation <- ''
if( numberOfOutputs == 1 ){
    convActivation <- 'sigmoid'
  } else {
    convActivation <- 'softmax'
  }

outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs,
                   kernel_size = c( 1, 1, 1 ), activation = convActivation,
                   kernel_regularizer = regularizer_l2( weightDecay ) )

unetModel <- keras_model( inputs = inputs, outputs = outputs )

return( unetModel )
}

#Model <- keras_model(inputs = inputs, outputs = outputs); Model

#################################################################################
#################################################################################
#inputImageSize = c( 256, 256, 32, 1)

createNoBrainerUnetModel3D <- function( inputImageSize ){

  numberOfOutputs <- 1
  nFilters <- 1
  convolutionKernelSize <- c( 3, 3, 3 )
  deconvolutionKernelSize <- c( 2, 2, 2 )
  ps = c( 2, 2, 2 )

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path
  outputs <- inputs %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_3d( filters = nFilters * 2, kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_3d( filters = nFilters * 4, kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )

  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

#中間層
  outputs <- outputs %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )

# Decoding path
  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = nFilters , kernel_size = deconvolutionKernelSize, strides = 2, padding = 'same' )
  outputs <- list( skip3, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  ) %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same' )

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = nFilters , kernel_size = deconvolutionKernelSize, strides = 2, padding = 'same', activation = 'relu'  )
  outputs <- list( skip2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  ) %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  )

  outputs <- outputs %>% layer_conv_3d_transpose( filters = nFilters , kernel_size = deconvolutionKernelSize, strides = 2, padding = 'same', activation = 'relu'  )
  outputs <- list( skip1, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  ) %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  )

  convActivation <- ''
  if( numberOfOutputs == 1 ){convActivation <- 'sigmoid'} else {convActivation <- 'softmax'}

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}

#unetModel %>% plot_model_modi(width=4, height=1.2)
unetModel
