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
createResUnetModel3D <- function( inputImageSize,
                                  numberOfOutputs = 1,
                                  numberOfFiltersAtBaseLayer = 32,
                                  bottleNeckBlockDepthSchedule = c( 3, 4 ),
                                  convolutionKernelSize = c( 3, 3, 3 ),
                                  deconvolutionKernelSize = c( 2, 2, 2 ),
                                  dropoutRate = 0.0,
                                  weightDecay = 0.0001,
                                  mode = c( 'classification', 'regression' )
                                )
{

  simpleBlock3D <- function( input, numberOfFilters, downsample = FALSE,
    upsample = FALSE, convolutionKernelSize = c( 3, 3, 3 ),
    deconvolutionKernelSize = c( 2, 2, 2 ), weightDecay = 0.0, dropoutRate = 0.0 )
    {
    numberOfOutputFilters <- numberOfFilters

    output <- input %>% layer_batch_normalization()
    output <- output %>% layer_activation_thresholded_relu( theta = 0 )

    if( downsample )
      {
      output <- output %>% layer_max_pooling_3d( pool_size = c( 2, 2, 2 ) )
      }

    output <- output %>% layer_conv_3d( filters = numberOfFilters,
      kernel_size = convolutionKernelSize, padding = 'same',
      kernel_regularizer = regularizer_l2( weightDecay ) )

    if( upsample )
      {
      output <- output %>%
        layer_conv_3d_transpose( filters = numberOfFilters,
          kernel_size = deconvolutionKernelSize, padding = 'same',
          kernel_initializer = initializer_he_normal(),
          kernel_regularizer = regularizer_l2( weightDecay ) )
      output <- output %>% layer_upsampling_3d( size = c( 2, 2, 2 ) )
      }

    if( dropoutRate > 0.0 )
      {
      output <- output %>% layer_dropout( rate = dropoutRate )
      }

    # Modify the input so that it has the same size as the output

    if( downsample )
      {
      input <- input %>% layer_conv_3d( filters = numberOfOutputFilters,
        kernel_size = c( 1, 1, 1 ), strides = c( 2, 2, 2 ), padding = 'same' )
      } else if( upsample ) {
      input <- input %>%
        layer_conv_3d_transpose( filters = numberOfOutputFilters,
          kernel_size = c( 1, 1, 1 ), padding = 'same' )
      input <- input %>% layer_upsampling_3d( size = c( 2, 2, 2 ) )
      } else if( numberOfFilters != numberOfOutputFilters ) {
      input <- input %>% layer_conv_3d( filters = numberOfOutputFilters,
        kernel_size = c( 1, 1, 1 ), padding = 'same' )
      }

    output <- skipConnection( input, output )

    return( output )
    }

  bottleNeckBlock3D <- function( input, numberOfFilters, downsample = FALSE,
    upsample = FALSE, deconvolutionKernelSize = c( 2, 2, 2 ), weightDecay = 0.0,
    dropoutRate = 0.0 )
    {
    output <- input

    numberOfOutputFilters <- numberOfFilters

    if( downsample )
      {
      output <- output %>% layer_batch_normalization()
      output <- output %>% layer_activation_thresholded_relu( theta = 0 )

      output <- output %>% layer_conv_3d(
        filters = numberOfFilters,
        kernel_size = c( 1, 1, 1 ), strides = c( 2, 2, 2 ),
        kernel_initializer = initializer_he_normal(),
        kernel_regularizer = regularizer_l2( weightDecay ) )
      }

    output <- output %>% layer_batch_normalization()
    output <- output %>% layer_activation_thresholded_relu( theta = 0 )

    output <- output %>% layer_conv_3d(
      filters = numberOfFilters, kernel_size = c( 1, 1, 1 ),
      kernel_initializer = initializer_he_normal(),
      kernel_regularizer = regularizer_l2( weightDecay ) )

    output <- output %>% layer_batch_normalization()
    output <- output %>% layer_activation_thresholded_relu( theta = 0 )

    if( upsample )
      {
      output <- output %>%
        layer_conv_3d_transpose( filters = numberOfFilters,
          kernel_size = deconvolutionKernelSize, padding = 'same',
          kernel_initializer = initializer_he_normal(),
          kernel_regularizer = regularizer_l2( weightDecay ) )
      output <- output %>% layer_upsampling_3d( size = c( 2, 2, 2 ) )
      }

    output <- output %>% layer_conv_3d(
      filters = numberOfFilters * 4, kernel_size = c( 1, 1, 1 ),
      kernel_initializer = initializer_he_normal(),
      kernel_regularizer = regularizer_l2( weightDecay ) )

    numberOfOutputFilters <- numberOfFilters * 4

    if( dropoutRate > 0.0 )
      {
      output <- output %>% layer_dropout( rate = dropoutRate )
      }

    # Modify the input so that it has the same size as the output

    if( downsample )
      {
      input <- input %>% layer_conv_3d( filters = numberOfOutputFilters,
        kernel_size = c( 1, 1, 1 ), strides = c( 2, 2, 2 ), padding = 'same' )
      } else if( upsample ) {
      input <- input %>%
        layer_conv_3d_transpose( filters = numberOfOutputFilters,
          kernel_size = c( 1, 1, 1 ), padding = 'same' )
      input <- input %>% layer_upsampling_3d( size = c( 2, 2, 2 ) )
      } else if( numberOfFilters != numberOfOutputFilters ) {
      input <- input %>% layer_conv_3d( filters = numberOfOutputFilters,
        kernel_size = c( 1, 1, 1 ), padding = 'valid' )
      }

    output <- skipConnection( input, output )

    return( output )
    }

  skipConnection <- function( source, target, mergeMode = 'sum' )
    {
    layerList <- list( source, target )

    if( mergeMode == 'sum' )
      {
      output <- layer_add( layerList )
      } else {
      channelAxis <- 1
      if( keras::backend()$image_data_format() == "channels_last" )
        {
        channelAxis <- -1
        }
      output <- layer_concatenate( layerList, axis = channelAxis )
      }

    return( output )
    }

  mode <- match.arg( mode )
  inputs <- layer_input( shape = inputImageSize )

  encodingLayersWithLongSkipConnections <- list()
  encodingLayerCount <- 1

  # Preprocessing layer

  model <- inputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer,
    kernel_size = convolutionKernelSize, activation = 'relu', padding = 'same',
    kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( weightDecay ) )

  encodingLayersWithLongSkipConnections[[encodingLayerCount]] <- model
  encodingLayerCount <- encodingLayerCount + 1

  # Encoding initialization path

  model <- model %>% simpleBlock3D( numberOfFiltersAtBaseLayer,
    downsample = TRUE,
    convolutionKernelSize = convolutionKernelSize,
    deconvolutionKernelSize = deconvolutionKernelSize,
    weightDecay = weightDecay, dropoutRate = dropoutRate )

  encodingLayersWithLongSkipConnections[[encodingLayerCount]] <- model
  encodingLayerCount <- encodingLayerCount + 1

  # Encoding main path

  numberOfBottleNeckLayers <- length( bottleNeckBlockDepthSchedule )
  for( i in seq_len( numberOfBottleNeckLayers ) )
    {
    numberOfFilters <- numberOfFiltersAtBaseLayer * 2 ^ ( i - 1 )

    for( j in seq_len( bottleNeckBlockDepthSchedule[i] ) )
      {
      if( j == 1 )
        {
        doDownsample <- TRUE
        } else {
        doDownsample <- FALSE
        }
      model <- model %>% bottleNeckBlock3D( numberOfFilters = numberOfFilters,
        downsample = doDownsample,
        deconvolutionKernelSize = deconvolutionKernelSize,
        weightDecay = weightDecay, dropoutRate = dropoutRate )

      if( j == bottleNeckBlockDepthSchedule[i] )
        {
        encodingLayersWithLongSkipConnections[[encodingLayerCount]] <- model
        encodingLayerCount <- encodingLayerCount + 1
        }
      }
    }
  encodingLayerCount <- encodingLayerCount - 1

  # Transition path

  numberOfFilters <- numberOfFiltersAtBaseLayer * 2 ^ numberOfBottleNeckLayers
  model <- model %>%
    bottleNeckBlock3D( numberOfFilters = numberOfFilters,
      downsample = TRUE,
      deconvolutionKernelSize = deconvolutionKernelSize,
      weightDecay = weightDecay, dropoutRate = dropoutRate )

  model <- model %>%
    bottleNeckBlock3D( numberOfFilters = numberOfFilters,
      upsample = TRUE,
      deconvolutionKernelSize = deconvolutionKernelSize,
      weightDecay = weightDecay, dropoutRate = dropoutRate )

  # Decoding main path

  numberOfBottleNeckLayers <- length( bottleNeckBlockDepthSchedule )
  for( i in seq_len( numberOfBottleNeckLayers ) )
    {
    numberOfFilters <- numberOfFiltersAtBaseLayer *
      2 ^ ( numberOfBottleNeckLayers - i )

    for( j in seq_len( bottleNeckBlockDepthSchedule[numberOfBottleNeckLayers - i + 1] ) )
      {
      if( j == bottleNeckBlockDepthSchedule[numberOfBottleNeckLayers - i + 1] )
        {
        doUpsample <- TRUE
        } else {
        doUpsample <- FALSE
        }
      model <- model %>% bottleNeckBlock3D( numberOfFilters = numberOfFilters,
        upsample = doUpsample,
        deconvolutionKernelSize = deconvolutionKernelSize,
        weightDecay = weightDecay, dropoutRate = dropoutRate )

      if( j == 1 )
        {
        model <- model %>% layer_conv_3d( filters = numberOfFilters * 4,
          kernel_size = c( 1, 1, 1 ), padding = 'same' )
        model <- skipConnection( encodingLayersWithLongSkipConnections[[encodingLayerCount]], model )
        encodingLayerCount <- encodingLayerCount - 1
        }
      }
    }

  # Decoding initialization path

  model <- model %>% simpleBlock3D( numberOfFiltersAtBaseLayer,
    upsample = TRUE,
    convolutionKernelSize = convolutionKernelSize,
    deconvolutionKernelSize = deconvolutionKernelSize,
    weightDecay = weightDecay, dropoutRate = dropoutRate )

  # Postprocessing layer

  model <- model %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer,
    kernel_size = convolutionKernelSize, activation = 'relu',
    padding = 'same',
    kernel_initializer = initializer_he_normal(),
    kernel_regularizer = regularizer_l2( weightDecay ) )

  encodingLayerCount <- encodingLayerCount - 1

  model <- skipConnection( encodingLayersWithLongSkipConnections[[encodingLayerCount]], model )

  model <- model %>% layer_batch_normalization()
  model <- model %>% layer_activation_thresholded_relu( theta = 0 )

  convActivation <- ''
  if( mode == 'classification' ) {
    convActivation <- 'softmax'
    } else if( mode == 'regression' ) {
    convActivation <- 'linear'
    } else {
    stop( 'Error: unrecognized mode.' )
    }

  outputs <- model %>%
    layer_conv_3d( filters = numberOfOutputs,
      kernel_size = c( 1, 1, 1 ), activation = convActivation,
      kernel_regularizer = regularizer_l2( weightDecay ) )

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}

