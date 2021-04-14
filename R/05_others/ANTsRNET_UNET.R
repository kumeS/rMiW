#This script from "ANTsRNet/R/createUnetModel.R"
#https://github.com/ANTsX/ANTsRNet/blob/master/R/createUnetModel.R

ANTsRNET_UnetModel2D <- function( inputImageSize,
                               numberOfOutputs = 1,
                               numberOfLayers = 4,
                               numberOfFiltersAtBaseLayer = 32,
                               convolutionKernelSize = c( 3, 3 ),
                               deconvolutionKernelSize = c( 2, 2 ),
                               poolSize = c( 2, 2 ),
                               strides = c( 2, 2 ),
                               dropoutRate = 0.0,
                               weightDecay = 0.0,
                               addAttentionGating = FALSE,
                               mode = c( 'classification', 'regression', 'sigmoid' )
                             )
{

  attentionGate2D <- function( x, g, interShape )
    {
    xTheta <- x %>% layer_conv_2d( filters = interShape, kernel_size = c( 1L, 1L ),
      strides = c( 1L, 1L ) )
    gPhi <- g %>% layer_conv_2d( filters = interShape, kernel_size = c( 1L, 1L ),
      strides = c( 1L, 1L ) )
    f <- layer_add( list( xTheta, gPhi ) ) %>% layer_activation_relu()
    fPsi <- f %>% layer_conv_2d( filters = 1L, kernel_size = c( 1L, 1L ),
      strides = c( 1L, 1L ) )
    alpha <- fPsi %>% layer_activation( activation = "sigmoid" )
    attention <- layer_multiply( list( x, alpha ) )
    return( attention )
    }

  mode <- match.arg( mode )

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path

  encodingConvolutionLayers <- list()
  for( i in seq_len( numberOfLayers ) )
    {
    numberOfFilters <- numberOfFiltersAtBaseLayer * 2 ^ ( i - 1 )

    if( i == 1 )
      {
      conv <- inputs %>% layer_conv_2d( filters = numberOfFilters,
        kernel_size = convolutionKernelSize, activation = 'relu',
        padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )
      } else {
      conv <- pool %>% layer_conv_2d( filters = numberOfFilters,
        kernel_size = convolutionKernelSize, activation = 'relu',
        padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )
      }
    if( dropoutRate > 0.0 )
      {
      conv <- conv %>% layer_dropout( rate = dropoutRate )
      }

    encodingConvolutionLayers[[i]] <- conv %>% layer_conv_2d(
      filters = numberOfFilters, kernel_size = convolutionKernelSize,
      activation = 'relu', padding = 'same' )

    if( i < numberOfLayers )
      {
      pool <- encodingConvolutionLayers[[i]] %>%
        layer_max_pooling_2d( pool_size = poolSize, strides = strides,
        padding = 'same' )
      }
    }

  # Decoding path

  outputs <- encodingConvolutionLayers[[numberOfLayers]]
  for( i in 2:numberOfLayers )
    {
    numberOfFilters <- numberOfFiltersAtBaseLayer * 2 ^ ( numberOfLayers - i )
    deconv <- outputs %>%
      layer_conv_2d_transpose( filters = numberOfFilters,
        kernel_size = deconvolutionKernelSize,
        padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )
    deconv <- deconv %>% layer_upsampling_2d( size = poolSize )

    if( addAttentionGating == TRUE )
      {
      outputs <- attentionGate2D( deconv,
        encodingConvolutionLayers[[numberOfLayers - i + 1]],
        as.integer( numberOfFilters / 4 ) )
      outputs <- layer_concatenate( list( deconv, outputs ), axis = 3 )
      } else {
      outputs <- layer_concatenate( list( deconv,
        encodingConvolutionLayers[[numberOfLayers - i + 1]] ),
        axis = 3 )
      }

    outputs <- outputs %>%
      layer_conv_2d( filters = numberOfFilters,
        kernel_size = convolutionKernelSize,
        activation = 'relu', padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )

    if( dropoutRate > 0.0 )
      {
      outputs <- outputs %>% layer_dropout( rate = dropoutRate )
      }

    outputs <- outputs %>%
      layer_conv_2d( filters = numberOfFilters,
        kernel_size = convolutionKernelSize,
        activation = 'relu', padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )
    }

  convActivation <- ''
  if( mode == 'sigmoid' )
    {
    convActivation <- 'sigmoid'
    } else if( mode == 'classification' ) {
    convActivation <- 'softmax'
    } else if( mode == 'regression' ) {
    convActivation <- 'linear'
    } else {
    stop( 'Error: unrecognized mode.' )
    }

  outputs <- outputs %>%
    layer_conv_2d( filters = numberOfOutputs,
      kernel_size = c( 1, 1 ), activation = convActivation,
      kernel_regularizer = regularizer_l2( weightDecay ) )

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}

ANTsRNET_UnetModel3D <- function( inputImageSize,
                               numberOfOutputs = 1,
                               numberOfLayers = 2,
                               numberOfFiltersAtBaseLayer = 1,
                               convolutionKernelSize = c( 3, 3, 3 ),
                               deconvolutionKernelSize = c( 2, 2, 2 ),
                               poolSize = c( 2, 2, 2 ),
                               strides = c( 2, 2, 2 ),
                               dropoutRate = 0.1,
                               weightDecay = 0.0,
                               addAttentionGating = FALSE,
                               mode = c( 'classification', 'regression', 'sigmoid' )
                             )
{

  attentionGate3D <- function( x, g, interShape )
    {
    xTheta <- x %>% layer_conv_3d( filters = interShape, kernel_size = c( 1L, 1L, 1L ),
      strides = c( 1L, 1L, 1L ) )
    gPhi <- g %>% layer_conv_3d( filters = interShape, kernel_size = c( 1L, 1L, 1L ),
      strides = c( 1L, 1L, 1L ) )
    f <- layer_add( list( xTheta, gPhi ) ) %>% layer_activation_relu()
    fPsi <- f %>% layer_conv_3d( filters = 1L, kernel_size = c( 1L, 1L, 1L ),
      strides = c( 1L, 1L, 1L ) )
    alpha <- fPsi %>% layer_activation( activation = "sigmoid" )
    attention <- layer_multiply( list( x, alpha ) )
    return( attention )
    }

  mode <- match.arg( mode )

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path

  encodingConvolutionLayers <- list()
  for( i in seq_len( numberOfLayers ) )
    {
    numberOfFilters <- numberOfFiltersAtBaseLayer * 2 ^ ( i - 1 )

    if( i == 1 )
      {
      conv <- inputs %>% layer_conv_3d( filters = numberOfFilters,
        kernel_size = convolutionKernelSize, activation = 'relu',
        padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )
      } else {
      conv <- pool %>% layer_conv_3d( filters = numberOfFilters,
        kernel_size = convolutionKernelSize, activation = 'relu',
        padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )
      }
    if( dropoutRate > 0.0 )
      {
      conv <- conv %>% layer_dropout( rate = dropoutRate )
      }

    encodingConvolutionLayers[[i]] <- conv %>% layer_conv_3d(
      filters = numberOfFilters, kernel_size = convolutionKernelSize,
      activation = 'relu', padding = 'same',
      kernel_regularizer = regularizer_l2( weightDecay ) )

    if( i < numberOfLayers )
      {
      pool <- encodingConvolutionLayers[[i]] %>%
        layer_max_pooling_3d( pool_size = poolSize, strides = strides,
        padding = 'same' )
      }
    }

  # Decoding path

  outputs <- encodingConvolutionLayers[[numberOfLayers]]
  for( i in 2:numberOfLayers )
    {
    numberOfFilters <- numberOfFiltersAtBaseLayer * 2 ^ ( numberOfLayers - i )
    deconv <- outputs %>%
      layer_conv_3d_transpose( filters = numberOfFilters,
        kernel_size = deconvolutionKernelSize,
        padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )
    deconv <- deconv %>% layer_upsampling_3d( size = poolSize )

    if( addAttentionGating == TRUE )
      {
      outputs <- attentionGate3D( deconv,
        encodingConvolutionLayers[[numberOfLayers - i + 1]],
        as.integer( numberOfFilters / 4 ) )
      outputs <- layer_concatenate( list( deconv, outputs ), axis = 4 )
      } else {
      outputs <- layer_concatenate( list( deconv,
        encodingConvolutionLayers[[numberOfLayers - i + 1]] ),
        axis = 4 )
      }

    outputs <- outputs %>%
      layer_conv_3d( filters = numberOfFilters,
        kernel_size = convolutionKernelSize,
        activation = 'relu', padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )

    if( dropoutRate > 0.0 )
      {
      outputs <- outputs %>% layer_dropout( rate = dropoutRate )
      }

    outputs <- outputs %>%
      layer_conv_3d( filters = numberOfFilters,
        kernel_size = convolutionKernelSize,
        activation = 'relu', padding = 'same',
        kernel_regularizer = regularizer_l2( weightDecay ) )
    }

  convActivation <- ''
  if( mode == 'sigmoid' )
    {
    convActivation <- 'sigmoid'
    } else if( mode == 'classification' ) {
    convActivation <- 'softmax'
    } else if( mode == 'regression' ) {
    convActivation <- 'linear'
    } else {
    stop( 'Error: unrecognized mode.' )
    }

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs,
      kernel_size = c( 1, 1, 1 ), activation = convActivation,
      kernel_regularizer = regularizer_l2( weightDecay ) )

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}
