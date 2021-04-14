#library(keras)
#py_plot_model <- reticulate::import(module = "tensorflow")$keras$utils$plot_model
#py_plot_model(create3DModel_2L_v1( inputImageSize=c(256, 256, 64,1)), to_file=paste0('Model_', formatC(length(list.files(pattern = "Model_")), width = 3, flag = "0"), '.png'), show_shapes=T, show_layer_names=T)
#system("open Model.png")
#browseURL("")
#browseURL("https://onlinelibrary.wiley.com/doi/epdf/10.1002/hbm.24811")
#browseURL("https://github.com/kumeS/ANTsRNet/blob/master/R/createCustomUnetModel.R")
#browseURL("https://arxiv.org/pdf/1804.03999.pdf")
#browseURL("https://arxiv.org/pdf/1512.03385.pdf")
#browseURL("https://arxiv.org/pdf/1606.06650.pdf")
#browseURL("https://keras.rstudio.com/reference/index.html")

#browseURL("https://arxiv.org/pdf/2011.01118.pdf")
#browseURL("https://arxiv.org/pdf/1902.04049.pdf")


#inputImageSize=c(256, 256, 64,1)

##################################################
## create 3D U-NET
##################################################
create3DModel_3L_v3 <- function( inputImageSize ){
  numberOfOutputs <- 1
  convolutionKernelSize <- c( 3, 3, 3 )
  deconvolutionKernelSize <- c( 2, 2, 2 )
  ps <-  c( 2, 2, 2 )
  strides <-  c( 2, 2, 2 )

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path
  outputs <- inputs %>%
    layer_conv_3d( filters = 8 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_3d( filters = 16 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = 16 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_3d( filters = 32 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = 32 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_3d( filters = 64 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

# intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = 64 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_3d( filters = 64 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

# Decoding path
  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 64 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same' )
  outputs <- list( skip3, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 64 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_3d( filters = 32 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 16 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 32 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_3d( filters = 16 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 32 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip1, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 16 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_3d( filters = 8 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

  if( numberOfOutputs == 1 ){convActivation <- 'sigmoid'} else {convActivation <- 'softmax'}

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  Model <- keras_model( inputs = inputs, outputs = outputs )

  return( Model )
}

########################################################################
########################################################################

create3DModel_3L_v2R <- function( inputImageSize ){
  numberOfOutputs <- 1
  convolutionKernelSize <- c( 3, 3, 3 )
  deconvolutionKernelSize <- c( 2, 2, 2 )
  ps <-  c( 2, 2, 2 )
  strides <-  c( 2, 2, 2 )

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path
  outputs <- inputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

# intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

# Decoding path
  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same' )
  outputs <- list( skip3, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip1, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

  if( numberOfOutputs == 1 ){convActivation <- 'sigmoid'} else {convActivation <- 'softmax'}

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  Model <- keras_model( inputs = inputs, outputs = outputs )

  return( Model )
}


create3DModel_3L_v2 <- function( inputImageSize ){
  numberOfOutputs <- 1
  convolutionKernelSize <- c( 3, 3, 3 )
  deconvolutionKernelSize <- c( 2, 2, 2 )
  ps <-  c( 2, 2, 2 )
  strides <-  c( 2, 2, 2 )

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path
  outputs <- inputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()

  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()

  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

# intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()

# Decoding path
  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same' )
  outputs <- list( skip3, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  ) %>%
    layer_batch_normalization()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip1, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  ) %>%
    layer_batch_normalization()

  if( numberOfOutputs == 1 ){convActivation <- 'sigmoid'} else {convActivation <- 'softmax'}

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  Model <- keras_model( inputs = inputs, outputs = outputs )

  return( Model )
}

########################################################################
########################################################################
create3DModel_3L_v1 <- function( inputImageSize ){
  numberOfOutputs <- 1
  convolutionKernelSize <- c( 3, 3, 3 )
  deconvolutionKernelSize <- c( 2, 2, 2 )
  ps <-  c( 2, 2, 2 )
  strides <-  c( 2, 2, 2 )

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path
  outputs <- inputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )

  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

# intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )

# Decoding path
  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same' )
  outputs <- list( skip3, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  ) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  ) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  )

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip1, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  ) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  )

  convActivation <- ''
  if( numberOfOutputs == 1 ){convActivation <- 'sigmoid'} else {convActivation <- 'softmax'}

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  Model <- keras_model( inputs = inputs, outputs = outputs )

  return( Model )
}

#




############################################################################################
## create 2D U-NET
############################################################################################
create2DModel_v1 <- function( inputImageSize ){
  numberOfOutputs <- 1
  nFilters <- 2
  convolutionKernelSize <- c( 3, 3 )
  deconvolutionKernelSize <- c( 2, 2 )
  ps = c( 2, 2 )
  #filter_sizes <- nFilters*2^seq.int(0, 3)

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path
  outputs <- inputs %>%
    layer_conv_2d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_2d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )

  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_2d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_2d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_2d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_2d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = ps )

# intermediate path
  outputs <- outputs %>%
    layer_conv_2d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_2d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )

# Decoding path
  outputs <- outputs %>%
    layer_conv_2d_transpose( filters = nFilters , kernel_size = deconvolutionKernelSize, strides = 2, padding = 'same' )
  outputs <- list( skip3, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_2d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  ) %>%
    layer_conv_2d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )

  outputs <- outputs %>%
    layer_conv_2d_transpose( filters = nFilters , kernel_size = deconvolutionKernelSize, strides = 2, padding = 'same', activation = 'relu'  )
  outputs <- list( skip2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_2d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  ) %>%
    layer_conv_2d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  )

  outputs <- outputs %>%
    layer_conv_2d_transpose( filters = nFilters , kernel_size = deconvolutionKernelSize, strides = 2, padding = 'same', activation = 'relu'  )
  outputs <- list( skip1, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_2d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  ) %>%
    layer_conv_2d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  )

  convActivation <- ''
  if( numberOfOutputs == 1 ){convActivation <- 'sigmoid'} else {convActivation <- 'softmax'}

  outputs <- outputs %>%
    layer_conv_2d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  Model <- keras_model( inputs = inputs, outputs = outputs )

  return( Model )
}

#3D model
#create3DModel(inputImageSize)
#inputImageSize <- c (256, 256, 24, 1); create3DModel(inputImageSize) %>% plot_model_modi(width=4, height=1.2)

#2D model
#create2DModel(inputImageSize)
#inputImageSize <- c (256, 256, 1); create2DModel(inputImageSize) %>% plot_model_modi(width=4, height=1.2)


########################################################################
########################################################################
########################################################################
#This script from "ANTsRNet/R/createUnetModel.R"
#https://github.com/ANTsX/ANTsRNet/blob/master/R/createUnetModel.R
########################################################################

ANTsRNet_createUnetModel2D <- function( inputImageSize,
                               numberOfOutputs = 1,
                               numberOfLayers = 3,
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

ANTsRNet_createUnetModel3D <- function( inputImageSize,
                               numberOfOutputs = 1,
                               numberOfLayers = 3,
                               numberOfFiltersAtBaseLayer = 32,
                               convolutionKernelSize = c( 3, 3, 3 ),
                               deconvolutionKernelSize = c( 2, 2, 2 ),
                               poolSize = c( 2, 2, 2 ),
                               strides = c( 2, 2, 2 ),
                               dropoutRate = 0.0,
                               weightDecay = 0.0,
                               addAttentionGating = T,
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

########################################################################
########################################################################
createResNetModel3D <- function( inputImageSize,
                                 inputScalarsSize = 0,
                                 numberOfClassificationLabels = 1000,
                                 layers = 1:4,
                                 residualBlockSchedule = c( 3, 4, 6, 3 ),
                                 lowestResolution = 64,
                                 cardinality = 1,
                                 squeezeAndExcite = FALSE,
                                 mode = c( 'classification', 'regression' )
                               )
{

  addCommonLayers <- function( model )
    {
    model <- model %>% layer_batch_normalization()
    model <- model %>% layer_activation_leaky_relu()

    return( model )
    }

  groupedConvolutionLayer3D <- function( model, numberOfFilters, strides )
    {
    K <- keras::backend()

    # Per standard ResNet, this is just a 2-D convolution
    if( cardinality == 1 )
      {
      groupedModel <- model %>% layer_conv_3d( filters = numberOfFilters,
        kernel_size = c( 3, 3, 3 ), strides = strides, padding = 'same' )
      return( groupedModel )
      }

    if( numberOfFilters %% cardinality != 0 )
      {
      stop( "numberOfFilters %% cardinality != 0" )
      }

    numberOfGroupFilters <- as.integer( numberOfFilters / cardinality )

    convolutionLayers <- list()
    for( j in 1:cardinality )
      {
      convolutionLayers[[j]] <- model %>% layer_lambda( function( z )
        {
        K$set_image_data_format( 'channels_last' )
        z[,,,, ( ( j - 1 ) * numberOfGroupFilters + 1 ):( j * numberOfGroupFilters )]
        } )
      convolutionLayers[[j]] <- convolutionLayers[[j]] %>%
        layer_conv_3d( filters = numberOfGroupFilters,
          kernel_size = c( 3, 3, 3 ), strides = strides, padding = 'same' )
      }

    groupedModel <- layer_concatenate( convolutionLayers )
    return( groupedModel )
    }

  squeezeAndExciteBlock3D <- function( model, ratio = 16 )
    {
    K <- keras::backend()

    initial <- model
    numberOfFilters <- K$int_shape( initial )[[2]]
    if( K$image_data_format() == "channels_last" )
      {
      numberOfFilters <- K$int_shape( initial )[[5]]
      }
    blockShape <- c( 1, 1, 1, numberOfFilters )

    block <- initial %>% layer_global_average_pooling_3d()
    block <- block %>% layer_reshape( target_shape = blockShape )
    block <- block %>% layer_dense( units = as.integer( numberOfFilters / ratio ),
      activation = 'relu', kernel_initializer = 'he_normal', use_bias = FALSE )
    block <- block %>% layer_dense( units = numberOfFilters, activation = 'sigmoid',
      kernel_initializer = 'he_normal', use_bias = FALSE )

    if( K$image_data_format() == "channels_first" )
      {
      block <- block %>% layer_permute( c( 5, 2, 3, 4 ) )
      }
    x <- list( initial, block ) %>% layer_multiply()

    return( x )
    }

  residualBlock3D <- function( model, numberOfFiltersIn, numberOfFiltersOut,
    strides = c( 1, 1, 1 ), projectShortcut = FALSE, squeezeAndExcite = FALSE )
    {
    shortcut <- model

    model <- model %>% layer_conv_3d( filters = numberOfFiltersIn,
      kernel_size = c( 1, 1, 1 ), strides = c( 1, 1, 1 ), padding = 'same' )
    model <- addCommonLayers( model )

    # ResNeXt (identical to ResNet when `cardinality` == 1)
    model <- groupedConvolutionLayer3D( model, numberOfFilters = numberOfFiltersIn,
      strides = strides )
    model <- addCommonLayers( model )

    model <- model %>% layer_conv_3d( filters = numberOfFiltersOut,
      kernel_size = c( 1, 1, 1 ), strides = c( 1, 1, 1 ), padding = 'same' )
    model <- model %>% layer_batch_normalization()

    if( projectShortcut == TRUE || prod( strides == c( 1, 1, 1 ) ) == 0 )
      {
      shortcut <- shortcut %>% layer_conv_3d( filters = numberOfFiltersOut,
        kernel_size = c( 1, 1, 1 ), strides = strides, padding = 'same' )
      shortcut <- shortcut %>% layer_batch_normalization()
      }

    if( squeezeAndExcite == TRUE )
      {
      model <- squeezeAndExciteBlock3D( model )
      }

    model <- layer_add( list( shortcut, model ) )

    model <- model %>% layer_activation_leaky_relu()

    return( model )
    }

  mode <- match.arg( mode )
  inputImage <- layer_input( shape = inputImageSize )

  nFilters <- lowestResolution

  outputs <- inputImage %>% layer_conv_3d( filters = nFilters,
    kernel_size = c( 7, 7, 7 ), strides = c( 2, 2, 2 ), padding = 'same' )
  outputs <- addCommonLayers( outputs )
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 3, 3, 3 ),
    strides = c( 2, 2, 2 ), padding = 'same' )

  for( i in seq_len( length( layers ) ) )
    {
    nFiltersIn <- lowestResolution * 2 ^ ( layers[i] )
    nFiltersOut <- 2 * nFiltersIn
    for( j in seq_len( residualBlockSchedule[i] ) )
      {
      projectShortcut <- FALSE
      if( i == 1 && j == 1 )
        {
        projectShortcut <- TRUE
        }
      if( i > 1 && j == 1 )
        {
        strides <- c( 2, 2, 2 )
        } else {
        strides <- c( 1, 1, 1 )
        }
      outputs <- residualBlock3D( outputs, numberOfFiltersIn = nFiltersIn,
        numberOfFiltersOut = nFiltersOut, strides = strides,
        projectShortcut = projectShortcut, squeezeAndExcite = squeezeAndExcite )
      }
    }
  outputs <- outputs %>% layer_global_average_pooling_3d()

  layerActivation <- ''
  if( mode == 'classification' ) {
    layerActivation <- 'softmax'
    } else if( mode == 'regression' ) {
    layerActivation <- 'linear'
    } else {
    stop( 'Error: unrecognized mode.' )
    }

  resNetModel <- NULL
  if( inputScalarsSize > 0 )
    {
    inputScalars <- layer_input( shape = c( inputScalarsSize ) )
    concatenatedLayer <- layer_concatenate( list( outputs, inputScalars ) )
    outputs <- concatenatedLayer %>%
      layer_dense( units = numberOfClassificationLabels, activation = layerActivation )
    resNetModel <- keras_model( inputs = list( inputImage, inputScalars ),
                                outputs = outputs )
    } else {
    outputs <- outputs %>%
      layer_dense( units = numberOfClassificationLabels, activation = layerActivation )
    resNetModel <- keras_model( inputs = inputImage, outputs = outputs )
    }

  return( resNetModel )
}

########################################################################
########################################################################

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


createDenseUnetModel3D <- function( inputImageSize,
                                    numberOfOutputs = 1L,
                                    numberOfLayersPerDenseBlock = c( 3, 4, 12, 8 ),
                                    growthRate = 48,
                                    initialNumberOfFilters = 96,
                                    reductionRate = 0.0,
                                    depth = 2,
                                    dropoutRate = 0.0,
                                    weightDecay = 1e-4,
                                    mode = c( 'classification', 'regression' )
)
{
  K <- keras::backend()

  inputImageSize = as.integer( inputImageSize )
  mode <- match.arg( mode )

  concatenationAxis <- 1
  if( K$image_data_format() == 'channels_last' )
  {
    concatenationAxis <- -1
  }

  convolutionFactory3D <- function( model, numberOfFilters, kernelSize = c( 3L, 3L, 3L ),
                                    dropoutRate = 0.0, weightDecay = 1e-4 )
  {
    # Bottleneck layer
    concatenationAxis = as.integer(concatenationAxis)

    model <- model %>% layer_batch_normalization( axis = concatenationAxis )
    model <- model %>% layer_scale( axis = concatenationAxis )
    model <- model %>% layer_activation( activation = 'relu' )
    model <- model %>% layer_conv_3d( filters = as.integer(numberOfFilters * 4),
                                      kernel_size = c( 1L, 1L, 1L ), use_bias = FALSE )

    if( dropoutRate > 0.0 )
    {
      model <- model %>% layer_dropout( rate = dropoutRate )
    }

    # Convolution layer

    model <- model %>% layer_batch_normalization( axis = concatenationAxis,
                                                  epsilon = 1.1e-5 )
    model <- model %>% layer_scale( axis = concatenationAxis )
    model <- model %>% layer_activation( activation = 'relu' )
    model <- model %>% layer_zero_padding_3d( padding = c( 1L, 1L, 1L ) )
    model <- model %>% layer_conv_3d( filters = numberOfFilters,
                                      kernel_size = kernelSize,
                                      use_bias = FALSE )

    if( dropoutRate > 0.0 )
    {
      model <- model %>% layer_dropout( rate = dropoutRate )
    }

    return( model )
  }

  transition3D <- function( model, numberOfFilters, compressionRate = 1.0,
                            dropoutRate = 0.0, weightDecay = 1e-4 )
  {
    model <- model %>% layer_batch_normalization( axis = concatenationAxis,
                                                  gamma_regularizer = regularizer_l2( weightDecay ),
                                                  beta_regularizer = regularizer_l2( weightDecay ) )
    model <- model %>% layer_scale( axis = concatenationAxis )
    model <- model %>% layer_activation( activation = 'relu' )
    model <- model %>% layer_conv_3d( filters =
                                        as.integer( numberOfFilters * compressionRate ),
                                      kernel_size = c( 1L, 1L, 1L ), use_bias = FALSE )

    if( dropoutRate > 0.0 )
    {
      model <- model %>% layer_dropout( rate = dropoutRate )
    }

    model <- model %>% layer_average_pooling_3d( pool_size = c( 2L, 2L, 2L ),
                                                 strides = c( 2L, 2L, 2L ) )
    return( model )
  }

  createDenseBlocks3D <- function( model, numberOfFilters, depth, growthRate,
                                   dropoutRate = 0.0, weightDecay = 1e-4 )
  {
    denseBlockLayers <- list( model )
    for( i in seq_len( depth ) )
    {
      model <- convolutionFactory3D( model, numberOfFilters = growthRate,
                                     kernelSize = c( 3L, 3L, 3L ), dropoutRate = dropoutRate,
                                     weightDecay = weightDecay )
      denseBlockLayers[[i+1]] <- model
      model <- layer_concatenate( denseBlockLayers, axis = concatenationAxis )
      numberOfFilters <- numberOfFilters + growthRate
    }

    return( list( model = model, numberOfFilters = numberOfFilters ) )
  }

  if( ( depth - 4 ) %% 3 != 0 )
  {
    stop( "Depth must be equal to 3*N+4 where N is an integer." )
  }
  numberOfLayers = as.integer( ( depth - 4 ) / 3 )

  numberOfDenseBlocks <- length( numberOfLayersPerDenseBlock )

  inputs <- layer_input( shape = inputImageSize )

  boxLayers <- list()
  boxCount <- 1

  # Initial convolution

  outputs <- inputs %>% layer_zero_padding_3d( padding = c( 3L, 3L, 3L ) )
  outputs <- outputs %>% layer_conv_3d( filters = initialNumberOfFilters,
                                        kernel_size = c( 7L, 7L, 7L ),
                                        strides = c( 2L, 2L, 2L ), use_bias = FALSE )
  outputs <- outputs %>% layer_batch_normalization( epsilon = 1.1e-5,
                                                    axis = concatenationAxis )
  outputs <- outputs %>% layer_scale( axis = concatenationAxis )
  outputs <- outputs %>% layer_activation( activation = "relu" )

  boxLayers[[boxCount]] <- outputs
  boxCount <- boxCount + 1

  outputs <- outputs %>% layer_zero_padding_3d( padding = c( 1L, 1L, 1L ) )
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 3L, 3L, 3L ),
                                               strides = c( 2L, 2L, 2L ) )

  # Add dense blocks

  nFilters <- initialNumberOfFilters

  for( i in seq_len( numberOfDenseBlocks - 1 ) )
  {
    denseBlockLayer <- createDenseBlocks3D( outputs,
                                            numberOfFilters = nFilters, depth = numberOfLayersPerDenseBlock[i],
                                            growthRate = growthRate, dropoutRate = dropoutRate,
                                            weightDecay = weightDecay )
    outputs <- denseBlockLayer$model

    boxLayers[[boxCount]] <- outputs
    boxCount <- boxCount + 1

    outputs <- transition3D( outputs,
                             numberOfFilters = denseBlockLayer$numberOfFilters,
                             compressionRate = 1.0 - reductionRate,
                             dropoutRate = dropoutRate,
                             weightDecay = weightDecay )

    nFilters <- as.integer( denseBlockLayer$numberOfFilters * ( 1 - reductionRate ) )
  }

  denseBlockLayer <- createDenseBlocks3D( outputs, numberOfFilters = nFilters,
                                          depth = numberOfLayersPerDenseBlock[numberOfDenseBlocks],
                                          growthRate = growthRate, dropoutRate = dropoutRate,
                                          weightDecay = weightDecay )
  outputs <- denseBlockLayer$model
  nFilters <- denseBlockLayer$numberOfFilters

  outputs <- outputs %>% layer_batch_normalization( epsilon = 1.1e-5,
                                                    axis = concatenationAxis )
  outputs <- outputs %>% layer_scale( axis = concatenationAxis )
  outputs <- outputs %>% layer_activation( activation = "relu" )

  boxLayers[[boxCount]] <- outputs
  boxCount <- boxCount - 1

  localNumberOfFilters <- tail( unlist( K$int_shape( boxLayers[[boxCount+1]] ) ), 1 )
  localLayer <- boxLayers[[boxCount]] %>% layer_conv_3d( filters = localNumberOfFilters,
                                                         kernel_size = c( 1L, 1L, 1L ),
                                                         padding = 'same',
                                                         kernel_initializer = 'normal' )
  boxCount <- boxCount - 1

  for( i in seq_len( numberOfDenseBlocks - 1 ) )
  {
    upsamplingLayer <- outputs %>% layer_upsampling_3d( size = c( 2L, 2L, 2L ) )
    outputs <- layer_add( list( localLayer, upsamplingLayer ) )

    localLayer <- boxLayers[[boxCount]]
    boxCount <- boxCount - 1

    localNumberOfFilters <- tail( unlist( K$int_shape( boxLayers[[boxCount+1]] ) ), 1 )
    outputs <- outputs %>% layer_conv_3d( filters = localNumberOfFilters,
                                          kernel_size = c( 3L, 3L, 3L ), padding = 'same',
                                          kernel_initializer = 'normal' )

    if( i == numberOfDenseBlocks )
    {
      outputs <- outputs %>% layer_dropout( rate = 0.3 )
    }

    outputs <- outputs %>% layer_batch_normalization()
    outputs <- outputs %>% layer_activation( activation = "relu" )
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
                   kernel_size = c( 1L, 1L, 1L ), activation = convActivation,
                   kernel_initializer = 'normal' )

  denseUnetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( denseUnetModel )
}


createResNetModel2D <- function( inputImageSize,
                                 inputScalarsSize = 0,
                                 numberOfClassificationLabels = 1000,
                                 layers = 1:4,
                                 residualBlockSchedule = c( 3, 4, 6, 3 ),
                                 lowestResolution = 64,
                                 cardinality = 1,
                                 squeezeAndExcite = FALSE,
                                 mode = c( 'classification', 'regression' )
                               )
{

  addCommonLayers <- function( model )
    {
    model <- model %>% layer_batch_normalization()
    model <- model %>% layer_activation_leaky_relu()

    return( model )
    }

  groupedConvolutionLayer2D <- function( model, numberOfFilters, strides )
    {
    K <- keras::backend()

    # Per standard ResNet, this is just a 2-D convolution
    if( cardinality == 1 )
      {
      groupedModel <- model %>% layer_conv_2d( filters = numberOfFilters,
        kernel_size = c( 3, 3 ), strides = strides, padding = 'same' )
      return( groupedModel )
      }

    if( numberOfFilters %% cardinality != 0 )
      {
      stop( "numberOfFilters %% cardinality != 0" )
      }

    numberOfGroupFilters <- as.integer( numberOfFilters / cardinality )

    convolutionLayers <- list()
    for( j in 1:cardinality )
      {
      convolutionLayers[[j]] <- model %>% layer_lambda( function( z )
        {
        K$set_image_data_format( 'channels_last' )
        z[,,, ( ( j - 1 ) * numberOfGroupFilters + 1 ):( j * numberOfGroupFilters )]
        } )
      convolutionLayers[[j]] <- convolutionLayers[[j]] %>%
        layer_conv_2d( filters = numberOfGroupFilters,
          kernel_size = c( 3, 3 ), strides = strides, padding = 'same' )
      }

    groupedModel <- layer_concatenate( convolutionLayers )
    return( groupedModel )
    }

  squeezeAndExciteBlock2D <- function( model, ratio = 16 )
    {
    K <- keras::backend()

    initial <- model
    numberOfFilters <- K$int_shape( initial )[[2]]
    if( K$image_data_format() == "channels_last" )
      {
      numberOfFilters <- K$int_shape( initial )[[4]]
      }
    blockShape <- c( 1, 1, numberOfFilters )

    block <- initial %>% layer_global_average_pooling_2d()
    block <- block %>% layer_reshape( target_shape = blockShape )
    block <- block %>% layer_dense( units = as.integer( numberOfFilters / ratio ),
      activation = 'relu', kernel_initializer = 'he_normal', use_bias = FALSE )
    block <- block %>% layer_dense( units = numberOfFilters, activation = 'sigmoid',
      kernel_initializer = 'he_normal', use_bias = FALSE )

    if( K$image_data_format() == "channels_first" )
      {
      block <- block %>% layer_permute( c( 4, 2, 3 ) )
      }
    x <- list( initial, block ) %>% layer_multiply()

    return( x )
    }

  residualBlock2D <- function( model, numberOfFiltersIn, numberOfFiltersOut,
    strides = c( 1, 1 ), projectShortcut = FALSE, squeezeAndExcite = FALSE )
    {
    shortcut <- model

    model <- model %>% layer_conv_2d( filters = numberOfFiltersIn,
      kernel_size = c( 1, 1 ), strides = c( 1, 1 ), padding = 'same' )
    model <- addCommonLayers( model )

    # ResNeXt (identical to ResNet when `cardinality` == 1)
    model <- groupedConvolutionLayer2D( model, numberOfFilters = numberOfFiltersIn,
      strides = strides )
    model <- addCommonLayers( model )

    model <- model %>% layer_conv_2d( filters = numberOfFiltersOut,
      kernel_size = c( 1, 1 ), strides = c( 1, 1 ), padding = 'same' )
    model <- model %>% layer_batch_normalization()

    if( projectShortcut == TRUE || prod( strides == c( 1, 1 ) ) == 0 )
      {
      shortcut <- shortcut %>% layer_conv_2d( filters = numberOfFiltersOut,
        kernel_size = c( 1, 1 ), strides = strides, padding = 'same' )
      shortcut <- shortcut %>% layer_batch_normalization()
      }

    if( squeezeAndExcite == TRUE )
      {
      model <- squeezeAndExciteBlock2D( model )
      }

    model <- layer_add( list( shortcut, model ) )

    model <- model %>% layer_activation_leaky_relu()

    return( model )
    }
  mode <- match.arg( mode )

  inputImage <- layer_input( shape = inputImageSize )

  nFilters <- lowestResolution

  outputs <- inputImage %>% layer_conv_2d( filters = nFilters,
    kernel_size = c( 7, 7 ), strides = c( 2, 2 ), padding = 'same' )
  outputs <- addCommonLayers( outputs )
  outputs <- outputs %>% layer_max_pooling_2d( pool_size = c( 3, 3 ),
    strides = c( 2, 2 ), padding = 'same' )

  for( i in seq_len( length( layers ) ) )
    {
    nFiltersIn <- lowestResolution * 2 ^ ( layers[i] )
    nFiltersOut <- 2 * nFiltersIn
    for( j in seq_len( residualBlockSchedule[i] ) )
      {
      projectShortcut <- FALSE
      if( i == 1 && j == 1 )
        {
        projectShortcut <- TRUE
        }
      if( i > 1 && j == 1 )
        {
        strides <- c( 2, 2 )
        } else {
        strides <- c( 1, 1 )
        }
      outputs <- residualBlock2D( outputs, numberOfFiltersIn = nFiltersIn,
        numberOfFiltersOut = nFiltersOut, strides = strides,
        projectShortcut = projectShortcut, squeezeAndExcite = squeezeAndExcite )
      }
    }
  outputs <- outputs %>% layer_global_average_pooling_2d()

  layerActivation <- ''
  if( mode == 'classification' ) {
    layerActivation <- 'softmax'
    } else if( mode == 'regression' ) {
    layerActivation <- 'linear'
    } else {
    stop( 'Error: unrecognized mode.' )
    }

  resNetModel <- NULL
  if( inputScalarsSize > 0 )
    {
    inputScalars <- layer_input( shape = c( inputScalarsSize ) )
    concatenatedLayer <- layer_concatenate( list( outputs, inputScalars ) )
    outputs <- concatenatedLayer %>%
      layer_dense( units = numberOfClassificationLabels, activation = layerActivation )
    resNetModel <- keras_model( inputs = list( inputImage, inputScalars ),
                                outputs = outputs )
    } else {
    outputs <- outputs %>%
      layer_dense( units = numberOfClassificationLabels, activation = layerActivation )
    resNetModel <- keras_model( inputs = inputImage, outputs = outputs )
    }

  return( resNetModel )
}



createNoBrainerUnetModel3D <- function( inputImageSize )
{

  numberOfOutputs <- 1
  numberOfFiltersAtBaseLayer <- 16
  convolutionKernelSize <- c( 3, 3, 3 )
  deconvolutionKernelSize <- c( 2, 2, 2 )

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path

  outputs <- inputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 2,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 2, 2, 2 ) )

  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 2,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 4,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 2, 2, 2 ) )

  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 4,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 8,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 2, 2, 2 ) )

  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 8,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 16,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()

  # Decoding path

  outputs <- outputs %>% layer_conv_3d_transpose(
    filters = numberOfFiltersAtBaseLayer * 16,
    kernel_size = deconvolutionKernelSize, strides = 2, padding = 'same' )

  outputs <- list( skip3, outputs ) %>% layer_concatenate()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 8,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 8,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()

  outputs <- outputs %>% layer_conv_3d_transpose(
    filters = numberOfFiltersAtBaseLayer * 8,
    kernel_size = deconvolutionKernelSize, strides = 2, padding = 'same' )

  outputs <- list( skip2, outputs ) %>% layer_concatenate()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 4,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 4,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()

  outputs <- outputs %>% layer_conv_3d_transpose(
    filters = numberOfFiltersAtBaseLayer * 4,
    kernel_size = deconvolutionKernelSize, strides = 2, padding = 'same' )

  outputs <- list( skip1, outputs ) %>% layer_concatenate()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 2,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()
  outputs <- outputs %>% layer_conv_3d( filters = numberOfFiltersAtBaseLayer * 2,
    kernel_size = convolutionKernelSize, padding = 'same' )
  outputs <- outputs %>% layer_activation_relu()

  convActivation <- ''
  if( numberOfOutputs == 1 )
    {
    convActivation <- 'sigmoid'
    } else {
    convActivation <- 'softmax'
    }

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs,
      kernel_size = 1, activation = convActivation )

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}


createHippMapp3rUnetModel3D <- function( inputImageSize,
                                         doFirstNetwork = TRUE )
{
  layer_convB_3d <- function( input, numberOfFilters, kernelSize = 3, strides = 1 )
    {
    block <- input %>% layer_conv_3d( filters = numberOfFilters,
      kernel_size = kernelSize, strides = strides, padding = 'same' )
    block <- block %>% layer_instance_normalization( axis = 5 )
    block <- block %>% layer_activation_leaky_relu()

    return( block )
    }

  residualBlock3D <- function( input, numberOfFilters )
    {
    block <- layer_convB_3d( input, numberOfFilters )
    block <- block %>% layer_spatial_dropout_3d( rate = 0.3 )
    block <- layer_convB_3d( block, numberOfFilters )

    return( block )
    }

  upsampleBlock3D <- function( input, numberOfFilters )
    {
    block <- input %>% layer_upsampling_3d()
    block <- layer_convB_3d( block, numberOfFilters )

    return( block )
    }

  featureBlock3D <- function( input, numberOfFilters )
    {
    block <- layer_convB_3d( input, numberOfFilters )
    block <- layer_convB_3d( block, numberOfFilters, kernelSize = 1 )

    return( block )
    }

  numberOfFiltersAtBaseLayer <- 16

  numberOfLayers <- 6
  if( doFirstNetwork == FALSE )
    {
    numberOfLayers <- 5
    }

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path

  add <- NULL

  encodingConvolutionLayers <- list()
  for( i in seq_len( numberOfLayers ) )
    {
    numberOfFilters = numberOfFiltersAtBaseLayer * 2 ^ ( i - 1 )

    conv <- NULL
    if( i == 1 )
      {
      conv <- layer_convB_3d( inputs, numberOfFilters )
      } else {
      conv <- layer_convB_3d( add, numberOfFilters, strides = 2 )
      }
    residualBlock <- residualBlock3D( conv, numberOfFilters )
    add <- list( conv, residualBlock ) %>% layer_add()

    encodingConvolutionLayers[[i]] <- add
    }

  # Decoding path

  outputs <- unlist( tail( encodingConvolutionLayers, 1 ) )[[1]]

  # 256
  numberOfFilters <- numberOfFiltersAtBaseLayer * 2 ^ ( numberOfLayers - 2 )
  outputs <- upsampleBlock3D( outputs, numberOfFilters )

  if( doFirstNetwork == TRUE )
    {
    # 256, 128
    outputs <- list( encodingConvolutionLayers[[5]], outputs ) %>%
      layer_concatenate()
    outputs <- featureBlock3D( outputs, numberOfFilters )
    numberOfFilters <- numberOfFilters / 2
    outputs <- upsampleBlock3D( outputs, numberOfFilters )
    }

  # 128, 64
  outputs <- list( encodingConvolutionLayers[[4]], outputs ) %>%
    layer_concatenate()
  outputs <- featureBlock3D( outputs, numberOfFilters )
  numberOfFilters <- numberOfFilters / 2
  outputs <- upsampleBlock3D( outputs, numberOfFilters )

  # 64, 32
  outputs <- list( encodingConvolutionLayers[[3]], outputs ) %>%
    layer_concatenate()
  feature64 <- featureBlock3D( outputs, numberOfFilters )
  numberOfFilters <- numberOfFilters / 2
  outputs <- upsampleBlock3D( feature64, numberOfFilters )
  back64 <- NULL
  if( doFirstNetwork == TRUE )
    {
    back64 <- layer_convB_3d( feature64, 1, 1 )
    } else {
    back64 <- feature64 %>% layer_conv_3d( filters = 1, kernel_size = 1 )
    }
  back64 <- back64 %>% layer_upsampling_3d()

  # 32, 16
  outputs <- list( encodingConvolutionLayers[[2]], outputs ) %>%
    layer_concatenate()
  feature32 <- featureBlock3D( outputs, numberOfFilters )
  numberOfFilters <- numberOfFilters / 2
  outputs <- upsampleBlock3D( feature32, numberOfFilters )
  back32 <- NULL
  if( doFirstNetwork == TRUE )
    {
    back32 <- layer_convB_3d( feature32, 1, 1 )
    } else {
    back32 <- feature32 %>% layer_conv_3d( filters = 1, kernel_size = 1 )
    }
  back32 <- list( back64, back32 ) %>% layer_add()
  back32 <- back32 %>% layer_upsampling_3d()

  # final
  outputs <- list( encodingConvolutionLayers[[1]], outputs ) %>%
    layer_concatenate()
  outputs <- layer_convB_3d( outputs, numberOfFilters, 3 )
  outputs <- layer_convB_3d( outputs, numberOfFilters, 1 )
  if( doFirstNetwork == TRUE )
    {
    outputs <- layer_convB_3d( outputs, 1, 1 )
    } else {
    outputs <- outputs %>% layer_conv_3d( filters = 1, kernel_size = 1 )
    }
  outputs <- list( back32, outputs ) %>% layer_add()
  outputs <- outputs %>% layer_activation( 'sigmoid' )

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}

createWideResNetModel2D <- function( inputImageSize,
                                     numberOfClassificationLabels = 1000,
                                     depth = 2,
                                     width = 1,
                                     residualBlockSchedule = c( 16, 32, 64 ),
                                     poolSize = c( 8, 8 ),
                                     dropoutRate = 0.0,
                                     weightDecay = 0.0005,
                                     mode = 'classification'
                                   )
{
  mode <- match.arg( mode )

  channelAxis <- 1
  if( keras::backend()$image_data_format() == "channels_last" )
    {
    channelAxis <- -1
    }

  initialConvolutionLayer <- function( model, numberOfFilters )
    {
    model <- model %>% layer_conv_2d( filters = numberOfFilters,
      kernel_size = c( 3, 3 ), padding = 'same',
      kernel_initializer = initializer_he_normal(),
      kernel_regularizer = regularizer_l2( weightDecay ) )

    model <- model %>% layer_batch_normalization( axis = channelAxis, momentum = 0.1,
      epsilon = 1.0e-5, gamma_initializer = "uniform" )
    model <- model %>% layer_activation( activation = "relu" )

    return( model )
    }

  customConvolutionLayer <- function( initialModel, base, width, strides = c( 1, 1 ),
    dropoutRate = 0.0, expand = TRUE )
    {
    numberOfFilters <- as.integer( base * width )

    if( expand == TRUE )
      {
      model <- initialModel %>% layer_conv_2d( filters = numberOfFilters, kernel_size = c( 3, 3 ),
        padding = 'same', strides = strides, kernel_initializer = initializer_he_normal(),
        kernel_regularizer = regularizer_l2( weightDecay ), use_bias = FALSE )
      } else {
      model <- initialModel
      }

    model <- model %>% layer_batch_normalization( axis = channelAxis, momentum = 0.1,
      epsilon = 1.0e-5, gamma_initializer = "uniform" )
    model <- model %>% layer_activation( activation = "relu" )

    model <- model %>% layer_conv_2d( filters = numberOfFilters, kernel_size = c( 3, 3 ),
      padding = 'same', kernel_initializer = initializer_he_normal(),
      kernel_regularizer = regularizer_l2( weightDecay ), use_bias = FALSE )

    if( expand == TRUE )
      {
      skipLayer <- initialModel %>% layer_conv_2d( filters = numberOfFilters,
        kernel_size = c( 1, 1 ), padding = 'same', strides = strides,
        kernel_initializer = initializer_he_normal(),
        kernel_regularizer = regularizer_l2( weightDecay ), use_bias = FALSE )

      model <- layer_add( list( model, skipLayer ) )
      } else {
      if( dropoutRate > 0.0 )
        {
        model <- model %>% layer_dropout( rate = dropoutRate )
        }

      model <- model %>% layer_batch_normalization( axis = channelAxis, momentum = 0.1,
        epsilon = 1.0e-5, gamma_initializer = "uniform" )
      model <- model %>% layer_activation( activation = "relu" )

      model <- model %>% layer_conv_2d( filters = numberOfFilters, kernel_size = c( 3, 3 ),
        padding = 'same', kernel_initializer = initializer_he_normal(),
        kernel_regularizer = regularizer_l2( weightDecay ), use_bias = FALSE )

      model <- layer_add( list( initialModel, model ) )
      }

    return( model )
    }

  inputs <- layer_input( shape = inputImageSize )

  outputs <- initialConvolutionLayer( inputs, residualBlockSchedule[1] )
  numberOfConvolutions <- 4

  for( i in seq_len( length( residualBlockSchedule ) ) )
    {
    baseNumberOfFilters <- residualBlockSchedule[i]

    outputs <- customConvolutionLayer( outputs, base = baseNumberOfFilters, width = width,
      strides = c( 1, 1 ), dropoutRate = 0.0, expand = TRUE )
    numberOfConvolutions <- numberOfConvolutions + 2

    for( j in seq_len( depth ) )
      {
      outputs <- customConvolutionLayer( outputs, base = baseNumberOfFilters,
        width = width, dropoutRate = dropoutRate, expand = FALSE )
      numberOfConvolutions <- numberOfConvolutions + 2
      }

    outputs <- outputs %>% layer_batch_normalization( axis = channelAxis, momentum = 0.1,
      epsilon = 1.0e-5, gamma_initializer = "uniform" )
    outputs <- outputs %>% layer_activation( activation = "relu" )
    }

  outputs <- outputs %>% layer_average_pooling_2d( pool_size = poolSize )
  outputs <- outputs %>% layer_flatten()

  layerActivation <- ''
  if( mode == 'classification' ) {
    layerActivation <- 'softmax'
    } else if( mode == 'regression' ) {
    layerActivation <- 'linear'
    } else {
    stop( 'Error: unrecognized mode.' )
    }

  outputs <- outputs %>% layer_dense( units = numberOfClassificationLabels,
    kernel_regularizer = regularizer_l2( weightDecay ), activation = layerActivation )

  wideResNetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( wideResNetModel )
}

createWideResNetModel3D <- function( inputImageSize,
                                     numberOfClassificationLabels = 1000,
                                     depth = 2,
                                     width = 1,
                                     residualBlockSchedule = c( 16, 32, 64 ),
                                     poolSize = c( 8, 8, 8 ),
                                     dropoutRate = 0.0,
                                     weightDecay = 0.0005,
                                     mode = c( 'classification', 'regression' )
                                   )
{

  mode <- match.arg( mode )

  channelAxis <- 1
  if( keras::backend()$image_data_format() == "channels_last" )
    {
    channelAxis <- -1
    }

  initialConvolutionLayer <- function( model, numberOfFilters )
    {
    model <- model %>% layer_conv_3d( filters = numberOfFilters,
      kernel_size = c( 3, 3, 3 ), padding = 'same',
      kernel_initializer = initializer_he_normal(),
      kernel_regularizer = regularizer_l2( weightDecay ) )

    model <- model %>% layer_batch_normalization( axis = channelAxis, momentum = 0.1,
      epsilon = 1.0e-5, gamma_initializer = "uniform" )
    model <- model %>% layer_activation( activation = "relu" )

    return( model )
    }

  customConvolutionLayer <- function( initialModel, base, width, strides = c( 1, 1, 1 ),
    dropoutRate = 0.0, expand = TRUE )
    {
    numberOfFilters <- as.integer( base * width )

    if( expand == TRUE )
      {
      model <- initialModel %>% layer_conv_3d( filters = numberOfFilters, kernel_size = c( 3, 3, 3 ),
        padding = 'same', strides = strides, kernel_initializer = initializer_he_normal(),
        kernel_regularizer = regularizer_l2( weightDecay ), use_bias = FALSE )
      } else {
      model <- initialModel
      }

    model <- model %>% layer_batch_normalization( axis = channelAxis, momentum = 0.1,
      epsilon = 1.0e-5, gamma_initializer = "uniform" )
    model <- model %>% layer_activation( activation = "relu" )

    model <- model %>% layer_conv_3d( filters = numberOfFilters, kernel_size = c( 3, 3, 3 ),
      padding = 'same', kernel_initializer = initializer_he_normal(),
      kernel_regularizer = regularizer_l2( weightDecay ), use_bias = FALSE )

    if( expand == TRUE )
      {
      skipLayer <- initialModel %>% layer_conv_3d( filters = numberOfFilters,
        kernel_size = c( 1, 1, 1 ), padding = 'same', strides = strides,
        kernel_initializer = initializer_he_normal(),
        kernel_regularizer = regularizer_l2( weightDecay ), use_bias = FALSE )

      model <- layer_add( list( model, skipLayer ) )
      } else {
      if( dropoutRate > 0.0 )
        {
        model <- model %>% layer_dropout( rate = dropoutRate )
        }

      model <- model %>% layer_batch_normalization( axis = channelAxis, momentum = 0.1,
        epsilon = 1.0e-5, gamma_initializer = "uniform" )
      model <- model %>% layer_activation( activation = "relu" )

      model <- model %>% layer_conv_3d( filters = numberOfFilters, kernel_size = c( 3, 3, 3 ),
        padding = 'same', kernel_initializer = initializer_he_normal(),
        kernel_regularizer = regularizer_l2( weightDecay ), use_bias = FALSE )

      model <- layer_add( list( initialModel, model ) )
      }

    return( model )
    }

  inputs <- layer_input( shape = inputImageSize )

  outputs <- initialConvolutionLayer( inputs, residualBlockSchedule[1] )
  numberOfConvolutions <- 4

  for( i in seq_len( length( residualBlockSchedule ) ) )
    {
    baseNumberOfFilters <- residualBlockSchedule[i]

    outputs <- customConvolutionLayer( outputs, base = baseNumberOfFilters, width = width,
      strides = c( 1, 1, 1 ), dropoutRate = 0.0, expand = TRUE )
    numberOfConvolutions <- numberOfConvolutions + 2

    for( j in seq_len( depth ) )
      {
      outputs <- customConvolutionLayer( outputs, base = baseNumberOfFilters,
        width = width, dropoutRate = dropoutRate, expand = FALSE )
      numberOfConvolutions <- numberOfConvolutions + 2
      }

    outputs <- outputs %>% layer_batch_normalization( axis = channelAxis, momentum = 0.1,
      epsilon = 1.0e-5, gamma_initializer = "uniform" )
    outputs <- outputs %>% layer_activation( activation = "relu" )
    }

  outputs <- outputs %>% layer_average_pooling_3d( pool_size = poolSize )
  outputs <- outputs %>% layer_flatten()

  layerActivation <- ''
  if( mode == 'classification' ) {
    layerActivation <- 'softmax'
    } else if( mode == 'regression' ) {
    layerActivation <- 'linear'
    } else {
    stop( 'Error: unrecognized mode.' )
    }

  outputs <- outputs %>% layer_dense( units = numberOfClassificationLabels,
    kernel_regularizer = regularizer_l2( weightDecay ), activation = layerActivation )

  wideResNetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( wideResNetModel )
}


CycleGanModel <- R6::R6Class( "CycleGanModel",

  inherit = NULL,

  lock_objects = FALSE,

  public = list(

    dimensionality = 2,

    inputImageSize = c( 128, 128, 3 ),

    numberOfChannels = 3,

    lambdaCycleLossWeight = 10.0,

    lambdaIdentityLossWeight = 1.0,

    numberOfFiltersAtBaseLayer = c( 32, 64 ),

    initialize = function( inputImageSize,
      lambdaCycleLossWeight = 10.0, lambdaIdentityLossWeight = 1.0,
      numberOfFiltersAtBaseLayer = c( 32, 64 ) )
      {
      self$inputImageSize <- inputImageSize
      self$numberOfChannels <- tail( self$inputImageSize, 1 )

      self$discriminatorPatchSize <- NULL

      self$dimensionality <- NA
      if( length( self$inputImageSize ) == 3 )
        {
        self$dimensionality <- 2
        } else if( length( self$inputImageSize ) == 4 ) {
        self$dimensionality <- 3
        } else {
        stop( "Incorrect size for inputImageSize.\n" )
        }

      optimizer <- optimizer_adam( lr = 0.0002, beta_1 = 0.5 )

      # Build discriminators for domains A and B

      self$discriminatorA <- self$buildDiscriminator()
      self$discriminatorA$compile( loss = 'mse',
        optimizer = optimizer, metrics = list( 'acc' ) )
      self$discriminatorA$trainable <- FALSE

      self$discriminatorB <- self$buildDiscriminator()
      self$discriminatorB$compile( loss = 'mse',
        optimizer = optimizer, metrics = list( 'acc' ) )
      self$discriminatorB$trainable <- FALSE

      # Build U-net like generators

      self$generatorAtoB <- self$buildGenerator()
      self$generatorBtoA <- self$buildGenerator()

      # Images

      imageA <- layer_input( shape = self$inputImageSize )
      imageB <- layer_input( shape = self$inputImageSize )

      fakeImageB <- self$generatorAtoB( imageA )
      fakeImageA <- self$generatorBtoA( imageB )

      reconstructedImageA <- self$generatorBtoA( fakeImageB )
      reconstructedImageB <- self$generatorAtoB( fakeImageA )

      identityImageA <- self$generatorBtoA( imageA )
      identityImageB <- self$generatorAtoB( imageB )

      # Check the images

      validityA <- self$discriminatorA( fakeImageA )
      validityB <- self$discriminatorB( fakeImageB )

      # Combined model

      self$combinedModel <- keras_model( inputs = list( imageA, imageB ),
        outputs = list( validityA, validityB, reconstructedImageA,
          reconstructedImageB, identityImageA, identityImageB  ) )
      self$combinedModel$compile( loss = list( 'mse', 'mse', 'mae', 'mae',
        'mae', 'mae' ), loss_weights = c( 1.0, 1.0,
          self$lambdaCycleLossWeight, self$lambdaCycleLossWeight,
          self$lambdaIdentityLossWeight, self$lambdaIdentityLossWeight ),
        optimizer = optimizer )
      },

    buildGenerator = function()
      {
      buildEncodingLayer <- function( input, numberOfFilters, kernelSize = 4 )
        {
        encoder <- input
        if( self$dimensionality == 2 )
          {
          encoder <- encoder %>% layer_conv_2d( numberOfFilters,
             kernel_size = kernelSize, strides = 2, padding = 'same' )
          } else {
          encoder <- encoder %>% layer_conv_3d( numberOfFilters,
             kernel_size = kernelSize, strides = 2, padding = 'same' )
          }
        encoder <- encoder %>% layer_activation_leaky_relu( alpha = 0.2 )
        encoder <- encoder %>% layer_instance_normalization()
        return( encoder )
        }

      buildDecodingLayer <- function( input, skipInput, numberOfFilters,
        kernelSize = 4, dropoutRate = 0 )
        {
        decoder <- input
        if( self$dimensionality == 2 )
          {
          decoder <- decoder %>% layer_upsampling_2d( size = 2 )
          decoder <- decoder %>% layer_conv_2d( numberOfFilters,
             kernel_size = kernelSize, strides = 1, padding = 'same',
            activation = 'relu' )
          } else {
          decoder <- decoder %>% layer_upsampling_3d( size = 2 )
          decoder <- decoder %>% layer_conv_3d( numberOfFilters,
             kernel_size = kernelSize, strides = 1, padding = 'same',
            activation = 'relu' )
          }
        if( dropoutRate > 0.0 )
          {
          decoder <- decoder %>% layer_dropout( rate = dropoutRate )
          }
        decoder <- decoder %>% layer_instance_normalization()
        decoder <- list( decoder, skipInput ) %>% layer_concatenate()
        return( decoder )
        }

      input <- layer_input( shape = self$inputImageSize )

      encodingLayers <- list()

      encodingLayers[[1]] <- buildEncodingLayer( input,
        self$numberOfFiltersAtBaseLayer[1], kernelSize = 4 )
      encodingLayers[[2]] <- buildEncodingLayer( encodingLayers[[1]],
        self$numberOfFiltersAtBaseLayer[1] * 2, kernelSize = 4 )
      encodingLayers[[3]] <- buildEncodingLayer( encodingLayers[[2]],
        self$numberOfFiltersAtBaseLayer[1] * 4, kernelSize = 4 )
      encodingLayers[[4]] <- buildEncodingLayer( encodingLayers[[3]],
        self$numberOfFiltersAtBaseLayer[1] * 8, kernelSize = 4 )

      decodingLayers <- list()
      decodingLayers[[1]] <- buildDecodingLayer( encodingLayers[[4]],
        encodingLayers[[3]], self$numberOfFiltersAtBaseLayer[1] * 4 )
      decodingLayers[[2]] <- buildDecodingLayer( decodingLayers[[1]],
        encodingLayers[[2]], self$numberOfFiltersAtBaseLayer[1] * 2 )
      decodingLayers[[3]] <- buildDecodingLayer( decodingLayers[[2]],
        encodingLayers[[1]], self$numberOfFiltersAtBaseLayer[1] )

      if( self$dimensionality == 2 )
        {
        decodingLayers[[4]] <- decodingLayers[[3]] %>%
          layer_upsampling_2d( size = 2 )
        decodingLayers[[4]] <- decodingLayers[[4]] %>%
          layer_conv_2d( self$numberOfChannels,
           kernel_size = 4, strides = 1, padding = 'same',
          activation = 'tanh' )
        } else {
        decodingLayers[[4]] <- decodingLayers[[3]] %>%
          layer_upsampling_3d( size = 2 )
        decodingLayers[[4]] <- decodingLayers[[4]] %>%
          layer_conv_3d( self$numberOfChannels,
           kernel_size = 4, strides = 1, padding = 'same',
          activation = 'tanh' )
        }

      generator <- keras_model( inputs = input, outputs = decodingLayers[[4]] )

      return( generator )
      },

    buildDiscriminator = function()
      {
      buildLayer <- function( input, numberOfFilters, kernelSize = 4,
        normalization = TRUE )
        {
        layer <- input
        if( self$dimensionality == 2 )
          {
          layer <- layer %>% layer_conv_2d( numberOfFilters,
             kernel_size = kernelSize, strides = 2, padding = 'same' )
          } else {
          layer <- layer %>% layer_conv_3d( numberOfFilters,
             kernel_size = kernelSize, strides = 2, padding = 'same' )
          }
        layer <- layer %>% layer_activation_leaky_relu( alpha = 0.2 )
        if( normalization == TRUE )
          {
          layer <- layer %>% layer_instance_normalization()
          }
        return( layer )
        }

      image <- layer_input( shape = c( self$inputImageSize ) )

      layers <- list()
      layers[[1]] <- image %>% buildLayer( self$numberOfFiltersAtBaseLayer[2],
        normalization = FALSE )
      layers[[2]] <- layers[[1]] %>%
        buildLayer( self$numberOfFiltersAtBaseLayer[2] * 2 )
      layers[[3]] <- layers[[2]] %>%
        buildLayer( self$numberOfFiltersAtBaseLayer[2] * 4 )
      layers[[4]] <- layers[[3]] %>%
        buildLayer( self$numberOfFiltersAtBaseLayer[2] * 8 )

      validity <- NA
      if( self$dimensionality == 2 )
        {
        validity <- layers[[4]] %>%
          layer_conv_2d( 1,  kernel_size = 4, strides = 1, padding = 'same')
        } else {
        validity <- layers[[4]] %>%
          layer_conv_3d( 1,  kernel_size = 4, strides = 1, padding = 'same')
        }

      if( is.null( self$discriminatorPatchSize ) )
        {
        K <- keras::backend()
        self$discriminatorPatchSize <- unlist( K$int_shape( validity ) )
        }

      discriminator <- keras_model( inputs = image, outputs = validity )

      return( discriminator )
      },

    train = function( X_trainA, X_trainB, numberOfEpochs, batchSize = 128,
      sampleInterval = NA, sampleFilePrefix = 'sample' )
      {
      valid <- array( data = 1, dim = c( batchSize, self$discriminatorPatchSize ) )
      fake <- array( data = 0, dim = c( batchSize, self$discriminatorPatchSize ) )

      for( epoch in seq_len( numberOfEpochs ) )
        {
        indicesA <- sample.int( dim( X_trainA )[1], batchSize )
        indicesB <- sample.int( dim( X_trainB )[1], batchSize )

        imagesA <- NULL
        imagesB <- NULL
        if( self$dimensionality == 2 )
          {
          imagesA <- X_trainA[indicesA,,,, drop = FALSE]
          imagesB <- X_trainB[indicesB,,,, drop = FALSE]
          } else {
          imagesA <- X_trainA[indicesA,,,,, drop = FALSE]
          imagesB <- X_trainB[indicesB,,,,, drop = FALSE]
          }

        # train discriminator

        fakeImagesB <- self$generatorAtoB$predict( imagesA )
        fakeImagesA <- self$generatorBtoA$predict( imagesB )

        dALossReal <- self$discriminatorA$train_on_batch( imagesA, valid )
        dALossFake <- self$discriminatorA$train_on_batch( fakeImagesA, fake )

        dBLossReal <- self$discriminatorB$train_on_batch( imagesB, valid )
        dBLossFake <- self$discriminatorB$train_on_batch( fakeImagesB, fake )

        dLoss <- list()
        for( i in seq_len( length( dALossReal ) ) )
          {
          dLoss[[i]] <- 0.25 * ( dALossReal[[i]] + dALossFake[[i]] +
            dBLossReal[[i]] + dBLossFake[[i]] )
          }

        # train generator

        gLoss <- self$combinedModel$train_on_batch( list( imagesA, imagesB ),
          list( valid, valid, imagesA, imagesB, imagesA, imagesB ) )

        cat( "Epoch ", epoch, ": [Discriminator loss: ", dLoss[[1]],
             " acc: ", dLoss[[2]], "] ", "[Generator loss: ", gLoss[[1]], ", ",
             mean( unlist( gLoss )[2:3] ), ", ", mean( unlist( gLoss )[4:5] ),
             ", ", mean( unlist( gLoss )[6] ), "]\n",
             sep = '' )

        if( self$dimensionality == 2 )
          {
          if( ! is.na( sampleInterval ) )
            {
            if( ( ( epoch - 1 ) %% sampleInterval ) == 0 )
              {
              # Do a 2x3 grid
              #
              # imageA  |  translated( imageA ) | reconstructed( imageA )
              # imageB  |  translated( imageB ) | reconstructed( imageB )

              indexA <- sample.int( dim( X_trainA )[1], 1 )
              imageA <- X_trainA[indexA,,,, drop = FALSE]

              indexB <- sample.int( dim( X_trainB )[1], 1 )
              imageB <- X_trainB[indexB,,,, drop = FALSE]

              X <- list()
              X[[1]] <- imageA
              X[[2]] <- self$generatorAtoB$predict( X[[1]] )
              X[[3]] <- self$generatorBtoA$predict( X[[2]] )

              X[[4]] <- imageB
              X[[5]] <- self$generatorBtoA$predict( X[[4]] )
              X[[6]] <- self$generatorAtoB$predict( X[[5]] )

              for( i in seq_len( length( X ) ) )
                {
                X[[i]] <- ( X[[i]] - min( X[[i]] ) ) /
                  ( max( X[[i]] ) - min( X[[i]] ) )
                X[[i]] <- drop( X[[i]] )
                }
              XrowA <- image_append(
                         c( image_read( X[[1]] ),
                            image_read( X[[2]] ),
                            image_read( X[[3]] ) ) )
              XrowB <- image_append(
                         c( image_read( X[[4]] ),
                            image_read( X[[5]] ),
                            image_read( X[[6]] ) ) )
              XAB <- image_append( c( XrowA, XrowB ), stack = TRUE )

              sampleDir <- dirname( sampleFilePrefix )
              if( ! dir.exists( sampleDir ) )
                {
                dir.create( sampleDir, showWarnings = TRUE, recursive = TRUE )
                }

              imageFileName <- paste0( sampleFilePrefix, "_iteration" , epoch, ".jpg" )
              cat( "   --> writing sample image: ", imageFileName, "\n" )
              image_write( XAB, path = imageFileName, format = "jpg")
              }
            }
          }
        }
      }
    )
  )



