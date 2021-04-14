##################################################
## create 3D U-NET
##################################################
#library(keras)
create3DModel <- function( inputImageSize3d , nFilters=2){
  numberOfOutputs <- 1
  nFilters <- nFilters
  convolutionKernelSize <- c( 3, 3, 3 )
  deconvolutionKernelSize <- c( 2, 2, 2 )
  ps = c( 2, 2, 2 )

  inputs <- layer_input( shape = inputImageSize3d )

  # Encoding path
  outputs <- inputs %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )

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

# intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )

# Decoding path
  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = nFilters , kernel_size = deconvolutionKernelSize, strides = 2, padding = 'same' )
  outputs <- list( skip3, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  ) %>%
    layer_conv_3d( filters = nFilters , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )

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

create3DModel( inputImageSize3d=c(256, 256, 24,1) )
##################################################
## create 2D U-NET
##################################################
create2DModel <- function( inputImageSize2d ){
  numberOfOutputs <- 1
  nFilters <- 2
  convolutionKernelSize <- c( 3, 3 )
  deconvolutionKernelSize <- c( 2, 2 )
  ps = c( 2, 2 )
  #filter_sizes <- nFilters*2^seq.int(0, 3)

  inputs <- layer_input( shape = inputImageSize2d )

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

  unetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( unetModel )
}

#3D model
#create3DModel(inputImageSize3d)
#inputImageSize3d <- c (256, 256, 24, 1); create3DModel(inputImageSize3d) %>% plot_model_modi(width=4, height=1.2)

#2D model
#create2DModel(inputImageSize2d)
#inputImageSize2d <- c (256, 256, 1); create2DModel(inputImageSize2d) %>% plot_model_modi(width=4, height=1.2)
