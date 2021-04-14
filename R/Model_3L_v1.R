#library(keras)
#py_plot_model <- reticulate::import(module = "tensorflow")$keras$utils$plot_model
#py_plot_model(create3DModel_3L_v3( inputImageSize=c(256, 256, 64,1)), to_file=paste0('Model_', formatC(length(list.files(pattern = "Model_")), width = 3, flag = "0"), '.png'), show_shapes=T, show_layer_names=T)
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

############################################################################################
############################################################################################
create3DModel_3L_v3 <- function( inputImageSize ){
  numberOfOutputs <- 1
  convolutionKernelSize <- c( 3, 3, 3 )
  deconvolutionKernelSize <- c( 2, 2, 2 )
  N <- 2
  ps <-  c( 2, 2, 2 )
  strides <-  c( 2, 2, 2 )

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path
  outputs <- inputs %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()

  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()

  skip1_i01 <- skip1 %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()

skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()

  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  # intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()

  # Decoding path
  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = N*8 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same' )
  outputs <- list( skip3, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = N*8, kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = N*4 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip1_i01, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()

  if( numberOfOutputs == 1 ){convActivation <- 'sigmoid'} else {convActivation <- 'softmax'}

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  skip1_i01_out <- skip1_i01 %>%
    layer_conv_3d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  Model <- keras_model( inputs = inputs, outputs = c(outputs, skip1_i01_out) )

  return( Model )
}

########################################################################
########################################################################
create3DModel_3L_v2RRRRRRR <- function( inputImageSize ){
  numberOfOutputs <- 1
  convolutionKernelSize <- c( 3, 3, 3 )
  deconvolutionKernelSize <- c( 2, 2, 2 )
  N <- 3
  ps <-  c( 2, 2, 2 )
  strides <-  c( 2, 2, 2 )

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path
  outputs <- inputs %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  # intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  # Decoding path
  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = N*8 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same' )
  outputs <- list( skip3, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = N*8, kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = N*4 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip1, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  if( numberOfOutputs == 1 ){convActivation <- 'sigmoid'} else {convActivation <- 'softmax'}

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  Model <- keras_model( inputs = inputs, outputs = outputs )

  return( Model )
}

create3DModel_3L_v2RRRRR <- function( inputImageSize ){
  numberOfOutputs <- 1
  convolutionKernelSize <- c( 3, 3, 3 )
  deconvolutionKernelSize <- c( 2, 2, 2 )
  N <- 2
  ps <-  c( 2, 2, 2 )
  strides <-  c( 2, 2, 2 )

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path
  outputs <- inputs %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  # intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  # Decoding path
  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = N*8 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same' )
  outputs <- list( skip3, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = N*8, kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = N*4 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip1, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  if( numberOfOutputs == 1 ){convActivation <- 'sigmoid'} else {convActivation <- 'softmax'}

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  Model <- keras_model( inputs = inputs, outputs = outputs )

  return( Model )
}

create3DModel_3L_v2RRRRRR <- function( inputImageSize ){
  numberOfOutputs <- 1
  convolutionKernelSize <- c( 3, 3, 3 )
  deconvolutionKernelSize <- c( 2, 2, 2 )
  N <- 4
  ps <-  c( 2, 2, 2 )
  strides <-  c( 2, 2, 2 )

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path
  outputs <- inputs %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  # intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  # Decoding path
  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = N*8 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same' )
  outputs <- list( skip3, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = N*8, kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = N*4 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip1, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  if( numberOfOutputs == 1 ){convActivation <- 'sigmoid'} else {convActivation <- 'softmax'}

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  Model <- keras_model( inputs = inputs, outputs = outputs )

  return( Model )
}

create3DModel_3L_v2RRRRR <- function( inputImageSize ){
  numberOfOutputs <- 1
  convolutionKernelSize <- c( 3, 3, 3 )
  deconvolutionKernelSize <- c( 2, 2, 2 )
  N <- 2
  ps <-  c( 2, 2, 2 )
  strides <-  c( 2, 2, 2 )

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path
  outputs <- inputs %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  # intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  # Decoding path
  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = N*8 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same' )
  outputs <- list( skip3, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = N*8, kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = N*8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = N*4 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip1, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = N*4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N*2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  if( numberOfOutputs == 1 ){convActivation <- 'sigmoid'} else {convActivation <- 'softmax'}

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  Model <- keras_model( inputs = inputs, outputs = outputs )

  return( Model )
}

create3DModel_3L_v2RRRR <- function( inputImageSize ){
  numberOfOutputs <- 1
  convolutionKernelSize <- c( 3, 3, 3 )
  deconvolutionKernelSize <- c( 2, 2, 2 )
  N <- 3
  ps <-  c( 2, 2, 2 )
  strides <-  c( 2, 2, 2 )

  inputs <- layer_input( shape = inputImageSize )

  # Encoding path
  outputs <- inputs %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  # intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  # Decoding path
  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = N , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same' )
  outputs <- list( skip3, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = N, kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = N , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip1, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = N , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
  #layer_activation_relu()

  if( numberOfOutputs == 1 ){convActivation <- 'sigmoid'} else {convActivation <- 'softmax'}

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  Model <- keras_model( inputs = inputs, outputs = outputs )

  return( Model )
}


create3DModel_3L_v2RRR <- function( inputImageSize ){
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
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
    #layer_activation_relu()

  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
    #layer_activation_relu()

  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
    #layer_activation_relu()

  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  # intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
    #layer_activation_relu()

  # Decoding path
  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same' )
  outputs <- list( skip3, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
    #layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2, kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
    #layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip1, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
    #layer_activation_relu()

  if( numberOfOutputs == 1 ){convActivation <- 'sigmoid'} else {convActivation <- 'softmax'}

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  Model <- keras_model( inputs = inputs, outputs = outputs )

  return( Model )
}

create3DModel_3L_v2RR <- function( inputImageSize ){
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
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.1) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
    #layer_activation_relu()

  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.1) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
    #layer_activation_relu()

  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.1) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
    #layer_activation_relu()

  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  # intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.1) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
    #layer_activation_relu()

  # Decoding path
  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same' )
  outputs <- list( skip3, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.1) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
    #layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.1) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
    #layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip1, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.1) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization()
    #layer_activation_relu()

  if( numberOfOutputs == 1 ){convActivation <- 'sigmoid'} else {convActivation <- 'softmax'}

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  Model <- keras_model( inputs = inputs, outputs = outputs )

  return( Model )
}


create3DModel_3L_v2R <- function( inputImageSize ){
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
    #layer_activation_relu() %>%
    layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
    #layer_batch_normalization() %>%
    #layer_activation_relu()

  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
    #layer_batch_normalization() %>%
    #layer_activation_relu()

  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
    #layer_batch_normalization() %>%
    #layer_activation_relu()

  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  # intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
    #layer_batch_normalization() %>%
    #layer_activation_relu()

  # Decoding path
  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same' )
  outputs <- list( skip3, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
    #layer_batch_normalization() %>%
    #layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
    #layer_batch_normalization() %>%
    #layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip1, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    layer_spatial_dropout_3d(rate = 0.05) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
    #layer_batch_normalization() %>%
    #layer_activation_relu()

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
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.1) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
    #layer_batch_normalization() %>%
    #layer_activation_relu()

  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.1) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
    #layer_batch_normalization() %>%
    #layer_activation_relu()

  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.1) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
    #layer_batch_normalization() %>%
    #layer_activation_relu()

  skip3 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  # intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.1) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
    #layer_batch_normalization() %>%
    #layer_activation_relu()

  # Decoding path
  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same' )
  outputs <- list( skip3, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.1) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
    #layer_batch_normalization() %>%
    #layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.1) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
    #layer_batch_normalization() %>%
    #layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip1, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    #layer_activation_relu() %>%
    #layer_spatial_dropout_3d(rate = 0.1) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
    #layer_batch_normalization() %>%
    #layer_activation_relu()

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
