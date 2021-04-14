#library(keras)
#py_plot_model <- reticulate::import(module = "tensorflow")$keras$utils$plot_model
#py_plot_model(create3DModel_2L_v3( inputImageSize=c(256, 256, 64,1)), to_file=paste0('Model_', formatC(length(list.files(pattern = "Model_")), width = 3, flag = "0"), '.png'), show_shapes=T, show_layer_names=T)
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
create3DModel_2L_v4 <- function( inputImageSize ){
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

# intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )

# Decoding path
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


############################################################################################
############################################################################################
create3DModel_2L_v3 <- function( inputImageSize ){
  numberOfOutputs <- 1
  convActivation <- ''
  if( numberOfOutputs == 1 ){convActivation <- 'sigmoid'} else {convActivation <- 'softmax'}

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
  skip1_out1 <- skip1 %>%
    layer_conv_3d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  skip1_2 <- skip1 %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
  skip1_out2 <- skip1_2 %>%
    layer_conv_3d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )
  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

# intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )

# Decoding path
  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 4 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  ) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  )

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 2 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip1_2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  ) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu'  )


  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  Model <- keras_model( inputs = inputs, outputs = c(outputs, skip1_out1, skip1_out2) )

  return( Model )
}

############################################################################################
############################################################################################
#inputImageSize=c(256, 256, 64,1)
create3DModel_2L_v2 <- function( inputImageSize ){
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
    layer_activation_relu() %>%
    layer_conv_3d( filters = 4 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

  skip1 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

  outputs <- outputs %>%
    layer_conv_3d( filters = 4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_3d( filters = 4 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

  skip2 <- outputs
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = ps )

# intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = 8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_3d( filters = 8 , kernel_size = convolutionKernelSize, padding = 'same'  ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

# Decoding path
  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 8 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu' )
  outputs <- list( skip2, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 8 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_3d( filters = 4 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

  outputs <- outputs %>%
    layer_conv_3d_transpose( filters = 4 , kernel_size = deconvolutionKernelSize, strides = strides, padding = 'same', activation = 'relu'  )
  outputs <- list( skip1, outputs ) %>% layer_concatenate()

  outputs <- outputs %>%
    layer_conv_3d( filters = 4 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu() %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same' ) %>%
    layer_batch_normalization() %>%
    layer_activation_relu()

  convActivation <- ''
  if( numberOfOutputs == 1 ){convActivation <- 'sigmoid'} else {convActivation <- 'softmax'}

  outputs <- outputs %>%
    layer_conv_3d( filters = numberOfOutputs, kernel_size = 1, activation = convActivation )

  Model <- keras_model( inputs = inputs, outputs = outputs )

  return( Model )
}

############################################################################################
############################################################################################
create3DModel_2L_v1 <- function( inputImageSize ){
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

# intermediate path
  outputs <- outputs %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' ) %>%
    layer_conv_3d( filters = 2 , kernel_size = convolutionKernelSize, padding = 'same', activation = 'relu' )

# Decoding path
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
