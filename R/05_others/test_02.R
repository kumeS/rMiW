#BottleneckResidual
inputImageSize2d <- c( 256, 256, 1 )
N <- 1

inputs <- layer_input( shape = inputImageSize2d )

outputs1 <- inputs %>%
  layer_conv_2d(filters=N, kernel_size = 1, strides=1, padding = 'same') %>%
  layer_batch_normalization() %>%
  layer_activation_relu() %>%
  layer_conv_2d(filters=N, kernel_size = 3, strides=1, padding = 'same') %>%
  layer_batch_normalization() %>%
  layer_activation_relu() %>%
  layer_conv_2d(filters=N, kernel_size = 1, strides=1, padding = 'same') %>%
  layer_batch_normalization()

outputs2 <- inputs %>%
  layer_conv_2d(filters=N, kernel_size = 1, padding = 'same') %>%
  layer_batch_normalization()

outputs <- layer_add(c(outputs1, outputs2))  %>%
      layer_activation_relu()

Model <- keras_model( inputs = inputs, outputs = outputs )
Model %>% plot_model_modi(width=4, height=1.2)

tf <- reticulate::import(module = "tensorflow"); py_plot_model <- tf$keras$utils$plot_model
py_plot_model(Model, to_file='Model.png', show_shapes=T, show_layer_names=T)
