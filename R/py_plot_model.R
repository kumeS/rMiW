##' @description visualize a model using plot_model in tensorflow
##'
##' @title Create a network diagram for the model object.
##' @param model the model object in Keras.
##' @param file a file name to be saved.
##' @return A image file in the current directory.
##' @author Satoshi Kume
##' @export
##' @examples
##' \dontrun{
##' py_plot_model(model)
##' }

Py_plot_model <- function(model, file='Model.png'){
Pyplot_model <- reticulate::import(module = "tensorflow")$keras$utils$plot_model
Pyplot_model(model, to_file=file, show_shapes=T, show_layer_names=T)
}





