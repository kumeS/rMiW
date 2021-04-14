# shape = SHAPE
# file = paste(TRAIN_PATH, "/", dir(Teacher_PATH)[1], sep="")

preprocess_2d_image_train_test <- function(file, shape){
  image <- readImage(file, type="png")
  image <- resize(image, w = shape[1], h = shape[2], filter = "bilinear")  ## none or bilinear
  #image <- normalize(image)
  #image <- clahe(image)
  #image <- image^1.0
  #EBImage::display(image)
  array(image, dim=c(shape[1], shape[2], shape[3]))
}

preprocess_2d_image_GT <- function(file, shape){
  image <- readImage(file, type="png")
  image <- resize(image, w = shape[1], h = shape[2], filter = "none")
  array(image, dim=c(shape[1], shape[2], shape[3]))                                    ## return as array
}

preprocess_3d_image_train_test <- function(file, shape){
  image <- readImage(file, type="png")
  image <- resize(image, w = shape[1], h = shape[2], filter = "bilinear")  ## none or bilinear
  #image <- normalize(image)
  #image <- clahe(image)
  #image <- image^1.0
  #EBImage::display(image)
  array(image, dim=c(1, shape[1], shape[2], shape[3]))
}

preprocess_3d_image_GT <- function(file, shape){
  image <- readImage(file, type="png")
  image <- resize(image, w = shape[1], h = shape[2], filter = "none")
  array(image, dim=c(1, shape[1], shape[2], shape[3]))                                    ## return as array
}
