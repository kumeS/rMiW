##' @title Data conversion of list to tensor.
##'
##' @description Convert list data to 4D/5D array
##' # Note below
##' # 2D convolution layer (e.g. spatial convolution over images).
##' # Input shape
##' 4D tensor with shape: (samples, channels, rows, cols) if data_format='channels_first'
##' 4D tensor with shape: (samples, rows, cols, channels) if data_format='channels_last'
##' # Output shape
##' 4D tensor with shape: (samples, filters, new_rows, new_cols) if data_format='channels_first'
##' 4D tensor with shape: (samples, new_rows, new_cols, filters) if data_format='channels_last'
##' rows and cols values might have changed due to padding.
##' 3D convolution layer (e.g. spatial convolution over volumes).
##' # Input shape
##' 5D tensor with shape: (samples, channels, conv_dim1, conv_dim2, conv_dim3) if data_format='channels_first'
##' 5D tensor with shape: (samples, conv_dim1, conv_dim2, conv_dim3, channels) if data_format='channels_last'
##' # Output shape
##' 5D tensor with shape: (samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3) if data_format='channels_first'
##' 5D tensor with shape: (samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters) if data_format='channels_last'
##' new_conv_dim1, new_conv_dim2 and new_conv_dim3 values might have changed due to padding.
##'
##' @usage list2tensor(xList)
##' @param xList list data of 3D array
##' @return 4D Array.
##' @author Satoshi Kume
##' @export list2tensor
##'

list2tensor <- function(xList) {
  xTensor <- base::simplify2array(xList)
  base::aperm(xTensor, c(4, 1, 2, 3))
}

##' @title Data conversion of list to tensor.
##' @usage list3tensor(yList)
##' @param yList list data of 4D array.
##' @return 5D Array.
##' @author Satoshi Kume
##' @export list3tensor
##'

list3tensor <- function(yList) {
  xTensor <- base::simplify2array(yList)
  base::aperm(xTensor, c(1, 2, 3, 5, 4))
}

##' @title Data conversion of list to tensor.
##' @usage list4tensor(yList)
##' @param yList list data of 4D array.
##' @return 5D Array.
##' @author Satoshi Kume
##' @export list4tensor
##'

list4tensor <- function(yList) {
  xTensor <- base::simplify2array(yList)
  base::aperm(xTensor, c(5, 1, 2, 3, 4))
}


