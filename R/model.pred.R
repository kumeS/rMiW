##' @title Calculate the prediction results with a binary label
##'
##' @param model a model object in keras
##' @param x a 4D array of images
##'
##' @importFrom stats predict
##' @importFrom purrr map
##' @importFrom purrr array_branch
##' @importFrom mmand threshold
##'
##' @export model.pred
##' @author Satoshi Kume
##'

model.pred <- function(model,
                       x){

Y_hat <- stats::predict(model,
                        x = x,
                        verbose=1)
#str(Y_hat)
Y_hat5 <- purrr::map(purrr::array_branch(Y_hat, 1),
                     .f = function(z) {mmand::threshold(z, 0.5, binarise = TRUE)})
#str(Y_hat5)

#Convert to array
Y_hatA <- rMiW::list2tensor(Y_hat5)
#str(Y_hatA)

return(Y_hatA)

}




