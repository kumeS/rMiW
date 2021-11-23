##' @title Create an array data of 3D images
##'
##' @param Directory a path of dataset directory
##' @param SaveAs a character of save file (.Rds).
##'
##' @importFrom  purrr map
##' @export readImageDataset3D
##' @author Satoshi Kume
##'

readImageDataset3D <- function(Directory, SaveAs=NULL){

Files <- dir(Directory, full.names=T)
if(sum(grepl("[.]png$", Files)) > 0){ extension <- "png" }else{
if(sum(grepl("[.]tif$", Files)) > 0){ extension <- "tiff" }
if(sum(grepl("[.]tiff$", Files)) > 0){ extension <- "tiff" }
}

#read images
X <- purrr::map(Files, processing_2D_image, shape = NULL, type=extension)
#str(X)
DatX <- base::simplify2array(X)
#str(DatX)
xTensor <- base::aperm(DatX, c(4, 1, 2, 3))
#str(xTensor)

if(!is.null(SaveAs)){
if(is.character(SaveAs)){
  SaveAsRds <- sub(".Rds$", "", SaveAs)
  saveRDS(paste0(SaveAsRds, ".Rds"))
}}

return(xTensor)

}


