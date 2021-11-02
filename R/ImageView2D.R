##' @title An image visualization function for the 4D array.
##'
##' @param ImgArray_x a 4D array
##' @param ImgArray_y a 4D array
##' @param ImgN a slice of images
##' @param lab a label information; three elements of a vector
##'
##' @importFrom EBImage Image
##' @importFrom EBImage resize
##' @importFrom EBImage channel
##' @importFrom EBImage paintObjects
##' @importFrom EBImage combine
##' @importFrom EBImage toRGB
##' @importFrom EBImage display
##'
##' @export ImageView2D
##' @author Satoshi Kume
##'


ImageView2D <- function(ImgArray_x,
                        ImgArray_y,
                        ImgN=1,
                        lab=c("Original", "Overlay", "Prediction")){
    if(class(ImgArray_x) == "Image"){ImgArray_x <- array(ImgArray_x, dim=dim(ImgArray_x))}
    if(class(ImgArray_y) == "Image"){ImgArray_y <- array(ImgArray_y, dim=dim(ImgArray_y))}
    if(length(dim(ImgArray_x)) != 4){
        if(length(dim(ImgArray_x)) == 3){
            ImgArray_x <- array(ImgArray_x, dim=c(1, dim(ImgArray_x)))
        }else{
            return(message("wrong dimensions of ImgArray_x"))
        }}
    if(length(dim(ImgArray_y)) != 4){
        if(length(dim(ImgArray_y)) == 3){
            ImgArray_y <- array(ImgArray_y, dim=c(1, dim(ImgArray_y)))
        }else{
            return(message("wrong dimensions of ImgArray_y"))
        }}
    Lab01=lab[1]; Lab02=lab[2]; Lab03=lab[3]
    Opac=c(0.2, 0.2); Width = 500; Height=250; XYsize = 256; Thres=0.4

    parold <- par(); on.exit(parold)
    #options(EBImage.display = "raster")
    #str(ImgArray_x); str(ImgArray_y)

    ImgArray <- list(ImgArray_x,
                     ImgArray_y)
    names(ImgArray) <- c("X", "Y")
    #str(ImgArray)
    if(!all(unique(as.character(unlist(ImgArray$Y))) == c("0", "1"))){
          ImgArray$Y[ImgArray$Y < Thres] <- 0
          ImgArray$Y[ImgArray$Y >= Thres] <- 1
    }

    par(bg = 'grey')
    Ximg <- (ImgArray$X[ImgN,,,] - range(ImgArray$X[ImgN,,,])[1])
    Ximg <- Ximg / range(Ximg)[2]
    #display(Ximg); display(ImgArray$X[ImgN,,,]); display(X1)
    X0 <- EBImage::Image(Ximg, colormode="Grayscale")
    X1 <- EBImage::resize(X0, XYsize, XYsize, filter="none")
    #range(ImgArray$X[ImgN,,,]); range(Ximg); range(X0); range(X1)
    Y0 <- EBImage::Image(ImgArray$Y[ImgN,,,], colormode="Grayscale")
    Y1 <- EBImage::resize(Y0, XYsize, XYsize, filter="none")
    #str(Y1); str(X1)

    Image_color01 <- EBImage::paintObjects(Y1,
                                           EBImage::toRGB(X1),
                                           opac=Opac,
                                           col=c("red","red"), thick=TRUE, closed=FALSE)
    EBImage::display(EBImage::combine(EBImage::toRGB(X1),
                                      EBImage::resize(Image_color01, XYsize, XYsize, filter="none"),
                                      EBImage::toRGB(Y1)),
                         nx=3, all=TRUE, spacing = 0.01, margin = 70, method = "raster")
    text(x = XYsize*0.5, y = -XYsize*0.175,
             label = Lab01, adj = c(0,1), col = "black", cex = 1.2, pos=1, srt=0)
    text(x = XYsize*0.5+XYsize, y = -XYsize*0.175,
             label = Lab02, adj = c(0,1), col = "black", cex = 1.2, pos=1, srt=0)
    text(x = XYsize*0.5+XYsize*2, y = -XYsize*0.175,
             label = Lab03, adj = c(0,1), col = "black", cex = 1.2, pos=1, srt=0)
    text(x = XYsize*0.5+XYsize, y = XYsize + XYsize*0.05,
             label = paste("Image section: ", ImgN),
             adj = c(0,1), col = "black", cex = 1.2, pos=1, srt=0)

}
