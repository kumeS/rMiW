##' @title An image visualization function after the model prediction.
##'
##' @param ImgArray_x a 4D array
##' @param ImgArray_y a 4D array
##' @param ImgArray_pred a 4D array
##' @param ImgN a slice of images
##' @param lab a label information; three elements of a vector
##' @param SaveAll a logical; locally save the all results ot not.
##'
##' @importFrom EBImage Image
##' @importFrom EBImage resize
##' @importFrom EBImage channel
##' @importFrom EBImage paintObjects
##' @importFrom EBImage combine
##' @importFrom EBImage toRGB
##' @importFrom EBImage display
##'
##' @export ImageView2D_pred
##' @author Satoshi Kume
##'

ImageView2D_pred <- function(ImgArray_x,
                             ImgArray_y,
                             ImgArray_pred,
                             ImgN=1,
                             lab=c("Ground truth", "Prediction", "Overlay", "Binary"),
                             SaveAll=FALSE){
if(length(dim(ImgArray_x)) != 4){return(message("wrong dimensions of ImgArray_x"))}
if(length(dim(ImgArray_y)) != 4){return(message("wrong dimensions of ImgArray_y"))}
if(length(dim(ImgArray_pred)) != 4){return(message("wrong dimensions of ImgArray_pred"))}
if(class(ImgArray_x) == "Image"){ ImgArray_x <- array(ImgArray_x, dim=dim(ImgArray_x)) }
if(class(ImgArray_y) == "Image"){ ImgArray_y <- array(ImgArray_y, dim=dim(ImgArray_y))}
if(class(ImgArray_pred) == "Image"){ ImgArray_pred <- array(ImgArray_pred, dim=dim(ImgArray_pred))}

Lab01=lab[1]; Lab02=lab[2]; Lab03=lab[3]; Lab04=lab[4]
Opac=c(0.2, 0.2); Width = 500; Height=250; XYsize = 256; Thres=0.4

ImgArray <- list(ImgArray_x,
                 ImgArray_y,
                 ImgArray_pred)
names(ImgArray) <- c("X", "Y", "Pred")

parold <- par(); on.exit(parold)
#str(ImgArray)

if(!SaveAll){
#ImgN <- 2
Ximg <- (ImgArray$X[ImgN,,,] - range(ImgArray$X[ImgN,,,])[1])
Ximg <- Ximg/ range(Ximg)[2]
Image_color01 <- EBImage::paintObjects(ImgArray$Y[ImgN,,,],
                                       EBImage::toRGB(Ximg),
                                       opac=Opac,
                                       col=c("red","red"), thick=T, closed=F)
Image_color02 <- EBImage::paintObjects(ImgArray$Pred[ImgN,,,],
                                       EBImage::toRGB(Ximg),
                                       opac=Opac,
                                       col=c("blue","blue"),
                                       thick=T, closed=F)
par(bg = 'grey')
EBImage::display(EBImage::combine(EBImage::resize(Image_color01, XYsize, XYsize, filter="none"),
                                  EBImage::resize(EBImage::toRGB(ImgArray$Y[ImgN,,,]), XYsize, XYsize, filter="none"),
                                  EBImage::resize(Image_color02, XYsize, XYsize, filter="none"),
                                  EBImage::resize(EBImage::toRGB(ImgArray$Pred[ImgN,,,]), XYsize, XYsize, filter="none")),
                  nx=2, all=TRUE, spacing = 0.01, margin = 70, method = "raster")

#A <- table(c(XYT$TY[[ABC]]), c(TX_lab[ABC,,,]))
intersection <- sum( matrix(ImgArray$Y[ImgN,,,]) * matrix(ImgArray$Pred[ImgN,,,]))
union <- sum( matrix(ImgArray$Y[ImgN,,,]) + matrix(ImgArray$Pred[ImgN,,,]) ) - intersection
result <- (intersection + 1) / ( union + 1)

text(x = -XYsize/6, y = XYsize/2, label = Lab01,
      adj = c(0,1), col = "black", cex = 1.25, pos=1, srt=90)
text(x = -XYsize/6, y = XYsize*1.45, label = Lab02,
      adj = c(0,1), col = "black", cex = 1.25, pos=1, srt=90)
text(x = XYsize/2, y = -XYsize/5, label = Lab03,
      adj = c(0,1), col = "black", cex = 1.25, pos=1, srt=0)
text(x = XYsize*1.5, y = -XYsize/5, label = Lab04,
      adj = c(0,1), col = "black", cex = 1.25, pos=1, srt=0)
text(x = XYsize, y = XYsize*2,
      label = paste("Image num.: ", ImgN, "  IOU:", round(result, 3)),
      adj = c(0,1), col = "black", cex = 1.25, pos=1, srt=0)
}else{
len <- length(dir()[grepl("ImageView2D_pred", dir())]) + 1
FileNam <- paste0("ImageView2D_pred_", formatC(len, flag = "0", width = 3))
dir.create(FileNam)

for(n in 1:dim(ImgArray$X)[1]){
#n <- 2
Ximg <- (ImgArray$X[n,,,] - range(ImgArray$X[n,,,])[1])
Ximg <- Ximg/ range(Ximg)[2]
Image_color01 <- EBImage::paintObjects(ImgArray$Y[n,,,],
                                       EBImage::toRGB(Ximg),
                                       opac=Opac,
                                       col=c("red","red"), thick=T, closed=F)
Image_color02 <- EBImage::paintObjects(ImgArray$Pred[n,,,],
                                       EBImage::toRGB(Ximg),
                                       opac=Opac,
                                       col=c("blue","blue"),
                                       thick=T, closed=F)
try(dev.off(), silent=T)
png(paste(FileNam, "/Image_",
          formatC(n, width = 4, flag = "0"), ".png", sep=""),
    width = 400, height = 400)
par(bg = 'grey')
EBImage::display(EBImage::combine(EBImage::resize(Image_color01, XYsize, XYsize, filter="none"),
                                  EBImage::resize(EBImage::toRGB(ImgArray$Y[n,,,]), XYsize, XYsize, filter="none"),
                                  EBImage::resize(Image_color02, XYsize, XYsize, filter="none"),
                                  EBImage::resize(EBImage::toRGB(ImgArray$Pred[n,,,]), XYsize, XYsize, filter="none")),
                  nx=2, all=TRUE, spacing = 0.01, margin = 70, method = "raster")

#A <- table(c(XYT$TY[[ABC]]), c(TX_lab[ABC,,,]))
intersection <- sum( matrix(ImgArray$Y[n,,,]) * matrix(ImgArray$Pred[n,,,]))
union <- sum( matrix(ImgArray$Y[n,,,]) + matrix(ImgArray$Pred[n,,,]) ) - intersection
result <- (intersection + 1) / ( union + 1)

text(x = -XYsize/6, y = XYsize/2, label = Lab01,
      adj = c(0,1), col = "black", cex = 1.25, pos=1, srt=90)
text(x = -XYsize/6, y = XYsize*1.45, label = Lab02,
      adj = c(0,1), col = "black", cex = 1.25, pos=1, srt=90)
text(x = XYsize/2, y = -XYsize/5, label = Lab03,
      adj = c(0,1), col = "black", cex = 1.25, pos=1, srt=0)
text(x = XYsize*1.5, y = -XYsize/5, label = Lab04,
      adj = c(0,1), col = "black", cex = 1.25, pos=1, srt=0)
text(x = XYsize, y = XYsize*2,
      label = paste("Image num.: ", n, "  IOU:", round(result, 3)),
      adj = c(0,1), col = "black", cex = 1.25, pos=1, srt=0)
try(dev.off(), silent=T)
}
}
}

