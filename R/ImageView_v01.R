##ImageView
#str(TrainImgDataset)
ImageView3Ddataset <- function(ImgArray=TrainImgDataset, Sample=1, ImgSection = 1, XYsize = 256,
                               Lab01="Original", Lab02="Merge", Lab03="Supervised",
                               random=T, SaveAll=F){
  if(SaveAll != T){
    par(bg = 'grey')
    if(random == T){
      ImgSection <- floor(runif(1, min=1, max=dim(ImgArray$Original)[4]))
    }

    Image_color01 <- paintObjects(resize(ImgArray$GroundTruth[Sample,,,ImgSection,], XYsize, XYsize, filter="none")/2,
                                  toRGB(resize(ImgArray$Original[Sample,,,ImgSection,], XYsize, XYsize, filter="none")), opac=c(0.25, 0.25),
                                  col=c("red","red"), thick=T, closed=F)
    EBImage::display(EBImage::combine(resize(toRGB(ImgArray$Original[Sample,,,ImgSection,]), XYsize, XYsize, filter="none"),
                                      resize(Image_color01, XYsize, XYsize, filter="none"),
                                      resize(toRGB(ImgArray$GroundTruth[Sample,,,ImgSection,]), XYsize, XYsize, filter="none")),
                     nx=3, all=TRUE, spacing = 0.01, margin = 70)
    text(x = XYsize*0.5, y = -XYsize*0.25,
         label = Lab01, adj = c(0,1), col = "black", cex = 1.2, pos=1, srt=0)
    text(x = XYsize*0.5+XYsize, y = -XYsize*0.25,
         label = Lab02, adj = c(0,1), col = "black", cex = 1.2, pos=1, srt=0)
    text(x = XYsize*0.5+XYsize*2, y = -XYsize*0.25,
         label = Lab03, adj = c(0,1), col = "black", cex = 1.2, pos=1, srt=0)
    text(x = XYsize*0.5+XYsize, y = XYsize,
         label = paste("Image section: ", ImgSection),
         adj = c(0,1), col = "black", cex = 1, pos=1, srt=0)
  }else{
    if(length(list.files(pattern = "3Dimg_")) == 0){
      dir.create("3Dimg_000")
      for(n in seq_len(dim(ImgArray$Original)[4])){
        png(paste(getwd(), "/3Dimg_000/",
                  formatC(n, width = 4, flag = "0"), ".png", sep=""),
            width = 1000, height = 1000)
        par(bg = 'grey')
        Image_color01 <- paintObjects(resize(ImgArray$GroundTruth[Sample,,,n,], XYsize, XYsize, filter="none")/2,
                                      toRGB(resize(ImgArray$Original[Sample,,,n,], XYsize, XYsize, filter="none")), opac=c(0.25, 0.25),
                                      col=c("red","red"), thick=T, closed=F)
        EBImage::display(EBImage::combine(resize(toRGB(ImgArray$Original[Sample,,,n,]), XYsize, XYsize, filter="none"),
                                          resize(Image_color01, XYsize, XYsize, filter="none"),
                                          resize(toRGB(ImgArray$GroundTruth[Sample,,,n,]), XYsize, XYsize, filter="none")),
                         nx=3, all=TRUE, spacing = 0.01, margin = 70)
        text(x = XYsize*0.5, y = -XYsize*0.2,
             label = Lab01, adj = c(0,1), col = "black", cex = 3, pos=1, srt=0)
        text(x = XYsize*0.5+XYsize, y = -XYsize*0.2,
             label = Lab02, adj = c(0,1), col = "black", cex = 3, pos=1, srt=0)
        text(x = XYsize*0.5+XYsize*2, y = -XYsize*0.2,
             label = Lab03, adj = c(0,1), col = "black", cex = 3, pos=1, srt=0)
        text(x = XYsize*0.5+XYsize, y = XYsize + XYsize*0.05,
             label = paste("Image section: ", n),
             adj = c(0,1), col = "black", cex = 3, pos=1, srt=0)
        try(dev.off(), silent=T)}
      } else {
        m <- length(list.files(pattern = "3Dimg_"))
        dir.create(paste("3Dimg_", formatC(m, width = 3, flag = "0"), sep=""))
        for(n in seq_len(dim(ImgArray$Original)[4])){
          png(paste(getwd(), "/", paste("3Dimg_", formatC(m, width = 3, flag = "0"), sep="")
                    , "/", formatC(n, width = 4, flag = "0"), ".png", sep=""),
              width = 1000, height = 1000)
          par(bg = 'grey')
          Image_color01 <- paintObjects(resize(ImgArray$GroundTruth[Sample,,,n,], XYsize, XYsize, filter="none")/2,
                                        toRGB(resize(ImgArray$Original[Sample,,,n,], XYsize, XYsize, filter="none")), opac=c(0.25, 0.25),
                                        col=c("red","red"), thick=T, closed=F)
          EBImage::display(EBImage::combine(resize(toRGB(ImgArray$Original[Sample,,,n,]), XYsize, XYsize, filter="none"),
                                            resize(Image_color01, XYsize, XYsize, filter="none"),
                                            resize(toRGB(ImgArray$GroundTruth[Sample,,,n,]), XYsize, XYsize, filter="none")),
                           nx=3, all=TRUE, spacing = 0.01, margin = 70)
          text(x = XYsize*0.5, y = -XYsize*0.2,
               label = Lab01, adj = c(0,1), col = "black", cex = 3, pos=1, srt=0)
          text(x = XYsize*0.5+XYsize, y = -XYsize*0.2,
               label = Lab02, adj = c(0,1), col = "black", cex = 3, pos=1, srt=0)
          text(x = XYsize*0.5+XYsize*2, y = -XYsize*0.2,
               label = Lab03, adj = c(0,1), col = "black", cex = 3, pos=1, srt=0)
          text(x = XYsize*0.5+XYsize, y = XYsize + XYsize*0.05,
               label = paste("Image section: ", n),
               adj = c(0,1), col = "black", cex = 3, pos=1, srt=0)
          try(dev.off(), silent=T)}
      }
    }
}


ImageView3Dresult <- function(ImgArray=TrainImgDataset, Sample=1, ImgSection = 1, XYsize = 256,
                              random=T, SaveAll=F){
ImageView3Ddataset(ImgArray=ImgArray, Sample=Sample, ImgSection = ImgSection, XYsize = XYsize,
                               Lab01="Test", Lab02="Merge", Lab03="Prediction",
                               random=random, SaveAll=SaveAll)
}





