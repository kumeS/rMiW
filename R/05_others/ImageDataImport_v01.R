##Demo3dTRAINDataImport

## Dataset
## 00_HumanNB4_Cell_All_ver191119/Mix_086_097

#WIDTH  = 256; HEIGHT = 256; CHANNELS = 1

Demo3dTRAINDataImport <- function(WIDTH  = 256, HEIGHT = 256, Z=30, CHANNELS = 1,
                                  path01="02_TrakEM2_tifs",
                                  path02="03_All"){
    TrainDIR <- paste0("./data/", path01, "/", path02)
    TRAIN_PATH = paste(TrainDIR, "/OriginalData", sep="")
    Teacher_PATH = paste(TrainDIR, "/01_membranes_GT_Binary", sep="")

    #TestDIR <- "./data/02_TrakEM2_tifs/03_All"
    #TEST_PATH = paste(TestDIR, "/OriginalData", sep="")
    #Test_GT_PATH = paste(TestDIR, "/01_membranes_GT", sep="")

    SHAPE = c(WIDTH, HEIGHT, CHANNELS)

    ImageFileTrain = paste(TRAIN_PATH, "/", dir(TRAIN_PATH), sep="")[1:Z]
    X = map(ImageFileTrain, preprocess_3d_image_train_test, shape = SHAPE)

    #str(X)
    #EBImage::display(X[[1]][1,,,1])
    xTensor1 <- simplify2array(X)
    #str(xTensor1)
    xTensor2 <- aperm(xTensor1, c(1, 2, 3, 5, 4))
    #str(xTensor2)

    ## Multi samples array
    #xTensor3 <- abind::abind(xTensor2, xTensor2, along=1)
    #str(xTensor3)
    #dimnames(xTensor3) <- list("samples", "conv_dim1", "conv_dim2", "conv_dim3", "channels")
    #samples, conv_dim1, conv_dim2, conv_dim3, channels
    #xTensor4 <- simplify2array(xTensor3)
    #str(xTensor4)

    ImageFileTeacher = paste(Teacher_PATH, "/", dir(Teacher_PATH), sep="")[1:Z]
    Y = map(ImageFileTeacher, preprocess_3d_image_GT, shape = SHAPE)
    #str(Y)
    #EBImage::display(Y[[1]][1,,,1])

    yTensor1 <- simplify2array(Y)
    #str(yTensor1)
    yTensor2 <- aperm(yTensor1, c(1, 2, 3, 5, 4))
    #str(yTensor2)
    #yTensor3 <- aperm(yTensor1, c(5, 2, 3, 4, ))

    DemoImage <- list(Training=xTensor2, TrainingGT=yTensor2)

    return (DemoImage)
}

