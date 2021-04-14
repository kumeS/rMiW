##Demo3dTRAINDataImport

## Dataset
## 00_HumanNB4_Cell_All_ver191119/Mix_086_097

#WIDTH  = 256; HEIGHT = 256; CHANNELS = 1;  Z=30
TrakEM2_TrainDataImport <- function(WIDTH  = 256, HEIGHT = 256, Z=24, CHANNELS = 1,
                                    data="../data/",
                                    path01="02_TrakEM2_tifs",
                                    path02="01_TrainingData_24slices",
                                    TRAIN_path="OriginalData",
                                    Teacher_path="01_membranes_GT_Binary"){
DataImport(WIDTH  = WIDTH, HEIGHT = HEIGHT, Z=Z, CHANNELS = CHANNELS,
                      data=data, path01=path01, path02=path02,
                      TRAIN_path=TRAIN_path, Teacher_path=Teacher_path)
}

TrakEM2_TrainDataImport2 <- function(WIDTH  = 256, HEIGHT = 256, Z=24, CHANNELS = 1,
                                    data="../data/",
                                    path01="02_TrakEM2_tifs",
                                    path02="01_TrainingData_6slices",
                                    TRAIN_path="OriginalData",
                                    Teacher_path="01_membranes_GT_Binary"){
DataImport(WIDTH  = WIDTH, HEIGHT = HEIGHT, Z=Z, CHANNELS = CHANNELS,
                      data=data, path01=path01, path02=path02,
                      TRAIN_path=TRAIN_path, Teacher_path=Teacher_path)
}

TrakEM2_TestDataImport <- function(WIDTH  = 256, HEIGHT = 256, Z=6, CHANNELS = 1,
                                    data="../data/",
                                    path01="02_TrakEM2_tifs",
                                    path02="02_TestData",
                                    TRAIN_path="OriginalData",
                                    Teacher_path="01_membranes_GT_Binary"){
DataImport(WIDTH  = WIDTH, HEIGHT = HEIGHT, Z=Z, CHANNELS = CHANNELS,
                      data=data, path01=path01, path02=path02,
                      TRAIN_path=TRAIN_path, Teacher_path=Teacher_path)
}


DataImport <- function(WIDTH  = 256, HEIGHT = 256, Z=6, CHANNELS = 1,
                       data="../data/",
                       path01="02_TrakEM2_tifs",
                       path02="01_TrainingData_6slices",
                       TRAIN_path="OriginalData",
                       Teacher_path="01_membranes_GT_Binary"){
    DataDIR <- paste0(data, path01, "/", path02)
    TRAIN_PATH = paste0(DataDIR, "/", TRAIN_path)
    Teacher_PATH = paste0(DataDIR, "/", Teacher_path)

    SHAPE = c(WIDTH, HEIGHT, CHANNELS)

    m0 <- paste0(TRAIN_PATH, "/", dir(TRAIN_PATH))
    m1 <- length(paste0(TRAIN_PATH, "/", dir(TRAIN_PATH)))
    DatX <- c()
    for(n in seq_len(m1)){
    #n <- 2
    ImageFileTrain = paste0(m0[n], "/", dir(m0[n]))[1:Z]
    X = map(ImageFileTrain, preprocess_2d_image_train_test, shape = SHAPE)
    DatX[[n]] <- simplify2array(X)
    }
    #str(X)
    #str(DatX)
    xTensor1 <- simplify2array(DatX)
    #str(xTensor1)
    xTensor2 <- aperm(xTensor1, c(5, 1, 2, 4, 3))
    #str(xTensor2)

    m2 <- paste0(Teacher_PATH, "/", dir(Teacher_PATH))
    m3 <- length(paste0(Teacher_PATH, "/", dir(Teacher_PATH)))
    DatY <- c()
    for(n in seq_len(m3)){
    ImageFileTrain = paste0(m2[n], "/", dir(m2[n]))[1:Z]
    Y = map(ImageFileTrain, preprocess_2d_image_GT, shape = SHAPE)
    DatY[[n]] <- simplify2array(Y)
    }
    #str(Y)
    #str(DatY)
    yTensor1 <- simplify2array(DatY)
    #str(yTensor1)
    yTensor2 <- aperm(yTensor1, c(5, 1, 2, 4, 3))
    #str(yTensor2)

    Img <- list(Original=xTensor2, GroundTruth=yTensor2)
    #str(Img)
    return (Img)
}


d2TRAINDataImport <- function(WIDTH  = 256, HEIGHT = 256, Z=24, CHANNELS = 1,
                              data="../data/",
                              path01="02_TrakEM2_tifs",
                              path02="01_TrainingData_24slices",
                              TRAIN_path="OriginalData",
                              Teacher_path="01_membranes_GT_Binary"){
    DataDIR <- paste0(data, path01, "/", path02)
    TRAIN_PATH = paste0(DataDIR, "/", TRAIN_path)
    Teacher_PATH = paste0(DataDIR, "/", Teacher_path)

    SHAPE = c(WIDTH, HEIGHT, CHANNELS)

    ImageFileTrain = paste(TRAIN_PATH, "/", dir(TRAIN_PATH), sep="")[1:Z]
    X = map(ImageFileTrain, preprocess_2d_image_train_test, shape = SHAPE)

    xTensor1 <- simplify2array(X)
    xTensor2 <- aperm(xTensor1, c(4, 1, 2, 3))

    ImageFileTeacher = paste(Teacher_PATH, "/", dir(Teacher_PATH), sep="")[1:Z]
    Y = map(ImageFileTeacher, preprocess_2d_image_GT, shape = SHAPE)
    yTensor1 <- simplify2array(Y)
    yTensor2 <- aperm(yTensor1, c(4, 1, 2, 3))

    Img <- list(Original=xTensor2, GroundTruth=yTensor2)

    return (Img)
}

