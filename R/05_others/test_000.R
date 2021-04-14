source("https://gist.githubusercontent.com/kumeS/3750335e751ed05e6ab37e9db513b19d/raw/c825445da920cac47aac6f46095e11d921d62f39/ANTsRNET_UNET.R")
source("https://gist.githubusercontent.com/kumeS/41fed511efb45bd55d468d4968b0f157/raw/b7205c6285422e5166f70b770e1e8674d65f5ea2/DL_plot_modi_v1.2.R")

#2d U-NET
model <- createUnetModel2D( c( 256, 256, 1 ),
                            numberOfLayers = 3,
                            addAttentionGating = F)
model %>% plot_model_modi(width=4, height=1.2)

#2d U-NET + Attention
model <- createUnetModel2D( c( 256, 256, 1 ),
                            numberOfLayers = 3,
                            addAttentionGating = T)

model %>% plot_model_modi(width=4, height=1.2)

#3d U-NET
model <- createUnetModel3D( c( 256, 256, 24, 1 ),
                            numberOfLayers = 3,
                            addAttentionGating = F)

model %>% plot_model_modi(width=4, height=1.2)

#3d U-NET + Attention
model <- createUnetModel3D( c( 256, 256, 24, 1 ),
                            numberOfLayers = 3,
                            addAttentionGating = T)
model %>% plot_model_modi(width=4, height=1.2)


