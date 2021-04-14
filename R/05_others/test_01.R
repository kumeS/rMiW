library(keras)
install.packages("animation")

a <- application_xception()
a
a %>% plot_model_modi(width=4, height=1.2) %>% htmlwidgets::saveWidget(file="../Model/application_xception.html")

a <- application_inception_v3()
a
a %>% plot_model_modi(width=4, height=1.2) %>% htmlwidgets::saveWidget(file="../Model/application_inception_v3.html")

a <- application_inception_resnet_v2()
a
a %>% plot_model_modi(width=4, height=1.2) %>% htmlwidgets::saveWidget(file="../Model/application_inception_resnet_v2.html")

a <- application_vgg16()
a
a %>% plot_model_modi(width=4, height=1.2) %>% htmlwidgets::saveWidget(file="../Model/application_vgg16.html")

a <- application_vgg19()
a
a %>% plot_model_modi(width=4, height=1.2) %>% htmlwidgets::saveWidget(file="../Model/application_vgg19.html")

a <- application_resnet50()
a
a %>% plot_model_modi(width=4, height=1.2) %>% htmlwidgets::saveWidget(file="../Model/application_resnet50.html")

a <- application_mobilenet()
a
a %>% plot_model_modi(width=4, height=1.2) %>% htmlwidgets::saveWidget(file="../Model/application_mobilenet.html")

a <- application_mobilenet_v2()
a
a %>% plot_model_modi(width=4, height=1.2) %>% htmlwidgets::saveWidget(file="../Model/application_mobilenet_v2.html")

a <- application_densenet()
a
a %>% plot_model_modi(width=4, height=1.2) %>% htmlwidgets::saveWidget(file="../Model/application_densenet.html")

a <- application_densenet121()
a
a %>% plot_model_modi(width=4, height=1.2) %>% htmlwidgets::saveWidget(file="../Model/application_densenet121.html")

a <- application_densenet169()
a
a %>% plot_model_modi(width=4, height=1.2) %>% htmlwidgets::saveWidget(file="../Model/application_densenet169.html")

a <- application_densenet201()
a
a %>% plot_model_modi(width=4, height=1.2) %>% htmlwidgets::saveWidget(file="../Model/application_densenet201.html")

a <- application_nasnet()
a
a %>% plot_model_modi(width=4, height=1.2) %>% htmlwidgets::saveWidget(file="../Model/application_nasnet.html")

a <- application_nasnetlarge()
a
a %>% plot_model_modi(width=4, height=1.2) %>% htmlwidgets::saveWidget(file="../Model/application_nasnetlarge.html")

a <- application_nasnetmobile()
a
a %>% plot_model_modi(width=4, height=1.2) %>% htmlwidgets::saveWidget(file="../Model/application_nasnetmobile.html")

