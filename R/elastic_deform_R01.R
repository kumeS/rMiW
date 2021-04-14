#system("pip install elasticdeform")

#Project description
#Elastic deformations for N-dimensional images (Python, SciPy, NumPy, TensorFlow)
#Documentation Status Build Status Build status
#This library implements elastic grid-based deformations for N-dimensional images.
#The elastic deformation approach is described in
#Ronneberger, Fischer, and Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation" (https://arxiv.org/abs/1505.04597)
#Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation" (https://arxiv.org/abs/1606.06650)
#The procedure generates a coarse displacement grid with a random displacement for each grid point. This grid is then interpolated to compute a displacement for each pixel in the input image. The input image is then deformed using the displacement vectors and a spline interpolation.
#In addition to the normal, forward deformation, this package also provides a function that can backpropagate the gradient through the deformation. This makes it possible to use the deformation as a layer in a convolutional neural network. For convenience, a TensorFlow wrapper is provided in elasticdeform.tf.

#プロジェクトの説明
#N次元画像の弾性変形 (Python, SciPy, NumPy, TensorFlow)
#ドキュメントステータス ビルドステータス ビルドステータス
#このライブラリは、N次元画像のための弾性グリッドベースの変形を実装しています。
#弾性変形のアプローチは
#Ronneberger, Fischer, and Brox, "U-Net. バイオメディカル画像セグメンテーションのための畳み込みネットワーク" (https://arxiv.org/abs/1505.04597)
#Çiçekら, "3D U-Net. スパースアノテーションからの密な体積分割の学習" (https://arxiv.org/abs/1606.06650)
#この処理は，各グリッド点に対してランダムな変位を持つ粗い変位グリッドを生成します．このグリッドは，入力画像の各ピクセルの変位を計算するために補間されます．そして，入力画像は，この変位ベクトルとスプライン補間を用いて変形されます．
#このパッケージでは，通常の前方変形に加えて，変形を介して勾配をバックプロパゲーションする機能も提供しています．これにより，畳み込みニューラルネットワークのレイヤーとして変形を使用することが可能になります．便宜上、elasticdeform.tfにはTensorFlowラッパーが用意されています。



reticulate::use_python("/usr/local/bin/python", required =T)
ed <- reticulate::import(module = "elasticdeform")
np <- reticulate::import(module = "numpy")
ib <- reticulate::import_builtins()

reticulate::py_help(np$zeros)
reticulate::py_help(ed$deform_random_grid)

X = np$zeros(c(200L, 200L))
head(X)
diag(X) <- 1
image(X)
diag(X[-c(1),]) <- 1
diag(X[-c(1:2),-c(1:2)]) <- 1
image(X)

Y <- X[,200:1]
image(Y)
diag(Y) <- 1
diag(Y[-c(1),]) <- 1
diag(Y[-c(1:2),-c(1:2)]) <- 1
image(Y)

# グレイ・カラー
greys <- grey.colors(100, start = 0.1, end = 1, gamma = 1, rev = T)

# Basic example
# 3x3グリット・ランダム変形
X_deformed0 <- ed$deform_random_grid(Y, sigma=25, points=3L)
image(X_deformed0, col = greys, axes = F)

# Multiple inputs
X_deformed1 <- ed$deform_random_grid(list(X, Y))

quartz(width=6, height=3)
par(mfrow=c(1,2), mai = c(0.1, 0.1, 0.1, 0.1))
image(X_deformed1[[1]], col = greys, axes = F)
image(X_deformed1[[2]], col = greys, axes = F)

X_deformed2 <- ed$deform_random_grid(list(X, Y), order=c(0, 0))
quartz(width=6, height=3)
par(mfrow=c(1,2), mai = c(0.1, 0.1, 0.1, 0.1))
image(X_deformed2[[1]], col = greys, axes = F)
image(X_deformed2[[2]], col = greys, axes = F)

#Multi-channel images
Img <- EBImage::readImage("https://www.r-project.org/Rlogo.png")

plot(Img)
str(Img0)

Img0 <- ed$deform_random_grid(Img, axis=c(0L))
Img0 <- ed$deform_random_grid(Img0, axis=c(1L))
str(Img0)
EBImage::display(EBImage::Image(Img0, colormode="Color"))

Img1 = np$random$rand(200L, 300L, 3L)
EBImage::display(EBImage::Image(Img1, colormode="Grayscale"))
Img1 <- ed$deform_random_grid(Img1, axis=c(0L))
EBImage::display(EBImage::Image(Img1, colormode="Grayscale"))

#Cropping
X = np$random$rand(200L, 300L)

crop <- list(ib$slice(50L, 150L), ib$slice(0L, 100L))

displacement = np$random$randn(2L, 3L, 3L) * 25

# deform full image
X_deformed = ed$deform_grid(X, displacement)
EBImage::display(EBImage::Image(X_deformed, colormode="Grayscale"))

# compute only the cropped region
X_deformed_crop <- ed$deform_grid(X, displacement, crop=crop)
EBImage::display(EBImage::Image(X_deformed_crop, colormode="Grayscale"))

# the deformation is the same
table(X_deformed[c(51:150),c(0:100)] == X_deformed_crop)

#Rotate and zoom
# 3x3グリット・ランダム変形 + 回転30 & zoom x2
X_deformed1 <- ed$deform_random_grid(Y, sigma=25, points=3L, rotate=30, zoom=2)
image(X_deformed1, col = greys, axes = F)


#Gradient
X = np$random$rand(200L, 300L)

# generate a deformation grid
displacement = np$random$randn(2L, 3L, 3L) * 25

# perform forward deformation
X_deformed = ed$deform_grid(X, displacement)

# obtain the gradient w.r.t. X_deformed (e.g., with backpropagation)
dX_deformed = np$random$randn(200L, 300L)

# compute the gradient w.r.t. X
#dX = ed$deform_grid_gradient(dX_deformed, displacement)
#バグる




