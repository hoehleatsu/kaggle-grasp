source("data_read.R")

visualize <- function() {
  ##Show correlation between features
  require("corrplot")
  require("caret")

  corMatMy <- cor(train[,features])
  corrplot(corMatMy, order = "hclust")

  highlyCor <- findCorrelation(corMatMy,0.70)
  ##Apply correlation filter at 0.70,
  ##then we remove all the variable correlated with more 0.7.
  filteredFeatures <- features[-highlyCor]
  train.filtered <- train[,filteredFeatures]
  corMatMy <- cor(train.filtered)
  corrplot(corMatMy, order = "hclust")

  ##Principal components
  p <-  prcomp(train[,features])
  print(p)

  require("FactoMineR")
  pca <- PCA(train[,features], scale.unit=FALSE, ncp=5, graph=TRUE)

  prop.table(table(train$Event))

  invisible()
}
