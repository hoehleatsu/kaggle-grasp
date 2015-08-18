##Load packages
require("xgboost")
require("Matrix")

##Improved data i/o. Now all functionality for reading & standardizing features is in this function.
source("data_read.R")

#############################################################################
## let us write learning procedures that take in a training set and return a classifier
## doing this gives us a bit more modularity
## this function returns a list of classifiers indexed by the event type
train_boost_classifier <- function(trainingSet, param = list(objective = "binary:logistic",
                                                             booster = "gblinear",
                                                             nthread = 1,
                                                             eval_metric = "auc",
                                                             alpha = 0.0001,
                                                             lambda = 1), eventTypes = theEventTypes) {

  ##Use all variables in the model (we use an interaction with
  ##subject, which corresponds to a separate model for each
  ##individual)
  RHS <- paste0("(",paste0(features,collapse=" + "),")*subject")

  formula <- as.formula(paste0(eventTypes[1] , "~", RHS))

  ##Generate design matrices for use in xgboost. We use sparse.matrix, but this is not directly
  ##helpfull, coz all features are of numeric type (except maybe subject & series)
  XTrain <- sparse.model.matrix(formula, data = trainingSet)

  ##Run OneVsAll classification to accomodate the multinomial response distribution.
  Map(function(ev) {
    dTrain <- xgb.DMatrix(XTrain, label = as.data.frame(trainingSet)[,ev,drop=TRUE])
    model <- xgb.train(param, dTrain, nround=100,verbose=1,print.every.n = 10, watchlist=list(train=dTrain))
    model$formula <- formula
    model
  }, eventTypes)
}
##Run the classifier on the normalized training set.
classifiers <- train_boost_classifier(train)

# then we can write a generic performance evaluation on a validation set
library("AUC")
measure_auc <- function(model, data_set, true_values) {
  auc(roc(predict(model, newdata = data_set),factor(true_values)))
}
measure_auc(classifiers[["HandStart"]],
            sparse.model.matrix(classifiers[["HandStart"]]$formula, data = test),
            test$HandStart)

# then we can also plot a learning curve (this tells us if we need more data or have a bias in our model)
# of course simply taking the first X rows is not optimal, but ok to show what is going on
# it also gives us some kind of bounds on performance
learning_curve_df <- data.frame()
sparse_test_set <- sparse.model.matrix(classifiers[["HandStart"]]$formula, data = test)
for (no_row_offset in 1:20) {
  sample_size <- 5000 * no_row_offset
  training_set <- train %>% head(sample_size)
  classifiers <- train_boost_classifier(training_set, eventTypes = "HandStart")
  score_train <- measure_auc(classifiers[["HandStart"]],
                            sparse.model.matrix(classifiers[["HandStart"]]$formula, data = training_set),
                            training_set$HandStart)
  score_test <- measure_auc(classifiers[["HandStart"]], sparse_test_set,
              test$HandStart)
  learning_curve_df <- rbind(learning_curve_df, data.frame(sample_size = sample_size,
                                        score_train = score_train,
                                        score_test = score_test))
}

##Plot learning curve
library("ggplot2")
ggplot(data = learning_curve_df, aes(x = sample_size)) +
  geom_line(aes(y = score_train), color = "Train") +
    geom_line(aes(y = score_test), color = "Test")

# a large gap indicates that we can benefit from more data
# a small gap indicates that more data will not make our model better; maybe a more complex model could help

######################################################################


######################################################################
## Simple classification using xgboost. No feature extraction done
## so far.
##
## Params:
##  train, test -
######################################################################
classifyBoost <- function(train, test) {

  ##Parameters for the model based L2 boosting
  param <- list(objective = "binary:logistic",
                booster = "gblinear",
                nthread = 1,
                eval_metric = "auc",
                alpha = 0.0001,
                lambda = 1)

  ## ##Parameters for tree based base-learners
  ## param <- list("objective" = "binary:logistic",
  ##               "bst:eta" = 0.1,
  ##               "bst:max_depth" = 9,
  ##               "eval_metric" = "auc",
  ##               "silent" = 0,
  ##               "nthread" = 1)

  ##Use all variables in the model (we use an interaction with
  ##subject, which corresponds to a separate model for each
  ##individual)
  RHS <- paste0("(",paste0(features,collapse=" + "),")*subject")

  formula <- as.formula(paste0(eventTypes[1] , "~", RHS))

  ##Generate design matrices for use in xgboost. We use sparse.matrix, but this is not directly
  ##helpfull, coz all features are of numeric type (except maybe subject & series)
  XTrain <- sparse.model.matrix(formula, data=train)
  XTest <- sparse.model.matrix(formula, data=test)
  for (ev in eventTypes) eegTest[,ev] <- 0 #just add the column
  XSubmission <- sparse.model.matrix(formula, data=eegTest)

  res <- list()

  ##Fit simple glm for each event-type (atm, we ignore subject & series). This is the one-vs-all principle
  ##for multinomial data.
  for (ev in eventTypes) {
    cat("Type = ",ev,"\n")
    ##Formula for the logit regression model
    formula <- as.formula(paste0(ev,"~", RHS))

    dTrain <- xgb.DMatrix(XTrain, label=as.data.frame(train)[,ev,drop=TRUE])
    dTest <- xgb.DMatrix(XTest, label=as.data.frame(test)[,ev,drop=TRUE])
    ##Run xgboost using the minimum number of rounds on the full data set.
    mBoost <- xgb.train(param, dTrain, nround=100,verbose=1,print.every.n = 10, watchlist=list(train=dTrain,test=dTest))

    ##Do the prediction
    my_predict <- predict(mBoost,newdata=XSubmission)

    ##Make a data.frame with the result
    res[[paste0(ev)]] <-  data.frame(my_predict)
  }

  ##Combine into one data.frame
  my_solution <- bind_cols(eegTest[,"id"],bind_cols(res))
  colnames(my_solution) <- c("id",eventTypes)

  ##Done
  return(my_solution)
}

######################################################################
## First attempt to use neural network classification.
##
## Comment (2015-08-19):
## Something not quite ok yet. Might have to do with the categorical
## 'subject' in the model equation. At least AUC is quite low
## and it doesn't appear to react on different classes and has almost
## all classification into the 'HandStart' class. ToDo: Check more
## on neural networks.
######################################################################

classifyNNET <- function(train, test) {
  require("nnet")
  ##Use subject factor (not sure this works as I think for neural networks)
  RHS <- paste0("(",paste0(features,collapse=" + "),")*subject")
  ##Without subject (only then size = 3 works)
  ##RHS <- paste0(features,collapse=" + ")

  ##Fit neural network
  m_nn <- nnet( as.formula(paste0("Event ~ ",RHS)), data=train, size = 2, decay = .1, maxit=500)
  p <- predict(m_nn, newdata=eegTest)
  head(p)

  if (FALSE) {
    require("AUC")
    auc(roc(p[,"HandStart"], factor(test[,"HandStart"],levels=c(0,1))))
    predClass <- predict(m_nn, newdata=eegTest, type="class")
    table(predClass, test$Event)
    ##measure_auc(m_nn, test, test$Event)
  }

  ##Export as a data frame
  my_solution <- bind_cols(eegTest[,"id"],as.data.frame(p))
  ##Done
  return(my_solution)
}


######################################################################
## Helper function to write submission table to file & zip it.
######################################################################

makeSubmission <- function(my_submission) {
  ## Reduce file size by limiting number of decimal places
  idIndex <- pmatch("id",colnames(my_submission))
  my_submission[,-idIndex] <- round(my_submission[,-idIndex],digits=4)
  write.csv(my_submission,file='../Results/submission1.csv',row.names=FALSE)
  zip("../Results/submission1.csv.zip",files="../Results/submission1.csv")
}

## Perform classification and write to file.
makeSubmission(my_submission <- classifyBoost(train, test))
## Try neural network fit.
makeSubmission(my_submission <- classifyNNET(train, test))
head(my_submission)
makeSubmission(my_submission)

