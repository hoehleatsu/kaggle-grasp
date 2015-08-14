##Load packages
require("xgboost")
require("Matrix")

##Improved data i/o
source("data_read.R")

##Parameters for the model based L2 boosting
param <- list(objective = "binary:logistic",
              booster = "gblinear",
              nthread = 1,
              eval_metric = "auc",
              alpha = 0.0001,
              lambda = 1)

##Load the training data
eegTrain <- get_datasets(verbose=TRUE)
##Load the competition test data.
eegTest <- get_datasets(subjects = 1:12, series = 9:10, base_path = "../Data/test", verbose=TRUE)

##Add column illustrating if in validation dataset or not.
eegTrain <- generate_validation_training_set(eegTrain)
dim(eegTrain)

##Data needs to be reduced (say 1%) to enable possibility to test code. Remove later
eegTrain$is_part_of_reduced_set <- runif(nrow(eegTrain)) < 0.01

##No need for this, but we do it to reduce size of data set.
train <- subset(eegTrain, ( is_part_of_training_set) & is_part_of_reduced_set)
test  <- subset(eegTrain, (!is_part_of_training_set) & is_part_of_reduced_set)

######################################################################
## Simple classification using xgboost. No feature extraction done
## so far.
##
## Params:
##  train, test -
######################################################################
classify <- function(train, test) {

  ##Event types
  eventTypes <- c("HandStart","FirstDigitTouch","BothStartLoadPhase","LiftOff","Replace","BothReleased")
  ##Use all variables in the model
  RHS <- "Fp1 + Fp2 + F7 + F3 + Fz + F4 + F8 + FC5 + FC1 + FC2 + FC6 + T7 + C3 + Cz + C4 + T8 + TP9 + CP5 + CP1 + CP2 + CP6 + TP10 + P7 + P3 + Pz + P4 + P8 + PO9 + O1 + Oz + O2 + PO10"
  res <- list()

  ##Generate design matrices for use in xgboost. We use sparse.matrix, but this is not directly
  ##helpfull, coz all features are of numeric type (except maybe subject & series)
  XTrain <- sparse.model.matrix(formula, data=train)
  XTest <- sparse.model.matrix(formula, data=test)
  ##Create predictions for test series
  for (ev in eventTypes) eegTest[,ev] <- 0 #just add the column
  XSubmission <- sparse.model.matrix(formula, data=eegTest)

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

## Reduce file size by limiting number of decimal places
my_submission <- classify(train, test)
my_submission[,-1]= round(my_submission[,-1],digits=4)
write.csv(my_submission,file='../Results/submission1.csv',row.names=FALSE)
zip("../Results/submission1.csv.zip",files="../Results/submission1.csv")
