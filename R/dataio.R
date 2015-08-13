rm(list=ls(all=TRUE)) #start with empty workspace

######################################################################
## Author: ... + Michael Höhle + ...
## Date:   13 Aug 2015
##
## Started with script from
## https://www.kaggle.com/karthikmurali11/grasp-and-lift-eeg-detection/logistic-regression-with-r-0-65/code
## and modified it, which included better looping (instead of hardcoded
## 1-2-3-4) and use of the readr package for faster I/O
##
## Todos:
##  - downsample as in https://www.kaggle.com/bitsofbits/grasp-and-lift-eeg-detection/naive-nnet (??)
##  - feature extraction form signals -> filters & frequencies, smooth?
##  - include previous frames in the regression
##  - study other scripts.
######################################################################

library("dplyr")
library("tidyr")
library("stringr")
library("readr")

###Register parallel
require("parallel")
require("doParallel")
require("foreach")
registerDoParallel(cores=4)
getDoParWorkers()

##Helper functions
paste00 <- function(x,...) paste0(x,...,collapse="")

##initialize my functions
merge_sort <- function (eeg, events) {
  ## merge events with data
  data <- merge(eeg,events,'id')
  data$value <- NULL
  ##seperate id into frame number and id
  data <- tidyr::separate(data,id,c('subject','series','frame'),sep='_')
  data$frame <- as.numeric(data$frame)
  ## order based on id
  data <- data[order(data$frame),]
  ##data$seriesno <- NULL
  data <- tidyr::unite(data,col=id,subject,series,frame,sep='_',remove = TRUE)
  return(data)
}

######################################################################
## First attempt of a model: simple logistic regression containing
## only main effects operating on the raw signals (no feature
## extraction).
##
## Params:
##  train - training df
##  test  - testing df
######################################################################
glm_regression <- function(train, test){

  ##Event types
  eventTypes <- c("HandStart","FirstDigitTouch","BothStartLoadPhase","LiftOff","Replace","BothReleased")
  ##Use all variables in th model
  RHS <- "Fp1 + Fp2 + F7 + F3 + Fz + F4 + F8 + FC5 + FC1 + FC2 + FC6 + T7 + C3 + Cz + C4 + T8 + TP9 + CP5 + CP1 + CP2 + CP6 + TP10 + P7 + P3 + Pz + P4 + P8 + PO9 + O1 + Oz + O2 + PO10"
  res <- list()

  for (ev in eventTypes) {
    cat("Type = ",ev,"\n")
    ##Formula for regression model
    formula <- as.formula(paste0(ev,"~", RHS))

    ##Fit the logistic regression model to training data
    reg_model <- glm(formula, data=train,family=binomial)

    ##Create predictions
    my_predict <- predict(reg_model,newdata=test,type='response')

    ##Make a data.frame with the result
    res[[paste0(ev)]] <-  data.frame(my_predict)
  }

  ##Combine into one data.frame
  my_solution <- tbl_df(data.frame(test$id,bind_cols(res)))
  colnames(my_solution) <- c("id",eventTypes)

  ##Done
  return(my_solution)
}

######################################################################
## Here the action starts.
######################################################################

set.seed(100)
options(scipen=999) #no scientific numbers
options (digits = 4) #hold decimal places to 4
## number of subjects you want to analyze
total_subj <- 12
## sub-sample training data to reduce computational load (min 1, max 8)
sub_sample <- 2#8
##initialize list
list_subj_predictions <- list()

## Loop through subjects
for (j in 1:total_subj) {
  cat(paste('Currently analysing subject',j),"\n")
  list_subj_traindata <- list() # initialize list
  subject <- j
  ## obtain all training data
  for (i in 1:sub_sample) {
    ##Read data
    file_name_eeg <- paste('../Data/train/subj',subject,'_series',i,
                           '_data.csv',sep='')
    file_name_events <- gsub("_data","_events",file_name_eeg)
    eeg <- read_csv(file_name_eeg, col_types=paste00("c",paste00(rep("n",32))))#read data.csv based on file name
    events <- read_csv(file_name_events,col_types=paste00("c",paste00(rep("n",6)))) #read events.csv based on file name

    ##Massage data
    list_subj_traindata[[i]] <- merge_sort(eeg,events)
    rm(eeg,events)
  }
  ##merge all training series data in 1 single data frame per subject
  train_variable_name <- paste('subj',subject,'_traindata',sep='')
  assign(train_variable_name,dplyr::rbind_all(list_subj_traindata))
  rm(list_subj_traindata)

  ##obtain and merge test data from all series
  list_subj_testdata <- list()
  ## i is series number (9 and 10 are always the test series)
  for (i in 9:10) {
    file_name_eeg <- paste('../Data/test/subj',subject,'_series',i,
                           '_data.csv',sep='')
    eeg <- read_csv(file_name_eeg, col_types=paste00("c",paste00(rep("n",32))))
    list_subj_testdata[[i-8]] <- eeg
    rm(eeg)
  }
  ##merge all series data in 1 single data frame per subject
  test_variable_name <- paste('subj',subject,'_testdata',sep='')
  assign(test_variable_name,rbind_all(list_subj_testdata))
  rm(list_subj_testdata)

  ##Logistic regression regression
  print('now performing regression')
  list_subj_predictions[[j]]<- glm_regression(get(train_variable_name), get(test_variable_name))

  rm(list=ls(pattern="^subj")) #remove variables in workspace to reduce memory usage
}

# combine all subject predictions into 1 dataframe for submission
my_submission <- rbind_all(list_subj_predictions)
my_submission[is.na(my_submission)] <- 0
## Reduce file size by limiting number of decimal places
my_submission[,-1]= round(my_submission[,-1],digits=4)
write.csv(my_submission,file='../Results/submission1.csv',row.names=FALSE)
zip("../Results/submission1.csv.zip",files="../Results/submission1.csv")
