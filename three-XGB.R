setwd("C:/Users/Fake News Thesis/Three")
train <- read.csv(file="three_training.csv",header = T)
test <- read.csv(file="three_testing.csv",header = T)
valid <- read.csv(file="three_validating.csv",header = T)
library(xgboost)
library(mlr)

train$Label <- as.numeric(train$Label)-1 #multinomial three (0-2)
test$Label <- as.numeric(test$Label)-1  #multinomial three (0-2)
valid$Label <- as.numeric(valid$Label)-1  #multinomial three (0-2)


#---- Model 1: PARTY + SPEAKER + STATE + SUBJECT + CREDIT HISTORY --------#
#Applying for Train dataset Advanced Features
dtrain <- xgb.DMatrix(data=as.matrix(train[c(-32,-33)]),label=train$Label)  #32=Search, 33=Label
dtest <- xgb.DMatrix(data=as.matrix(test[c(-32,-33)]),label=test$Label)   #32=Search, 33=Label
dvalid <- xgb.DMatrix(data=as.matrix(valid[c(-32,-33)]),label=valid$Label)   #32=Search, 33=Label

watchlist <- list(train=dtrain,test=dvalid)
# Calculate # of folds for cross-validation
# merror is for multiclass error with multi:softprob
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective = "multi:softprob", num_class = 3,
                 watchlist = watchlist)#tuned parameter by randomly selection


watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective="multi:softprob", num_class = 3,
                 watchlist = watchlist)#tuned parameter by randomly selection

#https://www.kaggle.com/general/17120
#Model 1: train-merror:0.397994	test-merror:0.420109 


#------------- Model 2: PARTY + SPEAKER + STATE + SUBJECT + CREDIT HISTORY + SEARCH ---------------------#
dtrain <- xgb.DMatrix(data=as.matrix(train[c(-33)]),label=train$Label)  #33=Label
dtest <- xgb.DMatrix(data=as.matrix(test[c(-33)]),label=test$Label)   #33=Label
dvalid <- xgb.DMatrix(data=as.matrix(valid[c(-33)]),label=valid$Label)   #33=Label
watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective="multi:softprob", num_class = 3,
                 watchlist = watchlist)#tuned parameter by randomly selection

#https://www.kaggle.com/general/17120
#Model 2: train-merror:0.387379	test-merror:0.425565 


#--------- Model 3: PARTY + SPEAKER + STATE + SUBJECT + CREDIT HISTORY + STATEMENT --------#

traindata <- cbind(statement.svd[1:10269,-1],train)
testdata <- cbind(statement.svd[10270:11552,-1],test)
validdata <- cbind(statement.svd[11553:12836,-1],valid)


dtrain <- xgb.DMatrix(data=as.matrix(traindata[c(-334)]),label=traindata$Label)
dtest <- xgb.DMatrix(data=as.matrix(testdata[c(-334)]),label=testdata$Label)
watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective="multi:softprob", num_class = 3,
                 watchlist = watchlist)#tuned parameter by randomly selection

#https://www.kaggle.com/general/17120
#Model 3: train-merror:0.346188	test-merror:0.438815 

#--------- Model 4: PARTY + SPEAKER + STATE + SUBJECT + CREDIT HISTORY + CONTEXT --------#

traindata <- cbind(context.svd[1:10269,-1],train)
testdata <- cbind(context.svd[10270:11552,-1],test)
validdata <- cbind(context.svd[11553:12836,-1],valid)


dtrain <- xgb.DMatrix(data=as.matrix(traindata[c(-333)]),label=traindata$Label)
dtest <- xgb.DMatrix(data=as.matrix(testdata[c(-333)]),label=testdata$Label)
watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective="multi:softprob", num_class = 3,
                 watchlist = watchlist)#tuned parameter by randomly selection

#https://www.kaggle.com/general/17120
#Model 4: train-merror:0.360016	test-merror:0.435698


#------------- Model 5: FULL MODEL ---------------------#
traindata <- cbind(statement.svd[1:10269,-1],context.svd[1:10269,-1],train)
testdata <- cbind(statement.svd[10270:11552,-1], context.svd[10270:11552,-1],test)

dtrain <- xgb.DMatrix(data=as.matrix(traindata[c(-634)]),label=traindata$Label)
dtest <- xgb.DMatrix(data=as.matrix(testdata[c(-634)]),label=testdata$Label)
watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective="multi:softprob", num_class = 3,
                 watchlist = watchlist)#tuned parameter by randomly selection

#https://www.kaggle.com/general/17120
#Model 5: train-merror:0.337326	test-merror:0.429462 

