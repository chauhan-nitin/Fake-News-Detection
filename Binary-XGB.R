setwd("C:/Users/Fake News Thesis/Binary")
train <- read.csv(file="binary_training.csv", header=T)
test <- read.csv(file="binary_testing.csv", header = T)
valid <- read.csv(file="binary_validating.csv", header=T)
library(xgboost)
library(mlr)


#---- Model 1: PARTY + SPEAKER + STATE + SUBJECT + CREDIT HISTORY --------#
#Applying for Train dataset Advanced Features
dtrain <- xgb.DMatrix(data=as.matrix(train[c(-31,-32)]),label=train$Label) #31=Search, 32=Label
dtest <- xgb.DMatrix(data=as.matrix(test[c(-31,-32)]),label=test$Label) #31=Search, 32=Label
dvalid <- xgb.DMatrix(data=as.matrix(valid[c(-31,-32)]),label=valid$Label) #31=Search, 32=Label

watchlist <- list(train=dtrain,test=dvalid)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "error", objective="binary:logistic", 
                 watchlist = watchlist)#tuned parameter by randomly selection
#train-error:0.304022	test-error:0.317757

watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "error", objective="binary:logistic", 
                 watchlist = watchlist)#tuned parameter by randomly selection


#https://www.kaggle.com/general/17120
#Model 1: train-error:0.304022	test-error:0.296181 

#------------- Model 2: PARTY + SPEAKER + STATE + SUBJECT + CREDIT HISTORY + SEARCH ---------------------#
dtrain <- xgb.DMatrix(data=as.matrix(train[c(-32)]),label=train$Label) #32=Label
dtest <- xgb.DMatrix(data=as.matrix(test[c(-32)]),label=test$Label) #32=Label
watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "error", objective="binary:logistic", 
                 watchlist = watchlist)#tuned parameter by randomly selecting
#https://www.kaggle.com/general/17120

#Model 2: train-error:0.301100	test-error:0.296181


#--------- Model 3: PARTY + SPEAKER + STATE + SUBJECT + CREDIT HISTORY +SEARCH + STATEMENT --------#

traindata <- cbind(statement.svd[1:10269,-1],train)
testdata <- cbind(statement.svd[10270:11552,-1],test)
validdata <- cbind(statement.svd[11553:12836,-1],valid)

dtrain <- xgb.DMatrix(data=as.matrix(traindata[c(-333)]),label=traindata$Label)
dtest <- xgb.DMatrix(data=as.matrix(testdata[c(-333)]),label=testdata$Label)
watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "error", objective="binary:logistic", 
                 watchlist = watchlist)#tuned parameter by randomly selection
# Model 3: train-error:0.279579	test-error:0.317225 


#--------- Model 4: PARTY + SPEAKER + STATE + SUBJECT + CREDIT HISTORY + SEARCH + CONTEXT --------#

traindata <- cbind(context.svd[1:10269,-1],train)
testdata <- cbind(context.svd[10270:11552,-1],test)
validdata <- cbind(context.svd[11553:12836,-1],valid)

dtrain <- xgb.DMatrix(data=as.matrix(traindata[c(-332)]),label=traindata$Label)
dtest <- xgb.DMatrix(data=as.matrix(testdata[c(-332)]),label=testdata$Label)
watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "error", objective="binary:logistic", 
                 watchlist = watchlist)#tuned parameter by randomly selection
#Model 4: train-error:0.286785	test-error:0.326578

#------------- Model 5: FULL MODEL ---------------------#
traindata <- cbind(statement.svd[1:10269,-1],context.svd[1:10269,-1],train)
testdata <- cbind(statement.svd[10270:11552,-1], context.svd[10270:11552,-1],test)
dtrain <- xgb.DMatrix(data=as.matrix(traindata[c(-633)]),label=traindata$Label)
dtest <- xgb.DMatrix(data=as.matrix(testdata[c(-633)]),label=testdata$Label)
watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "error", objective="binary:logistic", 
                 watchlist = watchlist)#tuned parameter by randomly selecting
#Model 5: train-error:0.269452	test-error:0.322681 

