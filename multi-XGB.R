setwd("C:/Users/Fake News Thesis/Multinomial")
train <- read.csv(file="multi_training.csv",header = T)
test <- read.csv(file="multi_testing.csv",header = T)
valid <- read.csv(file="multi_validating.csv",header = T)
library(xgboost)
library(mlr)

train$Label <- as.numeric(train$Label)-1 #multinomial (0-5)
test$Label <- as.numeric(test$Label)-1  #multinomial (0-5)
valid$Label <- as.numeric(valid$Label)-1  #multinomial (0-5)


#---- Model 1: PARTY + SPEAKER + STATE + SUBJECT + CREDIT HISTORY --------#
#Applying for Train dataset Advanced Features
dtrain <- xgb.DMatrix(data=as.matrix(train[c(-34,-35)]),label=train$Label)  #34=Search, 35=Label
dtest <- xgb.DMatrix(data=as.matrix(test[c(-34,-35)]),label=test$Label)   #34=Search, 35=Label
dvalid <- xgb.DMatrix(data=as.matrix(valid[c(-34,-35)]),label=valid$Label)   #34=Search, 35=Label

watchlist <- list(train=dtrain,test=dvalid)
# Calculate # of folds for cross-validation
# merror is for multiclass error with multi:softprob
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective = "multi:softprob", num_class = 6,
                 watchlist = watchlist)#tuned parameter by randomly selection


watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective="multi:softprob", num_class = 6,
                 watchlist = watchlist)#tuned parameter by randomly selection

#https://www.kaggle.com/general/17120
#Model 1: train-merror:0.531405	test-merror:0.586906


#------------- Model 2: PARTY + SPEAKER + STATE + SUBJECT + CREDIT HISTORY + SEARCH ---------------------#
dtrain <- xgb.DMatrix(data=as.matrix(train[c(-35)]),label=train$Label)  #35=Label
dtest <- xgb.DMatrix(data=as.matrix(test[c(-35)]),label=test$Label)   #35=Label
dvalid <- xgb.DMatrix(data=as.matrix(valid[c(-35)]),label=valid$Label)   #35=Label
watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective="multi:softprob", num_class = 6,
                 watchlist = watchlist)#tuned parameter by randomly selection

#https://www.kaggle.com/general/17120
#Model 2: train-merror:0.518454	test-merror:0.576773


#--------- Model 3: PARTY + SPEAKER + STATE + SUBJECT + CREDIT HISTORY + SEARCH + STATEMENT --------#

traindata <- cbind(statement.svd[1:10269,-1],train)
testdata <- cbind(statement.svd[10270:11552,-1],test)
validdata <- cbind(statement.svd[11553:12836,-1],valid)


dtrain <- xgb.DMatrix(data=as.matrix(traindata[c(-336)]),label=traindata$Label)
dtest <- xgb.DMatrix(data=as.matrix(testdata[c(-336)]),label=testdata$Label)
watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective="multi:softprob", num_class = 6,
                 watchlist = watchlist)#tuned parameter by randomly selection

#https://www.kaggle.com/general/17120
#Model 3: train-merror:0.433343	test-merror:0.596259

#--------- Model 4: PARTY + SPEAKER + STATE + SUBJECT + CREDIT HISTORY + SEARCH + CONTEXT --------#

traindata <- cbind(context.svd[1:10269,-1],train)
testdata <- cbind(context.svd[10270:11552,-1],test)
validdata <- cbind(context.svd[11553:12836,-1],valid)


dtrain <- xgb.DMatrix(data=as.matrix(traindata[c(-335)]),label=traindata$Label)
dtest <- xgb.DMatrix(data=as.matrix(testdata[c(-335)]),label=testdata$Label)
dvalid <- xgb.DMatrix(data=as.matrix(validdata[c(-335)]),label=validdata$Label)

watchlist <- list(train=dtrain,test=dvalid)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective="multi:softprob", num_class = 6,
                 watchlist = watchlist)#tuned parameter by randomly selection
#Model 4: train-merror:0.466842	test-merror:0.5862

watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective="multi:softprob", num_class = 6,
                 watchlist = watchlist)#tuned parameter by randomly selection

#https://www.kaggle.com/general/17120
#Model 4: train-merror:0.466842	test-merror:0.575994


#------------- Model 5: FULL MODEL ---------------------#
traindata <- cbind(statement.svd[1:10269,-1],context.svd[1:10269,-1],train)
testdata <- cbind(statement.svd[10270:11552,-1], context.svd[10270:11552,-1],test)

dtrain <- xgb.DMatrix(data=as.matrix(traindata[c(-636)]),label=traindata$Label)
dtest <- xgb.DMatrix(data=as.matrix(testdata[c(-636)]),label=testdata$Label)
watchlist <- list(train=dtrain,test=dtest)
bst <- xgb.train(data=dtrain, max_depth=5, eta=0.6, nthread=3, nrounds = 4,
                 eval_metric = "merror", objective="multi:softprob", num_class = 6,
                 watchlist = watchlist)#tuned parameter by randomly selection

#https://www.kaggle.com/general/17120
#Model 5: train-merror:0.432564	test-merror:0.586906

