# Random Forest
library("randomForest")
set.seed(1234)
setwd("C:/Users/Fake News Thesis/Three")
train <- read.csv(file="three_training.csv",header = T)
test <- read.csv(file="three_testing.csv",header = T)
valid <- read.csv(file="three_validating.csv",header = T)

#-------------------- Model 1: FULL MODEL ------------------

#Grid Search for Tuning Hyperparameters using CARET
library(caret)
?trainControl
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")#repeated cross-validation, 3-fold cross-validation, 10 resampling iterations, grid search
set.seed(1234)
tunegrid <- expand.grid(.mtry=c(1:15))
#Tuning hyperparameters using Validation set
rf_gridsearch <- train(factor(Label)~., data=valid[,-32], method="rf", metric="Accuracy", tuneGrid=tunegrid, trControl=control) #32=Search
print(rf_gridsearch)
plot(rf_gridsearch)
#7    0.5796557  0.36304949
#Creating the model based on the best value for mtry (Number of variables randomly sampled as candidates at each split).
?randomForest
rf <- randomForest(factor(train$Label) ~ ., data=train[,-32], keep.forest=TRUE, ntree=500, mtry=7) #32=Search
print(rf) #OOB estimate of  error rate: 43.02%
varImpPlot(rf)
#confusion Matrix
table(train$Label, predict(rf, train[,-33], type="response", norm.votes=TRUE)) #33=Label
RFpred <- predict(rf, test[,-33], type="response", norm.votes=TRUE) #33=Label
confusionMatrix(RFpred, factor(test$Label)) #This function from caret package gives every detail like accuracy,sensitivity ect
print("Accuracy of Grid Search Random Forest is 55.88%")


#Tuning using Algorithm Tools
#Algorithm Tune using TuneRF
x <- valid[,-33]
y <- valid[,33]
y <- as.factor(y)
bestmtry <- tuneRF(x,y, stepFactor=1.5, improve=1e-5, ntreeTry = 500)
print(bestmtry)
rf1 <- randomForest(factor(train$Label) ~ ., data=train[,-32], keep.forest=TRUE, ntree=500, mtry=5) #32=Search
print(rf1) # OOB estimate of  error rate: 42.77%
varImpPlot(rf1)
#confusion Matrix
table(train$Label, predict(rf1, train[,-33], type="response", norm.votes=TRUE)) #33=Label
RFpred <- predict(rf1, test[,-33], type="response", norm.votes=TRUE) #33=Label
confusionMatrix(RFpred, factor(test$Label)) #This function from caret package gives every detail like accuracy,sensitivity ect
print("Accuracy of Algorithm Tune Random Forest is 56.66%")


#-------------------- Model 2: FULL MODEL + Statement ------------------
traindata <- cbind(statement.svd[1:10269,-1],train)
testdata <- cbind(statement.svd[10270:11552,-1],test)
validdata <- cbind(statement.svd[11553:12836,-1],valid)

rf <- randomForest(factor(traindata$Label) ~ ., data=traindata[,-333], keep.forest=TRUE, ntree=500, mtry=5) #333 = Search
print(rf) #OOB estimate of  error rate: 47.73%
varImpPlot(rf)
#confusion Matrix
table(traindata$Label, predict(rf, traindata[,-334], type="response", norm.votes=TRUE)) # 334 = Label
RFpred <- predict(rf, testdata[,-334], type="response", norm.votes=TRUE) # 334 = Label
confusionMatrix(RFpred, factor(testdata$Label)) #This function from caret package gives every detail like accuracy,sensitivity ect
print("Accuracy of Grid Search Random Forest is 54.25%")


#----------------- Model 3: FULL MODEL + Context -------------------
traindata <- cbind(context.svd[1:10269,-1],train)
testdata <- cbind(context.svd[10270:11552,-1],test)
validdata <- cbind(context.svd[11553:12836,-1],valid)
rf <- randomForest(factor(traindata$Label) ~ ., data=traindata[,-332], keep.forest=TRUE, ntree=500, mtry=5) #332 = Search
print(rf) #OOB estimate of  error rate: 55.49%
varImpPlot(rf)
#confusion Matrix
table(traindata$Label, predict(rf, traindata[,-333], type="response", norm.votes=TRUE)) # 333 = Label
RFpred <- predict(rf, testdata[,-333], type="response", norm.votes=TRUE) # 333 = Label
confusionMatrix(RFpred, factor(testdata$Label)) #This function from caret package gives every detail like accuracy,sensitivity ect
print("Accuracy of Grid Search Random Forest is 44.51%")


#----------------- Model 4: FULL MODEL + SEARCH -------------------
rf1 <- randomForest(factor(train$Label) ~ ., data=train, keep.forest=TRUE, ntree=500, mtry=5)
print(rf1) #OOB estimate of  error rate: 42.73%
varImpPlot(rf1)
table(train$Label, predict(rf1, train[,-33], type="response", norm.votes=TRUE)) # 33 = Label
RFpred <- predict(rf1, test[,-33], type="response", norm.votes=TRUE) # 33 = Label
confusionMatrix(RFpred, factor(test$Label)) #This function from caret package gives every detail like accuracy,sensitivity ect
print("Accuracy of Algorithm Tune Random Forest is 57.29%")
