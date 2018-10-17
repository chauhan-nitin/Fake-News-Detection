# Random Forest
library("randomForest")
set.seed(1234)
setwd("C:/Users/Fake News Thesis/Multinomial")
train <- read.csv(file="multi_training.csv", header=T)
test <- read.csv(file="multi_testing.csv", header = T)
valid <- read.csv(file="multi_validating.csv", header=T)

#-------------------- Model 1: FULL MODEL ------------------

#Grid Search for Tuning Hyperparameters using CARET
library(caret)
?trainControl
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")#repeated cross-validation, 3-fold cross-validation, 10 resampling iterations, grid search
set.seed(1234)
tunegrid <- expand.grid(.mtry=c(1:15))
#Tuning hyperparameters using Validation set
rf_gridsearch <- train(factor(Label)~., data=valid[,-34], method="rf", metric="Accuracy", tuneGrid=tunegrid, trControl=control) #34=Search
print(rf_gridsearch)
plot(rf_gridsearch)
#4    0.4210956  0.2884003
#Creating the model based on the best value for mtry (Number of variables randomly sampled as candidates at each split).
?randomForest
rf <- randomForest(factor(train$Label) ~ ., data=train[,-34], keep.forest=TRUE, ntree=500, mtry=4) #34=Search
print(rf) #OOB estimate of  error rate: 58.66%
varImpPlot(rf)
#confusion Matrix
table(train$Label, predict(rf, train[,-35], type="response", norm.votes=TRUE)) #35=Label
RFpred <- predict(rf, test[,-35], type="response", norm.votes=TRUE) #35=Label
confusionMatrix(RFpred, factor(test$Label)) #This function from caret package gives every detail like accuracy,sensitivity ect
print("Accuracy of Grid Search Random Forest is 40.22%")


#Tuning using Algorithm Tools
#Algorithm Tune using TuneRF
x <- valid[,-35]
y <- valid[,35]
y <- as.factor(y)
bestmtry <- tuneRF(x,y, stepFactor=1.5, improve=1e-5, ntreeTry = 500)
print(bestmtry)
rf1 <- randomForest(factor(train$Label) ~ ., data=train[,-34], keep.forest=TRUE, ntree=500, mtry=4) #34=Search
print(rf1) #OOB estimate of  error rate: 58.38%
varImpPlot(rf1)
#confusion Matrix
table(train$Label, predict(rf1, train[,-35], type="response", norm.votes=TRUE)) #35=Label
RFpred <- predict(rf1, test[,-35], type="response", norm.votes=TRUE) #35=Label
confusionMatrix(RFpred, factor(test$Label)) #This function from caret package gives every detail like accuracy,sensitivity ect
print("Accuracy of Algorithm Tune Random Forest is 40.84%")


#-------------------- Model 2: FULL MODEL + Statement ------------------
traindata <- cbind(statement.svd[1:10269,-1],train)
testdata <- cbind(statement.svd[10270:11552,-1],test)
validdata <- cbind(statement.svd[11553:12836,-1],valid)

rf <- randomForest(factor(traindata$Label) ~ ., data=traindata[,-335], keep.forest=TRUE, ntree=500, mtry=4)
print(rf) #OOB estimate of  error rate:  66.53%
varImpPlot(rf)
#confusion Matrix
table(traindata$Label, predict(rf, traindata[,-336], type="response", norm.votes=TRUE))
RFpred <- predict(rf, testdata[,-336], type="response", norm.votes=TRUE)
confusionMatrix(RFpred, factor(testdata$Label)) #This function from caret package gives every detail like accuracy,sensitivity ect
print("Accuracy of Grid Search Random Forest is 37.02%")


#----------------- Model 3: FULL MODEL + Context -------------------
traindata <- cbind(context.svd[1:10269,-1],train)
testdata <- cbind(context.svd[10270:11552,-1],test)
validdata <- cbind(context.svd[11553:12836,-1],valid)
rf <- randomForest(factor(traindata$Label) ~ ., data=traindata[,-334], keep.forest=TRUE, ntree=500, mtry=6)
print(rf) #OOB estimate of  error rate: 70.81%
varImpPlot(rf)
#confusion Matrix
table(traindata$Label, predict(rf, traindata[,-335], type="response", norm.votes=TRUE))
RFpred <- predict(rf, testdata[,-335], type="response", norm.votes=TRUE)
confusionMatrix(RFpred, factor(testdata$Label)) #This function from caret package gives every detail like accuracy,sensitivity ect
print("Accuracy of Grid Search Random Forest is 28.37%")


#----------------- Model 4: FULL MODEL + SEARCH -------------------
rf1 <- randomForest(factor(train$Label) ~ ., data=train, keep.forest=TRUE, ntree=500, mtry=4)
print(rf1) #OOB estimate of  error rate: 57.64%
varImpPlot(rf1)
table(train$Label, predict(rf1, train[,-35], type="response", norm.votes=TRUE))
RFpred <- predict(rf1, test[,-35], type="response", norm.votes=TRUE)
confusionMatrix(RFpred, factor(test$Label)) #This function from caret package gives every detail like accuracy,sensitivity ect
print("Accuracy of Algorithm Tune Random Forest is 41.7%")
