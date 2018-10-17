library(keras)
install_keras()
setwd("C:/Users/Fake News Thesis/Multinomial")
train <- read.csv(file="multi_training.csv",header = T)
test <- read.csv(file="multi_testing.csv",header = T)
valid <- read.csv(file="multi_validating.csv",header = T)

levels(train$Label)
train$Label = factor(train$Label,levels(train$Label)[c(5,2,1,3,4,6)])
train$Label <- as.numeric(train$Label)
#META DATA
#Pants-Fire=5, False=2, Barely-True=1, Half-True=3, Mostly-True=4, True=6

levels(test$Label)
test$Label = factor(test$Label,levels(test$Label)[c(5,2,1,3,4,6)])
test$Label <- as.numeric(test$Label)
#META DATA
#Pants-Fire=5, False=2, Barely-True=1, Half-True=3, Mostly-True=4, True=6

levels(valid$Label)
valid$Label = factor(valid$Label,levels(valid$Label)[c(5,2,1,3,4,6)])
valid$Label <- as.numeric(valid$Label)
#META DATA
#Pants-Fire=5, False=2, Barely-True=1, Half-True=3, Mostly-True=4, True=6

#------------ Model 1: FULL MODEL -------------------#
#Checking the structure of Data
str(train)
newtrain <- as.matrix(train[,-34]) #Without Search = 34th Column
dimnames(newtrain) <- NULL

str(test)
newtest <- as.matrix(test[,-34]) #Without Search = 34th Column
dimnames(newtest) <- NULL

str(valid)
newvalid <- as.matrix(valid[,-34]) #Without Search = 34th Column
dimnames(newvalid) <- NULL

#Normalize the matrix
options(scipen = 999)
#Check the attributes used

newtrain[,1:33] <- normalize(newtrain[,1:33]) #Independent = Predictors
newtrain[,34] <- as.numeric(newtrain[,34]) -1  #Label = 0,1,2,3,4,5
summary(newtrain)
training <- newtrain[,1:33]
traintarget <- newtrain[,34]

newtest[,1:33] <- normalize(newtest[,1:33]) #Independent = Predictors
newtest[,34] <- as.numeric(newtest[,34]) -1  #Label = 0,1,2,3,4,5
summary(newtest)
testing <- newtest[,1:33]
testtarget <- newtest[,34]

newvalid[,1:33] <- normalize(newvalid[,1:33]) #Independent = Predictors
newvalid[,34] <- as.numeric(newvalid[,34]) -1 #Label = 0,1,2,3,4,5
summary(newvalid)
validating <- newvalid[,1:33]
validtarget <- newvalid[,34]

#One hot encoding
trainlabels <- to_categorical(traintarget)
testlabels <- to_categorical(testtarget)
validlabels <- to_categorical(validtarget)

#Create Sequential Model
model <- keras_model_sequential()
#Pipe symbol %>%
model %>% 
  layer_dense(units = 9, activation = 'relu', 
              input_shape = c(33))%>%
  layer_dense(units = 9, activation = 'relu') %>%
  layer_dense(units = 6, activation = 'softmax')
summary(model)        
#layer_dense(units = 2, activation = 'softmax') is Visible/Output layer
#33 independent * 9 layers= 297 + 9 constant values for each node = 306

#Compile the model
#For 6 class problem we will use "categorical_crossentropy"
model %>%
  compile(loss ='categorical_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy')

#Fit the model (Multi-layer Perceptron Neural Network)
#Using Train data on Validation set
history <- model %>%
  fit(training,
      trainlabels,
      epoch=200,
      validation_data = list(validating,validlabels)
  )
plot(history)

#Evaluate the model using Test Data
model1 <- model %>%
  evaluate(testing,testlabels)

#Prediction and Confusion Matrix - test data
prob <- model %>%
  predict_proba(testing)
pred <- model %>%
  predict_classes(testing)
table1 <- table(Predicted=pred, Actual=testtarget)


cbind(prob,pred,testtarget)

model1
table1

#------------ Model 2: FULL MODEL + SEARCH-------------------#
#Checking the structure of Data
str(train)
newtrain <- as.matrix(train)
dimnames(newtrain) <- NULL

str(test)
newtest <- as.matrix(test)
dimnames(newtest) <- NULL

str(valid)
newvalid <- as.matrix(valid)
dimnames(newvalid) <- NULL

#Normalize the matrix
options(scipen = 999)
#Check the attributes used

newtrain[,1:34] <- normalize(newtrain[,1:34])
newtrain[,35] <- as.numeric(newtrain[,35]) -1
summary(newtrain)
training <- newtrain[,1:34]
traintarget <- newtrain[,35]

newtest[,1:34] <- normalize(newtest[,1:34])
newtest[,35] <- as.numeric(newtest[,35]) -1
summary(newtest)
testing <- newtest[,1:34]
testtarget <- newtest[,35]

newvalid[,1:34] <- normalize(newvalid[,1:34])
newvalid[,35] <- as.numeric(newvalid[,35]) -1
summary(newvalid)
validating <- newvalid[,1:34]
validtarget <- newvalid[,35]

#One hot encoding
trainlabels <- to_categorical(traintarget)
testlabels <- to_categorical(testtarget)
validlabels <- to_categorical(validtarget)

#Create Sequential Model
model <- keras_model_sequential()
#Pipe symbol %>%
model %>% 
  layer_dense(units = 9, activation = 'relu', 
              input_shape = c(34))%>%
  layer_dense(units = 9, activation = 'relu') %>%
  layer_dense(units = 6, activation = 'softmax')
summary(model)        
#layer_dense(units = 2, activation = 'softmax') is Visible/Output layer
#31 independent * 8 layers=256 + 8 constant values for each node = 256
#2 nodes * 8 layers = 16 + 2 constant values = 18

#Compile the model
#For 2 class problem we will use "binary_crossentropy"
model %>%
  compile(loss ='categorical_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy')

#Fit the model (Multi-layer Perceptron Neural Network)
#Using Train data on Validation set
history <- model %>%
  fit(training,
      trainlabels,
      epoch=200,
      validation_data = list(validating,validlabels))
plot(history)

#Evaluate the model using Test Data
model2 <- model %>%
  evaluate(testing,testlabels)

#Prediction and Confusion Matrix - test data
prob <- model %>%
  predict_proba(testing)
pred <- model %>%
  predict_classes(testing)
table2 <- table(Predicted=pred, Actual=testtarget)


cbind(prob,pred,testtarget)

model2
table2
#model1 vs model2, SEARCH improves the accuracy by 5% (40 to 45) and reduces loss by 8.7%

#------------ Model 3: FULL MODEL + SEARCH + STATEMENT -------------------#
#Checking the structure of Data
traindata <- cbind(statement.svd[1:10269,-1],train)
traindata[1,336]
str(traindata)
newtrain <- as.matrix(traindata)
dimnames(newtrain) <- NULL

testdata <- cbind(statement.svd[10270:11552,-1],test)
str(testdata)
newtest <- as.matrix(testdata)
dimnames(newtest) <- NULL

validdata <- cbind(statement.svd[11553:12836,-1],valid)
str(validdata)
newvalid <- as.matrix(validdata)
dimnames(newvalid) <- NULL

#Normalize the matrix
options(scipen = 999)
#Check the attributes used

newtrain[,1:335] <- normalize(newtrain[,1:335]) #Independent = Predictor
newtrain[,336] <- as.numeric(newtrain[,336]) -1 #Label = 0,1,2,3,4,5
summary(newtrain)
training <- newtrain[,1:335]
traintarget <- newtrain[,336]

newtest[,1:335] <- normalize(newtest[,1:335]) #Independent = Predictor
newtest[,336] <- as.numeric(newtest[,336]) -1 #Label = 0,1,2,3,4,5
summary(newtest)
testing <- newtest[,1:335]
testtarget <- newtest[,336]

newvalid[,1:335] <- normalize(newvalid[,1:335]) #Independent = Predictor
newvalid[,336] <- as.numeric(newvalid[,336]) -1#Label = 0,1,2,3,4,5
summary(newvalid)
validating <- newvalid[,1:335]
validtarget <- newvalid[,336]

#One hot encoding
trainlabels <- to_categorical(traintarget)
testlabels <- to_categorical(testtarget)
validlabels <- to_categorical(validtarget)

#Create Sequential Model
model <- keras_model_sequential()
#Pipe symbol %>%
model %>% 
  layer_dense(units = 9, activation = 'relu', 
              input_shape = c(335))%>%
  layer_dense(units = 9, activation = 'relu') %>%
  layer_dense(units = 6, activation = 'softmax')
summary(model)        
#layer_dense(units = 2, activation = 'softmax') is Visible/Output layer
#31 independent * 8 layers=256 + 8 constant values for each node = 256
#2 nodes * 8 layers = 16 + 2 constant values = 18

#Compile the model
#For 2 class problem we will use "binary_crossentropy"
model %>%
  compile(loss ='categorical_crossentropy',
          optimizer = 'adam',
          metrics = 'accuracy')

#Fit the model (Multi-layer Perceptron Neural Network)
#Using Train data on Validation set
history <- model %>%
  fit(training,
      trainlabels,
      epoch=200,
      validation_data = list(validating,validlabels))
plot(history)

#Evaluate the model using Test Data
model3 <- model %>%
  evaluate(testing,testlabels)

#Prediction and Confusion Matrix - test data
prob <- model %>%
  predict_proba(testing)
pred <- model %>%
  predict_classes(testing)
table3 <- table(Predicted=pred, Actual=testtarget)


cbind(prob,pred,testtarget)

model3
table3
#No significant improvement over Model2 in terms of accuracy and loss
