library(keras)
install_keras()
setwd("C:/Users/Fake News Thesis/Three")
train <- read.csv(file="three_training.csv",header = T)
test <- read.csv(file="three_testing.csv",header = T)
valid <- read.csv(file="three_validating.csv",header = T)
levels(train$Label)
train$Label = factor(train$Label)
train$Label <- as.numeric(train$Label)
#META DATA
#False=1, Half-True=2, True=3

levels(test$Label)
test$Label = factor(test$Label)
test$Label <- as.numeric(test$Label)
#META DATA
#False=1, Half-True=2, True=3

levels(valid$Label)
valid$Label = factor(valid$Label)
valid$Label <- as.numeric(valid$Label)
#META DATA
#False=1, Half-True=2, True=3

#------------ Model 1: FULL MODEL -------------------#
#Checking the structure of Data
str(train)
newtrain <- as.matrix(train[,-32]) #Without Search = 32th Column
dimnames(newtrain) <- NULL

str(test)
newtest <- as.matrix(test[,-32]) #Without Search = 32th Column
dimnames(newtest) <- NULL

str(valid)
newvalid <- as.matrix(valid[,-32]) #Without Search = 32th Column
dimnames(newvalid) <- NULL

#Normalize the matrix
options(scipen = 999)
#Check the attributes used
#Total columns available is 32
newtrain[,1:31] <- normalize(newtrain[,1:31]) #Independent = Predictors
newtrain[,32] <- as.numeric(newtrain[,32]) -1  #Label = 0,1,2
summary(newtrain)
training <- newtrain[,1:31]
traintarget <- newtrain[,32]

newtest[,1:31] <- normalize(newtest[,1:31]) #Independent = Predictors
newtest[,32] <- as.numeric(newtest[,32]) -1  #Label = 0,1,2
summary(newtest)
testing <- newtest[,1:31]
testtarget <- newtest[,32]

newvalid[,1:31] <- normalize(newvalid[,1:31]) #Independent = Predictors
newvalid[,32] <- as.numeric(newvalid[,32]) -1 #Label = 0,1,2
summary(newvalid)
validating <- newvalid[,1:31]
validtarget <- newvalid[,32]

#One hot encoding
trainlabels <- to_categorical(traintarget)
testlabels <- to_categorical(testtarget)
validlabels <- to_categorical(validtarget)

#Create Sequential Model
model <- keras_model_sequential()
#Pipe symbol %>%
model %>% 
  layer_dense(units = 9, activation = 'relu', 
              input_shape = c(31))%>%
  layer_dense(units = 9, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')
summary(model)        
#31 independent * 9 layers= 279 + 9 constant values for each node = 288
#layer_dense(units = 3, activation = 'softmax') is Visible/Output layer with 3 classes

#Compile the model
#For 3 class problem we will use "categorical_crossentropy"
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

#Evaluate the model using Valid Data
model1 <- model %>%
  evaluate(validating,validlabels)

#Prediction and Confusion Matrix - Valid data
prob <- model %>%
  predict_proba(validating)
pred <- model %>%
  predict_classes(validating)
table1 <- table(Predicted=pred, Actual=validtarget)


cbind(prob,pred,validtarget)

model1

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

newtrain[,1:32] <- normalize(newtrain[,1:32]) #Independent = Predictors
newtrain[,33] <- as.numeric(newtrain[,33]) -1 #Label = 0,1,2
summary(newtrain)
training <- newtrain[,1:32]
traintarget <- newtrain[,33]

newtest[,1:32] <- normalize(newtest[,1:32]) #Independent = Predictors
newtest[,33] <- as.numeric(newtest[,33]) -1 #Label = 0,1,2
summary(newtest)
testing <- newtest[,1:32]
testtarget <- newtest[,33]

newvalid[,1:32] <- normalize(newvalid[,1:32]) #Independent = Predictors
newvalid[,33] <- as.numeric(newvalid[,33]) -1 #Label = 0,1,2
summary(newvalid)
validating <- newvalid[,1:32]
validtarget <- newvalid[,33]

#One hot encoding
trainlabels <- to_categorical(traintarget)
testlabels <- to_categorical(testtarget)
validlabels <- to_categorical(validtarget)

#Create Sequential Model
model <- keras_model_sequential()
#Pipe symbol %>%
model %>% 
  layer_dense(units = 9, activation = 'relu', 
              input_shape = c(32))%>%
  layer_dense(units = 9, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')
summary(model)        

#layer_dense(units = 3, activation = 'softmax') is Visible/Output layer with 3 classes
#32 independent * 9 layers=288 + 9 constant values for each node = 297


#Compile the model
#For 3 class problem we will use "categorical_crossentropy"
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

#Evaluate the model using Valid Data
model2 <- model %>%
  evaluate(validating,validlabels)

#Prediction and Confusion Matrix - Valid data
prob <- model %>%
  predict_proba(validating)
pred <- model %>%
  predict_classes(validating)
table2 <- table(Predicted=pred, Actual=validtarget)


cbind(prob,pred,validtarget)

model2
table2
#model1 vs model2, SEARCH doesn't improves the accuracy.
#accuracy(0.5939205 vs 0.5853468) and loss(0.8303358 vs 0.8315542)

#------------ Model 3: FULL MODEL + SEARCH + STATEMENT -------------------#
#Checking the structure of Data
traindata <- cbind(statement.svd[1:10269,-1],train)
traindata[1,334]
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

newtrain[,1:333] <- normalize(newtrain[,1:333]) #Independent = Predictor
newtrain[,334] <- as.numeric(newtrain[,334]) -1 #Label = 0,1,2
summary(newtrain)
training <- newtrain[,1:333]
traintarget <- newtrain[,334]

newtest[,1:333] <- normalize(newtest[,1:333]) #Independent = Predictor
newtest[,334] <- as.numeric(newtest[,334]) -1 #Label = 0,1,2
summary(newtest)
testing <- newtest[,1:333]
testtarget <- newtest[,334]

newvalid[,1:333] <- normalize(newvalid[,1:333]) #Independent = Predictor
newvalid[,334] <- as.numeric(newvalid[,334]) -1 #Label = 0,1,2
summary(newvalid)
validating <- newvalid[,1:333]
validtarget <- newvalid[,334]

#One hot encoding
trainlabels <- to_categorical(traintarget)
testlabels <- to_categorical(testtarget)
validlabels <- to_categorical(validtarget)

#Create Sequential Model
model <- keras_model_sequential()
#Pipe symbol %>%
model %>% 
  layer_dense(units = 9, activation = 'relu', 
              input_shape = c(333))%>%
  layer_dense(units = 9, activation = 'relu') %>%
  layer_dense(units = 3, activation = 'softmax')
summary(model)        
#layer_dense(units = 3, activation = 'softmax') is Visible/Output layer with 3 classes
#333 independent * 9 layers=2997 + 9 constant values for each node = 3006


#Compile the model
#For 3 class problem we will use "categorical_crossentropy"
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

#Evaluate the model using Valid Data
model3 <- model %>%
  evaluate(validating,validlabels)

#Prediction and Confusion Matrix - Valid data
prob <- model %>%
  predict_proba(validating)
pred <- model %>%
  predict_classes(validating)
table3 <- table(Predicted=pred, Actual=validtarget)


cbind(prob,pred,validtarget)

model3
table3
#Model 3 performs better over Model 1 & 2. The accuracy improves
#accuracy (0.6024942) and loss (0.8135661)
