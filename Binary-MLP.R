library(keras)
install_keras()
setwd("C:/Users/Fake News Thesis/Binary")
train <- read.csv(file="binary_training.csv",header = T)
test <- read.csv(file="binary_testing.csv",header = T)
valid <- read.csv(file="binary_validating.csv",header = T)

#------------ Model 1: FULL MODEL -------------------#
#Checking the structure of Data
str(train)
newtrain <- as.matrix(train[,-31]) #Without Search
dimnames(newtrain) <- NULL

str(test)
newtest <- as.matrix(test[,-31]) #Without Search
dimnames(newtest) <- NULL

str(valid)
newvalid <- as.matrix(valid[,-31]) #Without Search
dimnames(newvalid) <- NULL

#Normalize the matrix
options(scipen = 999)
#Check the attributes used

newtrain[,1:30] <- normalize(newtrain[,1:30]) #Independent = Predictors
newtrain[,31] <- as.numeric(newtrain[,31]) #Label = Target
summary(newtrain)
training <- newtrain[,1:30]
traintarget <- newtrain[,31]

newtest[,1:30] <- normalize(newtest[,1:30]) #Independent = Predictors
newtest[,31] <- as.numeric(newtest[,31]) #Label = Target
summary(newtest)
testing <- newtest[,1:30]
testtarget <- newtest[,31]

newvalid[,1:30] <- normalize(newvalid[,1:30]) #Independent = Predictors
newvalid[,31] <- as.numeric(newvalid[,31]) #Label = Target
summary(newvalid)
validating <- newvalid[,1:30]
validtarget <- newvalid[,31]

#One hot encoding
trainlabels <- to_categorical(traintarget)
testlabels <- to_categorical(testtarget)
validlabels <- to_categorical(validtarget)

#Create Sequential Model
# 30 independent variables
model <- keras_model_sequential()
#Pipe symbol %>%
model %>% 
          layer_dense(units = 8, activation = 'relu', 
                      input_shape = c(30))%>%
          layer_dense(units = 8, activation = 'relu') %>%
          layer_dense(units = 2, activation = 'softmax')
summary(model)        
#layer_dense(units = 2, activation = 'softmax') is Visible/Output layer
#30 independent * 8 layers=240 + 8 constant values for each node = 248
#2 nodes * 8 layers = 16 + 2 constant values = 18

#Compile the model
#For 2 class problem we will use "binary_crossentropy"
model %>%
          compile(loss ='binary_crossentropy',
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

newtrain[,1:31] <- normalize(newtrain[,1:31])
newtrain[,32] <- as.numeric(newtrain[,32])
summary(newtrain)
training <- newtrain[,1:31]
traintarget <- newtrain[,32]

newtest[,1:31] <- normalize(newtest[,1:31])
newtest[,32] <- as.numeric(newtest[,32])
summary(newtest)
testing <- newtest[,1:31]
testtarget <- newtest[,32]

newvalid[,1:31] <- normalize(newvalid[,1:31])
newvalid[,32] <- as.numeric(newvalid[,32])
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
  layer_dense(units = 8, activation = 'relu', 
              input_shape = c(31))%>%
  layer_dense(units = 8, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'softmax')
summary(model)        
#layer_dense(units = 2, activation = 'softmax') is Visible/Output layer
#31 independent * 8 layers=256 + 8 constant values for each node = 256
#2 nodes * 8 layers = 16 + 2 constant values = 18

#Compile the model
#For 2 class problem we will use "binary_crossentropy"
model %>%
  compile(loss ='binary_crossentropy',
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

#------------ Model 3: FULL MODEL + SEARCH + STATEMENT -------------------#
#Checking the structure of Data
traindata <- cbind(statement.svd[1:10269,-1],train)
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

newtrain[,1:332] <- normalize(newtrain[,1:332])
newtrain[,333] <- as.numeric(newtrain[,333])
summary(newtrain)
training <- newtrain[,1:332]
traintarget <- newtrain[,333]

newtest[,1:332] <- normalize(newtest[,1:332])
newtest[,333] <- as.numeric(newtest[,333])
summary(newtest)
testing <- newtest[,1:332]
testtarget <- newtest[,333]

newvalid[,1:332] <- normalize(newvalid[,1:332])
newvalid[,333] <- as.numeric(newvalid[,333])
summary(newvalid)
validating <- newvalid[,1:332]
validtarget <- newvalid[,333]

#One hot encoding
trainlabels <- to_categorical(traintarget)
testlabels <- to_categorical(testtarget)
validlabels <- to_categorical(validtarget)

#Create Sequential Model
model <- keras_model_sequential()
#Pipe symbol %>%
model %>% 
  layer_dense(units = 8, activation = 'relu', 
              input_shape = c(332))%>%
  layer_dense(units = 8, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'softmax')
summary(model)        
#layer_dense(units = 2, activation = 'softmax') is Visible/Output layer
#31 independent * 8 layers=256 + 8 constant values for each node = 256
#2 nodes * 8 layers = 16 + 2 constant values = 18

#Compile the model
#For 2 class problem we will use "binary_crossentropy"
model %>%
  compile(loss ='binary_crossentropy',
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
#model1 vs model2 vs model3
model1 #69.21% accuracy
model2 #69.60% accuracy
model3 #67.80% accuracy
