# Predictive Model

rm(list = ls(all = TRUE))

# set the working directory

setwd('C:\\Users\\D_und_V\\Documents\\MATLAB\\Practical')

# load caret 

library(caret)

install.packages("CHAID", repos="http://R-Forge.R-project.org")
library(CHAID)


# read the train set and the test set from csv, 
# the csv-missings ('#DIV/0') are replaced by 'NA'

train_orig <- read.csv(file="pml-training.csv", header=TRUE, 
                       as.is = TRUE, stringsAsFactors = FALSE, sep=',', na.strings=c('NA','','#DIV/0!'))
test_orig <- read.csv(file="pml-testing.csv", header=TRUE, as.is = TRUE, 
                      stringsAsFactors = FALSE, sep=',', na.strings=c('NA','','#DIV/0!'))

# set 
train_orig$classe <- as.factor(train_orig$classe)  

# the non-accelerometer measures are discard 
NAindex <- apply(train_orig,2,function(x) {sum(is.na(x))}) 
train_orig <- train_orig[,which(NAindex == 0)]
NAindex <- apply(test_orig,2,function(x) {sum(is.na(x))}) 
test_orig <- test_orig[,which(NAindex == 0)]

v <- which(lapply(train_orig, class) %in% "numeric")

preObj <-preProcess(train_orig[,v],method=c('knnImpute', 'center', 'scale'))
trainLess1 <- predict(preObj, train_orig[,v])
trainLess1$classe <- train_orig$classe

testLess1 <-predict(preObj,test_orig[,v])

# Create cross validation set

set.seed(12031987)

inTrain = createDataPartition(trainLess1$classe, p = 3/4, list=FALSE)
training = trainLess1[inTrain,]
crossValidation = trainLess1[-inTrain,]

# Train model with random forest due to its highly accuracy rate. The model is 
# build on a training set of 28 variables from the initial 160. 
# Cross validation is used as train control method.

modFit <- train(classe ~., method="rf", data=training, 
                trControl=trainControl(method='cv'), number=5, allowParallel=TRUE )

trainingPred <- predict(modFit, training)
confusionMatrix(trainingPred, training$classe)

cvPred <- predict(modFit, crossValidation)
confusionMatrix(cvPred, crossValidation$classe)

testingPred <- predict(modFit, testLess1)
testingPred

training$roll_belt <- factor(training$roll_belt) 
training$pitch_belt <- factor(training$pitch_belt) 


chaid.tree <-chaid(classe ~roll_belt+pitch_belt,data=training)