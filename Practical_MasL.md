Practical Machine Learning - Prediction Assignment Writeup
========================================================

# General information

The goal of this project is to analyse the data from the project "Human 
Activity Recognition" which was obtained by tests on humans doing some particular 
physical exercises, namely biceps-curls but with different kind of movements, called activity (there are five different activities, so five different kind of movements). 
The entire data is collected from the test on only 6 humans. 

The ultimative goal is to predict this activity from the variables which contain the sensor measurements about the movements. For this goal I use the machine learning algorithm called random forest which is actually an ensemble method and therefore very powerfull, so I could expect that it will do well in predicting. **Therefore I expect the out of sample error to be small, so 5% or less, this would mean a good model for me.** I use the cross calidation to estimate this error (see the details further).


# Exploratory analysis.

As the original data is in csv-format, I can look at it with UltraEdit or any othe suitable editor (Ecxel would also go, of course).
As I see, there are many missings in the data, which are represented by the symbols "#DIV/0!".
So my first task is to change this missings in the right way when reading the data to R.
I replace "#DIV/0!" by "NA"  when reading the data, as follows (here the R code):



```r
# some settings first of all 
rm(list = ls(all = TRUE))
setwd('C:\\Users\\D_und_V\\Documents\\MATLAB\\Practical')
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
# read data
train_orig <- read.csv(file="pml-training.csv", header=TRUE,
    as.is = TRUE, stringsAsFactors = FALSE, sep=',', na.strings=c('NA','','#DIV/0!'))
test_orig <- read.csv(file="pml-testing.csv", header=TRUE, as.is = TRUE,
            stringsAsFactors = FALSE, sep=',', na.strings=c('NA','','#DIV/0!'))
```


Then I compare the training and the test data to check if they have the same structure (here the R Code):


```r
# Do the training and the test data have different column names?
compar <- names(train_orig) != names(test_orig)
diffcol <- length(which(as.logical(compar), arr.ind = TRUE))
compar
```

```
##   [1] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
##  [12] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
##  [23] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
##  [34] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
##  [45] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
##  [56] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
##  [67] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
##  [78] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
##  [89] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
## [100] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
## [111] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
## [122] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
## [133] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
## [144] FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE FALSE
## [155] FALSE FALSE FALSE FALSE FALSE  TRUE
```
The only difference is that the last columns of the test data and the training data are different.
I look at these columns:

```r
tail(names(train_orig),1)
```

```
## [1] "classe"
```

```r
tail(names(test_orig),1)
```

```
## [1] "problem_id"
```
These columns are "problemID" in the training data set and "classe" in the test data set.
This difference is not the problem as I have to throw away irrelevant variables anyway, so I do this furhter for both the training and test data set, which would eliminate this needless last column too.


# Selecting variables.

After investigating the variables I can see that the most relevant variables have the endings "_x", "_y" or "_z"
in their names as it is natural because they are the meusurements of movements. I can hope that these variables are sufficient for good prediction, so I restrict the data to these variables. The only other variables which I need are "user_name" and of course "classe" which is the activity I have to predict. So I restrict the data as follows (here the R code):


```r
rx <- "(classe|user_name|.*_x$|.*_y$|.*_z$).*"
relevant <-grep(rx, names(train_orig), perl=TRUE, value=TRUE)
train_orig <- subset(train_orig, select=relevant)
names(train_orig)
```

```
##  [1] "user_name"         "gyros_belt_x"      "gyros_belt_y"     
##  [4] "gyros_belt_z"      "accel_belt_x"      "accel_belt_y"     
##  [7] "accel_belt_z"      "magnet_belt_x"     "magnet_belt_y"    
## [10] "magnet_belt_z"     "gyros_arm_x"       "gyros_arm_y"      
## [13] "gyros_arm_z"       "accel_arm_x"       "accel_arm_y"      
## [16] "accel_arm_z"       "magnet_arm_x"      "magnet_arm_y"     
## [19] "magnet_arm_z"      "gyros_dumbbell_x"  "gyros_dumbbell_y" 
## [22] "gyros_dumbbell_z"  "accel_dumbbell_x"  "accel_dumbbell_y" 
## [25] "accel_dumbbell_z"  "magnet_dumbbell_x" "magnet_dumbbell_y"
## [28] "magnet_dumbbell_z" "gyros_forearm_x"   "gyros_forearm_y"  
## [31] "gyros_forearm_z"   "accel_forearm_x"   "accel_forearm_y"  
## [34] "accel_forearm_z"   "magnet_forearm_x"  "magnet_forearm_y" 
## [37] "magnet_forearm_z"  "classe"
```


# Preprocessing variables.

I set the variable "classe" as factor:



```r
# set
train_orig$classe <- as.factor(train_orig$classe)
```

# Create cross validation set.

I divide the training data set into two parts, one for training and other for cross validation, in proportion 80-20 (here the R code):


```r
set.seed(77117711)
inTrain = createDataPartition(train_orig$classe, p = 4/5, list=FALSE)
training = train_orig[inTrain,]
crossValid = train_orig[-inTrain,]
```

# Model.

As said before our method of choice is the random forest model which is very powefull. 
So I build the model on a training set which has 36 variables and ca. 20000 observations. 
I use the cross validation as the control method.
The model building has a runtime of approximately 23 minutes. Here the R code:


```r
modFit <- train(classe ~., method="rf", data=training, 
                trControl=trainControl(method='cv'), number=5, allowParallel=TRUE )
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

# Accuracy of model.

I check the accuracy of mode for the training data set 


```r
trainingPred <- predict(modFit, training)
confusionMatrix(trainingPred, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4464    0    0    0    0
##          B    0 3038    0    0    0
##          C    0    0 2738    0    0
##          D    0    0    0 2573    0
##          E    0    0    0    0 2886
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9998, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

and the cross validation data set


```r
cvPred <- predict(modFit, crossValid)
confusionMatrix(cvPred, crossValid$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1114    7    1    2    0
##          B    1  747   10    0    0
##          C    1    5  673   22    1
##          D    0    0    0  619    1
##          E    0    0    0    0  719
## 
## Overall Statistics
##                                           
##                Accuracy : 0.987           
##                  95% CI : (0.9829, 0.9903)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9836          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9842   0.9839   0.9627   0.9972
## Specificity            0.9964   0.9965   0.9910   0.9997   1.0000
## Pos Pred Value         0.9911   0.9855   0.9587   0.9984   1.0000
## Neg Pred Value         0.9993   0.9962   0.9966   0.9927   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2840   0.1904   0.1716   0.1578   0.1833
## Detection Prevalence   0.2865   0.1932   0.1789   0.1580   0.1833
## Balanced Accuracy      0.9973   0.9904   0.9875   0.9812   0.9986
```

**The last table shows that the out of sample error is actually under 2% which is even better than we hoped.**



# Predicting on test data.


```r
testingPred <- predict(modFit, test_orig)
testingPred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

**With my model I was able to correcty predict all the 20 test cases.**
