install.packages("AppliedPredictiveModeling")
install.packages("caret")
library(AppliedPredictiveModeling)
library(caret)

## RAMA

P1 <- read.csv("SPweekly20.csv")
P2 <- read.csv("SPweekly30.csv")
RP <- read.csv("SPweeklyRandom20.csv")

#bad <- is.na(P1)
#PF1fix <- P1[!bad,]
#PF1 <- na.omit(PF1fix)
#dim(PF1)

#head(PF1)

#bad <- is.na(P2)
#PF2fix <- P2[!bad,]
#PF2 <- na.omit(PF2fix)
#dim(PF2)

#head(PF2)

#bad <- is.na(RP)
#RPfix <- RP[!bad,]
#RP1 <- na.omit(RPfix)
#dim(RP1)

#head(RP1)

## First, remove near-zero variance predictors then get rid of a few predictors 
## that duplicate values. For example, there are two possible values for the 
## housing variable: "Rent", "Own" and "ForFree". So that we don't have linear
## dependencies, we get rid of one of the levels (e.g. "ForFree")

#SP30 <- SP30[, -nearZeroVar(SP30)]
##SP11 <- P1[, -nearZeroVar(P1)]
#P2$Date <- NULL
P1$Date <- NULL
P1$SP <- NULL
P2$Date <- NULL
P2$SP <- NULL
RP$Date <- NULL
RP$SP <- NULL
#PF1$Change <- NULL
head(P1)
head(P2)
head(RP)
#SP$MMM <- NULL
#SP$MMMChange <- NULL
#SP$ABT <- NULL
#SP$ABTChange <- NULL
#SP$ACNclose <- NULL
#SP$ACNChange <- NULL
## Split the data into training (80%) and test sets (20%)
set.seed(100)
inTrainP1 <- createDataPartition(P1$Class, p = .8)[[1]]
P1Train <- P1[ inTrainP1, ]
P1Test  <- P1[-inTrainP1, ]
str(P1)

set.seed(100)
inTrainP2 <- createDataPartition(P2$Class, p = .8)[[1]]
P2Train <- P2[ inTrainP2, ]
P2Test  <- P2[-inTrainP2, ]
str(P2)

set.seed(100)
inTrainRP <- createDataPartition(RP$Class, p = .8)[[1]]
RPTrain <- RP[ inTrainRP, ]
RPTest  <- RP[-inTrainRP, ]
str(RP)

##Gautham

## apply Support vector machine model (SVM)
library(kernlab)
library(e1071)
set.seed(1056)
svmP1 <- train(Class ~ ., data = P1Train, method = "svmRadial", preProc = c("center", "scale"), tuneLength = 10,
                trControl = trainControl(method = "repeatedcv", repeats = 5))
svmP1

set.seed(1056)
svmP2 <- train(Class ~ ., data = P2Train, method = "svmRadial", preProc = c("center", "scale"), tuneLength = 10,
               trControl = trainControl(method = "repeatedcv", repeats = 5))
svmP2

set.seed(1056)
svmRP <- train(Class ~ ., data = RPTrain, method = "svmRadial", preProc = c("center", "scale"), tuneLength = 10,
                trControl = trainControl(method = "repeatedcv", repeats = 5))
svmRP
## generalized linear model (glm) - logistic regression 
set.seed(1056)
logisticRegP1 <- train(Class ~ ., data = P1Train, method = "glm", trControl = trainControl(method = "repeatedcv", repeats = 5))
logisticRegP1

set.seed(1056)
logisticRegP2 <- train(Class ~ ., data = P2Train, method = "glm", trControl = trainControl(method = "repeatedcv", repeats = 5))
logisticRegP2

set.seed(1056)
logisticRegRP <- train(Class ~ ., data = RPTrain, method = "glm", trControl = trainControl(method = "repeatedcv", repeats = 5))
logisticRegRP

## Model comparison
## use "resamples" to compare models that share a common set of resampled data set
## Since both cases started with the same seed, we can do pair-wised comparison

# resamp <- resamples(list(SVMPF1 = svmP1, LogisticPF1 = logisticRegP1,
                        # SVMPF2 = svmP2, LogisticPF2 =  logisticRegP2,
                         # svmRPF = svmRP, LogisticRPF = logisticRegRP ))

# summary(resamp)
# modelDifferences <- diff(resamp)
# summary(modelDifferences)


##Choose model in between - Box Plot - Lecture 9  - slide 17


### Predict the test set Using Log Regression
TestResultsP1 <- data.frame(obs = P1Test$Class)
TestResultsP1$prob <- predict(logisticRegP1, P1Test, type = "prob")[, "Down"]
TestResultsP1$pred <- predict(logisticRegP1, P1Test)
TestResultsP1$Label <- ifelse(TestResultsP1$obs == "Down", "True Outcome: Down",  "True Outcome: Up")

TestResultsP2 <- data.frame(obs = P2Test$Class)
TestResultsP2$prob <- predict(logisticRegP2, P2Test, type = "prob")[, "Down"]
TestResultsP2$pred <- predict(logisticRegP2, P2Test)
TestResultsP2$Label <- ifelse(TestResultsP2$obs == "Down", "True Outcome: Down",  "True Outcome: Up")


TestResultsRP <- data.frame(obs = RPTest$Class)
TestResultsRP$prob <- predict(logisticRegRP, RPTest, type = "prob")[, "Down"]
TestResultsRP$pred <- predict(logisticRegRP, RPTest)
TestResultsRP$Label <- ifelse(TestResultsRP$obs == "Down", "True Outcome: Down",  "True Outcome: Up")

### Create the confusion matrix from the test set.
confusionMatrix(data = TestResultsP1$pred, reference = TestResultsP1$obs)

confusionMatrix(data = TestResultsP2$pred, reference = TestResultsP2$obs)

confusionMatrix(data = TestResultsRP$pred, reference = TestResultsRP$obs)


## VLAD

### Plot the probability of bad credit
dev.new()
histogram(~prob|Label,
          data = TestResultsP1,
          layout = c(2, 1),
          nint = 20,
          xlab = "Probability of P20 Down",
          type = "count")

dev.new()
histogram(~prob|Label,
          data = TestResultsP2,
          layout = c(2, 1),
          nint = 20,
          xlab = "Probability of P30 Down",
          type = "count")

dev.new()
histogram(~prob|Label,
          data = TestResultsRP,
          layout = c(2, 1),
          nint = 20,
          xlab = "Probability of RandomP Down",
          type = "count")
### Calculate and plot the calibration curve
dev.new()
ResultCalib1 <- calibration(obs ~ prob, data = TestResultsP1)
xyplot(ResultCalib1)

dev.new()
ResultCalib2 <- calibration(obs ~ prob, data = TestResultsP2)
xyplot(ResultCalib2)

dev.new()
ResultCalibRandom <- calibration(obs ~ prob, data = TestResultsRP)
xyplot(ResultCalibRandom)
### ROC curves:

### Like glm(), roc() treats the last level of the factor as the event
### of interest so we use relevel() to change the observed class data
install.packages("pROC")
library(pROC)
P1ROC <- roc(relevel(TestResultsP1$obs, "Up"), TestResultsP1$prob)
P2ROC <- roc(relevel(TestResultsP2$obs, "Up"), TestResultsP2$prob)
RPROC <- roc(relevel(TestResultsRP$obs, "Up"), TestResultsRP$prob)
# coords(creditROC, "all")[,1:3]

auc(P1ROC)
ci.auc(P1ROC)

auc(P2ROC)
ci.auc(P2ROC)

auc(RPROC)
ci.auc(RPROC)
### Note the x-axis is reversed
#plot(creditROC)
### Old-school:
dev.new()
plot(P1ROC, legacy.axes = TRUE, main = "Portfolio 20 Stocks")

dev.new()
plot(P2ROC, legacy.axes = TRUE, main = "Portfolio 30 Stocks")

dev.new()
plot(RPROC, legacy.axes = TRUE, main = "Random Portfolio")

### Lift charts
library(caret)

resultstLift <- lift(obs ~ prob, data = TestResultsP1)
dev.new()
xyplot(resultstLift, main = "Portfolio 20 Stocks")

resultstLift <- lift(obs ~ prob, data = TestResultsP2, main = "Portfolio 30 Stocks")
dev.new()
xyplot(resultstLift, main = "Portfolio 30 Stocks")

resultstLift <- lift(obs ~ prob, data = TestResultsRP, main = "Random Portfolio")
dev.new()
xyplot(resultstLift, main = "Random Portfolio")
