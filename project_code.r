install.packages("caret")
install.packages("caret")
install.packages("ISLR")
install.packages("caTools")
install.packages("dplyr")
install.packages("mlbench")
install.packages("earth")
install.packages("kernlab")
install.packages("pROC")
install.packages("corrplot")
install.packages("tseries")
install.packages("Ecdat")


library(tseries)
library(Ecdat)
library(lattice)
library(ggplot2)
library(caret)
library(ISLR)
library(caTools)
library(dplyr)
library(Formula)
library(plotrix)
library(TeachingDemos)
library(plotmo)
library(mlbench)
library(nnet)
library(earth)
library(kernlab)
library(pROC)

##  The datatsets included here are the datasets after preprocessing and 
## the model EXTRA TREE REGRESSOR are done using python code due to the time complexity in R

X_train <- read.csv("~/Desktop/train_x.csv")
View(X_train)
X_train <- X_train[, 2:22]
View(X_train)

X_test <- read.csv("~/Desktop/test_x.csv")
View(X_test)
X_test <- X_test[, 2:22]
View(X_test)

#splitting our dependent variable into Y_train and Y_test
Y_train <- read.csv("~/Desktop/train_y.csv", header= FALSE)

Y_train <- Y_train[, 2]
View(Y_train)

Y_test <- read.csv("~/Desktop/test_y.csv", header= FALSE)
Y_test<- Y_test[, 2]
View(Y_test)


ctrl <- trainControl(method = "repeatedcv", repeats = 10)

# MARS model:
library(earth)
marsGrid = expand.grid(.degree=1:2, .nprune=2:30)
set.seed(100)
marsModel = train(x=X_train, y=Y_train, method="earth", preProc=c("center", "scale"),tuneGrid=marsGrid,trControl = ctrl  )
marsModel
marsModel$results
plot(marsModel)

# Lets see what variables are most important: 
varImp(marsModel)
marsPred = predict(marsModel, newdata=testData$x)
marsPR = postResample(pred=marsPred, obs=testData$y)
marsPR

#Ridge Regression
set.seed(100)
ctrl <- trainControl(method = "cv", number = 10)
ridge.fit <- train(X_train, Y_train, method = "ridge", trControl = ctrl, preProc = c("center", "scale"), tuneLength = 20) 
ridge.fit

summary(ridge.fit)
ridge_pred <- predict(ridge.fit, X_test)
ridge = data.frame(obs=Y_test, pred= ridge_pred)
defaultSummary(ridge)

plot(ridge.fit, main= "RIDGE REGRESSION" ,  xlab = "LAMBDA", ylab = "RMSE(Cross-Validation)")
ridge.fit$results
varImp(ridge.fit)

#lasso_regression
lasso.fit <- train(X_train, Y_train, method= "lasso", trControl = ctrl, preProc = c("center", "scale"), tuneLength = 30) 
lasso.fit

summary(lasso.fit)
lasso_pred <- predict(lasso.fit, X_test)
lasso = data.frame(obs=Y_test, pred= lasso_pred)
defaultSummary(lasso)

plot(lasso.fit, main= "LASSO" ,  xlab = "LAMBDA", ylab = "RMSE(Cross-Validation)")
lasso.fit$results
varImp(lasso.fit)

# A Support Vector Machine (SVM):

set.seed(300)
# tune against the cost C
svmRModel = train(x=X_train, y=Y_train, method="svmRadial", preProc=c("center", "scale"), tuneLength=10)

svmRModel
svmRModel$results
plot(svmRModel)

# Lets see what variables are most important:
varImp(svmRModel)

svmRPred = predict(svmRModel, newdata=X_test)
svmPR = postResample(pred=svmRPred, obs=Y_test)

svmPR

#Random Forest
install.packages("randomForest")
library(randomForest)
library(rpart)

random.fit <- randomForest(X_train, Y_train, control=cforest_unbiased(mtry=2,ntree=500))
random.fit

importance(random.fit)
varImp(random.fit)

rf_yHat = predict(random.fit, newdata= X_test)

## performance evaluation
rfPR = postResample(pred=rf_yHat, obs=Y_test)
rfPR

# KNN model
install.packages("caret")
library(caret)
set.seed(100)
knnModel = train(x=X_train, y=Y_train, method="knn",preProc=c("center","scale"), tuneLength=10)
knnModel
knnModel$results

# Lets see what variables are most important: 
varImp(knnModel)

# plot the RMSE performance against the k
plot(knnModel$results$k, knnModel$results$RMSE, type="o",xlab="# neighbors",ylab="RMSE", main="KNN plot")

# we try the model on the test data

knnPred = predict(knnModel, newdata = X_test)
knnPred
knnPR = postResample(pred=knnPred, obs = Y_test)
knnPR

# Decision trees
install.packages("rpart")
library(rpart)
install.packages("party")
library(party)
install.packages("partykit")
library(partykit)

# set up training data
trainData = data.frame( x=X_train, y=Y_train )
rPartModel = rpart( y ~ ., data=trainData, method="anova", control=rpart.control(cp=0.01,maxdepth=30)) 

# tree plotting   
rpartTree = as.party(rPartModel)
dev.new()
plot(rpartTree)


# predict test with this regression tree: 
rPart_yHat = predict(rPartModel,newdata=data.frame(x=X_test))

## performance evaluation
rtPR = postResample(pred=rPart_yHat, obs=Y_test)
rtPR

#PLS

library(pls)
set.seed(567)
pls <- train(X_train, Y_train, method="pls",tuneLength=10,  preProcess= c("zv","center","scale"), trControl= ctrl)
pls$results



# Lets see what variables are most important: 
varImp(pls)
 
plot(pls,main = "Number of Components vs RMSE")
pls$bestTune
summary(pls$resample)
plsPred = predict(pls, X_test)
postResample(pred = plsPred, obs = Y_test)



#enet
library(glmnet)
enetGrid <- expand.grid(alpha = c(0,  .1,  .2, .4, .6, .8, 1),
                        lambda = seq(.01, .1, length = 20) )
set.seed(123)
enetTune <- train(x = X_train, y = Y_train,
                  method = "glmnet",
                  tuneGrid = enetGrid,
                  trControl = ctrl)
enetTune

# Lets see what variables are most important: 
varImp(enetTune)
enetTune$bestTune
enetTune$results
plot(enetTune, xlab = "Lambda", ylab = "Cross-validated RMSE", main = "ENET")
enetPred = predict(enetTune, X_test)
postResample(pred = enetPred, obs = Y_test)




#GBM
gbmGrid <- expand.grid(n.trees = c(100,500), interaction.depth = c(1, 5),
                       n.minobsinnode = c(10), shrinkage = c(.01, .1))

ctrl <- trainControl(method = "repeatedcv", repeats = 10)

set.seed(476)
gbmFit <- train(x = X_train,
                y = Y_train,
                method = "gbm",
                tuneGrid = gbmGrid,
                verbose = FALSE,
                trControl = ctrl)
gbmFit



gbmFit$results
plot(gbmFit)
plot(gbmFit, type = "s", print.thres = c(.5), print.thres.pch = 3,
print.thres.pattern = "", print.thres.cex = 1.2,add = TRUE, col = "red", 
print.thres.col = "red", legacy.axes = TRUE)
gbm_pred <- predict(gbmFit, X_test)

gbmValues1 = data.frame(obs = Y_test, pred = gbm_pred)
defaultSummary(gbmValues1)



