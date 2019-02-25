require(fpp)
require(forecast)
library(fBasics)
library(data.table)
library(timeSeries)
library(ggplot2)
library(xts)
library(randomForest)
library(gbm)
library(rpart)
library(arm)
library(caret)
library(AUC)
library(fastAdaboost)
library(LiblineaR)
library(naivebayes)
library(caretEnsemble)
library(pROC)
library(moments)
library(rpart.plot)
library(RColorBrewer)	
library(party)

df <- fread("train_blood_don.csv", sep = ",", header = TRUE) #read the data set

#EDA

#Basic statistics
str(df)
summary(df)
#plot correlation matrix
panel.hist <- function(x, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5) )
  h <- hist(x, plot = FALSE)
  breaks <- h$breaks; nB <- length(breaks)
  y <- h$counts; y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col = "cyan", ...)
}
pairs(df[,c(2:5)], panel = panel.smooth,
      cex = 1.5, bg = "light blue",
      diag.panel = panel.hist, font.labels = 2) 

#Box plot
boxplot(df[,c("Total_Volume_Donated_cc")], 
        col = "green", range = 1.5, main = "Boxplot of Magnitude",ylab = "Magnitude", ylim = c(0, 12500))

boxplot(df[,c("Months_since_Last_Donation","Number_of_Donations","Months_since_First_Donation")], 
        col = "green", range = 1.5, main = "Boxplot of Magnitude",ylab = "Magnitude", ylim = c(0, 105))


df.sc <- scale(df) #scale
colMeans(df.sc) #column means is ~1
apply(df.sc,2,sd) #SD is 1

df.sc <- as.data.frame(df.sc)
df.sc$Months_since_First_Donation <-  (df.sc$Months_since_First_Donation+2)
df.sc$Months_since_Last_Donation <-  (df.sc$Months_since_Last_Donation+2)
df.sc$Number_of_Donations <-  (df.sc$Number_of_Donations+1)

#df.sc <- as.data.frame(df.sc)
df.sc$Total_Volume_Donated_cc <- NULL
df.sc$TARGET <- df$TARGET #put original target dim. scale center has changed it
df.sc$INDEX <- df$INDEX #same as above

#Running logistic regression
mylogit <- glm(TARGET ~ Months_since_Last_Donation + Number_of_Donations + Months_since_First_Donation,
               data = df.sc, family = binomial(link="logit"))
summary(mylogit)

df.test <- fread("test_blood_don.csv", sep = ",", header = TRUE)
#apply center and scalling and make all values possitve 

df.test.sc <- scale(df.test) #scale
colMeans(df.test.sc) #column means is ~1
apply(df.test.sc,2,sd) #SD is 1
df.test.sc <- as.data.frame(df.test.sc)

df.test.sc$Months_since_First_Donation <- (df.test.sc$Months_since_First_Donation+2)
df.test.sc$Months_since_Last_Donation <-  (df.test.sc$Months_since_Last_Donation+2)
df.test.sc$Number_of_Donations <-  (df.test.sc$Number_of_Donations+1)

df.test.sc$Total_Volume_Donated_cc <- NULL
#df.test$TARGET <- df$TARGET #put original target dim. scale center has changed it
df.test.sc$INDEX <- df.test$INDEX #same as above
#Model-1 - Logistic Regression

#predict new probabilities on test data
glm_response_scores <- as.data.frame(predict(mylogit, df.test.sc, type="response"))
m1 <- as.data.frame(glm_response_scores$`predict(mylogit, df.test.sc, type = "response")`)
colnames(m1)[1] <- "predictions"
write.csv(m1,"logistic.csv")

#Model-2 - Random Forest
myrandom <- randomForest(TARGET ~ Months_since_Last_Donation + Number_of_Donations + Months_since_First_Donation,
                         data = df.sc,importance = TRUE)
plot(varImp(myrandom,scale=F))
summary(myrandom)
#predict new probabilities on test data
myrandom.pred <- as.data.frame(predict(myrandom, df.test.sc))
m2 <- as.data.frame(myrandom.pred$`predict(myrandom, df.test.sc)`)
colnames(m2)[1] <- "predictions"
#confusionMatrix(df.sc$TARGET,myrandom.pred$`predict(myrandom, df.test)`)
write.csv(m2,"random_forest.csv")
#predictor importance



#Model-4 - rpart
myrpart <- rpart(TARGET ~ Months_since_Last_Donation + Number_of_Donations + Months_since_First_Donation,
                 data = df.sc)
summary(myrpart)
prp(myrpart)#plot a basic tree
new.tree.1 <- prp(myrpart,snip=TRUE)$obj # interactively trim the tree
prp(new.tree.1) # display the new tree
myrpart$cptable # get the complexity parameter table (CP table)
plotcp(myrpart) # get the plot for cp
#predict new probabilities on test data

myrpart.pred <- as.data.frame(predict(myrpart,df.test.sc, method = "class"))

m4 <- as.data.frame(myrpart.pred$`predict(myrpart, df.test.sc, method = "class")`)
colnames(m4)[1] <- "predictions"
write.csv(m4,"rpart_method.csv")

#Working Models
#Model-1 - Logistic Regression (m1)
#Model-2 - Random Forest (m2)
#Model-4 - rpart (m4)
 
#Ensemble-1
#Taking average of predictions
en1 <- (m1+m2+m4)/3
write.csv(en1,"ensemble.csv")

####********************#######
#Ensemble-2
#Taking average of predictions
en2 <- (m1+m2)/2
write.csv(en2,"ensemble_2.csv")
####********************#######
#Out of sample comparison using RMSE
postResample(pred=glm_response_scores[[1]], obs=ifelse(df[,TARGET]=='yes',1,0))
postResample(pred=myrandom.pred[[1]], obs=ifelse(df[,TARGET]=='yes',1,0))
postResample(pred=myrpart.pred[[1]], obs=ifelse(df[,TARGET]=='yes',1,0))

en.plot <- plot(cbind(en1$predictions,en2$predictions))

write.csv(((en1+en2)/2),"avg_en1_en2.csv")
#Ensemble-3

#Predictors for top layer models 
en.trainset <- as.data.frame(cbind(m1$predictions,m2$predictions,m4$predictions))
colnames(en.trainset) <- c('logistic_reg','random_forest','rpart') 
predictors_top<-c('logistic_reg','random_forest','rpart') 


################
#caret models
################
df <- fread("train_blood_don.csv", sep = ",", header = TRUE) 

df$Total_Volume_Donated_cc <- NULL
df$TARGET <- ifelse(df$TARGET==1,'yes','no')
df$TARGET <-(as.factor(df$TARGET))
outcomeName <- 'TARGET'

trainDF <- df
trainDF <-  as.data.frame(trainDF)

df$rowcount <- 1:nrow(df)
set.seed(1234)
splitIndex <- createDataPartition(df[,rowcount], p = .75, list = FALSE, times = 1)
trainDF <- df[ splitIndex,]
testDF  <- df[-splitIndex,]

objControl <- trainControl(method='repeatedcv', number=3, summaryFunction = twoClassSummary, 
                           classProbs = TRUE)
fit.rf<- train((TARGET)~(Months_since_Last_Donation+Number_of_Donations+Months_since_First_Donation), 
               data=trainDF, method="rf", metric="ROC", trControl=objControl)

fit.nnet<- train((TARGET)~(Months_since_Last_Donation+Number_of_Donations+Months_since_First_Donation), 
                 data=trainDF, method="nnet", metric="ROC", trControl=objControl)

fit.adaboost<- train((TARGET)~(Months_since_Last_Donation+Number_of_Donations+Months_since_First_Donation), 
                     data=trainDF, method="adaboost", metric="ROC", trControl=objControl)

fit.knn<- train((TARGET)~(Months_since_Last_Donation+Number_of_Donations+Months_since_First_Donation), 
                data=trainDF, method="knn", metric="ROC", trControl=objControl)

fit.gbm<- train((TARGET)~(Months_since_Last_Donation+Number_of_Donations+Months_since_First_Donation), 
                data=trainDF, method="gbm", metric="ROC", trControl=objControl)

fit.naives<- train((TARGET)~(Months_since_Last_Donation+Number_of_Donations+Months_since_First_Donation), 
                   data=trainDF, method="naive_bayes", metric="ROC", trControl=objControl)

#Model performance metrics
confusionMatrix(fit.rf)
confusionMatrix(fit.nnet)
confusionMatrix(fit.adaboost)
confusionMatrix(fit.knn)
confusionMatrix(fit.gbm)
confusionMatrix(fit.naives)

plot(fit.rf)
plot(fit.nnet)
plot(fit.adaboost)
plot(fit.knn)
plot(fit.gbm)
plot(fit.naives)

plot(varImp(fit.rf,scale=F))
plot(varImp(fit.nnet,scale=F))
plot(varImp(fit.adaboost,scale=F))
plot(varImp(fit.knn,scale=F))
plot(varImp(fit.gbm,scale=F))
plot(varImp(fit.naives,scale=F))

summary(fit.rf)
summary(fit.nnet)
summary(fit.adaboost)
summary(fit.knn)
summary(fit.gbm) #gives a colorful plot
summary(fit.naives)

results <- resamples(list(rf= fit.rf, nnet=fit.nnet, adaboost =fit.adaboost,
                          knn=fit.knn, gbm=fit.gbm, naives=fit.naives))

# box and whisker plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

# density plots of accuracy
scales <- list(x=list(relation="free"), y=list(relation="free"))
densityplot(results, scales=scales, pch = "|")

# dot plots of accuracy
scales <- list(x=list(relation="free"), y=list(relation="free"))
dotplot(results, scales=scales)

# pair-wise scatterplots of predictions to compare models
splom(results)

# difference in model predictions
diffs <- diff(results)
# summarize p-values for pair-wise comparisons
summary(diffs)

#Out of sample comparison metric - RMSE and in sample comparison metric - ROC
predictions.rf.insample <- predict(fit.rf, newdata=testDF,type='prob')
postResample(pred=predictions.rf.insample[[2]], obs=ifelse(testDF[,TARGET]=='yes',1,0))
auc.rf <- roc((testDF[,TARGET]), predictions.rf.insample[[2]])
print(auc.rf$auc)

predictions.nnet.insample <- predict(fit.nnet, newdata=testDF,type='prob')
postResample(pred=predictions.nnet.insample[[2]], obs=ifelse(testDF[,TARGET]=='yes',1,0))
auc.nnet <- roc((testDF[,TARGET]), predictions.nnet.insample[[2]])
print(auc.nnet$auc)

predictions.knn.insample <- predict(fit.knn, newdata=testDF,type='prob')
postResample(pred=predictions.knn.insample[[2]], obs=ifelse(testDF[,TARGET]=='yes',1,0))
auc.knn <- roc((testDF[,TARGET]), predictions.knn.insample[[2]])
print(auc.knn$auc)

predictions.gbm.insample <- predict(fit.gbm, newdata=testDF,type='prob')
postResample(pred=predictions.gbm.insample[[2]], obs=ifelse(testDF[,TARGET]=='yes',1,0))
auc.gbm <- roc((testDF[,TARGET]), predictions.gbm.insample[[2]])
print(auc.gbm$auc)

predictions.adaboost.insample <- predict(fit.adaboost, newdata=testDF,type='prob')
postResample(pred=predictions.adaboost.insample[[2]], obs=ifelse(testDF[,TARGET]=='yes',1,0))
auc.adaboost <- roc((testDF[,TARGET]), predictions.adaboost.insample[[2]])
print(auc.adaboost$auc)

predictions.naives.insample <- predict(fit.naives, newdata=testDF,type='prob')
postResample(pred=predictions.naives.insample[[2]], obs=ifelse(testDF[,TARGET]=='yes',1,0))
auc.naives <- roc((testDF[,TARGET]), predictions.naives.insample[[2]])
print(auc.naives$auc)

#Read test data
df.test <- fread("test_blood_don.csv", sep = ",", header = TRUE)
df.test$Total_Volume_Donated_cc <- NULL

#Out of sample predictions
predictions.rf <- predict(fit.rf, newdata=df.test,type='prob')
predictions.nnet <- predict(fit.nnet, newdata=df.test,type='prob')
predictions.knn <- predict(fit.knn, newdata=df.test,type='prob')
predictions.gbm <- predict(fit.gbm, newdata=df.test,type='prob')
predictions.adaboost <- predict(fit.adaboost, newdata=df.test,type='prob')
predictions.naives <- predict(fit.naives, newdata=df.test,type='prob')

#Out of sample comparison metric - RMSE
postResample(pred=predictions.rf[[2]], obs=ifelse(df[,TARGET]=='yes',1,0))
postResample(pred=predictions.nnet[[2]], obs=ifelse(df[,TARGET]=='yes',1,0))
postResample(pred=predictions.knn[[2]], obs=ifelse(df[,TARGET]=='yes',1,0))
postResample(pred=predictions.gbm[[2]], obs=ifelse(df[,TARGET]=='yes',1,0))
postResample(pred=predictions.adaboost[[2]], obs=ifelse(df[,TARGET]=='yes',1,0))
postResample(pred=predictions.naives[[2]], obs=ifelse(df[,TARGET]=='yes',1,0))



