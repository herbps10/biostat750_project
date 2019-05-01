library(tidyverse)
library(pcLasso)
library(glmnet)
library(mice)
library(GGally)
library(FNN)
library(caret)
library(neuralnet)
library(Metrics)

dat <- as_tibble(read.csv("data/train.csv", header = T))
#get rid of cols with lots of missing values
dat <- select(dat, -Id, -LotFrontage, -Alley, -FireplaceQu, -PoolQC, -Fence, -MiscFeature)
dat <- dat[complete.cases(dat),]
sum(is.na(dat)) # = 0. yay

y <- dat$SalePrice

dat_x <- select(dat, -SalePrice)
dat_x_scaled <- mutate_if(dat_x, is.numeric, scale)
x <- model.matrix(~., dat_x[,])
x_sc <- model.matrix(~., dat_x_scaled[,])
y_sc <- scale(y)

#### Helper functions used earlier to make sure we all had the same data
# write.csv(x_sc, "train_x_scaled.csv")
# write.csv(y_sc, "train_y_scaled.csv")
# write.csv(x, "train_x.csv")
# write.csv(y, "train_y.csv")
# x <- as.matrix(read.csv("train_x.csv", header = T) %>% select(-X))
# y <- as.matrix(read.csv("train_y.csv", header = T, row.names = 1))

## Make a correlation plot for the paper
# ggcorr(dat[,], label_alpha = TRUE, 
#                 size = 2.5) + ggtitle("Correlations between predictors") +
#  theme(legend.direction = "vertical")



###################################################    HERB: 
## To extract just the model for possible use in an ensemble,
## load the (scaled) training data and run this.
## Note that KNN needs separately specified X / Y training data.
knn12_model <- knn.reg(train = training_data_x, y = training_data_y, test = test_data_x, k = 12)
error <- Metrics::rmsle(tsty, knn12_model$pred)





###################################### Results for paper ##############
##### Kfold CV to choose optimal k and get test error estimate
set.seed(1)
numfolds <- 10
folds <- createFolds(y, k = numfolds, list = F)
min_k <- 1
max_k <- 40
rmls_cv <- as.data.frame(matrix(NA, nrow = (max_k - min_k + 1), ncol = 2))
colnames(rmls_cv) <- c("k","rmls")
krange <- seq(from = min_k, to = max_k)
for(i in krange){
  this_k_error <- vector()
  for(j in 1:numfolds){
    trnx <- x_sc[folds != j,]
    trny <- y[folds != j]
    tstx <- x_sc[folds == j,]
    tsty <- y[folds == j]
    knn <- knn.reg(train = trnx, y = trny, test = tstx, k = i)
    error <- Metrics::rmsle(tsty, knn$pred)
    this_k_error <- c(this_k_error, error)
  }
  print(this_k_error)
  rmls_cv[i,2] <- mean(this_k_error)
  rmls_cv[i,1] <- i
}
rmls_cv
plot(rmls_cv[,1],rmls_cv[,2])
ggplot() +
  geom_point(aes(x = rmls_cv[,1], y= rmls_cv[,2])) +
  geom_point(aes(x = 12, y = rmls_cv[12,2]), color = "red") +
  ylab("RMSLE") + xlab("k value")


########## Variable importance with KNN, using k = 12 from the CV
varrange <- ncol(x_sc)
numfolds <- 5
folds <- createFolds(y, k = numfolds, list = F)
vi_cv <- as.data.frame(matrix(NA, nrow = varrange, ncol = 2))
for(i in 1:varrange){
  x_minus_one <- x_sc[,-i]
  print(paste("omitting column number",i))
  this_x_error <- vector()
  for(j in 1:numfolds){ # 5-fold CV for this one to speed it up
    trnx <- x_minus_one[folds != j,]
    trny <- y[folds != j]
    tstx <- x_minus_one[folds == j,]
    tsty <- y[folds == j]
    knn <- knn.reg(train = trnx, y = trny, test = tstx, k = 12)
    #this_k_rmls <- c(this_k_rmls, mean(knn$pred - tsty)^2))
    error <- Metrics::rmsle(tsty, knn$pred)
    this_x_error <- c(this_x_error, error)
  }
  #print(this_x_error)
  vi_cv[i,2] <- mean(this_x_error)
  vi_cv[i,1] <- i
}
# Error rates with vars left out:
vi_cv

# Extracting variable names
values <- vi_cv[which(vi_cv$V2 > .1659),]
ordered <- dplyr::arrange(values, desc(V2))
ordered
cols <- ordered$V1
namez <- colnames(x_sc)
# Here are the top 10 variables
namez[cols]
################################### End results for paper #########




