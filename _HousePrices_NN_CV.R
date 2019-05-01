library(tidyverse)
library(pcLasso)
library(glmnet)
library(mice)
library(GGally)
library(FNN)
library(caret)
library(neuralnet)
library(NeuralNetTools)
library(Metrics)

dat <- as_tibble(read.csv("train.csv", header = T))
#get rid of cols with lots of missing values
dat <- select(dat, -Id, -LotFrontage, -Alley, -FireplaceQu, -PoolQC, -Fence, -MiscFeature)
dat <- dat[complete.cases(dat),]
sum(is.na(dat)) # = 0. yay


dat_NN_scaled <- mutate_if(dat, is.numeric, scale) 
#names(dat_NN_scaled) <- gsub(" ", "_", names(dat_NN_scaled))
#dat_NN_scaled <- dat_NN_scaled %>% dplyr::rename_all(funs(make.names(.)))
dat_NN_scaled_numeric <- model.matrix(~., dat_NN_scaled[,])[,-1]
oldnames <- colnames(dat_NN_scaled_numeric)
colnames(dat_NN_scaled_numeric) <- paste0("V",c(1:232))

n <- colnames(dat_NN_scaled_numeric)
#f <- as.formula(paste("SalePrice~", paste(sprintf("`%s`", n[!n %in% "SalePrice"]), collapse="+")))
f <- as.formula(paste("V232 ~", paste(n[!n %in% "V232"], collapse = " + ")))

##### Kfold CV for NN
test_nn <- function(vector_hidden_layers){
  set.seed(1)
  numfolds <- 10
  folds <- createFolds(1:nrow(dat_NN_scaled_numeric), k = numfolds, list = F)
  rmls_cv <- as.data.frame(matrix(NA, nrow = numfolds, ncol = 1))
  colnames(rmls_cv) <- c("rmls")
  predictions <- vector()
  truevals <- vector()
  for(j in 1:numfolds){
    trn <- dat_NN_scaled_numeric[folds != j,]
    tst <- dat_NN_scaled_numeric[folds == j,]
    ## PUT NN IN HERE
    starttime <- proc.time()
    nrnt <- neuralnet(f,
                      data = trn,
                      hidden = vector_hidden_layers,
                      linear.output = T,
                      rep = 1)
    print(proc.time() - starttime)
    p <- predict(nrnt, tst[,1:231])
    p_orig_val <- p*sd(dat$SalePrice) + mean(dat$SalePrice)
    print(head(p_orig_val))
    print(head(dat$SalePrice[folds == j]))
    predictions <- c(predictions, p_orig_val)
    truevals <- c(truevals, dat$SalePrice[folds == j])
    
    inds <- predictions > 0
    error <- Metrics::rmsle(truevals[inds], predictions[inds])
    print(error)
    #error <- Metrics::rmsle(dat$SalePrice[folds == j], p_orig_val)
    rmls_cv[j,1] <- error
  }
  return(mean(rmls_cv$rmls))
}

# Note that all of these remove negative predictions...
error_50_30_30_30 <- test_nn(c(50,30,30,30)) # .177
error_50_30_20_10 <- test_nn(c(50,30,20,10)) # .185
error_50_30_20 <- test_nn(c(50,30,20)) # .216

(error_60_50_40_40 <- test_nn(c(60,50,40,40))) # .179
(error_70_60_50_50 <- test_nn(c(70,60,50,50))) # .192
(error_80_70_60_60 <- test_nn(c(80,70,60,60))) # .193
(error_120_100_100_90 <- test_nn(c(120,100,100,90))) # .181

(error_120_100 <- test_nn(c(120,100))) # .645
(error_60_40 <- test_nn(c(60,40))) # .358
(error_80 <- test_nn(c(80))) # .776
(error_40 <- test_nn(c(40))) # .653

(error_50_60_60_70 <- test_nn(c(50,60,60,70))) # .201
(error_40_40_40_40 <- test_nn(c(40,40,40,40))) # .193
(error_50_60_60 <- test_nn(c(50,60,60))) # .222
(error_80_80_80_80 <- test_nn(c(80,80,80,80))) # .178

(error_60_50_40_40_30 <- test_nn(c(60,50,40,40,30))) # .177
(error_40_30_20_10_10 <- test_nn(c(40,30,20,10,10))) # .182

(error_60_50_40 <- test_nn(c(60,50,40))) # .195

(error_80_80_80_80_80_70_70_70_70_60_60 <- test_nn(c(80,80,80,80,80,70,70,70,70,60,60))) # .1697

(error_50x15 <- test_nn(rep(50, times = 15))) # .1695

(error_80x15 <- test_nn(rep(80, times = 15))) # .1785

# Swicthing to 5-fold CV
(error_50x20 <- test_nn(rep(50, times = 20))) # .1914
(error_70x50 <- test_nn(rep(70, times = 50))) # .reverting to mean.


####### After all  that, this one was slightly better than others:
(error_50x15 <- test_nn(rep(50, times = 15))) # .1695
nrnt_final <- neuralnet(f,
                  data = dat_NN_scaled_numeric,
                  hidden = rep(50, times = 15),
                  linear.output = T,
                  rep = 1)





