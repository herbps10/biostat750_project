
library(caret)
set.seed(1)
numfolds <- 10
folds <- createFolds(train.y, k = numfolds, list = F)

this_k_error <- vector()
for(j in 1:numfolds){
  trnx <- train.x[folds != j,]
  trny <- train.y[folds != j]
  tstx <- train.x[folds == j,]
  tsty <- train.y[folds == j]
  cart_model <- #### Insert CART here
  pred <- # insert prediction call here
  error <- Metrics::rmsle(tsty, cart_model$pred)
  this_k_error <- c(this_k_error, error)
}
print(this_k_error)
mean(this_k_error)