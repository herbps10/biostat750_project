library(tidyverse)

process_data <- function(data) {
  ids <- data$Id
  
  train <- !is.na(data$SalePrice)
  
  data <- data %>%
    select(-Id, -LotFrontage, -Alley, -FireplaceQu, -PoolQC, -Fence, -MiscFeature, -Utilities) %>%
    mutate_if(is.numeric, zoo::na.aggregate) %>%
    mutate_if(is.character, function(x) {
      x[is.na(x)] <- names(sort(table(x),decreasing=TRUE)[1])
      x
    })
  
  data <- data %>%
    mutate(
      TotalSF = TotalBsmtSF + `1stFlrSF` + `2ndFlrSF`,
      Total_Bathrooms = FullBath + (0.5 * HalfBath) + BsmtFullBath + (0.5 * BsmtHalfBath),
      haspool = ifelse(PoolArea > 0, 1, 0),
      has2ndfloor = ifelse(`2ndFlrSF` > 0, 1, 0),
      hasgarage = ifelse(GarageArea > 0, 1, 0),
      hasfireplace = ifelse(Fireplaces > 0, 1, 0),
      OverallQualCat = as.factor(OverallQual),
      OverallCondCat = as.factor(OverallCond),
      LogLotArea = log(LotArea),
      YrBltAndRemod = YearBuilt + YearRemodAdd
    )
  
  X_df <- data %>% 
    select(-SalePrice) %>%
    mutate_if(is.numeric, scale)
  
  X <- model.matrix(~ -1 + ., X_df, contrasts.arg = contrasts)
  y <- pull(data, SalePrice)
  
  X_train <- X[train,]
  y_train <- y[train]
  X_test <- X[!train,]
  
  list(ids = ids, X_train = X_train, y_train = y_train, X_test = X_test, test_ids = ids[!train])
}

data <- read_csv("data/train.csv", guess_max = 10000) %>%
  bind_rows(read_csv("data/test.csv", guess_max = 10000)) %>%
  process_data()

X_scaled <- data$X_train
y <- data$y_train

fit_lasso <- function(x, y, alpha) {
  glmnet::cv.glmnet(x, y, nfolds = 5, alpha = alpha)
}

fit_xgb <- function(x, y, eta, max_depth) {
  xgboost::xgboost(
    data = x,
    label = y,
    nrounds = 1000,
    nfold = 5,
    objective = "reg:linear",
    verbose = 0,
    early_stopping_rounds = 50,
    params = list(
      eta = eta,
      max_depth = max_depth
    )
  )
}

fit_bart <- function(x, y) {
  bartMachine::bartMachine(X = as.data.frame(x),
                           y = y, mem_cache_for_speed = FALSE,
                           num_trees = 50)
}

fit_knn <- function(x, y, k) {
  caret::knnreg(x = x, y = y, k = k)
}

fit_ensemble <- function(x, y) {
  ridge <- fit_lasso(x, y, 0)
  lasso <- fit_lasso(x, y, 1)
  xgb   <- fit_xgb(x, y, 0.05, 3)
  knn   <- fit_knn(x, y, 12)
  #bart  <- fit_bart(x, y)
  list(ridge = ridge, lasso = lasso, xgb = xgb, knn = knn) #bart = bart)
}

predict_ensemble <- function(weights, fit, x) {
  p_ridge = predict(fit$ridge, newx = x)[,1]
  p_lasso = predict(fit$lasso, newx = x)[,1]
  p_xgb   = predict(fit$xgb,   newdata = x)
  #p_bart  = predict(fit$bart,  new_data = as.data.frame(x))
  p_knn   = predict(fit$knn,   newdata = x)
  
  weights <- weights / sum(weights)
  p = weights[1] * p_ridge + weights[2] * p_lasso + weights[3] * p_xgb + weights[4] * p_knn #weights[4] * p_bart
  p
}

test_performance_ensemble <- function(weights, fit, x, y) {
  p <- predict_ensemble(weights, fit, x)
  Metrics::rmsle(p, y)
}

cross_validated_error <- function(weights, folds) {
  res <- folds %>%
    mutate(performance = pmap_dbl(list(fit, x_test, y_test), test_performance_ensemble, weights = weights))
  
  mean(res$performance)
}

k <- 10
indices <- caret::createFolds(y, k = k)
weights <- rep(1/4, 4)
ensemble_folds <- expand.grid(
  fold = 1:k
) %>%
  as_tibble() %>%
  mutate(indices = map(fold, function(i) indices[[i]])) %>%
  mutate(y_train = map(indices, function(i) y[-i]),
         x_train = map(indices, function(i) X_scaled[-i, ]),
         y_test  = map(indices, function(i) y[i]),
         x_test  = map(indices, function(i) X_scaled[i, ]),
         
         fit = pmap(list(x_train, y_train), fit_ensemble),
         
         performance = pmap_dbl(list(fit, x_test, y_test), test_performance_ensemble, weights = weights))

mean(ensemble_folds$performance)

optimal_weights <- optim(par = rep(1/4, 4),
                         fn = cross_validated_error,
                         lower = c(0, 0, 0, 0),
                         upper = c(1, 1, 1, 1),
                         method = "L-BFGS-B",
                         folds = ensemble_folds)

weights <- optimal_weights$par

overall_fit <- fit_ensemble(X_scaled, y)

##### Predict test set

prediction <- predict_ensemble(weights, overall_fit, data$X_test)

tibble(
  Id = data$test_ids,
  SalePrice = prediction
) %>%
  write_csv("test_predictions.csv")
