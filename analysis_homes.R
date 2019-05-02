library(tidyverse)

data <- read_csv("data/train.csv", guess_max = 10000) %>%
  bind_rows(read_csv("data/test.csv", guess_max = 10000)) %>%
  process_data()

X_scaled <- data$X_train
y <- data$y_train

##### Evaluate mean imputation
fit_mean <- function(x, y) {
  list(mean = mean(y))
}

test_performance_mean <- function(fit, x, y) {
  Metrics::rmsle(fit$mean, y)
}

k <- 10
indices <- caret::createFolds(y, k = k)
mean_folds <- expand.grid(
  fold = 1:k
) %>%
  as_tibble() %>%
  mutate(indices = map(fold, function(i) indices[[i]])) %>%
  mutate(y_train = map(indices, function(i) y[-i]),
         x_train = map(indices, function(i) X_scaled[-i, ]),
         y_test  = map(indices, function(i) y[i]),
         x_test  = map(indices, function(i) X_scaled[i, ]),
         
         fit = pmap(list(x_train, y_train), fit_mean),
         
         performance = pmap_dbl(list(fit, x_test, y_test), test_performance_mean))

mean(mean_folds$performance)

##### Evaluate LASSO, Ridge, ElasticNet
fit_lasso <- function(x, y, alpha) {
  print(paste("alpha = ", alpha))
  glmnet::cv.glmnet(x, y, nfolds = 5, alpha = alpha)
}

test_performance_lasso <- function(fit, x, y) {
  p = predict(fit, newx = x)[,1]
  p[p < 0] <- 0
  Metrics::rmsle(p, y)
}

k <- 10
indices <- caret::createFolds(y, k = k)
lasso_folds <- expand.grid(
  fold = 1:k,
  alpha = c(0, 0.25, 0.5, 0.75, 1)
) %>%
  as_tibble() %>%
  mutate(indices = map(fold, function(i) indices[[i]])) %>%
  mutate(y_train = map(indices, function(i) y[-i]),
         x_train = map(indices, function(i) X_scaled[-i, ]),
         y_test  = map(indices, function(i) y[i]),
         x_test  = map(indices, function(i) X_scaled[i, ]),
         
         fit = pmap(list(x_train, y_train, alpha), fit_lasso),
         
         performance = pmap_dbl(list(fit, x_test, y_test), test_performance_lasso))

lasso_folds %>% 
  group_by(alpha) %>%
  summarize(performance = mean(performance))

lasso_overall_fit <- fit_lasso(X_scaled, y, 0)
lasso_varimp <- caret::varImp(lasso_overall_fit$glmnet.fit, lambda = lasso_overall_fit$lambda.min)
tibble(
  name = rownames(lasso_varimp),
  value = lasso_varimp$Overall
) %>%
  arrange(-value)

##### Evaluate BART
fit_bart <- function(x, y, trees = 50, k = 2) {
  print(paste("trees =", trees))
  bartMachine::bartMachine(X = as.data.frame(x),
                           y = y, mem_cache_for_speed = FALSE,
                           num_trees = trees,
                           k = k)
}

test_performance_bart <- function(fit, x, y) {
  p = predict(fit, new_data = as.data.frame(x))
  Metrics::rmsle(p, y)
}

k <- 10
indices <- caret::createFolds(y, k = k)
bart_folds <- expand.grid(
  fold = 1:k,
  trees = c(50, 100, 200),
  k = c(2, 3, 5)
) %>%
  as_tibble() %>%
  mutate(indices = map(fold, function(i) indices[[i]])) %>%
  mutate(y_train = map(indices, function(i) y[-i]),
         x_train = map(indices, function(i) X_scaled[-i, ]),
         y_test  = map(indices, function(i) y[i]),
         x_test  = map(indices, function(i) X_scaled[i, ]),
         
         fit = pmap(list(x_train, y_train, trees, k), fit_bart),
         
         performance  = pmap_dbl(list(fit, x_test, y_test), test_performance_bart))

mean(bart_folds$performance)

bart_overall_fit <- fit_bart(X, y)

bart_varimp <- bartMachine::investigate_var_importance(bart_overall_fit)

##### Evaluate BART
fit_xgb <- function(x, y, eta, max_depth) {
  print(paste("eta =", eta, "max_depth=", max_depth))
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

test_performance_xgb <- function(fit, x, y) {
  p = predict(fit, newdata = x)
  Metrics::rmsle(p, y)
}

k <- 10
indices <- caret::createFolds(y, k = k)
xgboost_folds <- expand.grid(
  fold = 1:k,
  eta = 0.01,
  max_depth = 3
  #eta = c(.01, .05, .1, .3),
  #max_depth = c(1, 3, 5, 7)
) %>%
  as_tibble() %>%
  mutate(indices = map(fold, function(i) indices[[i]])) %>%
  mutate(y_train = map(indices, function(i) y[-i]),
         x_train = map(indices, function(i) X_scaled[-i, ]),
         y_test  = map(indices, function(i) y[i]),
         x_test  = map(indices, function(i) X_scaled[i, ]),
         
         fit = pmap(list(x_train, y_train, eta, max_depth), fit_xgb),
         
         performance  = pmap_dbl(list(fit, x_test, y_test), test_performance_xgb))

xgboost_folds %>%
  group_by(eta, max_depth) %>%
  summarize(performance = mean(performance))

xgb_overall_fit <- fit_xgb(X_scaled, y, eta = 0.01, max_depth = 3)
test_performance_xgb(xgb_overall_fit, X_scaled, y)

xgboost::xgb.importance(model = xgb_overall_fit)
