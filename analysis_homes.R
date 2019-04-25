library(tidyverse)

dat <- read_csv("~/Downloads/house-prices-advanced-regression-techniques/train.csv") %>%
  select(-Id, -Alley, -MiscFeature, -Fence, -PoolQC, -LotFrontage, -FireplaceQu)  %>%
  filter(complete.cases(.) == TRUE)

X <- model.matrix(SalePrice ~ ., dat)
y <- pull(dat, SalePrice)

##### Evaluate LASSO
fit_lasso <- function(x, y) {
  glmnet::cv.glmnet(x, y, nfolds = 5)
}

test_performance_lasso <- function(fit, x, y) {
  p = predict(fit, newx = x)[,1]
  Metrics::rmsle(p, y)
}

k <- 10
indices <- caret::createFolds(y, k = k)
lasso_folds <- expand.grid(
  fold = 1:k
) %>%
  as_tibble() %>%
  mutate(indices = map(fold, function(i) indices[[i]])) %>%
  mutate(y_train = map(indices, function(i) y[-i]),
         x_train = map(indices, function(i) X[-i, ]),
         y_test  = map(indices, function(i) y[i]),
         x_test  = map(indices, function(i) X[i, ]),
         
         fit = pmap(list(x_train, y_train), fit_lasso),
         
         performance = pmap_dbl(list(fit, x_test, y_test), test_performance_lasso))

mean(lasso_folds$performance)

##### Evaluate BART
fit_bart <- function(x, y) {
  bartMachine::bartMachine(X = as.data.frame(x),
                           y = y, mem_cache_for_speed = FALSE,
                           num_trees = 100)
}

test_performance_bart <- function(fit, x, y) {
  p = predict(fit, new_data = as.data.frame(x))
  Metrics::rmsle(p, y)
}

k <- 5
indices <- caret::createFolds(y, k = k)
bart_folds <- expand.grid(
  fold = 1:k
) %>%
  as_tibble() %>%
  mutate(indices = map(fold, function(i) indices[[i]])) %>%
  mutate(y_train = map(indices, function(i) y[-i]),
         x_train = map(indices, function(i) X[-i, ]),
         y_test  = map(indices, function(i) y[i]),
         x_test  = map(indices, function(i) X[i, ]),
         
         fit = pmap(list(x_train, y_train), fit_bart),
         
         performance  = pmap_dbl(list(fit, x_test, y_test), test_performance_bart))

mean(bart_folds$performance)
