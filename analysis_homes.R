library(tidyverse)

X <- read_csv("data/train_x.csv") %>% select(-X1, -`(Intercept)`) %>%
  select(-Exterior1stAsphShn, -ExterCondPo, -FoundationSlab, -HeatingWall)

# Feature engineering
X$TotalSF <- X$TotalBsmtSF + X$X1stFlrSF + X$X2ndFlrSF
X$Total_Bathrooms = X$FullBath + (0.5 * X$HalfBath) +
                              X$BsmtFullBath + (0.5 * X$BsmtHalfBath)
X$haspool <- ifelse(X$PoolArea > 0, 1, 0)
X$has2ndfloor <- ifelse(X$X2ndFlrSF > 0, 1, 0)
#X$hasgarage <- ifelse(X$GarageArea > 0, 1, 0)
X$hasfireplace <- ifelse(X$Fireplaces > 0, 1, 0)
X$OverallQualCat <- as.factor(X$OverallQual)
X$OverallCondCat <- as.factor(X$OverallCond)
X$LogLotArea <- log(X$LotArea)
X$YrBltAndRemod <- X$YearBuilt + X$YearRemodAdd

X <- model.matrix(~-1 + ., X)
X <- scale(X)
y <- read_csv("data/train_y.csv") %>% pull(x)

##### Evaluate LASSO, Ridge, ElasticNet
fit_lasso <- function(x, y, alpha) {
  glmnet::cv.glmnet(x, y, nfolds = 5, alpha = alpha)
}

test_performance_lasso <- function(fit, x, y) {
  p = predict(fit, newx = x)[,1]
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
         x_train = map(indices, function(i) X[-i, ]),
         y_test  = map(indices, function(i) y[i]),
         x_test  = map(indices, function(i) X[i, ]),
         
         fit = pmap(list(x_train, y_train, alpha), fit_lasso),
         
         performance = pmap_dbl(list(fit, x_test, y_test), test_performance_lasso))

lasso_folds %>% 
  group_by(alpha) %>%
  summarize(performance = mean(performance))

lasso_overall_fit <- fit_lasso(X, y, 1)
lasso_varimp <- caret::varImp(lasso_overall_fit$glmnet.fit, lambda = lasso_overall_fit$lambda.min)
tibble(
  name = rownames(lasso_varimp),
  value = lasso_varimp$Overall
) %>%
  arrange(-value)

##### Evaluate BART
fit_bart <- function(x, y) {
  bartMachine::bartMachine(X = as.data.frame(x),
                           y = y, mem_cache_for_speed = FALSE,
                           num_trees = 50)
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
