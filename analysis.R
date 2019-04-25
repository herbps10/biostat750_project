library(tidyverse)
library(pcLasso)

dat <- read_csv("data/data.csv") %>%
  select(-id, -X33) %>%
  mutate(diagnosis = as.numeric(diagnosis == "M"))

summary(prcomp(select(dat, -diagnosis)))

X = as.matrix(select(dat, -diagnosis))
y = pull(dat, diagnosis)

##### Evaluate LASSO
fit_lasso <- function(x, y) {
  glmnet::cv.glmnet(x, y, family = "binomial", nfolds = 5)
}

test_performance_lasso <- function(fit, x, y) {
  p = predict(fit, newx = x)[,1]
  as.numeric(pROC::roc(y ~ p)$auc)
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
         
         auc = pmap_dbl(list(fit, x_test, y_test), test_performance_lasso))

mean(lasso_folds$auc)

##### Evaluate pcLasso
fit_pclasso <- function(x, y, ratio) {
  cv.pcLasso(x, y, family = "binomial", ratio = ratio, nfolds = 5)
}

test_performance_pclasso <- function(fit, x, y) {
  p = predict(fit, xnew = x)
  as.numeric(pROC::roc(y ~ p)$auc)
}

k <- 10
indices <- caret::createFolds(y, k = k)
pclasso_folds <- expand.grid(
  ratio = seq(0.1, 1, 0.1),
  fold = 1:k
) %>%
  as_tibble() %>%
  mutate(indices = map(fold, function(i) indices[[i]])) %>%
  mutate(y_train = map(indices, function(i) y[-i]),
         x_train = map(indices, function(i) X[-i, ]),
         y_test  = map(indices, function(i) y[i]),
         x_test  = map(indices, function(i) X[i, ]),
         
         fit = pmap(list(x_train, y_train, ratio), fit_pclasso),
         
         auc = pmap_dbl(list(fit, x_test, y_test), test_performance_pclasso))

pclasso_folds %>%
  group_by(ratio) %>%
  summarize(auc = mean(auc))


##### Evaluate BART
fit_bart <- function(x, y) {
  bartMachine::bartMachine(X = as.data.frame(x), y = factor(y), mem_cache_for_speed = FALSE)
}

test_performance_bart <- function(fit, x, y) {
  p = predict(fit, new_data = as.data.frame(x))
  as.numeric(pROC::roc(y ~ p)$auc)
}

k <- 10
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
         
         auc = pmap_dbl(list(fit, x_test, y_test), test_performance_bart))