library(tidyverse)
library(pcLasso)
library(glmnet)
library(mice)

dat <- as_tibble(read.csv("house-prices/train.csv", header = T))

#get rid of cols with lots of missing values
dat <- select(dat, -Id, -LotFrontage, -Alley, -FireplaceQu, -PoolQC, -Fence, -MiscFeature)
dat <- dat[complete.cases(dat),]
sum(is.na(dat)) # = 0. yay

y <- dat$SalePrice
x <- select(dat, -SalePrice)
x <- model.matrix(~., x[,])

write.csv(x, "house-prices/train_x.csv")
write.csv(y, "house-prices/train_y.csv")

summary(prcomp(select(dat, -id, -SalePrice)))

lasso <- glmnet::cv.glmnet(x, y, family = "gaussian", alpha = 1, nfolds = 10)
best_lambda <- lasso$lambda[which.min(lasso$cvm)]

enet0.95 <- glmnet::cv.glmnet(x, y, alpha = 0.95, nfolds = 10)
enet0.5 <- glmnet::cv.glmnet(x, y, alpha = 0.5, nfolds = 10)
pclasso0.95 <- cv.pcLasso(x, y, ratio = 0.95, nfolds = 10)
pclasso0.75 <- cv.pcLasso(x, y, ratio = 0.75, nfolds = 10)
pclasso0.5 <- cv.pcLasso(x, y, ratio = 0.5, nfolds = 10)

plot(x = lasso$nzero, y = lasso$cvm, col = 'purple', type = 'l')
lines(x = enet0.95$nzero, y = enet0.95$cvm)
lines(x = pclasso0.95$nzero, y = pclasso0.95$cvm)
lines(x = pclasso0.75$nzero, y = pclasso0.75$cvm, col = 'red')
lines(x = pclasso0.5$nzero, y = pclasso0.5$cvm, col = 'green')
