library(ggplot2)
library(MASS)
library(glmnet)
library(readxl)

###############Linear regression with various regularizers###############

dt <- read.table("default_plus_chromatic_features_1059_tracks.txt", sep = ",")

lati <- dt$V117 
longti <- dt$V118

X <- dt[, -c(117, 118)]
######Latititude#####

mod1 <- lm(lati ~ as.matrix(X))
mod1_r2 <- summary(mod1)$r.squared #0.2928092
resi_mod1 <- resid(mod1)
fitted_mod1 <- fitted(mod1)
resi_fit_mod1 <- data.frame(fitted_mod1, resi_mod1)

mse1 <- sum((resi_mod1)^2) / nrow(dt)

ggplot(data = resi_fit_mod1) + geom_point(aes(x=fitted_mod1,y=resi_mod1)) + 
  ggtitle("Latitude vs Other features") + xlab("fitted values") + ylab("residuals")

##Box-Cox transformation
box_trans1 <-  lm((lati - min(lati) + 0.0001)  ~ as.matrix(X))
box_cox1 <- boxcox(box_trans1)
lambda1 <- box_cox1$x[which.max(box_cox1$y)]

trans_mod1 <-  lm((lati - min(lati) + 0.0001)^lambda1 ~ as.matrix(X))
box_trans2_r2 <- summary(trans_mod1)$r.squared #0.2977738

resi_trans_mod1 <- resid(trans_mod1)
fitted_trans_mod1 <- fitted(trans_mod1)
resi_fit_trans_mod1 <- data.frame(resi_trans_mod1, fitted_trans_mod1)

mse2 <- sum((resi_trans_mod1)^2) / nrow(dt)

ggplot(data = resi_fit_trans_mod1) + geom_point(aes(x=fitted_trans_mod1,y=resi_trans_mod1)) + 
  ggtitle("Latitude vs Other features\n(After Box-Cox transformation)") + xlab("fitted values") + ylab("residuals")

##Regularization Method##
mod_ri1 <- cv.glmnet(as.matrix(X), lati, alpha = 0) #Ridge
min(mod_ri1$cvm)
plot(mod_ri1)


mod_la1 <- cv.glmnet(as.matrix(X), lati, alpha = 1) #Lasso
min(mod_la1$cvm)
length(unique(coef(mod_la1, s = "lambda.min")[,1])) - 2
plot(mod_la1)

######Longtitude#####
mod2 <- lm(longti ~ as.matrix(X))
mod2_r2 <- summary(mod2)$r.squared  #0.3645767

resi_mod2 <- resid(mod2)
fitted_mod2 <- fitted(mod2)
resi_fit_mod2 <-  data.frame(fitted_mod2, resi_mod2)

mse3 <- sum(resi_mod2^2) / nrow(dt)

ggplot(data = resi_fit_mod2) + geom_point(aes(x=fitted_mod2,y=resi_mod2)) + 
  ggtitle("Longtitude vs Other features") + xlab("fitted values") + ylab("residuals")

##Box-Cox transformation##

box_trans2 <-  lm((longti - min(longti) + 0.0001)  ~ as.matrix(X))
box_cox2 <- boxcox(box_trans2)
lambda2 <- box_cox2$x[which.max(box_cox2$y)]

trans_mod2 <-  lm((longti - min(longti) + 0.0001)^lambda2 ~ as.matrix(X))
box_trans2_r2 <- summary(trans_mod2)$r.squared # 0.3616467

resi_trans_mod2 <- resid(trans_mod2)
fitted_trans_mod2 <- fitted(trans_mod2)
resi_fit_trans_mod2 <- data.frame(resi_trans_mod2, fitted_trans_mod2)

mse4 <- sum(resi_trans_mod2^2) / nrow(dt)

ggplot(data = resi_fit_trans_mod2) + geom_point(aes(x=fitted_trans_mod2,y=resi_trans_mod2)) + 
  ggtitle("Longtitude vs Other features\n(After Box Cox transformation)") + xlab("fitted values") + ylab("residuals")

##Regularization Method##
mod_ri2 <- cv.glmnet(as.matrix(X), longti, alpha = 0) #Ridge
min(mod_ri2$cvm)
plot(mod_ri2)

mod_la2 <- cv.glmnet(as.matrix(X), longti, alpha = 1) #Lasso
min(mod_la2$cvm)
plot(mod_la2)

length(unique(coef(mod_la2, s = "lambda.min")[,1])) - 2  #variables used, no intercept



####################Logistic Regression####################

dt1 <- read_excel("default of credit card clients.xls", skip = 1)
dt1$ID <- NULL
for (i in c(2:4, 6:11)){
  dt1[[names(dt1)[i]]] <- factor(dt1[[names(dt1)[i]]])
}

logi_mod1 <- glm(`default payment next month` ~., family = "binomial", data = dt1)

fitted_results <- ifelse(fitted(logi_mod1)> 0.5,1,0)

table(fitted_results, dt1$`default payment next month`)
accy <- (22280  +  2375) / (22280  +  2375 + 4261 + 1084)
                      
##Regularization Method##
default_pay <- as.matrix(dt1$`default payment next month`)
X1 <- dt1[, -ncol(dt1)]

trans <- formula(paste("~",paste(colnames(X1),collapse="+"),sep=""))
X_dummy <- model.matrix(trans,X1)
##Ridge
logi_mod_ri <- cv.glmnet(X_dummy, default_pay, family = "binomial", 
                         type.measure = "class", alpha = 0)
accy_ri <- 1 - min(logi_mod_ri$cvm)
plot(logi_mod_ri)
##Lasso
logi_mod_la <- cv.glmnet(X_dummy, default_pay, family = "binomial", 
                         type.measure = "class", alpha = 1)
accy_la <- 1 - min(logi_mod_la$cvm)
plot(logi_mod_la)
##Elastic net
logi_mod_elas <- cv.glmnet(X_dummy, default_pay, family = "binomial", 
                           type.measure = "class", alpha = 0.5)
accy_elas <- 1 - min(logi_mod_elas$cvm)
plot(logi_mod_elas)
##########A wide dataset, from cancer genetics##########

dt2 <- read_excel("Cancer Genetics.xlsx", col_names = FALSE)
dt2 <- t(dt2)
response <- read.table("response.txt")
response_ind <- factor(ifelse(response > 0, 0, 1))

##type.measure = "class"##
logi_mod1_la <- cv.glmnet(dt2, response_ind, alpha = 1, type.measure = "class", 
                         family = "binomial")
accy_la1 <- 1 - min(logi_mod1_la$cvm)
plot(logi_mod1_la)
##type.measure = "deviance"##
logi_mod1_la_dev <- cv.glmnet(dt2, response_ind, alpha = 1, type.measure = "deviance", 
                          family = "binomial")
dev <- min(logi_mod1_la_dev$cvm)
plot(logi_mod1_la_dev )
##type.measure = "auc"##
logi_mod1_la_auc <- cv.glmnet(dt2, response_ind, alpha = 1, type.measure = "auc", 
                              family = "binomial", nfold = 5)
auc <- max(logi_mod1_la_auc$cvm)

plot(logi_mod1_la_auc)

length(unique(coef(logi_mod1_la,s = "lambda.min")[,1]))-2


