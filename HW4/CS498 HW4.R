library(ggplot2)
library(grid)
library(gridExtra)
library(glmnet)

#####7.9#####
dt <- read.table("brunhild.txt", sep = "\t", header = T)
dt
##(a)##

mod1 <- lm(log(Sulfate) ~ log(Hours), data = dt)

ggplot(data = dt, aes(x = log(Hours), y = log(Sulfate))) + geom_point() + 
  geom_smooth(method="lm") + ggtitle("Hours vs Sulfate (Log-Log Coordinates)")

##(b)##

pt_ori_y <- (dt$Hours)^(-0.247)*exp(2.766) 
new_dt_ori <- data.frame(dt$Hours,pt_ori_y)

ggplot(data = dt, aes(x = Hours, y = Sulfate)) + geom_point() + geom_line(data = new_dt_ori, aes(x=dt.Hours,y=pt_ori_y)) + 
  ggtitle("Hours vs Sulfate (Original Coordinates)")

##(c)##

resid_log <- residuals(mod1)
fitted_log <- fitted(mod1)
res_fit_log <- data.frame(resid_log, fitted_log)

fitted_ori <- exp(fitted(mod1))
resid_ori <- dt$Sulfate - fitted_ori
res_fit_ori <- data.frame(resid_ori, fitted_ori)

p1 <- ggplot(data = res_fit_log, aes(x =fitted_log, y = resid_log)) + 
  geom_point() + ggtitle("Residual vs Fitted (Log-Log Coordinates)")

p2 <- ggplot(data = res_fit_ori, aes(x = fitted_ori, y = resid_ori)) + 
  geom_point() + ggtitle("Residual vs Fitted (Original Coordinates)")

grid.arrange(p1, p2, nrow = 2)

##(d)##
#In pdf file

#####7.10#####

dt2 <- read.table("physical.txt", sep = "\t", header = T)

##(a)##

mod3 <- lm(Mass ~., data = dt2)

fitted_mod3 <- fitted(mod3)
res_mod3 <- residuals(mod3)

res_fitted_mod3 <- data.frame(res_mod3, fitted_mod3)

ggplot(data = res_fitted_mod3, aes(x = fitted_mod3, y = res_mod3)) + 
  geom_point() + ggtitle("Residuals vs Fitted Values") +
  xlab("Fitted Value") +  ylab("Residual")
  

##(b)##

mod4 <- lm(Mass^(1/3) ~., data = dt2)

fitted_mod4 <- fitted(mod4)
res_mod4 <- residuals(mod4)
res_fitted_mod4 <-data.frame(res_mod4, fitted_mod4)

fitted_mod4_cube <- fitted(mod4)^3
res_mod4_cube <- dt2$Mass - fitted_mod4_cube
res_fitted_mod4_cube <-data.frame(res_mod4_cube, fitted_mod4_cube)

p3 <- ggplot(res_fitted_mod4_cube, aes(x=fitted_mod4, y = res_mod4)) +
  geom_point() + ggtitle("Residual vs Fitted Value(Cube Root Coordinates)")+
  xlab("Fitted Value") +  ylab("Residual")

p4 <- ggplot(res_fitted_mod4_cube, aes(x=fitted_mod4_cube, y = res_mod4_cube)) +
  geom_point() + ggtitle("Residual vs Fitted Value(Original Coordinates)")+
  xlab("Fitted Value") +  ylab("Residual")

grid.arrange(p3,p4, nrow = 2)

##(c)##

#In pdf file

#####7.11#####

dt3 <- read.table("abalone.data", sep = ",")
Names <- c("Sex", "Length", "Diameter", "Height", "Whole_weight", "Sucked_weight",
           "Viscera_weight", "Shell_weight", "rings")

colnames(dt3) <- Names

##(a)##

mod5 <- lm(rings ~ .-Sex, data = dt3)

res_ab <- residuals(mod5)
fitted_ab <- fitted(mod5)
res_fitted_ab <- data.frame(res_ab, fitted_ab)
mse <- sum((fitted_ab - dt3$rings)^2)/nrow(dt3)

ggplot(data = res_fitted_ab, aes(x= fitted_ab, y= res_ab)) + geom_point() + 
  ggtitle("Residual vs Fitted Values \n (Ignoring gender)")+
  xlab("Fitted Value") +  ylab("Residual")

##(b)##

mod6 <- lm(rings ~., data = dt3)

res_ab1 <- residuals(mod6)
fitted_ab1 <- fitted(mod6)
res_fitted_ab1 <- data.frame(res_ab1, fitted_ab1, Sex = dt3$Sex)
mse1 <- sum((dt3$rings-fitted_ab1)^2)/nrow(dt3)

ggplot(data = res_fitted_ab1, aes(x=fitted_ab1, y = res_ab1,color = Sex)) + 
  geom_point() + ggtitle("Residual vs Fitted Values \n (Including gender)")+
  xlab("Fitted Value") +  ylab("Residual")


##(c)##
#log of age, ignoring gender

mod7 <- lm(log(rings) ~. - Sex, data = dt3)
res_ab2 <- residuals(mod7)
fitted_ab2 <- fitted(mod7)
res_fitted_ab2 <- data.frame(res_ab2, fitted_ab2)
mse2 <-  sum((log(dt3$rings) - fitted_ab2)^2)/nrow(dt3)

ggplot(data = res_fitted_ab2, aes(x = fitted_ab2, y = res_ab2)) + 
  geom_point() + 
  ggtitle("Residual vs Fitted Values \n (Take log of age, ignoring gender)")+
  xlab("Fitted Value") +  ylab("Residual")

##(d)##
#log of age, including gender

mod8 <- lm(log(rings) ~., data = dt3)
res_ab3 <- residuals(mod8)
fitted_ab3 <- fitted(mod8)
res_fitted_ab3 <- data.frame(res_ab3, fitted_ab3, Sex = dt3$Sex)
mse3 <- sum((log(dt3$rings) - fitted_ab3)^2)/nrow(dt3)

ggplot(data = res_fitted_ab3, aes(x = fitted_ab3, y = res_ab3,color = Sex)) + geom_point() + 
  ggtitle("Residual vs Fitted Values\n(Take log of age,including gender)")+
  xlab("Fitted Value") +  ylab("Residual")

##(e)##

#In pdf file

##(f)##
## age ~ other measurements, ignoring sex (Ridge regression)
X <- as.matrix(dt3[, -c(1,ncol(dt3))])
y <- as.matrix(dt3[, ncol(dt3)])

re_mod_ri <- cv.glmnet(X,y, alpha = 0)
cv_error_ri <- min(re_mod_ri$cvm)
plot(re_mod_ri)
## age ~ other measurements, ignoring sex (Lasso regression)
re_mod_la <- cv.glmnet(X,y, alpha = 1)
cv_error_la <- min(re_mod_la$cvm)
plot(re_mod_la)
## age ~ other measurements, including sex (Ridge regression)
X1 <- dt3[, -ncol(dt3)]

trans <- formula(paste("~",paste(colnames(X1),collapse="+"),sep=""))

X_dummy <- model.matrix(trans,X1)
y <- as.matrix(dt3[, ncol(dt3)])

re_mod1_ri <- cv.glmnet(X_dummy,y, alpha = 0)
cv_error1_ri <- min(re_mod1_ri$cvm)
plot(re_mod1_ri)

## age ~ other measurements, including sex (Lasso regression)
re_mod1_la <- cv.glmnet(X_dummy,y, alpha = 1)
cv_error1_la <-  min(re_mod1_la$cvm)
plot(re_mod1_la)

## log(age) ~ other measurements, ignoring sex (Ridge regression)

re_mod2_ri <- cv.glmnet(X,log(y), alpha = 0)
cv_error2_ri <- min(re_mod2_ri$cvm)
plot(re_mod2_ri)

## log(age) ~ other measurements, ignoring sex (Lasso regression)

re_mod2_la <- cv.glmnet(X,log(y), alpha = 1)
cv_error2_la <- min(re_mod2_la$cvm)
plot(re_mod2_la)

## log(age) ~ other measurements, including sex (Ridge regression)

re_mod3_ri <- cv.glmnet(X_dummy,log(y), alpha = 0)
cv_error3_ri <-  min(re_mod3_ri$cvm)
plot(re_mod3_ri)
## log(age) ~ other measurements, including sex (Lasso regression)

re_mod3_la <- cv.glmnet(X_dummy,log(y), alpha = 1)
cv_error3_la <- min(re_mod3_la$cvm)
plot(re_mod3_la)




