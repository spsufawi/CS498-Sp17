#3.6(a)
library(ggplot2)
library(data.table)  #Load package we need
library(caret)
dt <- fread("K9.data", na.strings="?") #read the data set
dt <- dt[,-5410] #remove last column
new_dt <- dt  # make the same data frame
new_dt <- na.omit(new_dt) #remove all NA

binary <- ifelse(new_dt[,5409] == "active", 1, -1) #make an indicator variable

new_dt$binary <- binary #add the data frame

new_dt[,5409] <- NULL #remove it

nrow(new_dt) #31159

train_split <- createDataPartition(y=new_dt$binary , p=.9, list=FALSE)

training <- new_dt[train_split,]  # training

testing <- new_dt[-train_split, ] # testing


#############Cross Validation#################
##10fold Cross Validation

cv_dt <- training

lambda <- c(0.001, 0.01, 0.1) #choose from these three values

cv_error <- matrix(NA, ncol = 3, nrow = 10) 
for(q in 1:10){
  for(k in 1:3){

folder <- createDataPartition(cv_dt$binary, p = 0.9, list = FALSE)
cross_trainX <- cv_dt[folder, 1:5408]
cross_trainY <- cv_dt[folder, 5409]

cross_validate_X <- as.matrix(cv_dt[-folder, 1:5408])
cross_validate_Y <- as.matrix(cv_dt[-folder, 5409])
  
a <- matrix(runif(5408), ncol = 1)
b <- runif(1)

m <- 0.01
n <- 0.01

season <- 50
step <- 50

for(i in 1:season){
  for(j in 1:step){
    random <- sample(nrow(cross_trainX),1)
    random_X <- as.matrix(cross_trainX[random,])
    random_Y <- as.matrix(cross_trainY[random,])
    
    dim(a) #5408 1
    dim(random_X) # 1 5408
    eta <- m/(i + n)
    
    if(random_Y*(random_X %*% a + b) >=1){
      p_a <- lambda[k]*a
    } else
    { 
      p_a <-  lambda[k]*a - t(random_Y%*%random_X)
    }
    
   
    p_b <- ifelse(random_Y*(random_X %*% a + b) >=1, 0, -random_Y)
    
    a <- a - eta*p_a
    b <- b - eta*p_b
  }
}
 result <-  ifelse(cross_validate_X %*% a + as.vector(b) > 0, 1, -1)
 cv_error[q,k] <- mean(result == cross_validate_Y  )
  }
}

1 - apply(cv_error, 2, mean) #average

#################Use the best lambda and all training data set################

set.seed(1234)

trainingX <-  training[, 1:5408]
trainingY <-  training[, 5409]

a <- as.matrix(runif(5408), ncol = 1)
b <- runif(1)

m <- 0.01
n <- 0.01

season <- 150
step <- 100

accuracy <- numeric(season)

lambda <- 0.01
for(i in 1:season){
  for(j in 1:step){
random <- sample(nrow(training),1)
random_X <- as.matrix(trainingX[random,])
random_Y <- as.matrix(trainingY[random,])

dim(a) #5408 1
dim(random_X) # 1 5408
eta <- m/(i + n)

if(random_Y*(random_X %*% a + b) >=1){
   p_a <- lambda*a
} else
{ 
  p_a <-  lambda*a - t(random_Y%*%random_X)
}

#p_a <- ifelse(random_Y*(random_X %*% a + b) >=1, lambda*a, lambda*a - t(random_Y%*%random_X))
p_b <- ifelse(random_Y*(random_X %*% a + b) >=1, 0, -random_Y)

a <- a - eta*p_a
b <- b - eta*p_b
  }
  result_temp <- ifelse(as.matrix(testing[,1:5408]) %*% a + as.vector(b) > 0, 1, -1)
  accuracy[i] <- mean(result_temp==testing[,5409])
}

result <- ifelse(as.matrix(testing[,1:5408]) %*% a + as.vector(b) > 0, 1, -1)
mean(result==testing[,5409])

df <- data.frame(season = 1:season, accuracy)
ggplot(df, aes(x = season, y = accuracy)) + geom_line() + 
      scale_y_continuous(limits = c(0, 1))
########################(b)##########################

tr <- trainControl(method='cv' , number=10)
nb <- train(x = trainingX[1:5000,], y = factor(trainingY$binary[1:5000]), method = 'nb', trControl =tr)
result <- predict(nb, testing[,1:5408])
accuracy <- mean(result == testing[,5409])






#library(e1071)
#classifier <- naiveBayes(x=trainingX, y = factor(trainingY$binary), laplace=1)
#nb_result <- predict(classifier,testing[,1:5408], type = "class")
#mean(as.numeric(as.character(nb_result)) == testing[,5409])


                         
