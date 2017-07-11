
rm(list=ls(all=TRUE))
library(klaR)
library(caret)

# setwd('~/Current/Courses/LearningCourse/Pima')
# wdat<-read.csv('data.txt', header=FALSE)
# setwd("C:/Users/guoti/Dropbox/CS 498 -Applied Machine Learning Spring 2017/MyCode")
setwd("F:/Dropbox/CS 498 -Applied Machine Learning Spring 2017/MyCode")
wdat<-read.csv('pima-indians-diabetes.data.txt', header=FALSE)
bigx<-wdat[,-c(9)] # features
bigy<-wdat[,9] # label
trscore<-array(dim=10)
tescore<-array(dim=10)
###### my code
wtd<-createDataPartition(y=bigy, p=.8, list=FALSE) #caret
nbx<-bigx
ntrbx<-nbx[wtd, ] # training X: 615
ntrby<-bigy[wtd] # training Y: 615
trposflag<-ntrby>0
ptregs<-ntrbx[trposflag, ] # positive labels: 215
ntregs<-ntrbx[!trposflag,] # negative labels: 400
ntebx<-nbx[-wtd, ] # test X
nteby<-bigy[-wtd] # test Y: 153

ptrmean<-sapply(ptregs, mean, na.rm=TRUE) # positive label mean
ntrmean<-sapply(ntregs, mean, na.rm=TRUE) # negative label mean
ptrsd<-sapply(ptregs, sd, na.rm=TRUE) # positive label std
ntrsd<-sapply(ntregs, sd, na.rm=TRUE) # negative label std

ptroffsets<-t(t(ntrbx)-ptrmean)
ptrscales<-t(t(ptroffsets)/ptrsd) # scale it
ptrlogs<--(1/2)*rowSums(apply(ptrscales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd))
# assume normal: 1/sqrt(2*pi)/sigma * exp(x-mu)/2*sigma
# take log: log(x-mu)^2/2*sigma - log(sigma)
# log(x-mu)^2/sigma is ptrscales
# apply(ptrscales,c(1, 2), function(x)x^2) -- every element is squared
# train data
ntroffsets<-t(t(ntrbx)-ntrmean)
ntrscales<-t(t(ntroffsets)/ntrsd)
ntrlogs<--(1/2)*rowSums(apply(ntrscales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))

lvwtr<-ptrlogs>ntrlogs
gotrighttr<-lvwtr==ntrby # miss classfication rate 
trscore <-sum(gotrighttr)/(sum(gotrighttr)+sum(!gotrighttr))
# test data
pteoffsets<-t(t(ntebx)-ptrmean)
ptescales<-t(t(pteoffsets)/ptrsd)
ptelogs<--(1/2)*rowSums(apply(ptescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd))
nteoffsets<-t(t(ntebx)-ntrmean)
ntescales<-t(t(nteoffsets)/ntrsd)
ntelogs<--(1/2)*rowSums(apply(ntescales,c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))
lvwte<-ptelogs>ntelogs
gotright<-lvwte==nteby
tescore<-sum(gotright)/(sum(gotright)+sum(!gotright))

###use package

# use Klar and Caret
bigx<-wdat[,-c(9)]
bigy<-as.factor(wdat[,9])
wtd<-createDataPartition(y=bigy, p=.8, list=FALSE)
trax<-bigx[wtd,]
tray<-bigy[wtd]
model<-train(trax, tray, 'nb', trControl=trainControl(method='cv', number=10))
teclasses<-predict(model,newdata=bigx[-wtd,])
confusionMatrix(data=teclasses, bigy[-wtd])
