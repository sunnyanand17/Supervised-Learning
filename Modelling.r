#Load the training data in the table
train_data = read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt", header = FALSE, sep = "\t")
# Check if the data is loaded correctly
head(train_data)
summary(train_data)
str(train_data)
train_data = na.omit(train_data)
summary(train_data)
str(train_data)
#All the data is numerical and the categoricals data is not necessarily factored,
#in order to make sure the processing is fast.
test_data = read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticeval2000.txt", header = FALSE, sep = "\t")
# Check if the data is loaded correctly
head(test_data)
summary(test_data)
str(test_data)
test_data = na.omit(test_data)
summary(test_data)
str(test_data)
test_data_res = read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/tictgts2000.txt", header = FALSE)
head(test_data_res)
summary(test_data_res)
str(test_data_res)
#Feature selection techniques
set.seed(7)
# load the library
library(mlbench)
library(caret)
# calculate correlation matrix
correlationMatrix <- cor(train_data[,1:85])
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.90)
# print indexes of highly correlated attributes
print(highlyCorrelated)
library(lattice)
levelplot(correlationMatrix)
install.packages("VIF",dependencies = TRUE)
library(VIF)
res <- lm(V86~., data=train_data)
summary(res)
# checking multicolinearity for independent variables.
vif(lm(V86~., data=train_data))
set.seed(85)
# load the library
library(mlbench)
library(caret)
# load the data
data(train_data)
# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=2)
# run the RFE algorithm
results <- rfe(train_data[,44:85], train_data[,86], sizes=c(1:17), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))
#Each record consists of 86 attributes,
# containing sociodemographic data (attribute 1-43) and product ownership 
#(attributes 44-86).The sociodemographic data is derived from zip codes. 
#All customers living in areas with the same zip code have the same sociodemographic 
#attributes. Attribute 86, "CARAVAN:Number of mobile home policies", is the target
# variable. 
##The set of collected variables using the collinarity and feature selection technique
#eliminates all the varibales based on the sociodemographic attributes(1-43).
#The most important features which are selected are the ones below:
#PBRAND,  MOSHOOFD,  MOSTYPE,PPERSAUT and APERSAUT
#Let select all the variables
#V59+V47+V1+V16+V82+V78+V79

#Let us try to build a linear model and see if this works in predicting the number of customers who would buy
#insurance
train_data$V86 = as.factor(train_data$V86)
train_data$V59 = as.factor(train_data$V59)
train_data$V47 = as.factor(train_data$V47)
train_data$V1 = as.factor(train_data$V1)
train_data$V16 = as.factor(train_data$V16)
train_data$V82 = as.factor(train_data$V82)
train_data$V78 = as.factor(train_data$V78)
train_data$V79 = as.factor(train_data$V79)
summary(train_data)
fit <- glm(V86~V59+V47+V82+V68+V65, data=train_data)
summary(fit)
fitted(fit)
plot(fit)

test_data1 =cbind(test_data,test_data_res$V1)
summary(test_data1)
predict(fit,test_data)
table(predict(fit, test_data),test_data_res$V1)
fit1 <- lm(V86~V59+V47+V82+V78+V79, data=train_data)
summary(fit1)
fitted(fit1)
plot(fit1)
fit2 <- lm(V86~V59+V47+V82, data=train_data)
summary(fit2)
print(fitted(fit2)>.1)
plot(fit2)
fit3 <- lm(V86~V59+V47, data=train_data)
summary(fit3)
print(fitted(fit3))
plot(fit3)

#Let us try building a model on these set of categorical variables using a naive bayes
#approach
library("e1071")
dim(train_data)
train_data$V86 = as.factor(train_data$V86)
train_data$V59 = as.factor(train_data$V59)
train_data$V47 = as.factor(train_data$V47)
train_data$V1 = as.factor(train_data$V1)
train_data$V16 = as.factor(train_data$V16)
train_data$V82 = as.factor(train_data$V82)
train_data$V78 = as.factor(train_data$V78)
train_data$V79 = as.factor(train_data$V79)
train_data$V68 = as.factor(train_data$V68)
summary(test_data)
train_data1 = cbind(train_data,(train_data$V47 * train_data$V68))
model <- naiveBayes(V86 ~ V59 +V47 + V82, data = train_data)
table(predict(model, test_data),test_data_res$V1)

