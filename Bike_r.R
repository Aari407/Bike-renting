rm(list=ls(all=T))
setwd("D:/Project")#set working directory
x=c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "dummies", "Information", "MASS", "rpart", "gbm", "ROSE", "xlsx", "DataCombine", "xlsx")
lapply(x, require, character.only=TRUE)
df_train = read.csv('day.csv', header=TRUE, row.names="dteday")

-------------------------------------------------------------------------------------
#Exploratory Data Analysis
# Shape of the data
dim(df_train)
# Structure of the data
str(df_train)
# Lets see the colum names of the data
colnames(df_train)
#splitting columns into "continuous" and "catagorical"
num_list = c('temp', 'atemp', 'hum', 'windspeed', 'casual', 'cnt', 'registered')

cat_list = c('season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit')
df_train = subset(df_train, select = -c(instant))

---------------------------------------------------------------------------------------
#Missing Values Analysis
#Create dataframe with missing percentage
missing_val = data.frame(apply(df_train,2,function(x){sum(is.na(x))}))
sum(is.na(df_train))

----------------------------------------------------------------------------------
#Outlier Analysis
#Boxplot for continuous variables
for (i in 1:length(num_list))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (num_list[i]), x = "cnt"), data = subset(df_train))+ 
  stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
  outlier.size=1, notch=FALSE) + theme(legend.position="bottom")+
             labs(y=num_list[i],x="cnt")+ ggtitle(paste("Box plot of cnt",num_list[i])))
}

#Plotting plots together
gridExtra::grid.arrange(gn1,gn2, gn3, ncol=3)
gridExtra::grid.arrange(gn4,gn5,gn6, ncol=3)

#loop to remove outliers from all variables
for(i in num_list)
{
  print(i)
  #Extract outliers
  val = df_train[,i][df_train[,i] %in% boxplot.stats(df_train[,i])$out]
  #Remove outliers
  df_train = df_train[which(!df_train[,i] %in% val),]
}

-------------------------------------------------------------------------------------
#Feature Selection
# Correlation Plot 
corrgram(df_train[,num_list], order = F,
        upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

## Dimension Reduction
df_train = subset(df_train, select = -c(casual, temp, registered))
dim(df_train)
-----------------------------------------------------------------------------------
#Feature Scaling
#Updating the continuous and catagorical variable
num_list = c('atemp', 'hum', 'windspeed')

cat_list = c('mnth', 'yr', 'holiday', 'weekday', 'workingday','weathersit', 'season')

# Normalization

df_train[,'cnt'] = (df_train[,'cnt'] - min(df_train[,'cnt']))/(max(df_train[,'cnt'])-min(df_train[,'cnt']))

# Creating dummy variables for categorical variables
df_train = dummy.data.frame(df_train, cat_list)

----------------------------------------------------------------------------------
#Model Development
#Cleaning the environment
rmExcept("df_train")

#Divide data into train and test using stratified sampling method
set.seed(125)
train.index = sample(1:nrow(df_train), 0.8 * nrow(df_train))
X_train = df_train[ train.index,]
X_test  = df_train[-train.index,]

rmse <- function (actual, predicted) 
{
  return(sqrt(mean((actual-predicted)^2)))
}

---------------------------------------------------------------------------------
#Decision tree 
#model_dt1 = rpart(cnt ~., data = X_train, method = "anova")
#write rules into disk
#write(capture.output(summary(model_dt1)), "Rules.txt")
#predict_dt1 = predict(model_dt1, X_test[, names(X_test) != "cnt"])
# For test data 
#print(postResample(pred = predict_dt1, obs = X_test$cnt))
#print(mean(abs((X_test$cnt - predict_dt1)/X_test$cnt))*100)
#RMSE: 0.10778694
#Rsquared: 0.79653839

------------------------------------------------------------------------------------
#Linear Regression
#set.seed(123)
#Develop Model on training data
#model_lr1 = lm(cnt ~ ., data = X_train)
#Lets predict for test data
#predict_lr1 = predict(model_lr1, X_test[, names(X_test) != "cnt"])
# For test data 
#print(postResample(pred = predict_lr1, obs = X_test$cnt))
#print(mean(abs((X_test$cnt - predict_lr1)/X_test$cnt))*100)
#RMSE: 0.09353241
#Rsquared: 0.85164203

---------------------------------------------------------------------------------
#Random Forest
#set.seed(123)
#Develop Model on training data
#model_RF1 = randomForest(cnt~., data = X_train)
#Lets predict for test data
#predict_rf1 = predict(model_RF1,X_test[,names(X_test) != "cnt"])
#print(postResample(pred = predict_rf1, obs = X_test$cnt))
#print(mean(abs((X_test$cnt - predict_rf1)/X_test$cnt))*100)
#RMSE: 0.07298511
#Rsquared: 0.91083873

----------------------------------------------------------------------------------
#principal component analysis
prin_comp = prcomp(X_train)

#compute standard deviation of each principal component
std_dev = prin_comp$sdev

#compute variance
pr_var = std_dev^2

#proportion of variance explained
prop_varex = pr_var/sum(pr_var)

#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

#add a training set with principal components
train.data = data.frame(cnt = X_train$cnt, prin_comp$x)

# From the above plot selecting 20 components since it explains almost 95+ % data variance
train.data =train.data[,1:20]

#transform test into PCA
test.data = predict(prin_comp, newdata = X_test)
test.data = as.data.frame(test.data)

#select the first 20 components
test.data=test.data[,1:20]

--------------------------------------------------------------------------------
#Descision Tree
#Hyperparameter Tuning
set.seed(125) 
# Establish a list of possible values for minsplit and maxdepth
minsplit <- seq(1, 4, 1)
maxdepth <- seq(2, 30, 4)
# Create a data frame containing all combinations 
hyper_grid <- expand.grid(minsplit = minsplit, maxdepth = maxdepth)
head(hyper_grid)
nrow(hyper_grid)
num_models <- nrow(hyper_grid)
grade_models <- list()
# Write a loop over the rows of hyper_grid_rf to train the grid of models
for (i in 1:num_models) {
  
  minsplit <- hyper_grid$minsplit[i]
  maxdepth <- hyper_grid$maxdepth[i]
  
  grade_models[[i]] <- rpart(formula = cnt ~ ., 
                             data = train.data, 
                             method = "anova",
                             minsplit = minsplit,
                             maxdepth = maxdepth)
}
# Number of potential models in the grid
num_models <- length(grade_models)
# Create an empty vector to store RMSE values
rmse_values <- c()
# Write a loop over the models to compute validation RMSE
for (i in 1:num_models) {
  model <- grade_models[[i]]
  pred <- predict(object = model,
                  newdata = test.data)
  rmse_values[i] <- rmse(actual = X_test$cnt, 
                         predicted = pred)
}
# Identify the model with smallest validation set RMSE
best_model <- grade_models[[which.min(rmse_values)]]
best_model$control
# Compute test set RMSE on best_model
pred <- predict(object = best_model,newdata = test.data)
rmse(actual = X_test$cnt, predicted = pred)
print(postResample(pred = pred, obs = X_test$cnt))
print(mean(abs((X_test$cnt - pred)/X_test$cnt))*100) 

#RMSE: 0.12764151
#Rsquared: 0.69098383
#MAPE: 22.84871

----------------------------------------------------------------------------------
#Linear Regression
set.seed(125)
#Develop Model on training data
model_lr1 = lm(cnt ~ ., data = train.data)
#Lets predict for test data
predict_lr1 = predict(model_lr1, test.data)
# For test data 
print(postResample(pred = predict_lr1, obs = X_test$cnt))
print(mean(abs((X_test$cnt - predict_lr1)/X_test$cnt))*100)
#RMSE: 0.10402858
#Rsquared: 0.79094500
#MAPE: 20.06494

------------------------------------------------------------------------------------
#Random Forest
set.seed(125)
# Establish a list of possible values for mtry, nodesize, ntree and sampsize
ntree <- seq(500, 1000, 100)
mtry <- seq(4, ncol(train.data) * 0.8, 2)
nodesize <- seq(3, 8, 2)
sampsize <- nrow(train.data) * c(0.7, 0.8)
# Create a data frame containing all combinations 
hyper_grid_rf <- expand.grid(ntree=ntree, mtry=mtry, nodesize=nodesize, sampsize=sampsize)
head(hyper_grid_rf)
nrow(hyper_grid_rf)
num_models_rf <- nrow(hyper_grid_rf)
grade_models_rf <- list()

# Write a loop over the rows of hyper_grid_rf to train the grid of models
for (i in 1:num_models_rf) {
  
  ntree <- hyper_grid_rf$ntree[i]
  mtry <- hyper_grid_rf$mtry[i]
  nodesize <- hyper_grid_rf$nodesize[i]
  sampsize <- hyper_grid_rf$sampsize[i]
  # Train a Random Forest model 
  grade_models_rf[[i]] <- randomForest(formula = cnt ~ ., 
                                    data = train.data, 
                                    method = "anova",
                                    ntree= ntree, mtry=mtry, nodesize=nodesize, sampsize=sampsize)
}

# Number of potential models in the grid
num_models_rf <- length(grade_models_rf)
# Create an empty vector to store RMSE values
rmse_values_rf <- c()
# Write a loop over the models to compute validation RMSE
for (i in 1:num_models_rf) {
  model <- grade_models_rf[[i]]
  # Generate predictions on grade_valid
  pred_rf <- predict(object = model,
                  newdata = test.data)
  rmse_values_rf[i] <- rmse(actual = X_test$cnt, 
                         predicted = pred_rf)
}

# Identify the model with smallest validation set RMSE
best_model_rf <- grade_models_rf[[which.min(rmse_values_rf)]]
best_model_rf$control
# Compute test set RMSE on best_model
pred_rf <- predict(object = best_model_rf,newdata = test.data)
rmse(actual = X_test$cnt, predicted = pred_rf)
print(postResample(pred = pred_rf, obs = X_test$cnt))
print(mean(abs((X_test$cnt - pred_rf)/X_test$cnt))*100) 
#RMSE: 0.08795391
#Rsquared: 0.85053461
#MAPE: 15.04173


