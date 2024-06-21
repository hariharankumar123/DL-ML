
# Install packages

install.packages("ggcorrplot")
install.packages("lattice")
install.packages("glmnet")
install.packages("mltools")
install.packages("data.table")
install.packages("keras")
install.packages("tfruns")
install.packages("tidyr")
install.packages("GGally")
install.packages("purrr")
install.packages("knitr")
install.packages("scales")
install.packages("psych")
install.packages("dplyr")
install.packages("caret")
install.packages("ggplot2")
install.packages("corrplot")

#### LOAD LIBRARIES


library(ggcorrplot)
library(lattice)
library(glmnet)
library(mltools)
library(data.table)
library(keras)
library(tfruns)
library(tidyr)
library(GGally)
library(purrr)
library(knitr)
library(scales)
library(psych)
library(dplyr)
library(caret)
library(ggplot2)
library(corrplot)


############## LOADING THE DATASET

data <- read.csv("C:/Users/ASUS/Downloads/bike_sharing_dataset1/hour.csv")
head(data)


cat("Number of rows", dim(data)[1],"\n")

cat("Number of columns", dim(data)[2])


###### pRINT THE SUMMARY
str(data)
summary(data)


############### CHECK FOR MISSING OR NULL VALUES
colSums(is.na(data))


##########################################################################################
##########################################################################################
################# CREATING A VARIABLE ONLY WITH NUMERICAL VALUES #########################
##########################################################################################
##########################################################################################


numerical_vars <- c("temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt")


############ CHECK FOR LINEAR OR NON LINEAR RELATIONSHIP

plots <- lapply(numerical_vars, function(x) {
  ggplot(data, aes(x = !!sym(x), y = cnt)) + 
    geom_point(color = "steelblue") +
    labs(x = x, y = "Count")})

# Print all plots
for (plot in plots) {
  print(plot)
} 

# Correlation matrix
corr_matrix <- cor(data[,numerical_vars])
corrplot(corr_matrix, type = "upper", method = "circle", tl.col = "black", tl.srt = 45)


# Correlation matrix
cor_matrix <- cor(data[,numerical_vars])
corrplot(cor_matrix, type = "upper", method = "circle", tl.col = "black", tl.srt = 45)


cor(data[,numerical_vars])


ggplot(data, aes(x = as.Date(dteday), y = cnt)) +
  geom_line(color = "steelblue") +
  labs(x = "Date", y = "Count") +
  scale_x_date(date_labels = "%b %d", date_breaks = "1 week") +
  theme_bw()

##########################################################################################
##########################################################################################
########## SELECT CATEGORICAL VARIABLES AND TARGET VARIABLE AND CREATE A NEW DF###########
##########################################################################################
##########################################################################################

categorical_vars <- c("season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit")
df_cat <- data[, c(categorical_vars, "cnt")]


##########################################################################################
##########################################################################################
############## CONVERT CATEGORIC VARIABLE TO FACTORS  ####################################
##########################################################################################
##########################################################################################


df_cat[, categorical_vars] <- lapply(df_cat[, categorical_vars], factor)


##########################################################################################
##########################################################################################
########### CREATION OF PLOTS FOR EACH CATEGORIC VARIABLE ################################
##########################################################################################
##########################################################################################

plots <- lapply(categorical_vars, function(x) {
  ggplot(df_cat, aes(x = !!sym(x), y = cnt)) + 
    geom_boxplot(color = "steelblue") +
    labs(x = x, y = "Count")
})

# Print all plots
for (plot in plots) {
  print(plot)
}



ggplot(data, aes(x = cnt)) +
  geom_histogram(color = "white", fill = "steelblue", bins = 30) +
  labs(x = "Count", y = "Frequency", 
       title = "Distribution of Count Variable") +
  theme(plot.title = element_text(hjust = 0.5))



continuous_variables <- names(select_if(data, is.numeric))

# run a for loop through continuous variables and perform t-tests
for (var in continuous_variables) {
  print(paste0("T-test for association between Cnt and ", var, ":"))
  print(t.test(data[[var]], data$SalePrice))
  print("-----------------------------------------------------------------------")
}


###################################################################################
###################################################################################
################## DATA PRE-PROCESSING ############################################
###################################################################################
###################################################################################


# Convert season to x and y coordinates
data$x_season <- cos(2 * pi * (as.numeric(data$season) - 1) / max(data$season))
data$y_season <- sin(2 * pi * (as.numeric(data$season) - 1) / max(data$season))

# Convert month to x and y coordinates
data$x_month <- cos(2 * pi * (as.numeric(data$mnth) - 1) / max(data$mnth))
data$y_month <- sin(2 * pi * (as.numeric(data$mnth) - 1) / max(data$mnth))

# Convert day of week to x and y coordinates
data$x_day_of_week <- cos(2 * pi * (as.numeric(data$weekday) - 1) / max(data$weekday))
data$y_day_of_week <- sin(2 * pi * (as.numeric(data$weekday) - 1) / max(data$weekday))

# Convert hour to x and y coordinates
data$x_hour <- cos(2 * pi * data$hr / max(data$hr))
data$y_hour <- sin(2 * pi * data$hr / max(data$hr))


# Remove original circular variables
data <- subset(data, select = -c(season, mnth, hr, weekday))


data <- subset(data, select = -c(dteday))

str(data)


#### square root transformation.
data$cnt <- sqrt(data$cnt)


#### Plot of Dep var after sqr transformation
ggplot(data, aes(x = cnt)) +
  geom_histogram(color = "white", fill = "steelblue", bins = 30) +
  labs(x = "Count", y = "Frequency", 
       title = "Distribution of Count Variable after square-root transformation")



numerical_vars <- c("temp", "atemp", "hum", "windspeed", "casual", "registered")
preproc <- preProcess(data[, numerical_vars], methods= c("center", "scale"))
data[, numerical_vars] <- predict(preproc, data[, numerical_vars])
data


##### PRINT THE DATASET (5 ROWS)

head(data)


# Split the data into train and Test 

train_idx <- createDataPartition(data$cnt, p = 0.8, list = FALSE)
train_data <- data[train_idx, ]
test_data <- data[-train_idx, ]

##################################################################################
##################################################################################
################# MODEL  ######################################################### 
##################################################################################
##################################################################################



##################################################################################
############ RANDOM FOREST #######################################################
##################################################################################

set.seed(1)
# set number =10 ifor fold cross validation.

train_control <- trainControl(method = "cv", number = 10)


library(randomForest)

rf_model <- caret::train(cnt ~ ., 
                         data = train_data, 
                         method = "rf", 
                         trControl = train_control, 
                         preProcess = c("knnImpute", "nzv"), 
                         importance = TRUE)

rf_model



######## Summary of the Model
summary(rf_model)


rf_cnt_predictions <- predict(rf_model, newdata = test_data)

######### RMSE value rf_rmse <- RMSE(rf_cnt_predictions^2, test_data$cnt^2)
cat(paste("The RMSE Value of Random Forest model is: ", rf_rmse))

varImp(rf_model, scale = FALSE)


##################################################################################
##################################################################################
################ GRADIENT BOOSTING ###############################################
##################################################################################
##################################################################################

set.seed(1)
gradientBooting_model <- caret::train(
  cnt ~ .,
  data = train_data,
  method = "gbm",
  tuneLength = 5,
  trControl = train_control,
  na.action = na.pass,
  preProcess = c("knnImpute", "nzv")
  
)

gradientBooting_model

gradientBoosting_cnt_predictions <- predict(gradientBooting_model, newdata = test_data)

######### RMSE value 
gradientBoosting_rmse <- RMSE(gradientBoosting_cnt_predictions^2, test_data$cnt^2)
cat(paste("The RMSE Value of Gradient Boosted model is: ", gradientBoosting_rmse))

##################################################################################
##################################################################################
########## SUPPORT VECTOR MACHINE ################################################
##################################################################################
##################################################################################

svm_linear_model <- caret::train(
  cnt ~ .,
  data = train_data,
  na.action = na.pass,
  preProcess = c("knnImpute", "nzv"), 
  method = "svmLinear",
  trControl =  train_control,
  tuneLength = 5
  
)

#Print the summary of Linear SVM model
svm_linear_model

svm_linear_cnt_predictions <- predict(svm_linear_model, newdata = test_data)

# Find the RMSE value on orignal scaled target and predicted variables.
svm_linear_rmse <- RMSE(svm_linear_cnt_predictions^2, test_data$cnt^2)
cat(paste("The RMSE Value of LinearSVM model is: ", svm_linear_rmse))



###################################################################################
###################################################################################
########### CONCLUSION ############################################################
###################################################################################
###################################################################################

#install.packages("kableExtra")
library(kableExtra)
# Create data frame
model_performance <- data.frame(
  Model = c("Random Forest", "Gradient Boosted", "SVM"),
  RMSE = c(rf_rmse, gradientBoosting_rmse, svm_linear_rmse))

# Apply kable function and add Bootstrap styling
kable(model_performance, format = "html", row.names = FALSE, col.names = c("Model", "RMSE")) %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"))








# # Define color gradient function
# color_gradient <- scales::col_numeric(
#   palette = c("#FF0000", "#FFFF00", "#00FF00"),  # Define colors for gradient
#   domain = range(model_performance$RMSE), 
#   alpha = 1
# ) 
# 
# 
# # Apply kable function and add Bootstrap styling with color gradients
# kable(model_performance, format = "html", row.names = FALSE, col.names = c("Model", "RMSE")) %>%
#   kable_styling(bootstrap_options = c("striped", "hover", "condensed")) %>%
#   column_spec(2, color = color_gradient(model_performance$RMSE))
