#------------------------------------------------------------------------------
#
# Healthcare Analytics
# Project Title:  Predicting Survival of Heart Failure
# By:  Vivian Yuen-Lee
#
#------------------------------------------------------------------------------

# Load the required libraries
library(ggplot2)      # for data visualization
library(gridExtra)    # for combing multiple plots
library(caret)        # for generating confusion matrices
library(rpart)        # for decision trees
library(rpart.plot)   # for plotting the decision tree
library(class)        # for knn models
library(e1071)        # for Naive Bayes function

# Open source file. Obtain the heart failure dataset from the website and load it to R
hfdata <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")

#---------------------- Viewing and describing the data ----------------------#
# Examine the first 10 rows of data
head(hfdata, 10)
# Open the dataset with the Viewer
View(hfdata)
# Examine the structure of the dataset
str(hfdata)
# There are 299 observations of 13 variables
# The dependent/target variable is "DEATH_EVENT"

# Display the 5-point summary statistics for each variable
summary(hfdata)

#----------------------------- Data Exploration -----------------------------#
# Check for missing data in the dataset
sum(is.na(hfdata))
# Check for missing data in a specific column example
sum(is.na(heart_data$age)) 

# Display a frequency table for each of the categorical (or factor) variables:
# anaemia, diabetes, high_blood_pressure, sex, smoking, and DEATH_EVENT
# Check for invalid entries
table(hfdata$anaemia)             # frequency table for anaemia
table(hfdata$diabetes)            # frequency table for diabetes
table(hfdata$high_blood_pressure) # frequency table for high_blood_pressure
table(hfdata$sex)                 # frequency table for sex
table(hfdata$smoking)             # frequency table for smoking
table(hfdata$DEATH_EVENT)         # frequency table for DEATH_EVENT

# Generate histograms for each categorical variable 
hist1 <- ggplot(hfdata, aes(x = anaemia)) + geom_bar(fill = "steelblue") + 
  ggtitle("Histogram for anaemia")

hist2 <- ggplot(hfdata, aes(x = diabetes)) + geom_bar(fill = "darkorange") + 
  ggtitle("Histogram for diabetes")

hist3 <- ggplot(hfdata, aes(x = high_blood_pressure)) + 
  geom_bar(fill = "darkred") + ggtitle("Histogram for high_blood_pressure")

hist4 <- ggplot(hfdata, aes(x = sex)) + geom_bar(fill = "cyan") + 
  ggtitle("Histogram for sex") 

hist5 <- ggplot(hfdata, aes(x = smoking)) + geom_bar(fill = "chocolate3") + 
  ggtitle("Histogram for smoking")

hist6 <- ggplot(hfdata, aes(x = DEATH_EVENT)) + geom_bar(fill = "darkorchid3") +
  ggtitle("Histogram for DEATH_EVENT")
# Combined histograms to display on 1 plot
grid.arrange(hist1, hist2, hist3, hist4, hist5, hist6, nrow = 3)

# Generate boxplot for each of the numeric variables: age, 
# creatinine_phosphokinase, ejection_fraction, platelets, serum_creatinine, 
# serum_sodium, and time.
# Check for extreme values.
boxp1 <- ggplot(hfdata, aes(x = age)) + geom_boxplot(fill="aquamarine", 
                                                     outlier.color="red",
                                                     outlier.size=2) + 
  coord_flip() + ggtitle("Boxplot of age")

boxp2 <- ggplot(hfdata, aes(x = creatinine_phosphokinase)) + 
  geom_boxplot(fill="deeppink", outlier.color="red", outlier.size=2) + 
  coord_flip() + ggtitle("Boxplot of creatinine_phosphokinase")

boxp3 <- ggplot(hfdata, aes(x = ejection_fraction)) + 
  geom_boxplot(fill="burlywood", outlier.color="red", outlier.size=2) + 
  coord_flip() + ggtitle("Boxplot of ejection_fraction")

boxp4 <- ggplot(hfdata, aes(x = platelets)) + geom_boxplot(fill="chartreuse2", 
                                                           outlier.color="red",
                                                           outlier.size=2) + 
  coord_flip() + ggtitle("Boxplot of platelets")

boxp5 <- ggplot(hfdata, aes(x = serum_creatinine)) + geom_boxplot(fill="beige",
                                                                  outlier.color="red", outlier.size=2) + 
  coord_flip() + ggtitle("Boxplot of serum_creatinine")

boxp6 <- ggplot(hfdata, aes(x = serum_sodium)) + geom_boxplot(fill="cyan", 
                                                              outlier.color="red", outlier.size=2) + 
  coord_flip() + ggtitle("Boxplot of serum_sodium")

boxp7 <- ggplot(hfdata, aes(x = time)) + geom_boxplot(fill="darkorange2", 
                                                      outlier.color="red", 
                                                      outlier.size=2) + 
  coord_flip() + ggtitle("Boxplot of time")
# Combined box plots to display on 1 page
grid.arrange(boxp1, boxp2, boxp3, boxp4, boxp5, boxp6, boxp7, nrow = 4)

# Display a density plot for each of the numeric variables: age, 
# creatinine_phosphokinase, ejection_fraction, platelets, serum_creatinine, 
# serum_sodium, and time.
# Check the shape of the variable distribution.
denp1 <- ggplot(hfdata, aes(x = age)) + geom_density(color = "darkblue", 
                                                     size = 1.2) +
  ggtitle("Density plot of age")

denp2 <- ggplot(hfdata, aes(x = creatinine_phosphokinase)) + 
  geom_density(color = "darkred", size = 1.2) + 
  ggtitle("Density plot of creatinine_phosphokinase")

denp3 <- ggplot(hfdata, aes(x = ejection_fraction)) + 
  geom_density(color = "darkgreen", size = 1.2) +
  ggtitle("Density plot of ejection_fraction")

denp4 <- ggplot(hfdata, aes(x = platelets)) +
  geom_density(color = "darkmagenta", size = 1.2) +
  ggtitle("Density plot of platelets")

denp5 <- ggplot(hfdata, aes(x = serum_creatinine)) +
  geom_density(color = "chocolate4", size = 1.2) +
  ggtitle("Density plot of serum_creatinine")

denp6 <- ggplot(hfdata, aes(x = serum_sodium)) +
  geom_density(color = "deepskyblue4", size = 1.2) +
  ggtitle("Density plot of serum_sodium")

denp7 <- ggplot(hfdata, aes(x = time)) +
  geom_density(color = "darkolivegreen4", size = 1.2) +
  ggtitle("Density plot of time")
# Combined all density plots to display on 1 page
grid.arrange(denp1, denp2, denp3, denp4, denp5, denp6, denp7, nrow = 4)

#------------------- Model: Data Analysis and Preparation --------------------#
# Count the number of missing values in this dataset. 
sum(is.na(hfdata))            #  For entire dataset
# No missing values, hence imputation not required

# Data normalization is necessary for some algorithms such as KNN
# First, create a custom function that uses min-max normalization formula to 
# perform normalization.
normalize <- function(x){
  return( (x-min(x)) / (max(x)-min(x)) )
}
# Create new normalized dataset. Apply normalization to all variables by using 
# the lapply() function.
hfdataN <- as.data.frame(lapply(hfdata[], normalize))
# Verify all  variable have been normalized - Max should be 1.0000
summary(hfdataN)

# Several factor variables were misidentified as integers. Use the function 
# as.factor() to convert them to factors.
hfdata$anaemia     <- as.factor(hfdata$anaemia)     # converting "anaemia"
hfdata$diabetes    <- as.factor(hfdata$diabetes)    # converting "diabetes"
# converting "high_blood_pressure"
hfdata$high_blood_pressure <- as.factor(hfdata$high_blood_pressure)   
hfdata$sex         <- as.factor(hfdata$sex)         # converting "sex"
hfdata$smoking     <- as.factor(hfdata$smoking)     # converting "smoking"
hfdata$DEATH_EVENT <- as.factor(hfdata$DEATH_EVENT) # converting "DEATH_EVENT"

# Convert normalized variables to factors
hfdataN$anaemia     <- as.factor(hfdataN$anaemia)     # converting "anaemia"
hfdataN$diabetes    <- as.factor(hfdataN$diabetes)    # converting "diabetes"
hfdataN$high_blood_pressure <- as.factor(hfdataN$high_blood_pressure) # converting "high_blood_pressure"   
hfdataN$sex         <- as.factor(hfdataN$sex)         # converting "sex"
hfdataN$smoking     <- as.factor(hfdataN$smoking)     # converting "smoking"
hfdataN$DEATH_EVENT <- as.factor(hfdataN$DEATH_EVENT) # converting "DEATH_EVENT"

# Display the data structure again to confirm the factor variable changes
str(hfdata)
str(hfdataN)

#-------------------------- Model: Data Splitting ---------------------------#
# Use set.seed for reproducibility
set.seed(1234) 
# Create index to randomly split data into the 80% train - 20% test ratio
train_index <- sample(seq_len(nrow(hfdata)), size = 0.8*nrow(hfdata))

# Next, use the index to create a complete training set and use the remaining
# data to create a complete test set
hftrain <- hfdata[train_index, ]  # complete training set
hftest  <- hfdata[-train_index, ] # complete test set

# Use the same index to create a training & test set of normalized predictors
# training set predictors (normalized)
hftrain_predictors <- hfdataN[train_index, -13]
# Test set predictors (normalized)
hftest_predictors  <- hfdataN[-train_index, -13] 
# Finally, use the index to create training & test set labels
hftrain_labels <- hfdataN[train_index, 13]  # training set class variable
hftest_labels  <- hfdataN[-train_index, 13] # test set class variable

#--------------------------- Model: Decision Tree ----------------------------#
# The first model to be generated is a decision tree.  Set the initial seed 
# variable.
set.seed(200)
# Fitting a tree model to the data
hftree <- rpart(DEATH_EVENT ~ ., 
                data = hftrain,
                cp = 0.0)
# Take a look at the model
hftree
# Plot the decision tree
rpart.plot(hftree,
           type = 4,
           extra = 1,
           clip.right.labs = FALSE,
           main = "Decision Tree for Predicting Survival of Heart Failure")

# Use the decision tree model to predict classification outcomes in the test set
hftree_pred <- predict(hftree, hftest, type = "class")
# Use a confusion matrix to evaluate the performance of the tree
hftree_CM <- confusionMatrix(hftree_pred, hftest$DEATH_EVENT, positive = "1")
# View the confusion matrix
hftree_CM
# Print complexity parameter table for the decision tree
printcp(hftree)
# Use the tree model to find the optimal complexity parameter
set.seed(100)
bestcp <- hftree$cptable[which.min(hftree$cptable[,"xerror"]),"CP"]
bestcp   # display the optimized cp

# Prune the tree using the optimal complexity parameter
hftreeP <- prune(hftree, cp=bestcp)
# Plot the pruned decision tree
rpart.plot(hftreeP,
           type = 4,
           extra = 1,
           clip.right.labs = FALSE,
           main = "Pruned Decision Tree for Predicting Survival of Heart Failure")

# Use the pruned tree model to predict classification outcomes in the test set
hftreeP_pred <- predict(hftreeP, hftest, type = "class")
# Use a confusion matrix to evaluated the performance of the pruned tree
hftreeP_CM <- confusionMatrix(hftreeP_pred, hftest$DEATH_EVENT, positive = "1")
# View the pruned tree confusion matrix
hftreeP_CM
# Collect all performance metrics into a summary
hftree_metrics <- data.frame(Accuracy = hftree_CM$overall["Accuracy"],
                             Sensitivity = hftree_CM$byClass["Sensitivity"],
                             Specificity = hftree_CM$byClass["Specificity"],
                             row.names = "Decision Tree")
hftreeP_metrics <- data.frame(Accuracy = hftreeP_CM$overall["Accuracy"],
                              Sensitivity = hftreeP_CM$byClass["Sensitivity"],
                              Specificity = hftreeP_CM$byClass["Specificity"],
                              row.names = "Pruned Decision Tree")
hftree_metrics2 <- rbind(hftree_metrics, hftreeP_metrics)
hftree_metrics2

#------------------------ Model: K-Nearest Neighbors -------------------------#
# The second model is KNN. Since KNN uses euclidean distance to find the k nearest
# points, normalization is recommended when the features in the dataset have 
# varying scales.
# First, fit a tree model to the data
hfknn <- knn(train = hftrain_predictors,   # use normalized predictors 
             test = hftest_predictors, 
             cl = hftrain_labels,
             k = 15)   # the square root of # of training examples
# The number of training examples is 239.  Set k as the square root of 239, and
# round down to 15. Use a confusion matrix to evaluate the performance of the 
# KNN model.
hfknn_CM <- confusionMatrix(as.factor(hfknn), as.factor(hftest_labels), 
                            positive = "1")
hfknn_CM
# To improve model performance, an optimized value of K will be obtained through
# the train() function in the caret package with the method parameter set to
# "knn".  A 10-fold cross-validation will be performed.
hfknnT <- train(x = hftrain_predictors, y = hftrain_labels, 
                method = "knn",
                preProc = c("center", "scale"),
                trControl = trainControl(method="cv", number=10))
hfknnT   # view the model details

# Use the new KNN model to make predictions
hfknnT_pred <- predict(hfknnT, hftest_predictors, type = "raw")

# Use a confusion matrix to evaluate the performance of the tuned KNN model
hfknnT_CM <- confusionMatrix(hfknnT_pred, hftest_labels, positive = "1")
hfknnT_CM

# The optimized k value shows improvement over the original model. Therefore, 
# the improved model will be used. Collect all performance metrics into a summary.
hfknn_metrics <- data.frame(Accuracy = hfknnT_CM$overall["Accuracy"],
                            Sensitivity = hfknnT_CM$byClass["Sensitivity"],
                            Specificity = hfknnT_CM$byClass["Specificity"],
                            row.names = "K-Nearest Neighbors")
hfknn_metrics

#--------------------------- Model: Naive Bayes ----------------------------#
# The third model is a Naive Bayes. For this model, the Laplace estimator is 
# initially set to 0.
hfnb <- naiveBayes(DEATH_EVENT ~ ., 
                   data = hftrain,
                   laplace = 0)  
# Use the Naive Bayes model to predict classification outcomes in the test set 
hfnb_pred <- predict(hfnb, hftest, type="class")
# Use a confusion matrix to evaluated the performance of the Naive Bayes model
hfnb_CM <- confusionMatrix(hfnb_pred, hftest$DEATH_EVENT, positive = "1")
hfnb_CM

# In an attempt to improve model performance, a different Laplace estimator 
# will be used.
hfnbL <- naiveBayes(DEATH_EVENT ~ ., 
                    data = hftrain,
                    laplace = 1)   # set Laplace estimator to 1
# Make predictions on the test set 
hfnbL_pred <- predict(hfnbL, hftest, type="class")
# Assess model performance using a confusion matrix
hfnbL_CM <- confusionMatrix(hfnbL_pred, hftest$DEATH_EVENT, positive = "1")
hfnbL_CM
# There is no apparent improvement in the model's performance metrics.

# Collect all performance metrics into a summary
hfnb_metrics <- data.frame(Accuracy = hfnb_CM$overall["Accuracy"],
                           Sensitivity = hfnb_CM$byClass["Sensitivity"],
                           Specificity = hfnb_CM$byClass["Specificity"],
                           row.names = "Naive Bayes")
hfnb_metrics

#----------------------- Models Performance Comparison -----------------------#
# Consolidate the performance metrics into a metrics grid
perfcomp <- rbind(hftree_metrics2, hfknn_metrics, hfnb_metrics)
# Display this summary table. Models are listed from highest to lowest accuracy 
perfcomp[order(perfcomp$Accuracy, decreasing = TRUE), ]

# Generate a consolidated metrics plot
knn_val <- c(hfknnT_CM$overall['Accuracy'] * 100, hfknnT_CM$byClass['Specificity']*100 , hfknnT_CM$byClass['Sensitivity']*100)

DT_val <- c(hftree_CM$overall['Accuracy'] * 100, hftree_CM$byClass['Specificity']*100 , hftree_CM$byClass['Sensitivity']*100)

DT_pruned_val <-c(hftreeP_CM$overall['Accuracy'] * 100, hftreeP_CM$byClass['Specificity']*100 , hftreeP_CM$byClass['Sensitivity']*100)

nb_val <-c(hfnb_CM$overall['Accuracy'] * 100, hfnb_CM$byClass['Specificity']*100 , hfnb_CM$byClass['Sensitivity']*100)
plot_data <- as.matrix(data.frame(DT_pruned=DT_pruned_val, DT=DT_val, Knn=knn_val, NB=nb_val))
rownames(plot_data) <- c("Accuracy", "Specificity", "Sensitivity")
# Display the comparison plot
bp<-barplot(plot_data, col=c("#417dc1", "#0bb5ff","#8ee5ee" ), beside=TRUE, ylim=c(0,150),xlab="Models",ylab="Values in %", main="ML Models Performance")
legend("topright",                                    # Add legend to barplot
       legend = c("Accuracy", "Specificity", "Sensitivity"),
       fill = c("#417dc1", "#0bb5ff","#8ee5ee" ), cex = 0.7)

text(x=bp, y=plot_data, labels=round(plot_data,1), pos=3, xpd=NA, cex=0.7)

