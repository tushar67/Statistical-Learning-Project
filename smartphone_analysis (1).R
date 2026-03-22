# Smartphone usage and productivity - individual project
# Research question: can we identify different types of smartphone users
# and does usage behavior relate to productivity and stress?

library(tidyverse)
library(ggplot2)
library(corrplot)
library(factoextra)
library(cluster)
library(randomForest)
library(caret)
library(scales)
library(gridExtra)
library(ggcorrplot)
library(viridis)

set.seed(42)

data <- read.csv("/Users/tusharsinha/Downloads/Smartphone_Usage_Productivity_Dataset_50000.csv",
                 stringsAsFactors = FALSE)

# quick check
dim(data)
sum(is.na(data))   # no missing values
str(data)

# 1. EDA

# convert categorical columns to factors
data$Gender <- factor(data$Gender)
data$Occupation <- factor(data$Occupation)
data$Device_Type <- factor(data$Device_Type)

# basic summary ( to skip User_ID column)
summary(data[, -1])


#distributions of main outcome variables 

p1 <- ggplot(data, aes(x = Work_Productivity_Score)) +
  geom_histogram(binwidth = 1, fill = "#5B8DB8", color = "white") +
  labs(title = "Work Productivity Score", x = "Score (1-10)", y = "Count") +
  theme_minimal()

p2 <- ggplot(data, aes(x = Stress_Level)) +
  geom_histogram(binwidth = 1, fill = "#C1714F", color = "white") +
  labs(title = "Stress Level", x = "Level (1-10)", y = "Count") +
  theme_minimal()

grid.arrange(p1, p2, ncol = 2)


#categorical variables

p3 <- ggplot(data, aes(x = Gender, fill = Gender)) +
  geom_bar(show.legend = FALSE) +
  scale_fill_viridis_d() +
  labs(title = "Gender", y = "n") +
  theme_minimal()

p4 <- ggplot(data, aes(x = Occupation, fill = Occupation)) +
  geom_bar(show.legend = FALSE) +
  scale_fill_viridis_d() +
  labs(title = "Occupation", y = "n") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

p5 <- ggplot(data, aes(x = Device_Type, fill = Device_Type)) +
  geom_bar(show.legend = FALSE) +
  scale_fill_viridis_d() +
  labs(title = "Device Type", y = "n") +
  theme_minimal()

grid.arrange(p3, p4, p5, ncol = 3)


# does productivity vary across groups? 
# boxplots 

p6 <- ggplot(data, aes(x = Occupation, y = Work_Productivity_Score, fill = Occupation)) +
  geom_boxplot(show.legend = FALSE) +
  scale_fill_viridis_d() +
  labs(title = "Productivity by Occupation", y = "Productivity") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

p7 <- ggplot(data, aes(x = Gender, y = Work_Productivity_Score, fill = Gender)) +
  geom_boxplot(show.legend = FALSE) +
  scale_fill_viridis_d() +
  labs(title = "Productivity by Gender", y = "Productivity") +
  theme_minimal()

p8 <- ggplot(data, aes(x = Device_Type, y = Work_Productivity_Score, fill = Device_Type)) +
  geom_boxplot(show.legend = FALSE) +
  scale_fill_viridis_d() +
  labs(title = "Productivity by Device", y = "Productivity") +
  theme_minimal()

grid.arrange(p6, p7, p8, ncol = 3)

# correlation between numerical variables 

num_cols <- data %>%
  select(Age, Daily_Phone_Hours, Social_Media_Hours,
         Work_Productivity_Score, Sleep_Hours, Stress_Level,
         App_Usage_Count, Caffeine_Intake_Cups, Weekend_Screen_Time_Hours)

corr_mat <- cor(num_cols)

ggcorrplot(corr_mat,
           method = "circle",
           type = "lower",
           lab = TRUE,
           lab_size = 3,
           colors = c("#C1714F", "white", "#5B8DB8"),
           title = "Correlation matrix - numerical variables",
           ggtheme = theme_minimal())


# scatterplots for variables that i found interesting
p9 <- ggplot(data, aes(x = Daily_Phone_Hours, y = Work_Productivity_Score,
                       color = Occupation)) +
  geom_point(alpha = 0.15, size = 0.7) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.8) +
  scale_color_viridis_d() +
  labs(title = "Phone hours vs productivity (by occupation)") +
  theme_minimal()

p10 <- ggplot(data, aes(x = Sleep_Hours, y = Stress_Level, color = Gender)) +
  geom_point(alpha = 0.15, size = 0.7) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.8) +
  scale_color_viridis_d() +
  labs(title = "Sleep vs stress (by gender)") +
  theme_minimal()

grid.arrange(p9, p10, ncol = 2)


# 2. UNSUPERVISED - PCA + kmeans clustering

nums_scaled <- scale(num_cols)

# PCA
pca_out <- prcomp(nums_scaled, center = FALSE, scale. = FALSE)
summary(pca_out)

# scree plot - how many components explain most variance?
fviz_eig(pca_out,
         addlabels = TRUE,
         ylim = c(0, 40),
         barfill = "#5B8DB8",
         barcolor = "white",
         title = "Scree plot - variance explained by each PC")

# biplot - just plot a sample of observations so it's readable
fviz_pca_biplot(pca_out,
                geom.ind = "point",
                col.ind  = "gray65",
                col.var  = "#C1714F",
                repel    = TRUE,
                title    = "PCA biplot",
                ind.ind  = 1:1000)

# variable contributions
fviz_pca_var(pca_out,
             col.var = "contrib",
             gradient.cols = c("#5B8DB8", "white", "#C1714F"),
             repel = TRUE,
             title = "Variable contributions to PC1 and PC2")


# clustering
# I'll use the first 5 PC scores instead of raw variables
# this reduces noise and is standard practice

pc_scores <- pca_out$x[, 1:5]

set.seed(42)
idx_sample <- sample(nrow(pc_scores), 5000)
pc_sub     <- pc_scores[idx_sample, ]

fviz_nbclust(pc_sub, kmeans, method = "wss", k.max = 10) +
  labs(title = "Elbow method") +
  theme_minimal()

fviz_nbclust(pc_sub, kmeans, method = "silhouette", k.max = 10) +
  labs(title = "Silhouette method") +
  theme_minimal()

# choose k based on the plots above - (so, i'll go with 4)
k <- 4

km_out <- kmeans(pc_scores, centers = k, nstart = 25, iter.max = 100)

cat("Cluster sizes:\n")
print(table(km_out$cluster))

# add cluster assignment to original data
data$cluster <- factor(km_out$cluster)

# visualize
fviz_cluster(km_out,
             data = pc_scores,
             geom = "point",
             ellipse.type = "convex",
             palette = "viridis",
             alpha = 0.25,
             ggtheme = theme_minimal(),
             main = paste0("K-means clusters (k = ", k, ")"))

#cluster different

cluster_means <- data %>%
  group_by(cluster) %>%
  summarise(
    n                  = n(),
    avg_age            = mean(Age),
    avg_phone_hours    = mean(Daily_Phone_Hours),
    avg_social_media   = mean(Social_Media_Hours),
    avg_productivity   = mean(Work_Productivity_Score),
    avg_sleep          = mean(Sleep_Hours),
    avg_stress         = mean(Stress_Level),
    avg_apps           = mean(App_Usage_Count),
    avg_caffeine       = mean(Caffeine_Intake_Cups),
    avg_weekend_screen = mean(Weekend_Screen_Time_Hours)
  )

print(cluster_means)

# heatmap to compare clusters visually
cluster_long <- cluster_means %>%
  select(-n) %>%
  pivot_longer(-cluster, names_to = "variable", values_to = "mean_val") %>%
  group_by(variable) %>%
  mutate(z = scale(mean_val)[, 1])   # standardize so scales are comparable

ggplot(cluster_long, aes(x = cluster, y = variable, fill = z)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "#5B8DB8", mid = "white", high = "#C1714F", midpoint = 0) +
  labs(title = "Cluster profiles (z-scored means)",
       x = "Cluster", y = "", fill = "z-score") +
  theme_minimal()


# check if clusters differ in terms of categorical variables
p11 <- ggplot(data, aes(x = cluster, fill = Occupation)) +
  geom_bar(position = "fill") +
  scale_fill_viridis_d() +
  scale_y_continuous(labels = percent) +
  labs(title = "Occupation per cluster", y = "proportion") +
  theme_minimal()

p12 <- ggplot(data, aes(x = cluster, fill = Gender)) +
  geom_bar(position = "fill") +
  scale_fill_viridis_d() +
  scale_y_continuous(labels = percent) +
  labs(title = "Gender per cluster", y = "proportion") +
  theme_minimal()

p13 <- ggplot(data, aes(x = cluster, fill = Device_Type)) +
  geom_bar(position = "fill") +
  scale_fill_viridis_d() +
  scale_y_continuous(labels = percent) +
  labs(title = "Device type per cluster", y = "proportion") +
  theme_minimal()

grid.arrange(p11, p12, p13, ncol = 3)


# 3. SUPERVISED ANALYSIS


df_mod <- data %>%
  select(-User_ID, -cluster) %>%
  mutate(
    female       = as.integer(Gender == "Female"),
    other_gender = as.integer(Gender == "Other"),
    professional = as.integer(Occupation == "Professional"),
    student      = as.integer(Occupation == "Student"),
    business_own = as.integer(Occupation == "Business Owner"),
    ios_user     = as.integer(Device_Type == "iOS")
  ) %>%
  select(-Gender, -Occupation, -Device_Type)

# train/test split 70/30
split_idx <- createDataPartition(df_mod$Work_Productivity_Score, p = 0.7, list = FALSE)
train_df  <- df_mod[split_idx, ]
test_df   <- df_mod[-split_idx, ]

cat("train rows:", nrow(train_df), "  test rows:", nrow(test_df), "\n")

#regression: predict Work_Productivity_Score

# linear regression as a starting point
lm_fit  <- lm(Work_Productivity_Score ~ ., data = train_df)
summary(lm_fit)

lm_pred <- predict(lm_fit, newdata = test_df)
lm_rmse <- sqrt(mean((lm_pred - test_df$Work_Productivity_Score)^2))
lm_r2   <- cor(lm_pred, test_df$Work_Productivity_Score)^2
cat("Linear regression  ->  RMSE:", round(lm_rmse, 3), " R2:", round(lm_r2, 3), "\n")


# random forest 
cat("Running RF regression...\n")
rf_reg <- randomForest(Work_Productivity_Score ~ .,
                       data      = train_df,
                       ntree     = 300,
                       mtry      = floor(sqrt(ncol(train_df) - 1)),
                       importance = TRUE)

rf_pred <- predict(rf_reg, newdata = test_df)
rf_rmse <- sqrt(mean((rf_pred - test_df$Work_Productivity_Score)^2))
rf_r2   <- cor(rf_pred, test_df$Work_Productivity_Score)^2
cat("Random forest      ->  RMSE:", round(rf_rmse, 3), " R2:", round(rf_r2, 3), "\n")


# which variables matter most?
imp_reg <- as.data.frame(importance(rf_reg, type = 1))
imp_reg$variable <- rownames(imp_reg)
colnames(imp_reg)[1] <- "inc_mse"

ggplot(imp_reg, aes(x = reorder(variable, inc_mse), y = inc_mse)) +
  geom_col(fill = "#5B8DB8") +
  coord_flip() +
  labs(title = "Feature importance - productivity (RF)",
       x = "", y = "% increase in MSE when permuted") +
  theme_minimal()

# actual vs predicted
ggplot(data.frame(actual = test_df$Work_Productivity_Score, predicted = rf_pred),
       aes(x = actual, y = predicted)) +
  geom_jitter(alpha = 0.08, color = "#5B8DB8", width = 0.25) +
  geom_abline(slope = 1, intercept = 0, color = "#C1714F") +
  labs(title = "Actual vs predicted - productivity (RF)",
       x = "actual", y = "predicted") +
  theme_minimal()

#classification: high vs low stress

# split stress into two groups using the median (= 6)
# this makes it a binary classification problem

df_cls <- df_mod %>%
  mutate(stress_cat = factor(
    ifelse(Stress_Level >= 6, "high", "low"),
    levels = c("low", "high")
  )) %>%
  select(-Stress_Level, -Work_Productivity_Score)

cat("Class counts:\n")
print(table(df_cls$stress_cat))

train_cls <- df_cls[split_idx, ]
test_cls  <- df_cls[-split_idx, ]


# logistic regression first (simpler, interpretable)
logit_fit  <- glm(stress_cat ~ ., data = train_cls, family = binomial)
summary(logit_fit)

logit_prob <- predict(logit_fit, newdata = test_cls, type = "response")
logit_pred <- factor(ifelse(logit_prob >= 0.5, "high", "low"), levels = c("low", "high"))

cat("Logistic regression confusion matrix:\n")
print(confusionMatrix(logit_pred, test_cls$stress_cat, positive = "high"))


# random forest for classification
cat("Running RF classification...\n")
rf_cls <- randomForest(stress_cat ~ .,
                       data      = train_cls,
                       ntree     = 300,
                       mtry      = floor(sqrt(ncol(train_cls) - 1)),
                       importance = TRUE)

rf_cls_pred <- predict(rf_cls, newdata = test_cls)

cat("Random forest confusion matrix:\n")
cm_out <- confusionMatrix(rf_cls_pred, test_cls$stress_cat, positive = "high")
print(cm_out)


# feature importance 
imp_cls <- as.data.frame(importance(rf_cls, type = 1))
imp_cls$variable <- rownames(imp_cls)
colnames(imp_cls)[1] <- "inc_acc"

ggplot(imp_cls, aes(x = reorder(variable, inc_acc), y = inc_acc)) +
  geom_col(fill = "#C1714F") +
  coord_flip() +
  labs(title = "Feature importance - stress classification (RF)",
       x = "", y = "% decrease in accuracy when permuted") +
  theme_minimal()

# final comparison

cat("Regression results")
cat(sprintf("Linear regression  :  RMSE = %.3f   R2 = %.3f\n", lm_rmse, lm_r2))
cat(sprintf("Random forest      :  RMSE = %.3f   R2 = %.3f\n", rf_rmse, rf_r2))

cat("Classification results")
cat(sprintf("Logistic regression :  accuracy = %.3f\n", mean(logit_pred == test_cls$stress_cat)))
cat(sprintf("Random forest       :  accuracy = %.3f\n", mean(rf_cls_pred == test_cls$stress_cat)))
